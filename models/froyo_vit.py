from typing import List, Tuple

import pydantic
import torch
from torch import Tensor, nn
from typing_extensions import Self

from ..utils.nnmodel import ObservableModuleMixin, freeze_model_parameters
from .shapley import normalize_shapley_explanation
from .vanilla_vit import (
    VanillaViTClassifier,
    VanillaViTConfig,
    VanillaViTExplainer,
    VanillaViTLayer,
    VanillaViTModel,
)


class FroyoViTConfig(pydantic.BaseModel):
    """FroYo = Frozen Yoghurt. Everything in the backbone is frozen except for
    the last heads. It is otherwise identical with VanillaViT model."""

    attention_probs_dropout_prob: float
    explainer_attn_num_layers: int
    explainer_head_hidden_size: int
    explainer_normalize: bool
    hidden_dropout_prob: float
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    num_attention_heads: int
    num_hidden_layers: int
    num_labels: int
    img_channels: int
    img_px_size: int
    img_patch_size: int

    @property
    def is_decoder(self) -> bool:
        return False

    def into(self) -> VanillaViTConfig:
        return VanillaViTConfig(
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            explainer_attn_num_layers=self.explainer_attn_num_layers,
            explainer_head_hidden_size=self.explainer_head_hidden_size,
            explainer_normalize=self.explainer_normalize,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_labels=self.num_labels,
            img_channels=self.img_channels,
            img_px_size=self.img_px_size,
            img_patch_size=self.img_patch_size,
        )

    pass


class FroyoViTClassifier(VanillaViTClassifier):
    def __init__(self, config: FroyoViTConfig):
        super().__init__(config.into())

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        freeze_model_parameters(self, "vit")
        freeze_model_parameters(self, "classifier")
        return self

    pass


class FroyoViTSurrogate(VanillaViTClassifier):
    def __init__(self, config: FroyoViTConfig):
        super().__init__(config.into())

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        freeze_model_parameters(self, "vit")
        return self

    pass


class FroyoViTExplainer(VanillaViTExplainer):
    def __init__(self, config: FroyoViTConfig):
        super().__init__(config.into())

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        freeze_model_parameters(self, "vit")
        return self

    pass


class FroyoViTFinal(nn.Module, ObservableModuleMixin):
    def __init__(self, config: FroyoViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.vit = VanillaViTModel(config.into())
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)
        self.srg_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.srg_act = nn.Softmax(dim=-1)
        self.surrogate_null = nn.Parameter(
            torch.zeros((1, self.config.num_labels)), requires_grad=False
        )

        attn_layers: List[nn.Module] = []
        for i_ly in range(config.explainer_attn_num_layers):
            attn_layer = VanillaViTLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                layer_norm_eps=config.layer_norm_eps,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            attn_layers.append(attn_layer)
        self.explainer_attn = nn.ModuleList(attn_layers)

        mlp_max_width = int(config.explainer_head_hidden_size)
        mlp_list: List[nn.Module] = []
        mlp_list.append(nn.LayerNorm(config.hidden_size))
        mlp_list.append(nn.Linear(config.hidden_size, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, config.num_labels))
        self.explainer_mlp = nn.Sequential(*mlp_list)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output = self.vit(x, attention_mask)
        self.om_record_features(repr_cls=output, repr_srg=output, repr_exp=output)

        cls_logits = self.classifier(output[:, 0, :])
        cls_logits = self.act(cls_logits)

        if self.config.explainer_normalize:
            srg_logits = self.srg_classifier(output[:, 0, :])
            srg_logits = self.srg_act(srg_logits)
        else:
            srg_logits = ...

        for attn_layer in self.explainer_attn:
            output = attn_layer(output, attention_mask)
        # (batch_size, n_tokens, num_labels)
        output = self.explainer_mlp(output)
        # excluding class token.
        if self.config.explainer_normalize:
            surrogate_grand = srg_logits
            surrogate_null = self.surrogate_null
            output = normalize_shapley_explanation(
                output, surrogate_grand, surrogate_null
            )
        output = output[:, 1:, :].permute(0, 2, 1)
        return cls_logits, output

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        freeze_model_parameters(self, "vit")
        freeze_model_parameters(self, "classifier")
        return self

    pass
