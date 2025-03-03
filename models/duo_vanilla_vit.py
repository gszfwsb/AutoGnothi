from typing import List, Tuple

import pydantic
import torch
from torch import Tensor, nn

from ..utils.nnmodel import ObservableModuleMixin
from .shapley import normalize_shapley_explanation
from .vanilla_vit import (
    VanillaViTClassifier,
    VanillaViTConfig,
    VanillaViTLayer,
    VanillaViTModel,
    VanillaViTSurrogate,
)


class DuoVanillaViTConfig(pydantic.BaseModel):
    """Same as Vanilla ViT, except that we train it with dual objectives (both
    classification & explanation)."""

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


class DuoVanillaViTClassifier(VanillaViTClassifier):
    def __init__(self, config: DuoVanillaViTConfig):
        super().__init__(config.into())

    pass


class DuoVanillaViTSurrogate(VanillaViTSurrogate):
    def __init__(self, config: DuoVanillaViTConfig):
        super().__init__(config.into())

    pass


class DuoVanillaViTExplainer(nn.Module, ObservableModuleMixin):
    def __init__(self, config: DuoVanillaViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.vit = VanillaViTModel(config.into())
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

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
        pixel_values: Tensor,
        attention_mask: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output = self.vit(pixel_values, attention_mask)
        self.om_record_features(repr_cls=output, repr_exp=output)

        logits = self.classifier(output[:, 0, :])
        logits = self.act(logits)

        for attn_layer in self.explainer_attn:
            output = attn_layer(output, attention_mask)
        # (batch_size, n_tokens, num_labels)
        output = self.explainer_mlp(output)
        # excluding class token.
        if self.config.explainer_normalize:
            output = normalize_shapley_explanation(
                output, surrogate_grand, surrogate_null
            )
        output = output[:, 1:, :].permute(0, 2, 1)
        return output, logits

    pass


class DuoVanillaViTFinal(nn.Module, ObservableModuleMixin):
    def __init__(self, config: DuoVanillaViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config
        self.surrogate = VanillaViTSurrogate(config.into())
        self.surrogate_null = nn.Parameter(
            torch.zeros((1, self.config.num_labels)), requires_grad=False
        )
        self.explainer = DuoVanillaViTExplainer(config)

    def forward(
        self, pixel_values: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.config.explainer_normalize:
            surrogate_grand = self.surrogate(pixel_values, attention_mask)
            om_repr_srg = self.surrogate.om_take_observations()
        else:
            surrogate_grand = ...
            om_repr_srg = {}
        explainer, logits = self.explainer(
            pixel_values, attention_mask, surrogate_grand, self.surrogate_null
        )
        om_repr_exp = self.explainer.om_take_observations()
        self.om_record_features(
            repr_cls=om_repr_exp.get("repr_cls", None),
            repr_srg=om_repr_srg.get("repr_srg", None),
            repr_exp=om_repr_exp.get("repr_exp", None),
        )
        return logits, explainer

    def om_retain_observations(self, flag: bool = True) -> None:
        ObservableModuleMixin.om_retain_observations(self, flag)
        self.surrogate.om_retain_observations(flag)
        self.explainer.om_retain_observations(flag)

    pass
