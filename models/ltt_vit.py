from typing import Dict, List, Tuple, cast

import pydantic
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import Self

from ..utils.nnmodel import ObservableModuleMixin, freeze_model_parameters
from .shapley import normalize_shapley_explanation
from .vanilla_vit import VanillaViTConfig, VanillaViTEmbeddings, VanillaViTLayer


class LttViTConfig(pydantic.BaseModel):
    """differs from `transformers.ViTModel`"""

    attention_probs_dropout_prob: float
    explainer_s_attn_num_layers: int  # side head
    explainer_s_head_hidden_size: int  # side head
    explainer_normalize: bool  # side head
    hidden_dropout_prob: float
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    num_attention_heads: int
    num_hidden_layers: int
    num_labels: int
    s_attn_hidden_size: int  # side attention
    s_attn_intermediate_size: int  # side attention
    img_channels: int
    img_px_size: int
    img_patch_size: int

    def into(self) -> VanillaViTConfig:
        return VanillaViTConfig(
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            explainer_attn_num_layers=self.explainer_s_attn_num_layers,
            explainer_head_hidden_size=self.explainer_s_head_hidden_size,
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


class LttViTSurrogate(nn.Module, ObservableModuleMixin):
    def __init__(self, config: LttViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.vit = LttViTModel(config, num_side_branches=1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

        self.s_attn_classifier = nn.Linear(config.s_attn_hidden_size, config.num_labels)
        self.s_attn_act = nn.Softmax(dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_model_parameters(self, "vit.embeddings")
        freeze_model_parameters(self, "vit.encoder.layers")
        freeze_model_parameters(self, "vit.layernorm")
        freeze_model_parameters(self, "classifier")
        return self

    def ltt_freeze_layers_until(self, layer_id: int) -> None:
        self.vit.encoder.ltt_freeze_layers_until(layer_id)

    def forward(
        self, pixel_values: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        output, (srg_output,) = self.vit(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            side_layer_branches=[0],
        )
        self.om_record_features(repr_cls=output, repr_srg=srg_output)  # no worries
        logits = self.classifier(output[:, 0, :])
        logits = self.act(logits)
        srg_logits = self.s_attn_classifier(srg_output[:, 0, :])
        srg_logits = self.s_attn_act(srg_logits)
        return srg_logits, logits

    pass


class LttViTExplainer(nn.Module, ObservableModuleMixin):
    def __init__(self, config: LttViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.vit = LttViTModel(config, num_side_branches=1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

        s_explainer_attn: List[nn.Module] = []
        for i_ly in range(config.explainer_s_attn_num_layers):
            layer = VanillaViTLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.s_attn_hidden_size,
                intermediate_size=config.s_attn_intermediate_size,
                num_attention_heads=config.num_attention_heads,
                layer_norm_eps=config.layer_norm_eps,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            s_explainer_attn.append(layer)
        self.s_explainer_attn = nn.ModuleList(s_explainer_attn)

        mlp_max_width = int(config.explainer_s_head_hidden_size)
        mlp_list: List[nn.Module] = []
        mlp_list.append(nn.LayerNorm(config.s_attn_hidden_size))
        mlp_list.append(nn.Linear(config.s_attn_hidden_size, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, config.num_labels))
        self.s_explainer_mlp = nn.Sequential(*mlp_list)

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_model_parameters(self, "vit.embeddings")
        freeze_model_parameters(self, "vit.encoder.layers")
        freeze_model_parameters(self, "vit.layernorm")
        freeze_model_parameters(self, "classifier")
        return self

    def ltt_freeze_layers_until(self, layer_id: int) -> None:
        self.vit.encoder.ltt_freeze_layers_until(layer_id)

    def forward(
        self,
        pixel_values: Tensor,
        attention_mask: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        inp [ ]: pixel_values:    <batch_size, img_channels, img_px_size, img_px_size>
        inp [ ]: attention_mask:  <batch_size, n_tokens = 1 + n_players>
                                               n_players = (img_px_size / img_patch_size)^2
        inp [ ]: surrogate_grand: <batch_size, n_classes>
        inp [ ]: surrogate_null:  <1, n_classes>
        ret [0]: output:          <batch_size, n_players, n_classes>
        ret [1]: logits:          <batch_size, n_classes>
        """

        output, (exp_output,) = self.vit(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            side_layer_branches=[0],
        )
        self.om_record_features(repr_cls=output, repr_exp=exp_output)
        logits = self.classifier(output[:, 0, :])
        logits = self.act(logits)

        for layer in self.s_explainer_attn:
            exp_output = layer(exp_output, attention_mask)
        # <batch_size, n_tokens, n_classes>
        exp_output = self.s_explainer_mlp(exp_output)
        if self.config.explainer_normalize:
            exp_output = normalize_shapley_explanation(
                exp_output, surrogate_grand, surrogate_null
            )

        # excluding class token
        exp_output = exp_output[:, 1:, :].permute(0, 2, 1)

        return exp_output, logits

    pass


class LttViTFinal(nn.Module, ObservableModuleMixin):
    def __init__(self, config: LttViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config
        self.vit = LttViTModel(config, num_side_branches=2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

        self.s_attn_classifier = nn.Linear(config.s_attn_hidden_size, config.num_labels)
        self.s_attn_act = nn.Softmax(dim=-1)
        self.surrogate_null = nn.Parameter(
            torch.zeros((1, self.config.num_labels)), requires_grad=False
        )

        s_explainer_attn: List[nn.Module] = []
        for i_ly in range(config.explainer_s_attn_num_layers):
            layer = VanillaViTLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.s_attn_hidden_size,
                intermediate_size=config.s_attn_intermediate_size,
                num_attention_heads=config.num_attention_heads,
                layer_norm_eps=config.layer_norm_eps,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            s_explainer_attn.append(layer)
        self.s_explainer_attn = nn.ModuleList(s_explainer_attn)

        mlp_max_width = int(config.explainer_s_head_hidden_size)
        mlp_list: List[nn.Module] = []
        mlp_list.append(nn.LayerNorm(config.s_attn_hidden_size))
        mlp_list.append(nn.Linear(config.s_attn_hidden_size, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, config.num_labels))
        self.s_explainer_mlp = nn.Sequential(*mlp_list)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        freeze_model_parameters(self, ...)
        return self

    def forward(
        self,
        pixel_values: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        inp [ ]: pixel_values:    <batch_size, img_channels, img_px_size, img_px_size>
        inp [ ]: attention_mask:  <batch_size, n_tokens = 1 + n_players>
                                               n_players = (img_px_size / img_patch_size)^2
        ret [0]: logits:          <batch_size, n_classes>
        ret [1]: output:          <batch_size, n_players, n_classes>
        """

        if self.config.explainer_normalize:
            output, (srg_output, exp_output) = self.vit(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                side_layer_branches=[0, 1],
            )
            self.om_record_features(
                repr_cls=output, repr_srg=srg_output, repr_exp=exp_output
            )
        else:
            output, (exp_output,) = self.vit(
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                side_layer_branches=[1],
            )
            srg_output = ...
            self.om_record_features(repr_cls=output, repr_exp=exp_output)

        logits = self.classifier(output[:, 0, :])
        logits = self.act(logits)

        if self.config.explainer_normalize:
            srg_logits = self.s_attn_classifier(srg_output[:, 0, :])
            srg_logits = self.s_attn_act(srg_logits)
            surrogate_grand = srg_logits
            surrogate_null = self.surrogate_null
        else:
            surrogate_grand = ...
            surrogate_null = ...

        for layer in self.s_explainer_attn:
            exp_output = layer(exp_output, attention_mask)
        # <batch_size, n_tokens, n_classes>
        exp_output = self.s_explainer_mlp(exp_output)
        if self.config.explainer_normalize:
            exp_output = normalize_shapley_explanation(
                exp_output, surrogate_grand, surrogate_null
            )
        # excluding class token
        exp_output = exp_output[:, 1:, :].permute(0, 2, 1)

        return logits, exp_output

    pass


class LttViTModel(nn.Module):
    def __init__(self, config: LttViTConfig, num_side_branches: int):
        super().__init__()
        self.config = config
        self.num_side_branches = num_side_branches
        self.embeddings = VanillaViTEmbeddings(
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            img_px_size=config.img_px_size,
            img_patch_size=config.img_patch_size,
            img_channels=config.img_channels,
        )
        self.encoder = LttViTMultiEncoder(
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_side_branches=self.num_side_branches,
            s_attn_hidden_size=config.s_attn_hidden_size,
            s_attn_intermediate_size=config.s_attn_intermediate_size,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        s_attn_layernorm: List[nn.Module] = []
        for _ in range(num_side_branches):
            s_attn_layernorm.append(
                nn.LayerNorm(config.s_attn_hidden_size, eps=config.layer_norm_eps)
            )
        self.s_attn_layernorm = nn.ModuleList(s_attn_layernorm)

    def forward(
        self,
        pixel_values: Tensor,
        attention_mask: Tensor,
        side_layer_branches: List[int],
    ) -> Tuple[Tensor, List[Tensor]]:
        hidden_states = self.embeddings(pixel_values)
        hidden_states, side_layer_outputs = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            side_layer_branches=side_layer_branches,
        )
        hidden_states = self.layernorm(hidden_states)
        for _, i_b in enumerate(side_layer_branches):
            side_layer_outputs[_] = self.s_attn_layernorm[i_b](side_layer_outputs[_])
        return hidden_states, side_layer_outputs

    pass


class LttViTMultiEncoder(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_side_branches: int,
        s_attn_hidden_size: int,
        s_attn_intermediate_size: int,
    ):
        super().__init__()
        self.num_layers = num_hidden_layers
        self.num_branches = num_side_branches

        layers: List[nn.Module] = []
        for _ in range(num_hidden_layers):
            layer = VanillaViTLayer(
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
                num_attention_heads=num_attention_heads,
                repl_norm_1_ident=False,
                repl_norm_2_ident=False,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        s_attn_maps: Dict[str, nn.Module] = {}
        for i_b in range(num_side_branches):
            for i_ly in range(num_hidden_layers):
                layer = nn.Linear(hidden_size, s_attn_hidden_size)
                s_attn_maps[f"{i_b}_{i_ly}"] = layer
        self.s_attn_maps = nn.ModuleDict(s_attn_maps)

        s_attn_layers: Dict[str, nn.Module] = {}
        for i_b in range(num_side_branches):
            for i_ly in range(num_hidden_layers):
                layer = VanillaViTLayer(
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    hidden_dropout_prob=hidden_dropout_prob,
                    hidden_size=s_attn_hidden_size,
                    intermediate_size=s_attn_intermediate_size,
                    layer_norm_eps=layer_norm_eps,
                    num_attention_heads=num_attention_heads,
                    repl_norm_1_ident=False,
                    repl_norm_2_ident=False,
                )
                s_attn_layers[f"{i_b}_{i_ly}"] = layer
        self.s_attn_layers = nn.ModuleDict(s_attn_layers)

        # training hacks
        self._ltt_freeze_layer = num_hidden_layers

    def ltt_freeze_layers_until(self, layer_id: int) -> None:
        layer_id = max(1, min(len(self.layers), layer_id))
        self._ltt_freeze_layer = layer_id
        return

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        side_layer_branches: List[int],
    ) -> Tuple[Tensor, List[Tensor]]:
        side_states: List[Tensor] = []
        for i_b in range(self.num_branches):
            side_state: Tensor = cast(Tensor, None)
            if i_b in side_layer_branches:
                side_state = cast(Tensor, 0.0)  # so initial assignment works
            side_states.append(side_state)

        for i_ly in range(self.num_layers):
            hidden_states = self.layers[i_ly](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            for i_b in range(self.num_branches):
                if i_ly >= self._ltt_freeze_layer:
                    continue  # skip following layers
                if i_b not in side_layer_branches:
                    continue
                s_attn_map = self.s_attn_maps[f"{i_b}_{i_ly}"]
                side_states[i_b] = side_states[i_b] + F.gelu(s_attn_map(hidden_states))
                s_attn_layer = self.s_attn_layers[f"{i_b}_{i_ly}"]
                side_states[i_b] = s_attn_layer(
                    hidden_states=side_states[i_b],
                    attention_mask=attention_mask,
                )

        # so that side_states[i] ~= sorted(side_layer_branches)[i]
        side_states = [s for s in side_states if s is not None]
        return hidden_states, side_states

    pass

