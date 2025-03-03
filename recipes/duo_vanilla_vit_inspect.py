from typing import List, Tuple

import torch
from torch import Tensor, nn

from ..models.duo_vanilla_vit import (
    DuoVanillaViTClassifier,
    DuoVanillaViTConfig,
    DuoVanillaViTExplainer,
)
from ..models.shapley import normalize_shapley_explanation
from ..models.vanilla_vit import (
    VanillaViTEmbeddings,
    VanillaViTEncoder,
    VanillaViTLayer,
)
from ..recipes.types import ModelRecipe_Measurements_DualTaskSimilarity
from ..recipes.vanilla_bert import _fw_xs_preprocess
from ..utils.nnmodel import MergeStateDictRules, merge_state_dicts


class InspectingDuoVanillaViTClsExp(nn.Module):
    def __init__(self, config: DuoVanillaViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = VanillaViTEmbeddings(
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            img_px_size=config.img_px_size,
            img_patch_size=config.img_patch_size,
            img_channels=config.img_channels,
        )
        self.rest = InspectingRest(config)

    def forward(
        self,
        pixel_values: Tensor,
        attention_mask: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        hidden_states = self.embeddings(pixel_values)

        def _detach(x: Tensor) -> Tensor:
            device = x.device
            x = torch.tensor(x.detach().cpu().numpy(), device=device)
            return x

        hidden_states = _detach(hidden_states)
        hidden_states.requires_grad = True
        hidden_states.retain_grad()
        attention_mask = _detach(attention_mask)
        surrogate_grand = _detach(surrogate_grand)
        surrogate_null = _detach(surrogate_null)

        return self.rest(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
        )

    pass


class InspectingRest(nn.Module):
    def __init__(self, config: DuoVanillaViTConfig):
        super().__init__()
        self.config = config

        self.encoder = VanillaViTEncoder(
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        hidden_states: Tensor,
        attention_mask: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        output = self.layernorm(hidden_states)

        logits = self.classifier(output)
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
        return logits, output

    pass


def _conv_alt_models(
    m_config: DuoVanillaViTConfig,
    _m_o_classifier: DuoVanillaViTClassifier,
    m_o_explainer: DuoVanillaViTExplainer,
) -> Tuple[InspectingDuoVanillaViTClsExp, InspectingDuoVanillaViTClsExp]:
    rules: MergeStateDictRules = {
        "vit.embeddings.{_}": "embeddings.{_}",
        "vit.encoder.{_}": "rest.encoder.{_}",
        "vit.layernorm.{_}": "rest.layernorm.{_}",
        "classifier.{_}": "rest.classifier.{_}",
        "explainer_attn.{_}": "rest.explainer_attn.{_}",
        "explainer_mlp.{_}": "rest.explainer_mlp.{_}",
    }
    M_explainer = InspectingDuoVanillaViTClsExp(m_config)
    merge_state_dicts((rules, m_o_explainer), into=M_explainer)
    return M_explainer, M_explainer


def _fw_alt(
    _model_cls: InspectingDuoVanillaViTClsExp,
    model_exp: InspectingDuoVanillaViTClsExp,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Tensor]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    attr, logits = model_exp(xs, mask, token_type_ids, surrogate_grand, surrogate_null)
    return logits, attr


def duo_vanilla_vit_inspect() -> ModelRecipe_Measurements_DualTaskSimilarity[
    DuoVanillaViTConfig,
    DuoVanillaViTClassifier,
    DuoVanillaViTExplainer,
    InspectingDuoVanillaViTClsExp,
    InspectingDuoVanillaViTClsExp,
]:
    return ModelRecipe_Measurements_DualTaskSimilarity(
        allow=True,
        t_alt_classifier=InspectingDuoVanillaViTClsExp,
        t_alt_explainer=InspectingDuoVanillaViTClsExp,
        conv_alt_models=_conv_alt_models,
        grad_modules=lambda cls, exp: (cls.rest, exp.rest),
        fw_alt=_fw_alt,
    )
