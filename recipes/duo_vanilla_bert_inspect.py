from typing import List, Tuple, cast

import torch
from torch import Tensor, nn
from transformers.modeling_utils import ModuleUtilsMixin

from ..models.duo_vanilla_bert import (
    DuoVanillaBertClassifier,
    DuoVanillaBertConfig,
    DuoVanillaBertExplainer,
)
from ..models.shapley import normalize_shapley_explanation
from ..models.vanilla_bert import (
    VanillaBertEmbeddings,
    VanillaBertEncoder,
    VanillaBertLayer,
    VanillaBertPooler,
)
from ..recipes.types import ModelRecipe_Measurements_DualTaskSimilarity
from ..recipes.vanilla_bert import _fw_xs_preprocess
from ..utils.nnmodel import MergeStateDictRules, merge_state_dicts


class InspectingDuoVanillaBertClsExp(nn.Module):
    def __init__(self, _config: DuoVanillaBertConfig):
        super().__init__()
        config = _config.into()
        self.config = config
        self.bert_embeddings = VanillaBertEmbeddings(
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            layer_norm_eps=config.layer_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            vocab_size=config.vocab_size,
        )
        self.rest = InspectingRest(_config)

    def forward(
        self,
        xs: Tensor,
        mask: Tensor,
        token_type_ids: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        input_shape = cast(Tuple[int], xs.size())
        embedding_output = self.bert_embeddings(
            input_ids=xs,
            token_type_ids=token_type_ids,
        )

        def _detach(x: Tensor) -> Tensor:
            device = x.device
            x = torch.tensor(x.detach().cpu().numpy(), device=device)
            return x

        embedding_output = _detach(embedding_output)
        embedding_output.requires_grad = True
        embedding_output.retain_grad()
        mask = _detach(mask)
        token_type_ids = _detach(token_type_ids)
        surrogate_grand = _detach(surrogate_grand)
        surrogate_null = _detach(surrogate_null)

        return self.rest(
            input_shape=input_shape,
            embedding_output=embedding_output,
            attention_mask=mask,
            surrogate_grand=surrogate_grand,
            surrogate_null=surrogate_null,
        )

    pass


class InspectingRest(nn.Module, ModuleUtilsMixin):
    def __init__(self, _config: DuoVanillaBertConfig):
        super().__init__()
        config = _config.into()
        self.config = config

        self.bert_encoder = VanillaBertEncoder(
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
        )
        self.bert_pooler = VanillaBertPooler(hidden_size=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        attn_layers: List[nn.Module] = []
        for i_ly in range(config.explainer_attn_num_layers):
            attn_layer = VanillaBertLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_norm_eps=config.layer_norm_eps,
                num_attention_heads=config.num_attention_heads,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            attn_layers.append(attn_layer)
        self.explainer_attn = nn.ModuleList(attn_layers)

        self.explainer_dropout = nn.Dropout(config.hidden_dropout_prob)
        mlp_max_width = int(config.explainer_head_hidden_size)
        mlp_layers: List[nn.Module] = []
        mlp_layers.append(nn.Linear(config.hidden_size, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, config.num_labels))
        self.explainer_mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        input_shape: Tuple[int],
        embedding_output: Tensor,
        attention_mask: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        output = self.bert_encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
        )
        logits = self.bert_pooler(output)
        logits = self.dropout(logits)
        logits = self.classifier(logits)

        for attn_layer in self.explainer_attn:
            output = attn_layer(output, extended_attention_mask)
        output = self.explainer_dropout(output)
        # (batch_size, all_tokens, num_labels)
        output = self.explainer_mlp(output)
        # (batch_size, num_labels=2, num_tokens - 1)
        # excluding class token.
        if self.config.explainer_normalize:
            output = normalize_shapley_explanation(
                output, surrogate_grand, surrogate_null
            )
        output = output[:, 1:, :].permute(0, 2, 1)
        return output, logits

    pass


def _conv_alt_models(
    m_config: DuoVanillaBertConfig,
    _m_o_classifier: DuoVanillaBertClassifier,
    m_o_explainer: DuoVanillaBertExplainer,
) -> Tuple[InspectingDuoVanillaBertClsExp, InspectingDuoVanillaBertClsExp]:
    rules: MergeStateDictRules = {
        "bert.embeddings.{_}": "bert_embeddings.{_}",
        "bert.encoder.{_}": "rest.bert_encoder.{_}",
        "bert_pooler.{_}": "rest.bert_pooler.{_}",
        "classifier.{_}": "rest.classifier.{_}",
        "explainer_attn.{_}": "rest.explainer_attn.{_}",
        "explainer_mlp.{_}": "rest.explainer_mlp.{_}",
    }
    M_explainer = InspectingDuoVanillaBertClsExp(m_config)
    merge_state_dicts((rules, m_o_explainer), into=M_explainer)
    return M_explainer, M_explainer


def _fw_alt(
    _model_cls: InspectingDuoVanillaBertClsExp,
    model_exp: InspectingDuoVanillaBertClsExp,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Tensor]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    attr, logits = model_exp(xs, mask, token_type_ids, surrogate_grand, surrogate_null)
    return logits, attr


def duo_vanilla_bert_inspect() -> ModelRecipe_Measurements_DualTaskSimilarity[
    DuoVanillaBertConfig,
    DuoVanillaBertClassifier,
    DuoVanillaBertExplainer,
    InspectingDuoVanillaBertClsExp,
    InspectingDuoVanillaBertClsExp,
]:
    return ModelRecipe_Measurements_DualTaskSimilarity(
        allow=True,
        t_alt_classifier=InspectingDuoVanillaBertClsExp,
        t_alt_explainer=InspectingDuoVanillaBertClsExp,
        conv_alt_models=_conv_alt_models,
        grad_modules=lambda cls, exp: (cls.rest, exp.rest),
        fw_alt=_fw_alt,
    )
