import dataclasses
import pathlib
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..models.ltt_bert import (
    LttBertConfig,
    LttBertExplainer,
    LttBertFinal,
    LttBertSurrogate,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training
from .vanilla_bert import _fw_xs_preprocess, _gen_input, _gen_null, pre_conv_bert


@dataclasses.dataclass
class LttBertMisc:
    tokenizer: PreTrainedTokenizerBase
    pass


LttBertRecipe = ModelRecipe[
    LttBertConfig,
    LttBertMisc,
    LttBertSurrogate,
    LttBertSurrogate,
    LttBertExplainer,
    LttBertFinal,
]


def ltt_bert_recipe() -> LttBertRecipe:
    return ModelRecipe(
        id="ltt_bert",
        version="beta.1.01",
        t_config=LttBertConfig,
        t_classifier=LttBertSurrogate,
        t_surrogate=LttBertSurrogate,
        t_explainer=LttBertExplainer,
        t_final=LttBertFinal,
        load_misc=_load_misc,
        conv_pretrained_classifier=_conv_pretrained_classifier,
        conv_classifier_surrogate=_conv_classifier_surrogate,
        conv_surrogate_explainer=_conv_surrogate_explainer,
        conv_explainer_final=_conv_explainer_final,
        n_players=lambda cfg: cfg.max_position_embeddings - 1,
        gen_input=lambda cfg, misc, device: _gen_input(
            max_position_embeddings=cfg.max_position_embeddings,
            tokenizer=misc.tokenizer,
            device=device,
        ),
        gen_null=lambda cfg, misc, device: _gen_null(
            max_position_embeddings=cfg.max_position_embeddings,
            tokenizer=misc.tokenizer,
            device=device,
        ),
        training=ModelRecipe_Training(
            support_classifier=True,
            support_surrogate=True,
            support_explainer=True,
            exp_variant_duo=False,
            exp_variant_kernel_shap=False,
        ),
        fw_classifier=_fw_classifier,
        fw_surrogate=_fw_surrogate,
        fw_explainer=_fw_explainer,
        fw_final=_fw_final,
        measurements=ModelRecipe_Measurements(
            verify_final_coherency=True,
            allow_accuracy=True,
            allow_faithfulness=True,
            allow_cls_acc=True,
            allow_performance_cls=True,
            allow_performance_srg_exp=True,
            allow_performance_fin=True,
            allow_train_resources=True,
            allow_dual_task_similarity=False,
            allow_branches_cka=True,
        ),
    )


def _load_misc(m_path: pathlib.Path, cfg: LttBertConfig) -> LttBertMisc:
    tokenizer = AutoTokenizer.from_pretrained(m_path / "tokenizer")
    return LttBertMisc(
        tokenizer=tokenizer,
    )


def _conv_pretrained_classifier(
    cfg: LttBertConfig, model: nn.Module
) -> LttBertSurrogate:
    v_classifier = pre_conv_bert(cfg.into(), model)
    rules: MergeStateDictRules = {
        "bert.embeddings.{_}": ...,
        "bert.encoder.layers.{_}": ...,
        "bert_pooler.dense.{wb}": ...,
        "classifier.{wb}": ...,
        New(): "bert.encoder.s_attn_maps.0_{i}.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.attention.self.query.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.attention.self.key.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.attention.self.value.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.attention.output.dense.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.attention.output.LayerNorm.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.intermediate.dense.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.output.dense.{wb}",
        New(): "bert.encoder.s_attn_layers.0_{i}.output.LayerNorm.{wb}",
        New(): "bert_s_attn_pooler.dense.{wb}",
        New(): "s_attn_classifier.{wb}",
    }
    classifier = LttBertSurrogate(cfg)
    merge_state_dicts((rules, v_classifier), into=classifier)
    return classifier


def _conv_classifier_surrogate(
    cfg: LttBertConfig, _misc: LttBertMisc, classifier: LttBertSurrogate
) -> LttBertSurrogate:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,
        "bert_s_attn_pooler.{_}": ...,
        "s_attn_classifier.{_}": ...,
    }
    surrogate = LttBertSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: LttBertConfig,
    _misc: LttBertMisc,
    surrogate: LttBertSurrogate,
) -> LttBertExplainer:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "bert_s_attn_pooler.{_}": None,
        "classifier.{_}": ...,
        New(): "s_attn_attention_layers.{i}.attention.self.query.{wb}",
        New(): "s_attn_attention_layers.{i}.attention.self.key.{wb}",
        New(): "s_attn_attention_layers.{i}.attention.self.value.{wb}",
        New(): "s_attn_attention_layers.{i}.attention.output.dense.{wb}",
        New(): "s_attn_attention_layers.{i}.attention.output.LayerNorm.{wb}",
        New(): "s_attn_attention_layers.{i}.intermediate.dense.{wb}",
        New(): "s_attn_attention_layers.{i}.output.dense.{wb}",
        New(): "s_attn_attention_layers.{i}.output.LayerNorm.{wb}",
        "s_attn_classifier.{wb}": None,
        New(): "s_attn_explainer.0.{wb}",
        New(): "s_attn_explainer.2.{wb}",
        New(): "s_attn_explainer.4.{wb}",
    }
    explainer = LttBertExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: LttBertConfig,
    misc: LttBertMisc,
    classifier: LttBertSurrogate,
    surrogate: LttBertSurrogate,
    explainer: LttBertExplainer,
) -> LttBertFinal:
    # we need to replay the surrogate model
    device = classifier.bert.embeddings.word_embeddings.weight.device
    n_players = cfg.max_position_embeddings - 1
    nil_Xs = _gen_null(
        max_position_embeddings=cfg.max_position_embeddings,
        tokenizer=misc.tokenizer,
        device=device,
    )
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    surrogate.eval()
    with torch.no_grad():
        surrogate_null, _ = _fw_surrogate(surrogate, nil_Xs, nil_mask)

    rules_cls: MergeStateDictRules = {
        # keep original network, all others are side ladders
        "bert.embeddings.word_embeddings.weight": ...,
        "bert.embeddings.position_embeddings.weight": ...,
        "bert.embeddings.token_type_embeddings.weight": ...,
        "bert.embeddings.LayerNorm.{wb}": ...,
        "bert.encoder.layers.{i}.attention.self.query.{wb}": ...,
        "bert.encoder.layers.{i}.attention.self.key.{wb}": ...,
        "bert.encoder.layers.{i}.attention.self.value.{wb}": ...,
        "bert.encoder.layers.{i}.attention.output.dense.{wb}": ...,
        "bert.encoder.layers.{i}.attention.output.LayerNorm.{wb}": ...,
        "bert.encoder.layers.{i}.intermediate.dense.{wb}": ...,
        "bert.encoder.layers.{i}.output.dense.{wb}": ...,
        "bert.encoder.layers.{i}.output.LayerNorm.{wb}": ...,
        "bert_pooler.dense.{wb}": ...,
        "classifier.{wb}": ...,
        # learning classification on the side network
        "bert.encoder.s_attn_maps.0_{i}.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.attention.self.query.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.attention.self.key.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.attention.self.value.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.attention.output.dense.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.attention.output.LayerNorm.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.intermediate.dense.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.output.dense.{wb}": None,
        "bert.encoder.s_attn_layers.0_{i}.output.LayerNorm.{wb}": None,
        # discard classifier's side head
        "bert_s_attn_pooler.dense.{wb}": None,
        "s_attn_classifier.{wb}": None,
    }
    rules_srg: MergeStateDictRules = {
        "bert.embeddings.{_}": None,
        "bert.encoder.layers.{_}": None,
        "bert_pooler.{_}": None,
        "classifier.{_}": None,
        # respect surrogate as side network #0
        "bert.encoder.s_attn_maps.0_{i}.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.attention.self.query.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.attention.self.key.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.attention.self.value.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.attention.output.dense.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.attention.output.LayerNorm.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.intermediate.dense.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.output.dense.{wb}": ...,
        "bert.encoder.s_attn_layers.0_{i}.output.LayerNorm.{wb}": ...,
        # keep surrogate's side head
        "bert_s_attn_pooler.dense.{wb}": ...,
        "s_attn_classifier.{wb}": ...,
    }
    rules_exp: MergeStateDictRules = {
        "bert.embeddings.{_}": None,
        "bert.encoder.layers.{_}": None,
        "bert_pooler.{_}": None,
        "classifier.{_}": None,
        # respect explainer as side network #1
        "bert.encoder.s_attn_maps.0_{i}.{wb}": "bert.encoder.s_attn_maps.1_{i}.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.attention.self.query.{wb}": "bert.encoder.s_attn_layers.1_{i}.attention.self.query.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.attention.self.key.{wb}": "bert.encoder.s_attn_layers.1_{i}.attention.self.key.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.attention.self.value.{wb}": "bert.encoder.s_attn_layers.1_{i}.attention.self.value.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.attention.output.dense.{wb}": "bert.encoder.s_attn_layers.1_{i}.attention.output.dense.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.attention.output.LayerNorm.{wb}": "bert.encoder.s_attn_layers.1_{i}.attention.output.LayerNorm.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.intermediate.dense.{wb}": "bert.encoder.s_attn_layers.1_{i}.intermediate.dense.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.output.dense.{wb}": "bert.encoder.s_attn_layers.1_{i}.output.dense.{wb}",
        "bert.encoder.s_attn_layers.0_{i}.output.LayerNorm.{wb}": "bert.encoder.s_attn_layers.1_{i}.output.LayerNorm.{wb}",
        # keep explainer's side head
        "s_attn_attention_layers.{_}": ...,
        "s_attn_explainer.{_}": ...,
    }
    rules_extra: MergeStateDictRules = {"surrogate_null": ...}

    final = LttBertFinal(cfg)
    merge_state_dicts(
        (rules_cls, classifier),
        (rules_srg, surrogate),
        (rules_exp, explainer),
        (rules_extra, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _fw_classifier(
    model: LttBertSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    side_logits, logits = model(xs, mask, token_type_ids)
    return side_logits, logits


def _fw_surrogate(
    model: LttBertSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    side_logits, logits = model(xs, mask, token_type_ids)
    return side_logits, logits


def _fw_explainer(
    model: LttBertExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    side_attr, logits = model(xs, mask, token_type_ids, surrogate_grand, surrogate_null)
    return side_attr, logits


def _fw_final(model: LttBertFinal, xs: Tensor) -> Tuple[Tensor, Tensor]:
    device = xs.device
    mask = torch.ones_like(xs, device=device)
    token_type_ids = torch.zeros_like(xs, device=device)
    logits, attr = model(xs, mask, token_type_ids)
    return logits, attr
