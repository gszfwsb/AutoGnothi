import dataclasses
import pathlib
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..models.duo_vanilla_bert import (
    DuoVanillaBertClassifier,
    DuoVanillaBertConfig,
    DuoVanillaBertExplainer,
    DuoVanillaBertFinal,
    DuoVanillaBertSurrogate,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .duo_vanilla_bert_inspect import duo_vanilla_bert_inspect
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training
from .vanilla_bert import _fw_xs_preprocess, _gen_input, _gen_null, pre_conv_bert


@dataclasses.dataclass
class DuoVanillaBertMisc:
    tokenizer: PreTrainedTokenizerBase
    pass


DuoVanillaBertRecipe = ModelRecipe[
    DuoVanillaBertConfig,
    DuoVanillaBertMisc,
    DuoVanillaBertClassifier,
    DuoVanillaBertSurrogate,
    DuoVanillaBertExplainer,
    DuoVanillaBertFinal,
]


def duo_vanilla_bert_recipe() -> DuoVanillaBertRecipe:
    return ModelRecipe(
        id="duo_vanilla_bert",
        version="beta.1.01",
        t_config=DuoVanillaBertConfig,
        t_classifier=DuoVanillaBertClassifier,
        t_surrogate=DuoVanillaBertSurrogate,
        t_explainer=DuoVanillaBertExplainer,
        t_final=DuoVanillaBertFinal,
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
            exp_variant_duo=True,
            exp_variant_kernel_shap=False,
        ),
        fw_classifier=_fw_classifier,
        fw_surrogate=_fw_surrogate,
        fw_explainer=_fw_explainer,
        fw_final=_fw_final,
        measurements=ModelRecipe_Measurements(
            verify_final_coherency=False,
            allow_accuracy=True,
            allow_faithfulness=True,
            allow_cls_acc=True,
            allow_performance_cls=True,
            allow_performance_srg_exp=True,
            allow_performance_fin=True,
            allow_train_resources=True,
            allow_dual_task_similarity=duo_vanilla_bert_inspect(),
            allow_branches_cka=True,
        ),
    )


def _load_misc(m_path: pathlib.Path, cfg: DuoVanillaBertConfig) -> DuoVanillaBertMisc:
    tokenizer = AutoTokenizer.from_pretrained(m_path / "tokenizer")
    return DuoVanillaBertMisc(
        tokenizer=tokenizer,
    )


def _conv_pretrained_classifier(
    cfg: DuoVanillaBertConfig, model: nn.Module
) -> DuoVanillaBertClassifier:
    v_classifier = pre_conv_bert(cfg.into(), model)
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,
    }
    classifier = DuoVanillaBertClassifier(cfg)
    merge_state_dicts((rules, v_classifier), into=classifier)
    return classifier


def _conv_classifier_surrogate(
    cfg: DuoVanillaBertConfig,
    _misc: DuoVanillaBertMisc,
    classifier: DuoVanillaBertClassifier,
) -> DuoVanillaBertSurrogate:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,  # re-use pretrained knowledge whenever possible
        # "classifier.{_}": None,
        # New(): "classifier.{_}",
    }
    surrogate = DuoVanillaBertSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: DuoVanillaBertConfig,
    _misc: DuoVanillaBertMisc,
    surrogate: DuoVanillaBertSurrogate,
) -> DuoVanillaBertExplainer:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,
        New(): "explainer_attn.{i}.attention.self.query.{wb}",
        New(): "explainer_attn.{i}.attention.self.key.{wb}",
        New(): "explainer_attn.{i}.attention.self.value.{wb}",
        New(): "explainer_attn.{i}.attention.output.dense.{wb}",
        New(): "explainer_attn.{i}.attention.output.LayerNorm.{wb}",
        New(): "explainer_attn.{i}.intermediate.dense.{wb}",
        New(): "explainer_attn.{i}.output.dense.{wb}",
        New(): "explainer_attn.{i}.output.LayerNorm.{wb}",
        New(): "explainer_mlp.0.{wb}",
        New(): "explainer_mlp.2.{wb}",
        New(): "explainer_mlp.4.{wb}",
    }
    explainer = DuoVanillaBertExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: DuoVanillaBertConfig,
    misc: DuoVanillaBertMisc,
    classifier: DuoVanillaBertClassifier,
    surrogate: DuoVanillaBertSurrogate,
    explainer: DuoVanillaBertExplainer,
) -> DuoVanillaBertFinal:
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

    # rules_cl: MergeStateDictRules = {"{_}": "classifier.{_}"}
    rules_sr: MergeStateDictRules = {"{_}": "surrogate.{_}"}
    rules_ex: MergeStateDictRules = {"{_}": "explainer.{_}"}
    rules_misc: MergeStateDictRules = {"surrogate_null": ...}
    final = DuoVanillaBertFinal(cfg)
    merge_state_dicts(
        # (rules_cl, classifier),
        (rules_sr, surrogate),
        (rules_ex, explainer),
        (rules_misc, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _fw_classifier(
    model: DuoVanillaBertClassifier, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, logits


def _fw_surrogate(
    model: DuoVanillaBertSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, None


def _fw_explainer(
    model: DuoVanillaBertExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits, attr = model(xs, mask, token_type_ids, surrogate_grand, surrogate_null)
    return attr, logits


def _fw_final(
    model: DuoVanillaBertFinal,
    xs: Tensor,
) -> Tuple[Tensor, Tensor]:
    device = xs.device
    mask = torch.ones_like(xs, device=device)
    token_type_ids = torch.zeros_like(xs, device=device)
    logits, attr = model(xs, mask, token_type_ids)
    return logits, attr
