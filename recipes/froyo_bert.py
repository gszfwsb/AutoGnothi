import dataclasses
import pathlib
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..models.froyo_bert import (
    FroyoBertClassifier,
    FroyoBertConfig,
    FroyoBertExplainer,
    FroyoBertFinal,
    FroyoBertSurrogate,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training
from .vanilla_bert import _fw_xs_preprocess, _gen_input, _gen_null, pre_conv_bert


@dataclasses.dataclass
class FroyoBertMisc:
    tokenizer: PreTrainedTokenizerBase
    pass


FroyoBertRecipe = ModelRecipe[
    FroyoBertConfig,
    FroyoBertMisc,
    FroyoBertClassifier,
    FroyoBertSurrogate,
    FroyoBertExplainer,
    FroyoBertFinal,
]


def froyo_bert_recipe() -> FroyoBertRecipe:
    return ModelRecipe(
        id="froyo_bert",
        version="beta.1.01",
        t_config=FroyoBertConfig,
        t_classifier=FroyoBertClassifier,
        t_surrogate=FroyoBertSurrogate,
        t_explainer=FroyoBertExplainer,
        t_final=FroyoBertFinal,
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


def _load_misc(m_path: pathlib.Path, cfg: FroyoBertConfig) -> FroyoBertMisc:
    tokenizer = AutoTokenizer.from_pretrained(m_path / "tokenizer")
    return FroyoBertMisc(
        tokenizer=tokenizer,
    )


def _conv_pretrained_classifier(
    cfg: FroyoBertConfig, model: nn.Module
) -> FroyoBertClassifier:
    v_classifier = pre_conv_bert(cfg.into(), model)
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,
    }
    classifier = FroyoBertClassifier(cfg)
    merge_state_dicts((rules, v_classifier), into=classifier)
    return classifier


def _conv_classifier_surrogate(
    cfg: FroyoBertConfig, _misc: FroyoBertMisc, classifier: FroyoBertClassifier
) -> FroyoBertSurrogate:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,
    }
    surrogate = FroyoBertSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: FroyoBertConfig,
    _misc: FroyoBertMisc,
    surrogate: FroyoBertSurrogate,
) -> FroyoBertExplainer:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": None,
        "classifier.{_}": None,
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
    explainer = FroyoBertExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: FroyoBertConfig,
    misc: FroyoBertMisc,
    classifier: FroyoBertClassifier,
    surrogate: FroyoBertSurrogate,
    explainer: FroyoBertExplainer,
) -> FroyoBertFinal:
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
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,
    }
    rules_srg: MergeStateDictRules = {
        "bert.{_}": None,
        "bert_pooler.{_}": "srg_bert_pooler.{_}",
        "classifier.{_}": "srg_classifier.{_}",
    }
    rules_exp: MergeStateDictRules = {
        "bert.{_}": None,
        "explainer_attn.{_}": ...,
        "explainer_mlp.{_}": ...,
    }
    rules_misc: MergeStateDictRules = {"surrogate_null": ...}

    final = FroyoBertFinal(cfg)
    merge_state_dicts(
        (rules_cls, classifier),
        (rules_srg, surrogate),
        (rules_exp, explainer),
        (rules_misc, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _fw_classifier(
    model: FroyoBertClassifier, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, logits


def _fw_surrogate(
    model: FroyoBertSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, None


def _fw_explainer(
    model: FroyoBertExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    attr = model(xs, mask, token_type_ids, surrogate_grand, surrogate_null)
    return attr, None


def _fw_final(
    model: FroyoBertFinal,
    xs: Tensor,
) -> Tuple[Tensor, Tensor]:
    device = xs.device
    mask = torch.ones_like(xs, device=device)
    token_type_ids = torch.zeros_like(xs, device=device)
    logits, attr = model(xs, mask, token_type_ids)
    return logits, attr
