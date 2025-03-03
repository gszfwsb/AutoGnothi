import dataclasses
import pathlib
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..models.kernel_shap_bert import (
    KernelShapBertClassifier,
    KernelShapBertConfig,
    KernelShapBertExplainer,
    KernelShapBertFinal,
    KernelShapBertSurrogate,
    kernel_shap_torch,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training
from .vanilla_bert import _fw_xs_preprocess, _gen_input, _gen_null, pre_conv_bert


@dataclasses.dataclass
class KernelShapBertMisc:
    tokenizer: PreTrainedTokenizerBase
    pass


KernelShapBertRecipe = ModelRecipe[
    KernelShapBertConfig,
    KernelShapBertMisc,
    KernelShapBertClassifier,
    KernelShapBertSurrogate,
    KernelShapBertExplainer,
    KernelShapBertFinal,
]


def kernel_shap_bert_recipe() -> KernelShapBertRecipe:
    return ModelRecipe(
        id="kernel_shap_bert",
        version="beta.1.01",
        t_config=KernelShapBertConfig,
        t_classifier=KernelShapBertClassifier,
        t_surrogate=KernelShapBertSurrogate,
        t_explainer=KernelShapBertExplainer,
        t_final=KernelShapBertFinal,
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
            support_classifier=False,
            support_surrogate=False,
            support_explainer=True,
            exp_variant_duo=False,
            exp_variant_kernel_shap=True,
        ),
        fw_classifier=_fw_classifier,
        fw_surrogate=_fw_surrogate,
        fw_explainer=_fw_explainer,
        fw_final=_fw_final,
        measurements=ModelRecipe_Measurements(
            verify_final_coherency=False,  # since we have no trained explainer
            allow_accuracy=False,
            allow_faithfulness=True,
            allow_cls_acc=False,
            allow_performance_cls=False,
            allow_performance_srg_exp=False,
            allow_performance_fin=False,
            allow_train_resources=False,
            allow_dual_task_similarity=False,
            allow_branches_cka=False,
        ),
    )


def _load_misc(m_path: pathlib.Path, cfg: KernelShapBertConfig) -> KernelShapBertMisc:
    tokenizer = AutoTokenizer.from_pretrained(m_path / "tokenizer")
    return KernelShapBertMisc(
        tokenizer=tokenizer,
    )


def _conv_pretrained_classifier(
    cfg: KernelShapBertConfig, model: nn.Module
) -> KernelShapBertClassifier:
    v_classifier = pre_conv_bert(cfg.into(), model)
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,
    }
    classifier = KernelShapBertClassifier(cfg)
    merge_state_dicts((rules, v_classifier), into=classifier)
    return classifier


def _conv_classifier_surrogate(
    cfg: KernelShapBertConfig,
    _misc: KernelShapBertMisc,
    classifier: KernelShapBertClassifier,
) -> KernelShapBertSurrogate:
    surrogate = KernelShapBertSurrogate(cfg)
    merge_state_dicts(({"{_}": ...}, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: KernelShapBertConfig,
    _misc: KernelShapBertMisc,
    surrogate: KernelShapBertSurrogate,
) -> KernelShapBertExplainer:
    rules: MergeStateDictRules = {"{_}": None, New(): "Xs_train"}
    explainer = KernelShapBertExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: KernelShapBertConfig,
    _misc: KernelShapBertMisc,
    classifier: KernelShapBertClassifier,
    _surrogate: KernelShapBertSurrogate,
    explainer: KernelShapBertExplainer,
) -> KernelShapBertFinal:
    rules_cls: MergeStateDictRules = {"{_}": "classifier.{_}"}
    rules_exp: MergeStateDictRules = {"Xs_train": "explainer.Xs_train"}
    final = KernelShapBertFinal(cfg)
    merge_state_dicts(
        (rules_cls, classifier),
        (rules_exp, explainer),
        into=final,
    )
    return final


def _fw_classifier(
    model: KernelShapBertClassifier, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, logits


def _fw_surrogate(
    model: KernelShapBertSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, logits


def _fw_explainer(
    model: KernelShapBertExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    raise NotImplementedError("explainer model not available for KernelSHAP")


def _fw_final(model: KernelShapBertFinal, xs: Tensor) -> Tuple[Tensor, Tensor]:
    device = xs.device
    mask = torch.ones_like(xs, device=device)
    token_type_ids = torch.zeros_like(xs, device=device)
    logits = model(xs, mask, token_type_ids)
    attr = kernel_shap_torch(
        fw_classifier=lambda xs: model(xs, mask, token_type_ids),
        Xs_train=model.explainer.Xs_train,
        Xs_explain=xs,
        n_samples=model.config.kernel_shap_n_samples,
        batch_size=xs.shape[0],  # always use this and will never OOM
        silent=False,
    )
    return logits, attr
