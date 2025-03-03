import dataclasses
import pathlib
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from ..models.duo_vanilla_vit import (
    DuoVanillaViTClassifier,
    DuoVanillaViTConfig,
    DuoVanillaViTExplainer,
    DuoVanillaViTFinal,
    DuoVanillaViTSurrogate,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .duo_vanilla_vit_inspect import duo_vanilla_vit_inspect
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training
from .vanilla_vit import _fw_xs_preprocess, _gen_input, _gen_null, pre_conv_vit


@dataclasses.dataclass
class DuoVanillaViTMisc:
    pass


DuoVanillaViTRecipe = ModelRecipe[
    DuoVanillaViTConfig,
    DuoVanillaViTMisc,
    DuoVanillaViTClassifier,
    DuoVanillaViTSurrogate,
    DuoVanillaViTExplainer,
    DuoVanillaViTFinal,
]


def duo_vanilla_vit_recipe() -> DuoVanillaViTRecipe:
    return ModelRecipe(
        id="duo_vanilla_vit",
        version="beta.1.01",
        t_config=DuoVanillaViTConfig,
        t_classifier=DuoVanillaViTClassifier,
        t_surrogate=DuoVanillaViTSurrogate,
        t_explainer=DuoVanillaViTExplainer,
        t_final=DuoVanillaViTFinal,
        load_misc=_load_misc,
        conv_pretrained_classifier=_conv_pretrained_classifier,
        conv_classifier_surrogate=_conv_classifier_surrogate,
        conv_surrogate_explainer=_conv_surrogate_explainer,
        conv_explainer_final=_conv_explainer_final,
        n_players=lambda cfg: (cfg.img_px_size // cfg.img_patch_size) ** 2,
        gen_input=lambda cfg, misc, device: _gen_input(
            img_px_size=cfg.img_px_size,
            img_patch_size=cfg.img_patch_size,
            device=device,
        ),
        gen_null=lambda cfg, misc, device: _gen_null(
            img_px_size=cfg.img_px_size,
            img_patch_size=cfg.img_patch_size,
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
            allow_dual_task_similarity=duo_vanilla_vit_inspect(),
            allow_branches_cka=True,
        ),
    )


def _load_misc(m_path: pathlib.Path, cfg: DuoVanillaViTConfig) -> DuoVanillaViTMisc:
    return DuoVanillaViTMisc()


def _conv_pretrained_classifier(
    cfg: DuoVanillaViTConfig, model: nn.Module
) -> DuoVanillaViTClassifier:
    v_classifier = pre_conv_vit(cfg.into(), model)
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
    }
    classifier = DuoVanillaViTClassifier(cfg)
    merge_state_dicts((rules, v_classifier), into=classifier)
    return classifier


def _conv_classifier_surrogate(
    cfg: DuoVanillaViTConfig,
    _misc: DuoVanillaViTMisc,
    classifier: DuoVanillaViTClassifier,
) -> DuoVanillaViTSurrogate:
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
    }
    surrogate = DuoVanillaViTSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: DuoVanillaViTConfig,
    _misc: DuoVanillaViTMisc,
    surrogate: DuoVanillaViTSurrogate,
) -> DuoVanillaViTExplainer:
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
        New(): "explainer_attn.{i}.attention.self.query.{wb}",
        New(): "explainer_attn.{i}.attention.self.key.{wb}",
        New(): "explainer_attn.{i}.attention.self.value.{wb}",
        New(): "explainer_attn.{i}.attention.output.dense.{wb}",
        New(): "explainer_attn.{i}.intermediate.dense.{wb}",
        New(): "explainer_attn.{i}.output.dense.{wb}",
        New(): "explainer_attn.{i}.layernorm_before.{wb}",
        New(): "explainer_attn.{i}.layernorm_after.{wb}",
        New(): "explainer_mlp.0.{wb}",
        New(): "explainer_mlp.1.{wb}",
        New(): "explainer_mlp.3.{wb}",
        New(): "explainer_mlp.5.{wb}",
    }
    explainer = DuoVanillaViTExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: DuoVanillaViTConfig,
    misc: DuoVanillaViTMisc,
    classifier: DuoVanillaViTClassifier,
    surrogate: DuoVanillaViTSurrogate,
    explainer: DuoVanillaViTExplainer,
) -> DuoVanillaViTFinal:
    # we need to replay the surrogate model
    device = classifier.vit.embeddings.cls_token.device
    n_players = (cfg.img_px_size // cfg.img_patch_size) ** 2
    nil_Xs = _gen_null(
        img_px_size=cfg.img_px_size,
        img_patch_size=cfg.img_patch_size,
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
    final = DuoVanillaViTFinal(cfg)
    merge_state_dicts(
        # (rules_cl, classifier),
        (rules_sr, surrogate),
        (rules_ex, explainer),
        (rules_misc, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _fw_classifier(
    model: DuoVanillaViTClassifier, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask)
    return logits, logits


def _fw_surrogate(
    model: DuoVanillaViTSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask)
    return logits, None


def _fw_explainer(
    model: DuoVanillaViTExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    attr, logits = model(xs, mask, surrogate_grand, surrogate_null)
    return attr, logits


def _fw_final(
    model: DuoVanillaViTFinal,
    xs: Tensor,
) -> Tuple[Tensor, Tensor]:
    device = xs.device
    batch_size, *_ = xs.shape
    n_players = (model.config.img_px_size // model.config.img_patch_size) ** 2
    mask = torch.ones((batch_size, 1 + n_players), dtype=torch.long, device=device)
    logits, attr = model(xs, mask)
    return logits, attr
