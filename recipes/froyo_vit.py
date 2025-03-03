import dataclasses
import pathlib
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from ..models.froyo_vit import (
    FroyoViTClassifier,
    FroyoViTConfig,
    FroyoViTExplainer,
    FroyoViTFinal,
    FroyoViTSurrogate,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training
from .vanilla_vit import _fw_xs_preprocess, _gen_input, _gen_null, pre_conv_vit


@dataclasses.dataclass
class FroyoViTMisc:
    pass


FroyoViTRecipe = ModelRecipe[
    FroyoViTConfig,
    FroyoViTMisc,
    FroyoViTClassifier,
    FroyoViTSurrogate,
    FroyoViTExplainer,
    FroyoViTFinal,
]


def froyo_vit_recipe() -> FroyoViTRecipe:
    return ModelRecipe(
        id="froyo_vit",
        version="beta.1.01",
        t_config=FroyoViTConfig,
        t_classifier=FroyoViTClassifier,
        t_surrogate=FroyoViTSurrogate,
        t_explainer=FroyoViTExplainer,
        t_final=FroyoViTFinal,
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


def _load_misc(m_path: pathlib.Path, cfg: FroyoViTConfig) -> FroyoViTMisc:
    return FroyoViTMisc()


def _conv_pretrained_classifier(
    cfg: FroyoViTConfig, model: nn.Module
) -> FroyoViTClassifier:
    v_classifier = pre_conv_vit(cfg.into(), model)
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
    }
    classifier = FroyoViTClassifier(cfg)
    merge_state_dicts((rules, v_classifier), into=classifier)
    return classifier


def _conv_classifier_surrogate(
    cfg: FroyoViTConfig, _misc: FroyoViTMisc, classifier: FroyoViTClassifier
) -> FroyoViTSurrogate:
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
    }
    surrogate = FroyoViTSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: FroyoViTConfig,
    _misc: FroyoViTMisc,
    surrogate: FroyoViTSurrogate,
) -> FroyoViTExplainer:
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": None,
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
    explainer = FroyoViTExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: FroyoViTConfig,
    misc: FroyoViTMisc,
    classifier: FroyoViTClassifier,
    surrogate: FroyoViTSurrogate,
    explainer: FroyoViTExplainer,
) -> FroyoViTFinal:
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

    rules_cls: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
    }
    rules_srg: MergeStateDictRules = {
        "vit.{_}": None,
        "classifier.{_}": "srg_classifier.{_}",
    }
    rules_exp: MergeStateDictRules = {
        "vit.{_}": None,
        "explainer_attn.{_}": ...,
        "explainer_mlp.{_}": ...,
    }
    rules_misc: MergeStateDictRules = {"surrogate_null": ...}

    final = FroyoViTFinal(cfg)
    merge_state_dicts(
        (rules_cls, classifier),
        (rules_srg, surrogate),
        (rules_exp, explainer),
        (rules_misc, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _fw_classifier(
    model: FroyoViTClassifier, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask)
    return logits, logits


def _fw_surrogate(
    model: FroyoViTSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask)
    return logits, None


def _fw_explainer(
    model: FroyoViTExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    attr = model(xs, mask, surrogate_grand, surrogate_null)
    return attr, None


def _fw_final(
    model: FroyoViTFinal,
    xs: Tensor,
) -> Tuple[Tensor, Tensor]:
    device = xs.device
    batch_size, *_ = xs.shape
    n_players = (model.config.img_px_size // model.config.img_patch_size) ** 2
    mask = torch.ones((batch_size, 1 + n_players), dtype=torch.long, device=device)
    logits, attr = model(xs, mask)
    return logits, attr
