import dataclasses
import pathlib
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from ..models.ltt_vit import LttViTConfig, LttViTExplainer, LttViTFinal, LttViTSurrogate
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training
from .vanilla_vit import _fw_xs_preprocess, _gen_input, _gen_null, pre_conv_vit


@dataclasses.dataclass
class LttViTMisc:
    pass


LttViTRecipe = ModelRecipe[
    LttViTConfig,
    LttViTMisc,
    LttViTSurrogate,
    LttViTSurrogate,
    LttViTExplainer,
    LttViTFinal,
]


def ltt_vit_recipe() -> LttViTRecipe:
    return ModelRecipe(
        id="ltt_vit",
        version="beta.1.01",
        t_config=LttViTConfig,
        t_classifier=LttViTSurrogate,
        t_surrogate=LttViTSurrogate,
        t_explainer=LttViTExplainer,
        t_final=LttViTFinal,
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


def _load_misc(m_path: pathlib.Path, cfg: LttViTConfig) -> LttViTMisc:
    return LttViTMisc()


def _conv_pretrained_classifier(cfg: LttViTConfig, model: nn.Module) -> LttViTSurrogate:
    v_classifier = pre_conv_vit(cfg.into(), model)
    rules: MergeStateDictRules = {
        "vit.embeddings.{_}": ...,
        "vit.encoder.layers.{_}": ...,
        "vit.layernorm.{wb}": ...,
        "classifier.{_}": ...,
        New(): "vit.encoder.s_attn_maps.0_{i}.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.attention.self.query.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.attention.self.key.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.attention.self.value.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.attention.output.dense.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.intermediate.dense.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.output.dense.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.layernorm_before.{wb}",
        New(): "vit.encoder.s_attn_layers.0_{i}.layernorm_after.{wb}",
        New(): "vit.s_attn_layernorm.0.{wb}",
        New(): "s_attn_classifier.{wb}",
    }
    classifier = LttViTSurrogate(cfg)
    merge_state_dicts((rules, v_classifier), into=classifier)
    return classifier


def _conv_classifier_surrogate(
    cfg: LttViTConfig, _misc: LttViTMisc, classifier: LttViTSurrogate
) -> LttViTSurrogate:
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
        "s_attn_classifier.{_}": ...,
    }
    surrogate = LttViTSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: LttViTConfig,
    _misc: LttViTMisc,
    surrogate: LttViTSurrogate,
) -> LttViTExplainer:
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,
        "s_attn_classifier.{wb}": None,
        New(): "s_explainer_mlp.0.{wb}",
        New(): "s_explainer_mlp.1.{wb}",
        New(): "s_explainer_mlp.3.{wb}",
        New(): "s_explainer_mlp.5.{wb}",
    }
    explainer = LttViTExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: LttViTConfig,
    misc: LttViTMisc,
    classifier: LttViTSurrogate,
    surrogate: LttViTSurrogate,
    explainer: LttViTExplainer,
) -> LttViTFinal:
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
        # keep original network, all others are side ladders
        "vit.embeddings.cls_token": ...,
        "vit.embeddings.position_embeddings": ...,
        "vit.embeddings.patch_embeddings.projection.{wb}": ...,
        "vit.encoder.layers.{i}.attention.self.query.{wb}": ...,
        "vit.encoder.layers.{i}.attention.self.key.{wb}": ...,
        "vit.encoder.layers.{i}.attention.self.value.{wb}": ...,
        "vit.encoder.layers.{i}.attention.output.dense.{wb}": ...,
        "vit.encoder.layers.{i}.intermediate.dense.{wb}": ...,
        "vit.encoder.layers.{i}.output.dense.{wb}": ...,
        "vit.encoder.layers.{i}.layernorm_before.{wb}": ...,
        "vit.encoder.layers.{i}.layernorm_after.{wb}": ...,
        "vit.layernorm.{wb}": ...,
        "classifier.{wb}": ...,
        # discard side network and side head
        # learning classification on the side network
        "vit.encoder.s_attn_maps.0_{i}.{_}": None,
        "vit.encoder.s_attn_layers.0_{i}.{_}": None,
        "vit.s_attn_layernorm.0.{wb}": None,
        "s_attn_classifier.{wb}": None,
    }
    rules_srg: MergeStateDictRules = {
        "vit.embeddings.{_}": None,
        "vit.encoder.layers.{_}": None,
        "vit.layernorm.{wb}": None,
        "classifier.{_}": None,
        # respect surrogate as side network #0
        "vit.encoder.s_attn_maps.0_{i}.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.attention.self.query.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.attention.self.key.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.attention.self.value.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.attention.output.dense.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.intermediate.dense.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.output.dense.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.layernorm_before.{wb}": ...,
        "vit.encoder.s_attn_layers.0_{i}.layernorm_after.{wb}": ...,
        "vit.s_attn_layernorm.0.{wb}": ...,
        # keep surrogate's side head
        "s_attn_classifier.{wb}": ...,
    }
    rules_exp: MergeStateDictRules = {
        "vit.embeddings.{_}": None,
        "vit.encoder.layers.{_}": None,
        "vit.layernorm.{wb}": None,
        "classifier.{_}": None,
        # respect explainer as side network #1
        "vit.encoder.s_attn_maps.0_{i}.{wb}": "vit.encoder.s_attn_maps.1_{i}.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.attention.self.query.{wb}": "vit.encoder.s_attn_layers.1_{i}.attention.self.query.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.attention.self.key.{wb}": "vit.encoder.s_attn_layers.1_{i}.attention.self.key.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.attention.self.value.{wb}": "vit.encoder.s_attn_layers.1_{i}.attention.self.value.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.attention.output.dense.{wb}": "vit.encoder.s_attn_layers.1_{i}.attention.output.dense.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.intermediate.dense.{wb}": "vit.encoder.s_attn_layers.1_{i}.intermediate.dense.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.output.dense.{wb}": "vit.encoder.s_attn_layers.1_{i}.output.dense.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.layernorm_before.{wb}": "vit.encoder.s_attn_layers.1_{i}.layernorm_before.{wb}",
        "vit.encoder.s_attn_layers.0_{i}.layernorm_after.{wb}": "vit.encoder.s_attn_layers.1_{i}.layernorm_after.{wb}",
        "vit.s_attn_layernorm.0.{wb}": "vit.s_attn_layernorm.1.{wb}",
        # keep explainer's side head
        "s_explainer_attn.{_}": ...,
        "s_explainer_mlp.{_}": ...,
    }
    rules_extra: MergeStateDictRules = {"surrogate_null": ...}

    final = LttViTFinal(cfg)
    merge_state_dicts(
        (rules_cls, classifier),
        (rules_srg, surrogate),
        (rules_exp, explainer),
        (rules_extra, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _fw_classifier(
    model: LttViTSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    side_logits, logits = model(xs, mask)
    return side_logits, logits


def _fw_surrogate(
    model: LttViTSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    side_logits, logits = model(xs, mask)
    return side_logits, logits


def _fw_explainer(
    model: LttViTExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    side_attr, logits = model(xs, mask, surrogate_grand, surrogate_null)
    return side_attr, logits


def _fw_final(model: LttViTFinal, xs: Tensor) -> Tuple[Tensor, Tensor]:
    device = xs.device
    batch_size, *_ = xs.shape
    n_players = (model.config.img_px_size // model.config.img_patch_size) ** 2
    mask = torch.ones((batch_size, 1 + n_players), dtype=torch.long, device=device)
    logits, attr = model(xs, mask)
    return logits, attr
