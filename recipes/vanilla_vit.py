import dataclasses
import pathlib
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from transformers import ViTForImageClassification

from ..models.vanilla_vit import (
    VanillaViTClassifier,
    VanillaViTConfig,
    VanillaViTExplainer,
    VanillaViTFinal,
    VanillaViTSurrogate,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training


@dataclasses.dataclass
class VanillaViTMisc:
    pass


VanillaViTRecipe = ModelRecipe[
    VanillaViTConfig,
    VanillaViTMisc,
    VanillaViTClassifier,
    VanillaViTSurrogate,
    VanillaViTExplainer,
    VanillaViTFinal,
]


def vanilla_vit_recipe() -> VanillaViTRecipe:
    return ModelRecipe(
        id="vanilla_bert",
        version="beta.1.01",
        t_config=VanillaViTConfig,
        t_classifier=VanillaViTClassifier,
        t_surrogate=VanillaViTSurrogate,
        t_explainer=VanillaViTExplainer,
        t_final=VanillaViTFinal,
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


def _load_misc(m_path: pathlib.Path, cfg: VanillaViTConfig) -> VanillaViTMisc:
    return VanillaViTMisc()


def pre_conv_vit(cfg: VanillaViTConfig, model: nn.Module) -> VanillaViTClassifier:
    rules: MergeStateDictRules = {}
    if isinstance(model, VanillaViTClassifier):
        rules = {"{_}": ...}
    elif isinstance(model, ViTForImageClassification):
        rules = {
            "vit.embeddings.cls_token": ...,
            "vit.embeddings.position_embeddings": ...,
            "vit.embeddings.patch_embeddings.projection.{wb}": ...,
            "vit.encoder.layer.{i}.attention.attention.query.{wb}": "vit.encoder.layers.{i}.attention.self.query.{wb}",
            "vit.encoder.layer.{i}.attention.attention.key.{wb}": "vit.encoder.layers.{i}.attention.self.key.{wb}",
            "vit.encoder.layer.{i}.attention.attention.value.{wb}": "vit.encoder.layers.{i}.attention.self.value.{wb}",
            "vit.encoder.layer.{i}.attention.output.dense.{wb}": "vit.encoder.layers.{i}.attention.output.dense.{wb}",
            "vit.encoder.layer.{i}.intermediate.dense.{wb}": "vit.encoder.layers.{i}.intermediate.dense.{wb}",
            "vit.encoder.layer.{i}.output.dense.{wb}": "vit.encoder.layers.{i}.output.dense.{wb}",
            "vit.encoder.layer.{i}.layernorm_before.{wb}": "vit.encoder.layers.{i}.layernorm_before.{wb}",
            "vit.encoder.layer.{i}.layernorm_after.{wb}": "vit.encoder.layers.{i}.layernorm_after.{wb}",
            "vit.layernorm.{wb}": ...,
            "classifier.{wb}": None,
            New(): "classifier.{wb}",
        }
    classifier = VanillaViTClassifier(cfg)
    merge_state_dicts((rules, model), into=classifier)
    return classifier


def _conv_pretrained_classifier(
    cfg: VanillaViTConfig, model: nn.Module
) -> VanillaViTClassifier:
    classifier = pre_conv_vit(cfg, model)
    return classifier


def _conv_classifier_surrogate(
    cfg: VanillaViTConfig, _misc: VanillaViTMisc, classifier: VanillaViTClassifier
) -> VanillaViTSurrogate:
    rules: MergeStateDictRules = {
        "vit.{_}": ...,
        "classifier.{_}": ...,  # re-use pretrained knowledge whenever possible
        # "classifier.{_}": None,
        # New(): "classifier.{_}",
    }
    surrogate = VanillaViTSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: VanillaViTConfig, _misc: VanillaViTMisc, surrogate: VanillaViTSurrogate
) -> VanillaViTExplainer:
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
    explainer = VanillaViTExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: VanillaViTConfig,
    misc: VanillaViTMisc,
    classifier: VanillaViTClassifier,
    surrogate: VanillaViTSurrogate,
    explainer: VanillaViTExplainer,
) -> VanillaViTFinal:
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

    rules_cls: MergeStateDictRules = {"{_}": "classifier.{_}"}
    rules_srg: MergeStateDictRules = {"{_}": "surrogate.{_}"}
    rules_exp: MergeStateDictRules = {"{_}": "explainer.{_}"}
    rules_misc: MergeStateDictRules = {"surrogate_null": ...}

    final = VanillaViTFinal(cfg)
    merge_state_dicts(
        (rules_cls, classifier),
        (rules_srg, surrogate),
        (rules_exp, explainer),
        (rules_misc, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _gen_input(
    img_px_size: int, img_patch_size: int, device: torch.device
) -> Callable[[Any, Any], Tuple[Tensor, Tensor]]:
    """Collate function for input data
    TODO: ret [0]: xs: <batch_size, n_players, *>
    ret [0]: xs: <batch_size, img_in_channels, img_px_size, img_px_size>
    ret [1]: ys: <batch_size>"""

    def mask_input(raw_xs: List[Tensor], raw_ys: List[int]):
        xs = torch.stack(raw_xs, dim=0).to(device)
        ys = torch.tensor(raw_ys).to(device)
        return xs, ys

    return mask_input


def _gen_null(img_px_size: int, img_patch_size: int, device: torch.device) -> Tensor:
    _ = img_patch_size
    x = torch.zeros((1, 3, img_px_size, img_px_size), device=device)
    return x


def _fw_xs_preprocess(xs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    device = xs.device
    batch_size, *_ = xs.shape  # (batch_size, img_channels, img_px_size, img_px_size)
    mask_cls = torch.ones((batch_size, 1), dtype=mask.dtype, device=device)
    mask = torch.cat([mask_cls, mask], dim=1)
    return xs, mask


def _fw_classifier(
    model: VanillaViTClassifier, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask)
    return logits, logits


def _fw_surrogate(
    model: VanillaViTSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask)
    return logits, None


def _fw_explainer(
    model: VanillaViTExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask = _fw_xs_preprocess(xs, mask)
    attr = model(xs, mask, surrogate_grand, surrogate_null)
    return attr, None


def _fw_final(model: VanillaViTFinal, xs: Tensor) -> Tuple[Tensor, Tensor]:
    device = xs.device
    batch_size, *_ = xs.shape
    n_players = (model.config.img_px_size // model.config.img_patch_size) ** 2
    mask = torch.ones((batch_size, 1 + n_players), dtype=torch.long, device=device)
    logits, attr = model(xs, mask)
    return logits, attr
