from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pydantic
import torch
from torch import Tensor

from ..datasets.loader import DatasetLoader
from ..recipes.types import ModelRecipe, TClassifier, TConfig, TFinal, TMisc
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model


class FaithfulnessCurve(pydantic.BaseModel):
    auc: float
    avg: Dict[int, float]
    std: Dict[int, float]


CurvePoint = Dict[int, Dict[int, float]]  # (i ->) cls -> stop -> accuracy


class MeasureFaithfulnessReport(pydantic.BaseModel):
    """Reports the faithfulness of the final model's explanation. Types are:

    Curve: { [stop: int]: (metric: float) };
    AUC:   (auc: float);
    All:   { [stop: int]: [i]: (acc: float) };

    Requires: classifier [-1], final [-1]."""

    insertion: FaithfulnessCurve
    deletion: FaithfulnessCurve
    insertion_non_ok: FaithfulnessCurve
    deletion_non_ok: FaithfulnessCurve
    data_cls: List[int]
    data_ins: List[CurvePoint]
    data_del: List[CurvePoint]


def measure_faithfulness(
    env: ExpEnv,
    device: torch.device,
    d_loader: Optional[DatasetLoader],
    resolution: Optional[int],  # how many samples in insertion / deletion
) -> MeasureFaithfulnessReport:
    env.log("loading final model...")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.measurements.allow_faithfulness:
        raise ValueError("unsupported recipe action")
    infer_exp, infer_fn = _infer_perturbed_factory(
        env=env,
        m_recipe=m_recipe,
        m_config=m_config,
        device=device,
    )

    if d_loader is None:
        env.log("loading dataset...")
        d_config = (
            config.eval_faithfulness.dataset
            if config.eval_faithfulness.dataset is not None
            else config.dataset
        )
        d_loader = load_cfg_dataset(d_config, env.model_path)
    if resolution is None:
        resolution = config.eval_faithfulness.resolution

    env.log("[[[ running measurement... ]]]")
    ok_cls_l: List[int] = []
    ins_curves: List[CurvePoint] = []
    del_curves: List[CurvePoint] = []
    for i, (_inputs, _targets) in enumerate(d_loader.test(1)):
        ok_cls, explanation = infer_exp(_inputs, _targets)
        ok_cls_l.append(ok_cls)
        ins_curve = infer_fn(_inputs, _targets, ok_cls, explanation, resolution, 0)
        ins_curves.append(ins_curve)
        ins_val = [_auc(curve) for curve in ins_curve.values()]
        del_curve = infer_fn(_inputs, _targets, ok_cls, explanation, resolution, 1)
        del_curves.append(del_curve)
        del_val = [_auc(curve) for curve in del_curve.values()]
        env.log(
            f"> sample {i}: ok_cls {ok_cls}, ins^ {ins_val[ok_cls]:.6f}, del^ {del_val[ok_cls]:.6f}"
        )

    def _paint_curve(curves: List[Dict[int, float]]) -> FaithfulnessCurve:
        items: Dict[int, List[float]] = {}  # stop -> [metric]
        for curve in curves:
            for cl, point in curve.items():
                if cl not in items:
                    items[cl] = []
                items[cl].append(point)
        all_avg: Dict[int, float] = {}
        all_std: Dict[int, float] = {}
        for stop, metrics in items.items():
            metrics = np.array(metrics)
            all_avg[stop] = float(np.mean(metrics))
            all_std[stop] = float(np.std(metrics))
        all_auc = np.array(list(all_avg.values()))
        all_auc = ((all_auc[1:] + all_auc[:-1]) / 2).mean()  # same
        all_auc = float(all_auc.item())
        return FaithfulnessCurve(
            auc=all_auc,
            avg=all_avg,
            std=all_std,
        )

    cv_ins_ok: List[Dict[int, float]] = []
    cv_del_ok: List[Dict[int, float]] = []
    cv_ins_nok: List[Dict[int, float]] = []
    cv_del_nok: List[Dict[int, float]] = []
    for ok_cls, ins_curve, del_curve in zip(ok_cls_l, ins_curves, del_curves):
        for cl in ins_curve.keys():
            if cl == ok_cls:
                cv_ins_ok.append(ins_curve[cl])
                cv_del_ok.append(del_curve[cl])
            else:
                cv_ins_nok.append(ins_curve[cl])
                cv_del_nok.append(del_curve[cl])
    st_ins_ok = _paint_curve(cv_ins_ok)
    st_del_ok = _paint_curve(cv_del_ok)
    st_ins_nok = _paint_curve(cv_ins_nok)
    st_del_nok = _paint_curve(cv_del_nok)

    info = "FINAL RESULTS:\n"
    info += (
        f"  > insertion: target {st_ins_ok.auc:.6f}, non-target {st_ins_nok.auc:.6f}\n"
    )
    info += f"  > deletion: target {st_del_ok.auc:.6f}, non-target {st_del_nok.auc:.6f}"
    env.log(info)
    return MeasureFaithfulnessReport(
        insertion=st_ins_ok,
        deletion=st_del_ok,
        insertion_non_ok=st_ins_nok,
        deletion_non_ok=st_del_nok,
        data_cls=ok_cls_l,
        data_ins=ins_curves,
        data_del=del_curves,
    )


def _auc(curve: Dict[int, float]) -> float:
    all_vals = np.array(list(curve.values()))
    auc = ((all_vals[1:] + all_vals[:-1]) / 2).mean()
    return float(auc.item())


def _infer_perturbed_factory(
    env: ExpEnv,
    m_recipe: ModelRecipe[TConfig, TMisc, TClassifier, Any, Any, TFinal],
    m_config: TConfig,
    device: torch.device,
) -> Tuple[
    Callable[[Any, Any], Tuple[int, Tensor]],
    Callable[[Any, Any, int, Tensor, int, int], CurvePoint],
]:
    """
    ret [0]: (_inputs, _targets) -> ok_cls: int, explanation: Tensor
    ret [1]: (_inputs, _targets, ok_cls: int, explanation: Tensor, steps: int, mask_base: int)
               -> { [cls: int]: [stop: int]: metric: float }
    """
    config = env.config

    _, m_classifier = load_epoch_model(env, m_recipe, "classifier", device=device)
    _, m_surrogate = load_epoch_model(env, m_recipe, "surrogate", device=device)
    _epoch_final, m_final = load_epoch_model(env, m_recipe, "final", device=device)

    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    batch_size = config.eval_faithfulness.batch_size

    def _explain(_inputs: Any, _targets: Any) -> Tuple[int, Tensor]:
        Xs, Zs = gen_input(_inputs, _targets)
        m_final.eval()
        with torch.no_grad():
            _logits, exp_output = m_recipe.fw_final(m_final, Xs)
        ok_cls = int(Zs.item())
        return ok_cls, exp_output

    def _infer(
        _inputs: Any,
        _targets: Any,
        ok_cls: int,
        explanation: Tensor,
        steps: int,
        mask_base: int,
    ) -> CurvePoint:
        Xs, _Zs = gen_input(_inputs, _targets)
        # for each class...
        _batch_size_1, n_classes, *_ = explanation.shape

        result: Dict[int, Dict[int, float]] = {}
        for i_cls in range(n_classes):
            attr = explanation[0, i_cls]
            stops, masks = _get_perturbed_samples(
                explanations=attr,
                n_players=n_players,
                steps=steps,
                mask_base=mask_base,
            )
            stops, masks = stops.to(device), masks.to(device)

            ret: Dict[int, float] = {}
            for offset in range(0, len(stops), batch_size):
                b_stops = stops[offset : offset + batch_size]
                b_masks = masks[offset : offset + batch_size]
                b_size = b_stops.shape[0]
                b_Xs = torch.repeat_interleave(Xs, b_size, dim=0)
                with torch.no_grad():
                    # _, b_Ys = m_recipe.fw_classifier(m_classifier, b_Xs, b_masks)
                    b_Ys, _ = m_recipe.fw_surrogate(m_surrogate, b_Xs, b_masks)
                    # b_Ys = F.softmax(b_Ys, dim=1)
                b_Ysc = b_Ys[:, i_cls]
                for i in range(b_size):
                    stop, metrics = int(b_stops[i].item()), float(b_Ysc[i].item())
                    ret[stop] = metrics
            result[i_cls] = ret
        return result

    return _explain, _infer


def _get_perturbed_samples(
    explanations: Tensor,
    n_players: int,
    steps: int,
    mask_base: int,
) -> Tuple[Tensor, Tensor]:
    """
    inp [0]: explanations: <1, n_players>
    ret [0]: masked_samples: <steps>
    ret [1]: masks: <steps, n_players>
    """

    steps = min(n_players, steps)
    attribution = explanations.reshape((-1,)).cpu().numpy()
    ranking = np.argsort(attribution)[::-1]

    stops = np.linspace(0, n_players, steps, dtype=np.int64)  # maybe not duplicate
    masks_l: List[np.ndarray] = []
    for i in stops:
        mask = np.ones((n_players,), dtype=np.int64) * mask_base
        ids = ranking[:i]
        mask[ids] ^= 1
        masks_l.append(mask)

    stops_t = torch.tensor(np.array(stops), dtype=torch.long)
    masks_t = torch.tensor(np.array(masks_l), dtype=torch.long)
    return stops_t, masks_t
