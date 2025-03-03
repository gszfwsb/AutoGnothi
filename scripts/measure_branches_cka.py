import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, cast

import pydantic
import torch
import torch.utils.hooks
from torch import Tensor, nn

from ..datasets.loader import DatasetLoader
from ..models.cka import kernel_cka, linear_cka
from ..recipes.ltt_bert import LttBertRecipe
from ..recipes.types import ModelRecipe, TClassifier, TSurrogate
from ..utils.nnmodel import ObservableModuleMixin
from .env import ExpEnv
from .resources import (
    get_epoch_ckpts,
    get_recipe,
    load_cfg_dataset,
    load_epoch_ckpt,
    load_epoch_model,
)


class CkaStats(pydantic.BaseModel):
    linear_cka_all: List[List[float]]
    linear_cka_avg: List[float]
    linear_cka_std: List[float]
    kernel_cka_all: List[List[float]]
    kernel_cka_avg: List[float]
    kernel_cka_std: List[float]


class MeasureBranchesCkaReport(pydantic.BaseModel):
    """Measures the CKA similarity between the (original) Classifier task and
    the (trained) Explainer task. Measurements are conducted in-between the
    hidden transformer features of the two tasks.

    Requires: classifier [-1], surrogate [-1], explainer [ep], ObservableMixin."""

    epochs: List[int]
    classes: List[List[int]]
    all: CkaStats
    by_cls: Dict[str, CkaStats]


def measure_branches_cka(
    env: ExpEnv, device: torch.device, d_loader: Optional[DatasetLoader]
) -> MeasureBranchesCkaReport:
    env.log("loading models...")
    config = env.config
    _m_recipe, m_config = get_recipe(config)
    if not _m_recipe.measurements.allow_branches_cka:
        raise ValueError("unsupported recipe action")

    if d_loader is None:
        env.log("loading dataset...")
        d_config = (
            config.eval_branches_cka.dataset
            if config.eval_branches_cka is not None
            and config.eval_branches_cka.dataset is not None
            else config.dataset
        )
        d_loader = load_cfg_dataset(d_config, env.model_path)

    m_recipe = cast(LttBertRecipe, _m_recipe)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    gen_null = m_recipe.gen_null(m_config, m_misc, device)

    _epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )
    _epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )
    # surrogate_null
    nil_Xs = gen_null
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    m_surrogate.eval()
    with torch.no_grad():
        # <1, n_classes>
        surrogate_null, _ = m_recipe.fw_surrogate(m_surrogate, nil_Xs, nil_mask)

    # we need to go through all epochs of the explainer. if applicable.
    env.log("[[[ running measurement... ]]]")
    all_epochs: List[int] = []
    all_cls: List[List[int]] = []
    all_lin_cka: List[List[float]] = []
    all_krn_cka: List[List[float]] = []
    for _loading_epoch in get_epoch_ckpts(
        env.model_path, "explainer", config.train_explainer.epochs
    ):
        epoch_explainer, mpt_explainer = load_epoch_ckpt(
            env.model_path, "explainer", _loading_epoch, required=True
        )
        m_explainer = m_recipe.t_explainer(m_config)
        m_explainer.load_state_dict(mpt_explainer)
        m_explainer = m_explainer.to(device)

        ts_begin = time.time()
        cls_all, lin_cka_all, krn_cka_all = _explainer_cka_eval(
            env=env,
            device=device,
            n_mask_samples=config.train_explainer.n_mask_samples,
            n_players=n_players,
            surrogate_null=surrogate_null,
            d_items=d_loader.test(
                config.eval_branches_cka.batch_size
                if config.eval_branches_cka is not None
                else config.train_explainer.batch_size
            ),
            m_recipe=m_recipe,
            m_classifier=m_classifier,
            m_surrogate=m_surrogate,
            m_explainer=m_explainer,
            epoch=epoch_explainer,
            gen_input=gen_input,
        )
        ts_delta = time.time() - ts_begin

        all_epochs.append(epoch_explainer)
        all_cls.append(cls_all)
        all_lin_cka.append(lin_cka_all)
        all_krn_cka.append(krn_cka_all)
        lin_avg = torch.tensor(lin_cka_all).mean().item()
        krn_avg = torch.tensor(krn_cka_all).mean().item()
        info = f"  > epoch {epoch_explainer} done in {ts_delta:.2f}s // "
        info += f"cka: lin avg {lin_avg:.6f}, krn avg {krn_avg:.6f}"
        env.log(info)

    # stat by all classes
    stat_all = _stat(all_lin_cka, all_krn_cka)
    stat_by_cls: Dict[str, CkaStats] = {}
    for cl in sorted(set([cl for ep_cls in all_cls for cl in ep_cls])):
        cl_lin_cka: List[List[float]] = []
        cl_krn_cka: List[List[float]] = []
        for ep_cls, ep_lin_cka, ep_krn_cka in zip(all_cls, all_lin_cka, all_krn_cka):
            cl_lin_cka.append([item for c, item in zip(ep_cls, ep_lin_cka) if c == cl])
            cl_krn_cka.append([item for c, item in zip(ep_cls, ep_krn_cka) if c == cl])
        stat_by_cls[f"{cl}"] = _stat(cl_lin_cka, cl_krn_cka)

    return MeasureBranchesCkaReport(
        epochs=all_epochs,
        classes=all_cls,
        all=stat_all,
        by_cls=stat_by_cls,
    )


def _stat(lin_cka_all: List[List[float]], krn_cka_all: List[List[float]]) -> CkaStats:
    lin_cka_t = torch.tensor(lin_cka_all)
    krn_cka_t = torch.tensor(krn_cka_all)
    return CkaStats(
        linear_cka_all=lin_cka_all,
        linear_cka_avg=lin_cka_t.mean(dim=1).tolist(),
        linear_cka_std=lin_cka_t.std(dim=1).tolist(),
        kernel_cka_all=krn_cka_all,
        kernel_cka_avg=krn_cka_t.mean(dim=1).tolist(),
        kernel_cka_std=krn_cka_t.std(dim=1).tolist(),
    )


TObservableExplainer = TypeVar("TObservableExplainer", bound=nn.Module)


def _explainer_cka_eval(
    env: ExpEnv,
    device: torch.device,
    n_mask_samples: int,
    n_players: int,
    surrogate_null: Tensor,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[Any, Any, TClassifier, TSurrogate, TObservableExplainer, Any],
    m_classifier: TClassifier,
    m_surrogate: TSurrogate,
    m_explainer: TObservableExplainer,
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[List[int], List[float], List[float]]:
    all_cls: List[int] = []
    all_lin_cka: List[float] = []
    all_krn_cka: List[float] = []

    # see train for dimensions
    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)
        Xs_EXT = Xs.reshape((batch_size, 1, -1)).repeat(1, n_mask_samples, 1)
        Xs_EXT = Xs_EXT.reshape((batch_size * n_mask_samples, -1))

        m_surrogate.eval()
        with torch.no_grad():
            surrogate_grand, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_1)

        m_classifier.eval()
        with torch.no_grad():
            mixin = ObservableModuleMixin.using(m_classifier)
            mixin.om_retain_observations(True)
            _classifier, _ = m_recipe.fw_classifier(m_classifier, Xs, Xs_mask_1)
            repr_cls = mixin.om_observe()["repr_cls"]

        m_explainer.eval()
        with torch.no_grad():
            mixin = ObservableModuleMixin.using(m_explainer)
            mixin.om_retain_observations(True)
            _explainer_shap, _ = m_recipe.fw_explainer(
                m_explainer, Xs, Xs_mask_1, surrogate_grand, surrogate_null
            )
            repr_exp = mixin.om_observe()["repr_exp"]

        lin_cka = linear_cka(repr_cls, repr_exp)
        krn_cka = kernel_cka(repr_cls, repr_exp)
        all_cls.extend(Zs.tolist())
        all_lin_cka.extend(lin_cka.tolist())
        all_krn_cka.extend(krn_cka.tolist())

        info = f"  > epoch {epoch} :{batch_idx}:test // "
        info += f"cka: lin {lin_cka.mean().item():.6f}, krn {krn_cka.mean().item():.6f}"
        info += f" // fin {len(all_lin_cka)}"
        env.log(info)

    return all_cls, all_lin_cka, all_krn_cka
