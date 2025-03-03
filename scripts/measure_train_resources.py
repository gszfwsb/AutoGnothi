import gc
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import pydantic
import torch
import torch.autograd.profiler_util
import torch.utils.data
from torch import Tensor, nn

from ..datasets.loader import DatasetLoader
from ..models.shapley import (
    loss_logits_kl_divergence,
    loss_shapley_new,
    mask_purely_uniform,
    mask_shapley_new,
)
from ..recipes.types import ModelRecipe, TClassifier, TConfig, TExplainer, TSurrogate
from ..utils.units import MiBytes, Seconds
from .env import ExpEnv
from .measure_performance import _sync
from .resources import get_recipe, load_cfg_dataset


class SecondsStats(pydantic.BaseModel):
    all: List[Seconds]
    avg: Seconds
    std: Seconds

    @staticmethod
    def from_list(all: List[Seconds]) -> "SecondsStats":
        return SecondsStats(
            all=all,
            avg=torch.tensor(all).mean().item(),
            std=torch.tensor(all).std().item(),
        )


class MiBytesStats(pydantic.BaseModel):
    all: List[MiBytes]
    avg: MiBytes
    std: MiBytes

    @staticmethod
    def from_list(all: List[MiBytes]) -> "MiBytesStats":
        return MiBytesStats(
            all=all,
            avg=torch.tensor(all).mean().item(),
            std=torch.tensor(all).std().item(),
        )


class MeasureTrainResourcesReport(pydantic.BaseModel):
    init_tm: Seconds
    init_mem: MiBytes
    srg_tm: SecondsStats
    srg_mem: MiBytesStats
    exp_tm: SecondsStats
    exp_mem: MiBytesStats


def measure_train_resources(
    env: ExpEnv, device: torch.device, d_loader: Optional[DatasetLoader]
) -> MeasureTrainResourcesReport:
    env.log("loading models...")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.measurements.allow_train_resources:
        raise ValueError("unsupported recipe action")

    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    gen_null = m_recipe.gen_null(m_config, m_misc, device)

    if d_loader is None:
        env.log("loading dataset...")
        d_config = (
            config.eval_performance.dataset
            if config.eval_performance.dataset is not None
            else config.dataset
        )
        d_config = d_config.model_copy(deep=True)
        d_loader = load_cfg_dataset(d_config, env.model_path)

    def _load_models() -> (
        Tuple[nn.Module, nn.Module, torch.optim.AdamW, nn.Module, torch.optim.AdamW]  # type: ignore
    ):
        m_classifier = m_recipe.t_classifier(m_config).to(device=device)
        m_surrogate = m_recipe.t_surrogate(m_config).to(device=device)
        optim_srg = torch.optim.AdamW(  # type: ignore
            m_surrogate.parameters(), lr=config.train_surrogate.lr
        )
        m_explainer = m_recipe.t_explainer(m_config).to(device=device)
        optim_exp = torch.optim.AdamW(  # type: ignore
            m_explainer.parameters(), lr=config.train_explainer.lr
        )
        return m_classifier, m_surrogate, optim_srg, m_explainer, optim_exp

    # measure setup resources
    init, init_tm, init_mem = _measure_props(_load_models, device)
    m_classifier, m_surrogate, optim_srg, m_explainer, optim_exp = init
    env.log(f"init: {init_tm:.6f} s, {init_mem:.2f} MB")
    batch_size = config.eval_train_resources.batch_size
    max_size = config.eval_train_resources.max_samples

    # surrogate_null
    nil_Xs = gen_null
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    m_surrogate.eval()
    with torch.no_grad():
        surrogate_null, _ = m_recipe.fw_surrogate(m_surrogate, nil_Xs, nil_mask)

    # surrogate resources
    aggr_size = 0
    all_srg_tm: List[float] = []
    all_srg_mem: List[float] = []
    for _inputs, _targets in d_loader.train(batch_size):
        Xs, _Zs = gen_input(_inputs, _targets)
        size, *_ = Xs.shape
        _step = lambda: _surrogate_batch_train(  # noqa: E731
            device=device,
            n_players=n_players,
            m_recipe=m_recipe,
            m_classifier=m_classifier,
            m_surrogate=m_surrogate,
            optimizer=optim_srg,
            Xs=Xs,
        )
        _loss, tm, mem = _measure_props(_step, device)
        optim_srg.step()
        all_srg_tm.append(tm / size)
        all_srg_mem.append(mem // size)
        aggr_size += size
        env.log(f"> surrogate: {tm / size:.6f} s, {mem / size:.2f} MB ({aggr_size})")
        if aggr_size >= max_size:
            break

    # explainer resources
    aggr_size = 0
    all_exp_tm: List[float] = []
    all_exp_mem: List[float] = []
    for _inputs, _targets in d_loader.train(batch_size):
        Xs, _Zs = gen_input(_inputs, _targets)
        size, *_ = Xs.shape
        _step = lambda: _explainer_batch_train(  # noqa: E731
            device=device,
            n_mask_samples=config.train_explainer.n_mask_samples,
            n_players=n_players,
            surrogate_null=surrogate_null,
            m_recipe=m_recipe,
            m_surrogate=m_surrogate,
            m_explainer=m_explainer,
            optimizer=optim_srg,
            Xs=Xs,
        )
        _loss, tm, mem = _measure_props(_step, device)
        optim_srg.step()
        all_exp_tm.append(tm / size)
        all_exp_mem.append(mem / size)
        aggr_size += size
        env.log(f"> explainer: {tm / size:.6f} s, {mem / size:.2f} MB ({aggr_size})")
        if aggr_size >= max_size:
            break

    return MeasureTrainResourcesReport(
        init_tm=init_tm,
        init_mem=init_mem,
        srg_tm=SecondsStats.from_list(all_srg_tm),
        srg_mem=MiBytesStats.from_list(all_srg_mem),
        exp_tm=SecondsStats.from_list(all_exp_tm),
        exp_mem=MiBytesStats.from_list(all_exp_mem),
    )


###############################################################################
#   functions to measure


def _surrogate_batch_train(
    device: torch.device,
    n_players: int,
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, Any, Any],
    m_classifier: TClassifier,
    m_surrogate: TSurrogate,
    optimizer: torch.optim.Optimizer,  # type: ignore
    Xs: Tensor,
) -> Tensor:
    batch_size, *_ = Xs.shape
    Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)
    Xs_mask_rand = mask_purely_uniform(batch_size, n_players)
    Xs_mask_rand = Xs_mask_rand.to(device)

    optimizer.zero_grad()
    m_classifier.eval()
    with torch.no_grad():
        _, orig_Ys = m_recipe.fw_classifier(m_classifier, Xs, Xs_mask_1)

    optimizer.zero_grad()
    m_surrogate.train()
    adapt_Ys, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_rand)
    loss_kld = loss_logits_kl_divergence(orig_Ys, adapt_Ys)
    loss_kld.backward()
    # optimizer.step()
    return loss_kld


def _explainer_batch_train(
    device: torch.device,
    n_mask_samples: int,
    n_players: int,
    surrogate_null: Tensor,
    m_recipe: ModelRecipe[TConfig, Any, Any, TSurrogate, TExplainer, Any],
    m_surrogate: TSurrogate,
    m_explainer: TExplainer,
    optimizer: torch.optim.Optimizer,  # type: ignore
    Xs: Tensor,
) -> Tensor:
    batch_size, *_ = Xs.shape
    Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)
    Xs_mask_shap_ = mask_shapley_new(batch_size * n_mask_samples, n_players).to(device)
    Xs_mask_shap = Xs_mask_shap_.reshape((batch_size, n_mask_samples, n_players))
    # <batch_size * n_mask_samples, ...>
    Xs_EXT = []
    for b in range(batch_size):
        for _ in range(n_mask_samples):
            Xs_EXT.append(Xs[b])
    Xs_EXT = torch.stack(Xs_EXT, dim=0)

    optimizer.zero_grad()
    m_surrogate.eval()
    with torch.no_grad():
        surrogate_values, _ = m_recipe.fw_surrogate(m_surrogate, Xs_EXT, Xs_mask_shap_)
        surrogate_grand, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_1)
        surrogate_grand_EXT = surrogate_grand.reshape((batch_size, 1, -1))
        surrogate_grand_EXT = surrogate_grand_EXT.repeat(1, n_mask_samples, 1)
        surrogate_grand_EXT = surrogate_grand_EXT.reshape(
            (batch_size * n_mask_samples, -1)
        )

    optimizer.zero_grad()
    m_explainer.train()
    explainer_shap, _ = m_recipe.fw_explainer(
        m_explainer, Xs, Xs_mask_1, surrogate_grand, surrogate_null
    )
    loss_shap = loss_shapley_new(
        batch_size=batch_size,
        n_mask_samples=n_mask_samples,
        n_players=n_players,
        mask=Xs_mask_shap,
        v_0=surrogate_null,
        v_s=surrogate_values,
        v_1=surrogate_grand,
        phi=explainer_shap,
    )
    loss_shap.backward()
    # optimizer.step()
    return loss_shap


###############################################################################
#   utilities

T = TypeVar("T")


def _measure_props(
    func: Callable[[], T], device: torch.device
) -> Tuple[T, float, float]:
    # fn, device -> ret, time/s, memory/MB
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    tm_0 = time.perf_counter_ns()

    def _evt_mem(evt: torch.autograd.profiler_util.FunctionEvent) -> int:
        try:
            # not supported by torch 2.0.0
            return evt.self_device_memory_usage  # type: ignore
        except AttributeError:
            pass
        return evt.self_cuda_memory_usage

    # start running
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        ret = func()
        _sync()
        tm_1 = time.perf_counter_ns()
    cuda_mem = max(  # wtf?
        _evt_mem(evt) for evt in prof.key_averages() if evt.key != "[memory]"
    )
    # print(prof.key_averages().table(sort_by="self_device_memory_usage"))
    return ret, (tm_1 - tm_0) / 1e9, cuda_mem / 1e6
