import gc
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar, cast

import pydantic
import torch
from torch import Tensor, nn

from ..datasets.loader import DatasetLoader
from ..recipes.types import ModelRecipe, TMisc
from ..utils.units import GFLOPS, MParams, Seconds
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model


class ModelPerformance(pydantic.BaseModel):
    time: List[Seconds]
    time_avg: Seconds
    time_std: Seconds
    gflops: GFLOPS
    params_all: MParams
    params_trainable: MParams


class MeasurePerformanceReport(pydantic.BaseModel):
    """Measures runtime & theoretical performance of a model.

    Requires: classifier [-1], surrogate [-1], explainer [-1], final [-1]."""

    classifier: Optional[ModelPerformance]
    surrogate: Optional[ModelPerformance]
    explainer: Optional[ModelPerformance]
    final: Optional[ModelPerformance]


def measure_performance(
    env: ExpEnv, device: torch.device, d_loader: Optional[DatasetLoader]
) -> MeasurePerformanceReport:
    env.log("loading models...")
    config = env.config
    m_recipe, m_config = get_recipe(config)

    if d_loader is None:
        env.log("loading dataset...")
        d_config = (
            config.eval_performance.dataset
            if config.eval_performance.dataset is not None
            else config.dataset
        )
        d_loader = load_cfg_dataset(d_config, env.model_path)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    gen_null = m_recipe.gen_null(m_config, m_misc, device)

    if m_recipe.measurements.allow_performance_cls:
        results_cls = _measure_cls_perf(
            env=env,
            device=device,
            d_loader=d_loader,
            m_recipe=m_recipe,
            n_players=n_players,
            gen_input=gen_input,
            cfg_loops=config.eval_performance.loops,
            batch_size=1,
        )
    else:
        results_cls = None

    if m_recipe.measurements.allow_performance_srg_exp:
        results_srg, results_exp = _measure_srg_exp_perf(
            env=env,
            device=device,
            d_loader=d_loader,
            m_recipe=m_recipe,
            n_players=n_players,
            gen_input=gen_input,
            gen_null=gen_null,
            cfg_loops=config.eval_performance.loops,
            batch_size=1,
        )
    else:
        results_srg, results_exp = None, None

    if m_recipe.measurements.allow_performance_fin:
        results_fin = _measure_fin_perf(
            env=env,
            device=device,
            d_loader=d_loader,
            m_recipe=m_recipe,
            gen_input=gen_input,
            cfg_loops=config.eval_performance.loops,
            batch_size=1,
        )
    else:
        results_fin = None

    return MeasurePerformanceReport(
        classifier=results_cls,
        surrogate=results_srg,
        explainer=results_exp,
        final=results_fin,
    )


def _measure_cls_perf(
    env: ExpEnv,
    device: torch.device,
    d_loader: DatasetLoader,
    m_recipe: ModelRecipe[Any, Any, Any, Any, Any, Any],
    n_players: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
    cfg_loops: int,
    batch_size: int,
) -> ModelPerformance:
    _epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )

    tm_cls_ls: List[float] = []  # [s] per sample, not per inference
    Xs, mask_1 = cast(Tensor, ...), cast(Tensor, ...)
    for loop in range(cfg_loops):
        for i, (_inputs, _targets) in enumerate(d_loader.test(batch_size)):
            print(f"<cls> loop {loop} of {cfg_loops}, sample {i}   ", end="\r")
            Xs, Zs = gen_input(_inputs, _targets)
            size = Zs.shape[0]
            mask_1 = torch.ones((size, n_players), dtype=torch.long, device=device)
            tm_cls, _ = _measure_time(
                lambda: m_recipe.fw_classifier(m_classifier, Xs, mask_1)
            )
            tm_cls_ls.append(tm_cls / size)
    assert Xs is not ... and mask_1 is not ...

    flops_cls = _measure_flops(lambda: m_recipe.fw_classifier(m_classifier, Xs, mask_1))
    results = _stat_perf(m_classifier, tm_cls_ls, flops_cls)

    msg = f"PERFORMANCE RESULTS for {m_recipe.id} <cls>\n"
    msg += f"    params: all {results.params_all:.3f} M, trainable {results.params_trainable:.3f} M\n"
    msg += f"    flops: {results.gflops:.3f} G\n"
    msg += f"    time: mean {results.time_avg * 1e3:.3f} ms, std {results.time_std * 1e3:.3f} ms\n"
    for line in msg.strip().split("\n"):
        env.log(line)
    return results


def _measure_srg_exp_perf(
    env: ExpEnv,
    device: torch.device,
    d_loader: DatasetLoader,
    m_recipe: ModelRecipe[Any, TMisc, Any, Any, Any, Any],
    n_players: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
    gen_null: Tensor,
    cfg_loops: int,
    batch_size: int,
) -> Tuple[ModelPerformance, ModelPerformance]:
    _epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )
    _epoch_explainer, m_explainer = load_epoch_model(
        env, m_recipe, "explainer", device=device
    )

    # surrogate_null
    nil_Xs = gen_null
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    m_surrogate.eval()
    with torch.no_grad():
        surrogate_null, _ = m_recipe.fw_surrogate(m_surrogate, nil_Xs, nil_mask)

    # surrogate & explainer must be joint evaluated since explainer depends on
    # grand and null values
    tm_srg_ls: List[float] = []
    tm_exp_ls: List[float] = []
    Xs, mask_1, surrogate_grand = cast(Tensor, ...), cast(Tensor, ...), []
    for loop in range(cfg_loops):
        for i, (_inputs, _targets) in enumerate(d_loader.test(batch_size)):
            print(f"<srg,exp> loop {loop} of {cfg_loops}, sample {i}   ", end="\r")
            Xs, Zs = gen_input(_inputs, _targets)
            size = Zs.shape[0]
            mask_1 = torch.ones((size, n_players), dtype=torch.long, device=device)
            surrogate_grand = []
            tm_srg, _ = _measure_time(
                lambda: surrogate_grand.append(
                    m_recipe.fw_surrogate(m_surrogate, Xs, mask_1)[0]
                )
            )
            tm_exp, _ = _measure_time(
                lambda: m_recipe.fw_explainer(
                    m_explainer, Xs, mask_1, surrogate_grand[0], surrogate_null
                )
            )
            tm_srg_ls.append(tm_srg / size)
            tm_exp_ls.append(tm_exp / size)
    assert Xs is not ... and mask_1 is not ...

    flops_srg = _measure_flops(lambda: m_recipe.fw_surrogate(m_surrogate, Xs, mask_1))
    flops_exp = _measure_flops(
        lambda: m_recipe.fw_explainer(
            m_explainer, Xs, mask_1, surrogate_grand[0], surrogate_null
        )
    )
    results_srg = _stat_perf(m_surrogate, tm_srg_ls, flops_srg)
    results_exp = _stat_perf(m_explainer, tm_exp_ls, flops_exp)

    msg = f"PERFORMANCE RESULTS for {m_recipe.id} <srg>\n"
    msg += f"    params: all {results_srg.params_all:.3f} M, trainable {results_srg.params_trainable:.3f} M\n"
    msg += f"    flops: {results_srg.gflops:.3f} G\n"
    msg += f"    time: mean {results_srg.time_avg * 1e3:.3f} ms, std {results_srg.time_std * 1e3:.3f} ms\n"
    msg += f"PERFORMANCE RESULTS for {m_recipe.id} <exp>\n"
    msg += f"    params: all {results_exp.params_all:.3f} M, trainable {results_exp.params_trainable:.3f} M\n"
    msg += f"    flops: {results_exp.gflops:.3f} G\n"
    msg += f"    time: mean {results_exp.time_avg * 1e3:.3f} ms, std {results_exp.time_std * 1e3:.3f} ms\n"
    for line in msg.strip().split("\n"):
        env.log(line)
    return results_srg, results_exp


def _measure_fin_perf(
    env: ExpEnv,
    device: torch.device,
    d_loader: DatasetLoader,
    m_recipe: ModelRecipe[Any, TMisc, Any, Any, Any, Any],
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
    cfg_loops: int,
    batch_size: int,
) -> ModelPerformance:
    _epoch_final, m_final = load_epoch_model(env, m_recipe, "final", device=device)

    tm_fin_ls: List[float] = []
    Xs = cast(Tensor, ...)
    for loop in range(cfg_loops):
        for i, (_inputs, _targets) in enumerate(d_loader.test(batch_size)):
            print(f"<fin> loop {loop} of {cfg_loops}, sample {i}   ", end="\r")
            Xs, Zs = gen_input(_inputs, _targets)
            size = Zs.shape[0]
            tm_fin, _ = _measure_time(lambda: m_recipe.fw_final(m_final, Xs))
            tm_fin_ls.append(tm_fin / size)
    assert Xs is not ...

    flops_fin = _measure_flops(lambda: m_recipe.fw_final(m_final, Xs))
    results = _stat_perf(m_final, tm_fin_ls, flops_fin)

    msg = f"PERFORMANCE RESULTS for {m_recipe.id} <fin>\n"
    msg += f"    params: all {results.params_all:.3f} M, trainable {results.params_trainable:.3f} M\n"
    msg += f"    flops: {results.gflops:.3f} G\n"
    msg += f"    time: mean {results.time_avg * 1e3:.3f} ms, std {results.time_std * 1e3:.3f} ms\n"
    for line in msg.strip().split("\n"):
        env.log(line)
    return results


###############################################################################


T = TypeVar("T")


def _measure_time(func: Callable[[], T]) -> Tuple[float, T]:
    with torch.no_grad():
        _sync()
        _clear_garbage()
        _sync()
        tm_0 = time.perf_counter_ns()
        ret = func()
        _sync()
        tm_1 = time.perf_counter_ns()
        _clear_garbage()
    return (tm_1 - tm_0) / 1e9, ret


def _sync() -> None:
    if hasattr(torch.cpu, "synchronize"):
        torch.cpu.synchronize()  # type: ignore
    torch.cuda.synchronize()
    return


def _clear_garbage() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    return


def _measure_flops(func: Callable[[], Any]) -> int:
    _sync()
    _clear_garbage()
    _sync()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        func()
        _sync()
        _clear_garbage()
    tot_flops = [i.flops for i in prof.key_averages()]
    return sum(tot_flops)


def _stat_perf(model: nn.Module, tm: List[float], flops: int) -> ModelPerformance:
    tm_t = torch.tensor(tm)
    params_all = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ModelPerformance(
        time=tm,
        time_avg=torch.mean(tm_t).item(),
        time_std=torch.std(tm_t).item(),
        gflops=flops / 1e9,
        params_all=params_all / 1e6,
        params_trainable=params_trainable / 1e6,
    )
