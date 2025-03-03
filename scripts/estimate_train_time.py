import torch

from ..utils.units import Seconds
from .env import ExpEnv
from .measure_all import load_or_run_report
from .measure_train_resources import (
    MeasureTrainResourcesReport,
    measure_train_resources,
)
from .resources import get_recipe


def estimate_train_time(env: ExpEnv, device: torch.device) -> None:
    env.log("[[[ retrieving training resource report... ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.measurements.allow_train_resources:
        env.log("[[[ error: cannot measure training speed ]]]")
        raise ValueError("given model does not support measurement")
    report = load_or_run_report(
        env=env,
        t_report=MeasureTrainResourcesReport,
        filename="train_resources.json",
        run=lambda: measure_train_resources(env, device, d_loader=None),
    )

    train_size = getattr(config.dataset, "train_size", -1)
    if train_size < 0:
        train_size = int(input(">>> enter train set size: "))
    tm_surrogate = (
        report.init_tm * config.train_classifier.epochs
        + report.init_tm * config.train_surrogate.epochs
        + report.srg_tm.avg * train_size * config.train_classifier.epochs
        + report.srg_tm.avg * train_size * config.train_surrogate.epochs
    )
    tm_explainer = (
        report.init_tm * config.train_explainer.epochs
        + report.exp_tm.avg * train_size * config.train_explainer.epochs
    )

    env.log("[[[ estimated training time ]]]")
    env.log(f"> surrogate: {fmt_tm(tm_surrogate)}")
    env.log(f"> explainer: {fmt_tm(tm_explainer)}")
    return


def fmt_tm(tm: Seconds) -> str:
    min = int(tm // 60) % 60
    hr = int(tm / 60 / 60)
    if hr == 0:
        return f"     {min:02d}m"
    return f"{hr: 3d}h {min:02d}m"
