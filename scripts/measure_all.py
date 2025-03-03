import json
from typing import Callable, Optional, Type, TypeVar

import pydantic
import torch

from .env import ExpEnv
from .measure_accuracy import MeasureAccuracyReport, measure_accuracy
from .measure_branches_cka import MeasureBranchesCkaReport, measure_branches_cka
from .measure_cls_acc import MeasureClsAccReport, measure_cls_acc
from .measure_dual_task_similarity import (
    MeasureDualTaskSimilarityReport,
    measure_dual_task_similarity,
)
from .measure_faithfulness import MeasureFaithfulnessReport, measure_faithfulness
from .measure_performance import MeasurePerformanceReport, measure_performance
from .measure_train_resources import (
    MeasureTrainResourcesReport,
    measure_train_resources,
)
from .resources import get_recipe


def measure_all(
    env: ExpEnv,
    device: torch.device,
    run_accuracy: bool,
    run_faithfulness: bool,
    run_cls_acc: bool,
    run_performance: bool,
    run_train_resources: bool,
    run_branches_cka: bool,
    run_dual_task_similarity: bool,
) -> None:
    # used for allowed measurement detection
    m_recipe, _m_config = get_recipe(env.config)

    def _run_report(
        t_report: Type[TReport],
        filename: str,
        run: Callable[[], TReport],
        recipe_allow: bool,
        cli_allow: bool,
    ) -> Optional[TReport]:
        name = filename.split(".")[0]
        if recipe_allow:
            if cli_allow:
                env.log(f"[[[ Measuring: {name} ]]]")
                return load_or_run_report(env, t_report, filename, run)
            else:
                env.log(f"[[[ skip: {name} ]]]")
        return None

    _run_report(
        t_report=MeasureAccuracyReport,
        filename="accuracy.json",
        run=lambda: measure_accuracy(env, device, d_loader=None),
        recipe_allow=m_recipe.measurements.allow_accuracy,
        cli_allow=run_accuracy,
    )
    _run_report(
        t_report=MeasureFaithfulnessReport,
        filename="faithfulness.json",
        run=lambda: measure_faithfulness(env, device, d_loader=None, resolution=None),
        recipe_allow=m_recipe.measurements.allow_faithfulness,
        cli_allow=run_faithfulness,
    )
    _run_report(
        t_report=MeasureClsAccReport,
        filename="cls_acc.json",
        run=lambda: measure_cls_acc(env, device, d_loader=None),
        recipe_allow=m_recipe.measurements.allow_cls_acc,
        cli_allow=run_cls_acc,
    )
    _run_report(
        t_report=MeasurePerformanceReport,
        filename="performance.json",
        run=lambda: measure_performance(env, device, d_loader=None),
        recipe_allow=(
            m_recipe.measurements.allow_performance_cls
            or m_recipe.measurements.allow_performance_srg_exp
            or m_recipe.measurements.allow_performance_fin
        ),
        cli_allow=run_performance,
    )
    _run_report(
        t_report=MeasureTrainResourcesReport,
        filename="train_resources.json",
        run=lambda: measure_train_resources(env, device, d_loader=None),
        recipe_allow=m_recipe.measurements.allow_train_resources,
        cli_allow=run_train_resources,
    )
    _run_report(
        t_report=MeasureBranchesCkaReport,
        filename="branches_cka.json",
        run=lambda: measure_branches_cka(env, device, d_loader=None),
        recipe_allow=m_recipe.measurements.allow_branches_cka,
        cli_allow=run_branches_cka,
    )
    _run_report(
        t_report=MeasureDualTaskSimilarityReport,
        filename="dual_task_similarity.json",
        run=lambda: measure_dual_task_similarity(env, device, d_loader=None),
        recipe_allow=m_recipe.measurements.allow_dual_task_similarity is not False,
        cli_allow=run_dual_task_similarity,
    )
    env.log("[[[ done all measurements ]]]")
    return


TReport = TypeVar("TReport", bound=pydantic.BaseModel)


def load_or_run_report(
    env: ExpEnv,
    t_report: Type[TReport],
    filename: str,
    run: Callable[[], TReport],
) -> TReport:
    f_path = env.model_path / ".reports" / filename
    # attempting to load file
    if f_path.exists():
        with open(f_path, "r", encoding="utf-8") as f:
            raw = json.loads(f.read())
        report = t_report.model_validate(raw)
        return report

    # have to run & save
    report = run()
    f_path.parent.mkdir(parents=True, exist_ok=True)
    with open(f_path, "w", encoding="utf-8") as f:
        raw = report.model_dump_json(by_alias=True, exclude_unset=True)
        raw = json.dumps(json.loads(raw), indent=2)
        f.write(raw + "\n")
    return report
