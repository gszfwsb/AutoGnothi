import pathlib
from typing import Optional

import torch
import typer

from ..datasets.loader import preload_all_datasets
from ..params.loader import preload_all_params
from ..utils.nnmodel import show_model_fridge
from ..utils.tools import set_iterative_seed
from .env import ExpEnv
from .estimate_train_time import estimate_train_time
from .measure_accuracy import measure_accuracy
from .measure_all import measure_all
from .measure_branches_cka import measure_branches_cka
from .measure_cls_acc import measure_cls_acc
from .measure_dual_task_similarity import measure_dual_task_similarity
from .measure_faithfulness import measure_faithfulness
from .measure_performance import measure_performance
from .measure_train_resources import measure_train_resources
from .pretrain_classifier import pretrain_classifier
from .preview_text_shapley import preview_text_shapley
from .resources import get_recipe, load_id_dataset
from .run_image_explanation import run_image_explanation
from .run_text_explanation import run_text_explanation
from .train_all import (
    conv_classifier_surrogate,
    conv_explainer_final,
    conv_pretrained_classifier,
    conv_surrogate_explainer,
    train_all,
)
from .train_classifier import train_classifier
from .train_explainer import train_explainer
from .train_surrogate import train_surrogate


def _load_env(path: pathlib.Path) -> ExpEnv:
    return ExpEnv(
        model_path=path,
        get_logger_opts=lambda _: None,
    )


app = typer.Typer(
    pretty_exceptions_enable=False,
)


###############################################################################
#   main pipeline


@app.command("preload_all")
def cmd_preload_all() -> None:
    print("[[[ preloading all datasets... ]]]")
    preload_all_datasets()
    print("[[[ preloading all pretrained params... ]]]")
    preload_all_params()
    print("[[[ preload all ok ]]]")
    return


@app.command("pretrain_classifier")
def cmd_pretrain_classifier(
    model_path: pathlib.Path, device: str = typer.Option()
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    with _env.fork(lambda ec: ec.logger_classifier) as cl_env:
        pretrain_classifier(cl_env, _device)
    return


@app.command("estimate_train_time")
def cmd_estimate_train_time(
    model_path: pathlib.Path, device: str = typer.Option()
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    estimate_train_time(_env, _device)
    return


@app.command("conv_pretrained_classifier")
def cmd_conv_pretrained_classifier(model_path: pathlib.Path) -> None:
    _env = _load_env(model_path)
    conv_pretrained_classifier(_env)
    return


@app.command("train_classifier")
def cmd_train_classifier(
    model_path: pathlib.Path, device: str = typer.Option()
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    with _env.fork(lambda ec: ec.logger_classifier) as cl_env:
        train_classifier(cl_env, _device)
    return


@app.command("conv_classifier_surrogate")
def cmd_conv_classifier_surrogate(model_path: pathlib.Path) -> None:
    _env = _load_env(model_path)
    conv_classifier_surrogate(_env)
    return


@app.command("train_surrogate")
def cmd_train_surrogate(model_path: pathlib.Path, device: str = typer.Option()) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    with _env.fork(lambda ec: ec.logger_surrogate) as sg_env:
        train_surrogate(sg_env, _device)
    return


@app.command("conv_surrogate_explainer")
def cmd_conv_surrogate_explainer(model_path: pathlib.Path) -> None:
    _env = _load_env(model_path)
    conv_surrogate_explainer(_env)
    return


@app.command("train_explainer")
def cmd_train_explainer(model_path: pathlib.Path, device: str = typer.Option()) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    with _env.fork(lambda ec: ec.logger_explainer) as ex_env:
        train_explainer(ex_env, _device)
    return


@app.command("conv_explainer_final")
def cmd_conv_explainer_final(model_path: pathlib.Path) -> None:
    _env = _load_env(model_path)
    conv_explainer_final(_env)
    return


@app.command("train_all")
def cmd_train_all(model_path: pathlib.Path, device: str = typer.Option()) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    with _env.fork(lambda _: None) as env:
        train_all(env, _device)
    return


@app.command("measure_accuracy")
def cmd_measure_accuracy(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    measure_accuracy(_env, _device, _d_loader)
    return


@app.command("measure_faithfulness")
def cmd_measure_faithfulness(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
    resolution: Optional[int] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    measure_faithfulness(_env, _device, _d_loader, resolution)
    return


@app.command("measure_cls_acc")
def cmd_measure_cls_acc(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    measure_cls_acc(_env, _device, _d_loader)
    return


@app.command("measure_performance")
def cmd_measure_performance(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    measure_performance(_env, _device, _d_loader)
    return


@app.command("measure_train_resources")
def cmd_measure_train_resources(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    measure_train_resources(_env, _device, _d_loader)
    return


@app.command("measure_branches_cka")
def cmd_measure_branches_cka(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    measure_branches_cka(_env, _device, _d_loader)


@app.command("measure_dual_task_similarity")
def cmd_measure_dual_task_similarity(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    measure_dual_task_similarity(_env, _device, _d_loader)
    return


@app.command("measure_all")
def cmd_measure_all(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    run_accuracy: bool = typer.Option(True),
    run_faithfulness: bool = typer.Option(True),
    run_cls_acc: bool = typer.Option(True),
    run_performance: bool = typer.Option(True),
    run_train_resources: bool = typer.Option(True),
    run_branches_cka: bool = typer.Option(True),
    run_dual_task_similarity: bool = typer.Option(True),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    measure_all(
        env=_env,
        device=_device,
        run_accuracy=run_accuracy,
        run_faithfulness=run_faithfulness,
        run_cls_acc=run_cls_acc,
        run_performance=run_performance,
        run_train_resources=run_train_resources,
        run_branches_cka=run_branches_cka,
        run_dual_task_similarity=run_dual_task_similarity,
    )
    return


@app.command("run_all", help="Docker-friendly command to run all steps")
def cmd_run_all(
    model_name: str,
    device: str = typer.Option(""),
) -> None:
    if not (model_path := pathlib.Path(model_name)).exists():
        model_path = pathlib.Path(__file__).parent / f"../experiments/{model_name}/"
    default_device_name = "cuda" if torch.cuda.is_available() else "cpu"
    _env = _load_env(model_path)
    device = device or default_device_name
    _env.log(f"[[[ ! running on device: `{device}` ]]]")
    _device = torch.device(device)

    with _env.fork(lambda _: None) as env:
        train_all(env, _device)

    measure_all(
        env=_env,
        device=_device,
        run_accuracy=True,
        run_faithfulness=True,
        run_cls_acc=True,
        run_performance=True,
        run_train_resources=True,
        run_branches_cka=True,
        run_dual_task_similarity=True,
    )
    return


@app.command("run_image_explanation")
def cmd_run_image_explanation(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
    into: pathlib.Path = typer.Option(),
    limit: Optional[int] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    img_px_size: int = _env.config.net.params.img_px_size  # type: ignore
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset, img_px_size=img_px_size) if dataset else None
    run_image_explanation(_env, _device, _d_loader, into, limit)
    return


@app.command("run_text_explanation")
def cmd_run_text_explanation(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
    into: pathlib.Path = typer.Option(),
    limit: Optional[int] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    run_text_explanation(_env, _device, _d_loader, into, limit)
    return


###############################################################################
#   miscellaneous functions


@app.command("__show_fridge__")
def cmd_show_fridge(model_path: pathlib.Path) -> None:
    _env = _load_env(model_path)
    m_recipe, m_config = get_recipe(_env.config)

    m_classifier = m_recipe.t_classifier(m_config)
    m_classifier.train()
    show_model_fridge(m_classifier)

    m_surrogate = m_recipe.t_surrogate(m_config)
    m_surrogate.train()
    show_model_fridge(m_surrogate)

    m_explainer = m_recipe.t_explainer(m_config)
    m_explainer.train()
    show_model_fridge(m_explainer)

    return


@app.command("__preview_text_shapley__")
def cmd_preview_text_shapley(
    model_path: pathlib.Path,
    device: str = typer.Option(),
    dataset: Optional[str] = typer.Option(None),
) -> None:
    _env = _load_env(model_path)
    _device = torch.device(device)
    _d_loader = load_id_dataset(dataset) if dataset else None
    preview_text_shapley(_env, _device, _d_loader)
    return


def main() -> None:
    set_iterative_seed(42, "scripts.shell.main")
    try:
        app()
    except Exception as exc:
        # import traceback

        # exc = traceback.format_exc()
        # for line in exc.split("\n"):
        #     print(f"!!! {line}")
        raise exc from exc
    finally:
        print("[[[ RUN STOPPED ]]]")
    return
