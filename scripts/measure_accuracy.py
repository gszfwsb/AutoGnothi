import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

import pydantic
import torch
import torch.utils.data
from torch import Tensor

from ..datasets.loader import DatasetLoader
from ..models.shapley import mask_uniform_selective
from ..recipes.types import ModelRecipe, TConfig, TSurrogate
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model


class MeasureAccuracyReport(pydantic.BaseModel):
    """Reports the (final epoch) surrogate model's accuracy given a varying
    number of perturbed tokens / players / patches / features.

    Requires: surrogate [ep]."""

    masked_players: List[int]
    accuracy: List[float]


def measure_accuracy(
    env: ExpEnv, device: torch.device, d_loader: Optional[DatasetLoader]
) -> MeasureAccuracyReport:
    env.log("[[[ measuring model accuracy ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.measurements.allow_accuracy:
        raise ValueError("unsupported recipe action")

    if d_loader is None:
        env.log("loading dataset...")
        d_config = (
            config.eval_accuracy.dataset
            if config.eval_accuracy.dataset is not None
            else config.dataset
        )
        d_loader = load_cfg_dataset(d_config, env.model_path)

    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)

    epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )

    env.log("[[[ measuring surrogate... ]]]")
    all_masked_players: List[int] = torch.linspace(
        0, n_players, config.eval_accuracy.resolution, dtype=torch.long
    ).tolist()
    all_acc: List[float] = []
    for n_masked_players in all_masked_players:
        ts_begin = time.time()
        acc = _measure_surrogate_epoch(
            env=env,
            device=device,
            n_players=n_players,
            n_masked_players=n_masked_players,
            d_items=d_loader.test(config.train_surrogate.batch_size),
            m_recipe=m_recipe,
            m_surrogate=m_surrogate,
            epoch=epoch_surrogate,
            gen_input=gen_input,
        )
        ts_delta = time.time() - ts_begin

        all_acc.append(acc)
        info = f"  > mask {n_masked_players} done in {ts_delta:.2f}s // test_acc: {acc:.3f}"
        env.log(info)

    return MeasureAccuracyReport(
        masked_players=all_masked_players,
        accuracy=all_acc,
    )


def _measure_surrogate_epoch(
    env: ExpEnv,
    device: torch.device,
    n_players: int,
    n_masked_players: int,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, Any, TSurrogate, Any, Any],
    m_surrogate: TSurrogate,
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> float:
    correct, total = 0, 0
    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_k = mask_uniform_selective(batch_size, n_players, n_masked_players)
        Xs_mask_k = Xs_mask_k.to(device)

        m_surrogate.eval()
        with torch.no_grad():
            adapt_Ys, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_k)
        _, adapt_Zs = adapt_Ys.max(dim=1)
        correct += adapt_Zs.eq(Zs).sum().item()
        total += batch_size

        info = f"  > mask {n_masked_players} :{batch_idx}:test // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)
    return correct / total
