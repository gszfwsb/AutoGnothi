import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

import pydantic
import torch
import torch.utils.data
from torch import Tensor

from ..datasets.loader import DatasetLoader
from ..recipes.types import ModelRecipe, TConfig, TFinal
from ..utils.strings import ranged_modulo_test
from .env import ExpEnv
from .resources import (
    get_epoch_ckpts,
    get_recipe,
    load_cfg_dataset,
    load_epoch_ckpt,
    load_epoch_model,
)


class MeasureClsAccReport(pydantic.BaseModel):
    """Reports classifier accuracy, in the final model generated from all
    available iterated epochs. Accuracy is performed on a test set.

    Requires: classifier [ep], surrogate [ep], explainer [ep] | final [-1]."""

    epochs: List[int]
    accuracy: List[float]


def measure_cls_acc(
    env: ExpEnv, device: torch.device, d_loader: Optional[DatasetLoader]
) -> MeasureClsAccReport:
    env.log("[[[ measuring classifier accuracy ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.measurements.allow_cls_acc:
        raise ValueError("unsupported recipe action")

    if d_loader is None:
        env.log("loading dataset...")
        d_config = (
            config.eval_cls_acc.dataset
            if config.eval_cls_acc.dataset is not None
            else config.dataset
        )
        d_loader = load_cfg_dataset(d_config, env.model_path)

    m_misc = m_recipe.load_misc(env.model_path, m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)

    _epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )
    _epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )

    def measure_on(ep: int) -> bool:
        if config.eval_cls_acc.on_exp_epochs is None:
            return ep == config.train_explainer.epochs
        return ranged_modulo_test(config.eval_cls_acc.on_exp_epochs)(ep)

    env.log("[[[ measuring explainers... ]]]")
    all_epochs: List[int] = []
    all_acc: List[float] = []
    for _loading_epoch in get_epoch_ckpts(
        env.model_path, "explainer", config.train_explainer.epochs
    ):
        if not measure_on(_loading_epoch):
            continue
        epoch_explainer, mpt_explainer = load_epoch_ckpt(
            env.model_path, "explainer", _loading_epoch, required=True
        )
        m_explainer = m_recipe.t_explainer(m_config)
        m_explainer.load_state_dict(mpt_explainer)
        m_explainer = m_explainer.to(device)
        m_final = m_recipe.conv_explainer_final(
            m_config, m_misc, m_classifier, m_surrogate, m_explainer
        )
        m_final = m_final.to(device)

        ts_begin = time.time()
        acc = _measure_final_cls_epoch(
            env=env,
            d_items=d_loader.test(config.train_classifier.batch_size),
            m_recipe=m_recipe,
            m_final=m_final,
            epoch=epoch_explainer,
            gen_input=gen_input,
        )
        ts_delta = time.time() - ts_begin

        all_epochs.append(epoch_explainer)
        all_acc.append(acc)
        info = f"  > epoch {epoch_explainer} done in {ts_delta:.2f}s // test_acc: {acc:.3f}"
        env.log(info)

    return MeasureClsAccReport(
        epochs=all_epochs,
        accuracy=all_acc,
    )


def _measure_final_cls_epoch(
    env: ExpEnv,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, Any, Any, Any, TFinal],
    m_final: TFinal,
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> float:
    correct, total = 0, 0
    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape

        m_final.eval()
        with torch.no_grad():
            fin_Ys, _ = m_recipe.fw_final(m_final, Xs)
        _, fin_Zs = fin_Ys.max(dim=1)
        correct += fin_Zs.eq(Zs).sum().item()
        total += batch_size

        info = f"  > epoch {epoch} :{batch_idx}:test // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)
    return correct / total
