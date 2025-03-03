import math
import time
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
import torch.utils.data
from torch import Tensor

from ..recipes.types import ModelRecipe, TClassifier, TConfig
from ..utils.tools import set_iterative_seed
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model, save_epoch_ckpt


def train_classifier(
    env: ExpEnv,
    device: torch.device,
    set_model_mode: Optional[Callable[[TClassifier, bool], None]] = None,
) -> None:
    env.log("[[[ train classifier ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.training.support_classifier:
        env.log("[[[ skip: classifier cannot be trained ]]]")
        return

    d_loader = load_cfg_dataset(config.dataset, env.model_path)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)

    epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )
    if epoch_classifier >= config.train_classifier.epochs:
        env.log("[[[ classifier already trained ]]]")
        return
    optimizer = torch.optim.AdamW(  # type: ignore
        m_classifier.parameters(), lr=config.train_classifier.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.train_classifier.epochs
    )

    for epoch in range(epoch_classifier + 1, config.train_classifier.epochs + 1):
        info = f"### epoch {epoch}"
        set_iterative_seed(config.seed, f"train_classifier[epoch={epoch}]")
        env.log(info)

        torch.autograd.set_detect_anomaly(True)  # type: ignore

        # trick for ltt
        if config.train_classifier.EXPERIMENTAL_progressive_training:
            freeze_lys = math.ceil(epoch / 1)
            freeze_lys = min(freeze_lys, m_config.num_hidden_layers)
            env.log(f"  > freeze side branches exc. first {freeze_lys} layers")
            m_classifier.ltt_freeze_layers_until(freeze_lys)

        ts_begin = time.time()
        train_cls_loss, train_cls_acc = _classifier_epoch_train(
            env=env,
            device=device,
            n_players=n_players,
            d_items=d_loader.train(config.train_classifier.batch_size),
            m_recipe=m_recipe,
            m_classifier=m_classifier,
            m_classifier_set_mode=set_model_mode,
            optimizer=optimizer,
            epoch=epoch,
            gen_input=gen_input,
        )
        test_cls_loss, test_cls_acc = _classifier_epoch_eval(
            env=env,
            device=device,
            n_players=n_players,
            d_items=d_loader.test(config.train_classifier.batch_size),
            m_recipe=m_recipe,
            m_classifier=m_classifier,
            epoch=epoch,
            gen_input=gen_input,
        )
        scheduler.step()
        ts_delta = time.time() - ts_begin

        entry = {
            "epoch": epoch,
            "train_cls_loss": train_cls_loss,
            "train_cls_acc": train_cls_acc,
            "test_cls_loss": test_cls_loss,
            "test_cls_acc": test_cls_acc,
        }
        env.metrics(entry)
        info = f"  > epoch {epoch} done in {ts_delta:.2f}s // "
        info += f"train_loss: cls {train_cls_loss:.6f} // "
        info += f"test_loss: cls {test_cls_loss:.6f} // "
        info += f"test_acc: {test_cls_acc:.3f}"
        env.log(info)

        saved = save_epoch_ckpt(
            env.model_path, "classifier", config.train_classifier, epoch, m_classifier
        )
        if saved:
            env.flush_cfg()

    return


def _classifier_epoch_train(
    env: ExpEnv,
    device: torch.device,
    n_players: int,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, Any, Any, Any],
    m_classifier: TClassifier,
    m_classifier_set_mode: Optional[Callable[[TClassifier, bool], None]],
    optimizer: torch.optim.Optimizer,  # type: ignore
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[float, float]:
    """Classifier training epoch: backprop
    ret [0]: train_cls_loss (mean);
    ret [1]: train_cls_acc (0.0 ~ 1.0)."""

    cls_loss, correct, total = 0.0, 0, 0

    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)

        optimizer.zero_grad()
        m_classifier.train()
        if m_classifier_set_mode is not None:
            m_classifier_set_mode(m_classifier, True)
        base_Ys, _ = m_recipe.fw_classifier(m_classifier, Xs, Xs_mask_1)
        loss_cls = torch.nn.functional.cross_entropy(base_Ys, Zs)
        loss_cls.backward()
        optimizer.step()

        cls_loss += loss_cls.item()
        _, base_Zs = base_Ys.max(dim=1)
        correct += base_Zs.eq(Zs).sum().item()
        total += batch_size

        info = f"  > epoch {epoch} :{batch_idx}:train // "
        info += f"loss: cls {loss_cls.item() / batch_size:.6f} // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)

    return cls_loss / total, correct / total


def _classifier_epoch_eval(
    env: ExpEnv,
    device: torch.device,
    n_players: int,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, Any, Any, Any],
    m_classifier: TClassifier,
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[float, float]:
    """Classifier training epoch: evaluate
    ret [0]: test_cls_loss (mean).
    ret [1]: test_cls_acc (0.0 ~ 1.0)."""

    test_cls_loss, correct, total = 0.0, 0, 0

    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)

        m_classifier.eval()
        with torch.no_grad():
            base_Ys, _ = m_recipe.fw_classifier(m_classifier, Xs, Xs_mask_1)
            loss_cls = torch.nn.functional.cross_entropy(base_Ys, Zs)
            test_cls_loss += loss_cls.item()

        _, base_Zs = base_Ys.max(dim=1)
        correct += base_Zs.eq(Zs).sum().item()
        total += batch_size

        info = f"  > epoch {epoch} :{batch_idx}:test // "
        info += f"loss: cls {loss_cls.item() / batch_size:.6f} // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)

    return test_cls_loss / total, correct / total
