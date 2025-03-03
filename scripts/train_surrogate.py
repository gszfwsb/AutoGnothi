import math
import time
from typing import Any, Callable, Iterable, Tuple

import torch
import torch.utils.data
from torch import Tensor

from ..models.shapley import loss_logits_kl_divergence, mask_purely_uniform
from ..recipes.types import ModelRecipe, TClassifier, TConfig, TSurrogate
from ..utils.tools import set_iterative_seed
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model, save_epoch_ckpt


def train_surrogate(env: ExpEnv, device: torch.device) -> None:
    env.log("[[[ train surrogate ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.training.support_surrogate:
        env.log("[[[ skip: surrogate cannot be trained ]]]")
        return

    d_loader = load_cfg_dataset(config.dataset, env.model_path)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)

    _epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )
    epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )
    if epoch_surrogate >= config.train_surrogate.epochs:
        env.log("[[[ surrogate already trained ]]]")
        return

    optimizer = torch.optim.AdamW(  # type: ignore
        m_surrogate.parameters(), lr=config.train_surrogate.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.train_surrogate.epochs
    )

    for epoch in range(epoch_surrogate + 1, config.train_surrogate.epochs + 1):
        info = f"### epoch {epoch}"
        set_iterative_seed(config.seed, f"train_surrogate[epoch={epoch}]")
        env.log(info)

        torch.autograd.set_detect_anomaly(True)  # type: ignore

        # trick for ltt
        if config.train_surrogate.EXPERIMENTAL_progressive_training:
            freeze_lys = math.ceil(epoch / 3)
            freeze_lys = min(freeze_lys, m_config.num_hidden_layers)
            env.log(f"  > freeze side branches exc. first {freeze_lys} layers")
            m_surrogate.ltt_freeze_layers_until(freeze_lys)

        ts_begin = time.time()
        train_kld_loss, train_cls_loss, train_cls_acc = _surrogate_epoch_train(
            env=env,
            device=device,
            n_players=n_players,
            d_items=d_loader.train(config.train_surrogate.batch_size),
            m_recipe=m_recipe,
            m_classifier=m_classifier,
            m_surrogate=m_surrogate,
            optimizer=optimizer,
            epoch=epoch,
            gen_input=gen_input,
        )
        test_kld_loss, test_cls_loss, test_cls_acc = _surrogate_epoch_eval(
            env=env,
            device=device,
            n_players=n_players,
            d_items=d_loader.test(config.train_surrogate.batch_size),
            m_recipe=m_recipe,
            m_classifier=m_classifier,
            m_surrogate=m_surrogate,
            epoch=epoch,
            gen_input=gen_input,
        )
        scheduler.step()
        ts_delta = time.time() - ts_begin

        entry = {
            "epoch": epoch,
            "train_kld_loss": train_kld_loss,
            "train_cls_loss": train_cls_loss,
            "train_cls_acc": train_cls_acc,
            "test_kld_loss": test_kld_loss,
            "test_cls_loss": test_cls_loss,
            "test_cls_acc": test_cls_acc,
        }
        env.metrics(entry)
        info = f"  > epoch {epoch} done in {ts_delta:.2f}s // "
        info += f"train_loss: kld {train_kld_loss:.6f} cls {train_cls_loss:.6f} // "
        info += f"test_loss: kld {test_kld_loss:.6f} cls {test_cls_loss:.6f} // "
        info += f"test_acc: {test_cls_acc:.3f}"
        env.log(info)

        saved = save_epoch_ckpt(
            env.model_path, "surrogate", config.train_surrogate, epoch, m_surrogate
        )
        if saved:
            env.flush_cfg()

    return


def _surrogate_epoch_train(
    env: ExpEnv,
    device: torch.device,
    n_players: int,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, Any, Any],
    m_classifier: TClassifier,
    m_surrogate: TSurrogate,
    optimizer: torch.optim.Optimizer,  # type: ignore
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[float, float, float]:
    """Surrogate training epoch: backprop
    ret [0]: train_kld_loss (mean);
    ret [1]: train_cls_loss (mean);
    ret [2]: train_cls_acc (0.0 ~ 1.0)."""

    train_kld_loss, train_cls_loss, correct, total = 0.0, 0.0, 0, 0

    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
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
        with torch.no_grad():
            loss_cls = torch.nn.functional.cross_entropy(adapt_Ys, Zs)
        optimizer.step()

        train_kld_loss += loss_kld.item()
        train_cls_loss += loss_cls.item()
        _, adapt_Zs = adapt_Ys.max(dim=1)
        correct += adapt_Zs.eq(Zs).sum().item()
        total += batch_size

        info = f"  > epoch {epoch} :{batch_idx}:train // "
        info += f"loss: kld {loss_kld.item() / batch_size:.6f} cls {loss_cls.item() / batch_size:.6f} // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)

    return train_kld_loss / total, train_cls_loss / total, correct / total


def _surrogate_epoch_eval(
    env: ExpEnv,
    device: torch.device,
    n_players: int,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, Any, Any],
    m_classifier: TClassifier,
    m_surrogate: TSurrogate,
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[float, float, float]:
    """Surrogate training epoch: evaluate
    ret [0]: test_kld_loss (mean);
    ret [1]: test_cls_loss (mean);
    ret [2]: test_cls_acc (0.0 ~ 1.0)."""

    test_kld_loss, test_cls_loss, correct, total = 0.0, 0.0, 0, 0

    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)
        Xs_mask_rand = mask_purely_uniform(batch_size, n_players).to(device)

        m_classifier.eval()
        with torch.no_grad():
            _, orig_Ys = m_recipe.fw_classifier(m_classifier, Xs, Xs_mask_1)

        m_surrogate.eval()
        with torch.no_grad():
            adapt_Ys, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_rand)
            loss_kld = loss_logits_kl_divergence(orig_Ys, adapt_Ys)
            loss_cls = torch.nn.functional.cross_entropy(adapt_Ys, Zs)
            test_kld_loss += loss_kld.item()
            test_cls_loss += loss_cls.item()

        _, adapt_Zs = adapt_Ys.max(dim=1)
        correct += adapt_Zs.eq(Zs).sum().item()
        total += batch_size

        info = f"  > epoch {epoch} :{batch_idx}:test // "
        info += f"loss: kld {loss_kld.item() / batch_size:.6f} cls {loss_cls.item() / batch_size:.6f} // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)

    return test_kld_loss / total, test_cls_loss / total, correct / total
