import time
from typing import Any, Callable, Iterable, List, Tuple

import torch
import torch.utils.data
import wandb
from torch import Tensor

from ..models.shapley import loss_shapley_new, mask_shapley_new
from ..recipes.types import ModelRecipe, TConfig, TExplainer, TSurrogate
from ..utils.tools import set_iterative_seed
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model, save_epoch_ckpt


def train_duo_explainer(env: ExpEnv, device: torch.device) -> None:
    env.log("[[[ !!! *experimental* train (duo) classifier + explainer !!! ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.training.support_explainer or not m_recipe.training.exp_variant_duo:
        env.log("[[[ skip: explainer cannot be trained ]]]")
        return

    d_loader = load_cfg_dataset(config.dataset, env.model_path)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    gen_null = m_recipe.gen_null(m_config, m_misc, device)

    _epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )
    epoch_explainer, m_explainer = load_epoch_model(
        env, m_recipe, "explainer", device=device
    )

    optimizer = torch.optim.AdamW(  # type: ignore
        m_explainer.parameters(), lr=config.train_explainer.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.train_explainer.epochs
    )

    # surrogate_null
    nil_Xs = gen_null
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    m_surrogate.eval()
    with torch.no_grad():
        # <1, n_classes>
        surrogate_null, _ = m_recipe.fw_surrogate(m_surrogate, nil_Xs, nil_mask)

    for epoch in range(epoch_explainer + 1, config.train_explainer.epochs + 1):
        info = f"### epoch {epoch}"
        set_iterative_seed(config.seed, f"train_explainer[epoch={epoch}]")
        env.log(info)

        torch.autograd.set_detect_anomaly(True)  # type: ignore

        ts_begin = time.time()
        train_cls_loss, train_reg_loss, train_loss, train_cls_acc = (
            _duo_explainer_epoch_train(
                env=env,
                device=device,
                n_mask_samples=config.train_explainer.n_mask_samples,
                n_players=n_players,
                surrogate_null=surrogate_null,
                d_items=d_loader.train(config.train_explainer.batch_size),
                m_recipe=m_recipe,
                m_surrogate=m_surrogate,
                m_explainer=m_explainer,
                optimizer=optimizer,
                epoch=epoch,
                gen_input=gen_input,
            )
        )
        test_cls_loss, test_reg_loss, test_loss, test_plots, test_cls_acc = (
            _duo_explainer_epoch_eval(
                env=env,
                device=device,
                n_mask_samples=config.train_explainer.n_mask_samples,
                n_players=n_players,
                surrogate_null=surrogate_null,
                d_items=d_loader.test(config.train_explainer.batch_size),
                m_recipe=m_recipe,
                m_surrogate=m_surrogate,
                m_explainer=m_explainer,
                epoch=epoch,
                gen_input=gen_input,
            )
        )
        scheduler.step()
        ts_delta = time.time() - ts_begin

        entry = {
            "epoch": epoch,
            "train_cls_loss": train_cls_loss,
            "train_reg_loss": train_reg_loss,
            "train_loss": train_loss,
            "train_cls_acc": train_cls_acc,
            "test_cls_loss": test_cls_loss,
            "test_reg_loss": test_reg_loss,
            "test_loss": test_loss,
            "test_cls_acc": test_cls_acc,
            "test_plots": test_plots,
        }
        env.metrics(entry)
        info = f"  > epoch {epoch} done in {ts_delta:.2f}s // "
        info += f"train_loss: shap {train_reg_loss:.6f} // "
        info += f"test_loss: shap {test_reg_loss:.6f}"
        env.log(info)

        saved = save_epoch_ckpt(
            env.model_path, "explainer", config.train_explainer, epoch, m_explainer
        )
        if saved:
            env.flush_cfg()

    return


def _duo_explainer_epoch_train(
    env: ExpEnv,
    device: torch.device,
    n_mask_samples: int,
    n_players: int,
    surrogate_null: Tensor,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, Any, TSurrogate, TExplainer, Any],
    m_surrogate: TSurrogate,
    m_explainer: TExplainer,
    optimizer: torch.optim.Optimizer,  # type: ignore
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[float, float, float, float]:
    """Classifier+Explainer training epoch: backprop
      - Assuming num_tokens = cls_token + actual_features.
    ret [0]: train_cls_loss (mean);
    ret [1]: train_reg_loss (mean);
    ret [2]: train_loss (mean);
    ret [3]: train_cls_acc (0.0 ~ 1.0)."""

    cls_loss, reg_loss, tot_loss, correct, total = 0, 0, 0, 0, 0

    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)
        # <batch_size * n_mask_samples, n_players>
        Xs_mask_shap_ = mask_shapley_new(batch_size * n_mask_samples, n_players).to(
            device
        )
        # <batch_size, n_mask_samples, n_players>
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
            # <batch_size * n_mask_samples, n_classes>
            surrogate_values, _ = m_recipe.fw_surrogate(
                m_surrogate, Xs_EXT, Xs_mask_shap_
            )
            # <batch_size, n_classes>
            surrogate_grand, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_1)
            # <batch_size * n_mask_samples, n_classes>
            surrogate_grand_EXT = surrogate_grand.reshape((batch_size, 1, -1))
            surrogate_grand_EXT = surrogate_grand_EXT.repeat(1, n_mask_samples, 1)
            surrogate_grand_EXT = surrogate_grand_EXT.reshape(
                (batch_size * n_mask_samples, -1)
            )

        optimizer.zero_grad()
        m_explainer.train()
        # <batch_size, n_classes, n_players>
        explainer_shap, base_Ys = m_recipe.fw_explainer(
            m_explainer, Xs, Xs_mask_1, surrogate_grand, surrogate_null
        )
        assert base_Ys is not None
        loss_cls = torch.nn.functional.cross_entropy(base_Ys, Zs)
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
        loss = loss_cls + loss_shap
        loss.backward()
        optimizer.step()

        cls_loss += loss_cls.item()
        reg_loss += loss_shap.item()
        tot_loss += loss.item()
        _, base_Zs = base_Ys.max(dim=1)
        correct += base_Zs.eq(Zs).sum().item()
        total += batch_size

        info = f"  > epoch {epoch} :{batch_idx}:train // "
        info += f"loss: cls {loss_cls.item() / batch_size:.6f} "
        info += f"shap {loss_shap.item() / batch_size:.6f} "
        info += f"tot {loss.item() / batch_size:.6f} // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)

    return cls_loss / total, reg_loss / total, tot_loss / total, correct / total


def _duo_explainer_epoch_eval(
    env: ExpEnv,
    device: torch.device,
    n_mask_samples: int,
    n_players: int,
    surrogate_null: Tensor,
    d_items: Iterable[Tuple[Any, Any]],
    m_recipe: ModelRecipe[TConfig, Any, Any, TSurrogate, TExplainer, Any],
    m_surrogate: TSurrogate,
    m_explainer: TExplainer,
    epoch: int,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
) -> Tuple[float, float, float, float, List[wandb.Image]]:
    """Classifier+Explainer training epoch: evaluate
    ret [0]: test_cls_loss (mean);
    ret [1]: test_reg_loss (mean);
    ret [2]: test_loss (mean);
    ret [3]: test_cls_acc (mean);
    ret [4]: test_plots (List[wandb.Image])."""

    cls_loss, reg_loss, tot_loss, correct, total = 0, 0, 0, 0, 0
    test_plots: List[wandb.Image] = []

    # see train for dimensions
    for batch_idx, (_inputs, _targets) in enumerate(d_items):
        Xs, Zs = gen_input(_inputs, _targets)
        batch_size, *_ = Xs.shape
        Xs_mask_1 = torch.ones((batch_size, n_players), dtype=torch.long, device=device)
        Xs_mask_shap_ = mask_shapley_new(batch_size * n_mask_samples, n_players).to(
            device
        )
        Xs_mask_shap = Xs_mask_shap_.reshape((batch_size, n_mask_samples, n_players))
        Xs_EXT = []
        for b in range(batch_size):
            for _ in range(n_mask_samples):
                Xs_EXT.append(Xs[b])
        Xs_EXT = torch.stack(Xs_EXT, dim=0)

        m_surrogate.eval()
        with torch.no_grad():
            surrogate_values, _ = m_recipe.fw_surrogate(
                m_surrogate, Xs_EXT, Xs_mask_shap_
            )
            surrogate_grand, _ = m_recipe.fw_surrogate(m_surrogate, Xs, Xs_mask_1)
            surrogate_grand_EXT = surrogate_grand.reshape((batch_size, 1, -1))
            surrogate_grand_EXT = surrogate_grand_EXT.repeat(1, n_mask_samples, 1)
            surrogate_grand_EXT = surrogate_grand_EXT.reshape(
                (batch_size * n_mask_samples, -1)
            )

        m_explainer.eval()
        with torch.no_grad():
            explainer_shap, base_Ys = m_recipe.fw_explainer(
                m_explainer, Xs, Xs_mask_1, surrogate_grand, surrogate_null
            )
            assert base_Ys is not None
        loss_cls = torch.nn.functional.cross_entropy(base_Ys, Zs)
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
        loss = loss_cls + loss_shap

        cls_loss += loss_cls.item()
        reg_loss += loss_shap.item()
        tot_loss += loss.item()
        _, base_Zs = base_Ys.max(dim=1)
        correct += base_Zs.eq(Zs).sum().item()
        total += batch_size

        # ... todo: make plots

        info = f"  > epoch {epoch} :{batch_idx}:test // "
        info += f"loss: cls {loss_cls.item() / batch_size:.6f} "
        info += f"shap {loss_shap.item() / batch_size:.6f} "
        info += f"tot {loss.item() / batch_size:.6f} // "
        info += f"acc: {100.0 * correct / total:.3f}%, {correct}/{total}"
        env.log(info)

    return (
        cls_loss / total,
        reg_loss / total,
        tot_loss / total,
        correct / total,
        test_plots,
    )
