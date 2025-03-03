from typing import Any, Callable, Iterable, List, Optional, Tuple

import rich.console
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor

from ..datasets.loader import DatasetLoader
from ..recipes.types import ModelRecipe, TSurrogate
from ..utils.functional import batched
from .env import ExpEnv
from .run_text_explanation import print_label, print_text_attr, real_tokenize_text
from .resources import get_recipe, load_cfg_dataset, load_epoch_model

console = rich.get_console()


def preview_text_shapley(
    env: ExpEnv, device: torch.device, d_loader: Optional[DatasetLoader]
) -> None:
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if d_loader is None:
        d_loader = load_cfg_dataset(config.dataset, env.model_path)

    _epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )

    m_misc = m_recipe.load_misc(env.model_path, m_config)
    tokenizer = m_misc.tokenizer  # TODO: now supports text only
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    n_players = m_recipe.n_players(m_config)

    for i, (_inputs, _targets) in enumerate(d_loader.test(1)):
        Xs, Zs = gen_input(_inputs, _targets)
        sv, _v0, _vn = _get_shap(
            device=device,
            m_recipe=m_recipe,
            m_surrogate=m_surrogate,
            gen_input=gen_input,
            n_players=n_players,
            _inputs=_inputs,
            reps=8,
            batch_size=16,
        )

        tokens = real_tokenize_text(Xs[0].tolist(), tokenizer)
        attr_0 = [(w, sv[0, i].item()) for i, w in tokens if i < sv.shape[1]]
        attr_1 = [(w, sv[1, i].item()) for i, w in tokens if i < sv.shape[1]]
        print_label(0, int(Zs))
        print_text_attr(attr_0)
        console.print("")
        print_label(1, int(Zs))
        print_text_attr(attr_1)
        console.print("\n")

    return


def _get_shap(
    device: torch.device,
    m_recipe: ModelRecipe[Any, Any, Any, TSurrogate, Any, Any],
    m_surrogate: TSurrogate,
    gen_input: Callable[[Any, Any], Tuple[Tensor, Tensor]],
    n_players: int,
    _inputs: Any,
    reps: int,
    batch_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """(_inputs: str[]) -> (sv: <n_classes, n_players>, v0: <n_classes>, vn: <n_classes>)"""

    Xs, Zs = gen_input([_inputs], [0])
    n_classes = 2  # TODO: binary classification only
    perms: List[torch.Tensor] = []

    def _get_inputs() -> Iterable[torch.Tensor]:
        """ret [0]: input: <n_tokens>."""
        for _ in range(reps):
            yield Xs

    def _preprocess(_inp: torch.Tensor) -> torch.Tensor:
        """create exactly (?, n_players + 1) masks:
        inp [0]: _input: <n_tokens>;
        ret [0]: mask: <n_players + 1, n_tokens>."""

        perm = torch.randperm(n_players)
        perms.append(perm)
        masks = torch.zeros((n_players + 1, n_players), dtype=torch.long)
        for i in range(n_players + 1):
            masks[i, perm[:i]] = 1
        return masks.to(device)

    def _inference(mask: torch.Tensor) -> torch.Tensor:
        """inference:
        inp [0]: mask: <* <= batch_size, n_tokens>;
        ret [0]: logits: <*, n_classes>."""

        bs = mask.shape[0]
        _, n_tokens = Xs.shape
        inputs = Xs
        inputs = inputs.reshape(1, n_tokens)
        inputs = inputs.expand(bs, -1)
        with torch.no_grad():
            logits, _ = m_recipe.fw_surrogate(m_surrogate, inputs, mask)
        return logits

    sv = torch.zeros((n_players, n_classes))
    ret_v0: Optional[torch.Tensor] = None  # <n_classes>
    ret_vN: Optional[torch.Tensor] = None  # <n_classes>

    bar = tqdm.tqdm(total=n_players * reps, desc="explaining model (sampling)")
    for all_logits in batched(_get_inputs, _preprocess, _inference, batch_size):
        # all_logits: <n_players + 1, n_classes>
        vs = _evaluate_v(classes=all_logits)  # <n_players + 1, n_classes>
        # d_p_vs: v(S union {i}) - v(S), for each i in perm[i]
        d_p_vs = vs[1:] - vs[:-1]  # <n_players, n_classes>
        # d_vs: ..., for each i
        perm = perms.pop(0)
        d_vs = torch.zeros_like(d_p_vs)
        for i in range(n_players):
            d_vs[perm[i], :] = d_p_vs[i, :]
        sv += d_vs.to(sv.device)
        # save cached values
        ret_v0 = vs[0]
        ret_vN = vs[-1]
        bar.update(n_players)

    sv = sv.T.reshape(n_classes, -1) / reps

    return sv, ret_v0, ret_vN  # type: ignore


def _evaluate_v(
    classes: torch.Tensor,
) -> torch.Tensor:
    """Evaluate v as in v(S union {i}) - v(S) for each sample in the batch.

    classes: Tensor[batch_size, n_classes] of class probabilities;
    ret: Tensor[batch_size, n_classes] of v(S union {i}) - v(S) for each
        corresponding i and each selected label."""

    # batch_size = classes.shape[0]
    # [0]: softmax and make them sum up to 1
    p = F.softmax(classes, dim=1)
    # [1]: sharpen the probabilities
    p = torch.log(p / (1 - p + 1e-6))

    # # [2]: log sum exp trick -- it doesn't belong here
    # r = torch.logsumexp(classes, dim=1)

    return p
