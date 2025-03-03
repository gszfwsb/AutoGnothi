import json
import pathlib
from typing import Dict, List, Optional, Tuple

import pydantic
import rich.console
import torch
from transformers import PreTrainedTokenizerBase

from ..datasets.loader import DatasetLoader
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model


class RunTextExplanationResults(pydantic.BaseModel):
    items: Dict[int, List[Tuple[str, float]]]


console = rich.get_console()


def run_text_explanation(
    env: ExpEnv,
    device: torch.device,
    d_loader: Optional[DatasetLoader],
    into: pathlib.Path,
    limit: Optional[int],
) -> None:
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if d_loader is None:
        d_loader = load_cfg_dataset(config.dataset, env.model_path)

    _epoch_final, m_final = load_epoch_model(env, m_recipe, "final", device=device)

    m_misc = m_recipe.load_misc(env.model_path, m_config)

    tokenizer: PreTrainedTokenizerBase = m_misc.tokenizer  # HACK: FORCE CAST
    gen_input = m_recipe.gen_input(m_config, m_misc, device)
    result_buffer: List[List[Tuple[str, float]]] = []
    for i, (_inputs, _targets) in enumerate(d_loader.test(1)):
        if limit is not None and i >= limit:
            break
        Xs, _Zs = gen_input(_inputs, _targets)
        with torch.no_grad():
            logits, attr = m_recipe.fw_final(m_final, Xs)
        # align with original input
        _, _pred_Zs = logits.max(dim=1)
        Zs, pred_Zs = int(_Zs.item()), int(_pred_Zs.item())
        if Zs != pred_Zs:
            continue

        tokens = real_tokenize_text(Xs[0].tolist(), tokenizer)
        pairs = [(w, attr[0, Zs, i].item()) for i, w in tokens if i < attr.shape[2]]
        print(f"# {i}")
        print_label(Zs, int(Zs))
        print_text_attr(pairs)
        print("\n")
        result_buffer.append(pairs)

    env.log(f"saving into: {into}")
    results = RunTextExplanationResults(
        items={i: r for i, r in enumerate(result_buffer)}
    )
    with open(into, "w", encoding="utf-8") as f:
        j = results.model_dump_json()
        r = json.loads(j)
        j = json.dumps(r, indent=2)
        f.write(j + "\n")
    return


def real_tokenize_text(
    tks: List[int], tokenizer: PreTrainedTokenizerBase
) -> List[Tuple[int, str]]:
    ret: List[Tuple[int, str]] = []
    for i, tk in enumerate(tks):
        if tk in tokenizer.all_special_ids:
            continue
        s = tokenizer.decode(tk).strip()
        if not s:
            s = " "
        if s[0].isalpha():
            s = " " + s
        elif s.startswith("##"):
            s = s[2:]
        else:
            pass
        ret.append((i, s))
    ret[0] = (ret[0][0], ret[0][1].lstrip())
    ret[-1] = (ret[-1][0], ret[-1][1].rstrip())
    return ret


###############################################################################
#   CLI print


def print_label(label: int, pred: int) -> None:
    style = "bold green" if label == pred else "white"
    console.print(f"[{label}] ", style=style, end="", highlight=False)
    return


def print_text_attr(tks_scores: List[Tuple[str, float]]) -> None:
    attr = [at for _, at in tks_scores]
    for tk, at in tks_scores:
        cl_lim = max(abs(min(attr)), abs(max(attr)))
        # cl_lim = 1.0
        cl_begin = (18, 132, 255)  # < 0
        cl_mid = (224, 224, 224)
        cl_end = (237, 127, 127)  # > 0
        if at < -cl_lim:
            color = cl_begin
        elif -cl_lim <= at < 0:
            color = _mix_color(cl_begin, cl_mid, -at / cl_lim)
        elif 0 <= at < cl_lim:
            color = _mix_color(cl_mid, cl_end, 1.0 - at / cl_lim)
        elif cl_lim <= at:
            color = cl_end
        else:
            color = (0, 0, 0)
        console.print(
            f"{tk}",
            style=f"rgb({color[0]},{color[1]},{color[2]})",
            highlight=False,
            end="",
        )
    return


def _mix_color(
    cl: Tuple[int, int, int], cr: Tuple[int, int, int], r: float
) -> Tuple[int, int, int]:
    cc = list(int(cl[i] * r + cr[i] * (1 - r)) for i in range(3))
    return tuple(cc)  # type: ignore
