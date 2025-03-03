import json
import os
import pathlib
from typing import Dict, List, Tuple

import bs4
import numpy as np
import typer

from ..scripts.run_text_explanation import RunTextExplanationResults, _mix_color

app = typer.Typer()


@app.command()
def draw_text_explanation(
    into: pathlib.Path,
    files: List[pathlib.Path],
) -> None:
    data: Dict[str, RunTextExplanationResults] = {}
    for fn in files:
        print(f"> including: {fn.resolve().as_posix()}")
        with open(fn, "r", encoding="utf-8") as f:
            r = f.read()
            j = json.loads(r)
            results = RunTextExplanationResults.model_validate(j)
        data[os.path.splitext(fn.name)[0]] = results

    dom = "<html><head></head><body>"
    keys = sorted(list(data.values())[0].items.keys())
    for key in keys:
        items: Dict[str, List[Tuple[str, float]]] = {}
        for name, results in data.items():
            items[name] = results.items[key]
        # paint
        dom += f"<h2># {key}</h2>"
        attrs: List[float] = []
        for _, points in items.items():
            for _, grad in points:
                attrs.append(grad)
        cl_lim = max(abs(min(attrs)), abs(max(attrs)))
        for name, points in items.items():
            points = heuristic_merge(points, into=15)
            dom += _paint_text(cl_lim, name, points)
    dom += "</body></html>"

    output = into.resolve().as_posix()
    print(f"> writing: {output}")
    dom_fmt = bs4.BeautifulSoup(dom, "html.parser")
    with open(output, "w", encoding="utf-8") as f:
        f.write(dom_fmt.prettify())
    return


def heuristic_merge(tks: List[Tuple[str, float]], into: int) -> List[Tuple[str, float]]:
    grp: List[Tuple[int, int, List[float]]] = [
        (i, i, [sc]) for i, (_tk, sc) in enumerate(tks)
    ]

    def score_of(
        lhs: Tuple[int, int, List[float]], rhs: Tuple[int, int, List[float]]
    ) -> float:
        _l, _r, sc_l = lhs
        _l, _r, sc_r = rhs
        sc_a = np.array(sc_l + sc_r)
        sc = sc_a.std()
        return float(sc)

    while len(grp) > into:
        # find 2 smallest adjacent groups
        min_idx = 1
        for i in range(1, len(grp)):
            sc_i = score_of(grp[i - 1], grp[i])
            sc_min = score_of(grp[min_idx - 1], grp[min_idx])
            if abs(sc_i) < abs(sc_min):
                min_idx = i
        # merge the groups
        gl, gr = grp[min_idx - 1], grp[min_idx]
        gn = (gl[0], gr[1], gl[2] + gr[2])
        grp.pop(min_idx)
        grp.pop(min_idx - 1)
        grp.append(gn)
        grp.sort()

    s_tk = [tk for tk, _sc in tks]
    ret: List[Tuple[str, float]] = []
    for li, ri, sc in grp:
        sc = np.array(sc)
        sc_m = float(sc.mean())
        ret.append(("".join(s_tk[li : ri + 1]), sc_m))
    return ret


def _paint_text(cl_lim: float, name: str, points: List[Tuple[str, float]]) -> str:
    dom = f'<h3 style="font-weight: bold;">{name}</h3>'
    dom += "<p>"
    for word, grad in points:
        cl_begin = (18, 132, 255)  # < 0
        cl_mid = (255, 255, 255)
        cl_end = (237, 127, 127)  # > 0
        if grad < -cl_lim:
            color = cl_begin
        elif -cl_lim <= grad < 0:
            color = _mix_color(cl_begin, cl_mid, -grad / cl_lim)
        elif 0 <= grad < cl_lim:
            color = _mix_color(cl_mid, cl_end, 1.0 - grad / cl_lim)
        elif cl_lim <= grad:
            color = cl_end
        else:
            color = (0, 0, 0)
        dom += f'<span style="color: #000000; background-color: rgb({color[0]}, {color[1]}, {color[2]});">{word}</span>'
    dom += "</p>"
    return dom
