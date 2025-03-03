import json
import pathlib
from typing import Callable, Dict, List, Union

# [filename] -> [column_name] -> (file) -> value
GetResultsRules = Dict[str, Dict[str, Callable[[Dict], Union[int, float]]]]


def get_result(key: str, rules: GetResultsRules) -> Dict[str, Union[int, float, None]]:
    path = pathlib.Path(__file__).parent / f"../experiments/{key}/.reports/"
    result: Dict[str, Union[int, float, None]] = {}
    for fn, columns in rules.items():
        try:
            with open(path / fn, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
        except Exception:
            data = {}
        for col, rule in columns.items():
            try:
                result[col] = rule(data)
            except Exception:
                result[col] = None
    return result


def main() -> None:
    rules: GetResultsRules = {
        "cls_acc.json": {
            "cls_acc": lambda d: sum(d["accuracy"]) / len(d["accuracy"]),
        },
        "accuracy.json": {
            "srg_acc": lambda d: sum(d["accuracy"]) / len(d["accuracy"]),
        },
        "branches_cka.json": {
            "cka_linear_0": lambda d: d["all"]["linear_cka_avg"][0],
            "cka_linear_n": lambda d: d["all"]["linear_cka_avg"][-1],
            "cka_kernel_0": lambda d: d["all"]["kernel_cka_avg"][0],
            "cka_kernel_n": lambda d: d["all"]["kernel_cka_avg"][-1],
        },
        "faithfulness.json": {
            "insertion_auc": lambda d: d["insertion"]["auc"],
            "deletion_auc": lambda d: d["deletion"]["auc"],
        },
        "performance.json": {
            "params_all_cls": lambda d: d["classifier"]["params_all"],
            "params_all_srg": lambda d: d["surrogate"]["params_all"],
            "params_all_exp": lambda d: d["explainer"]["params_all"],
            "params_all_fin": lambda d: d["final"]["params_all"],
            "params_train_cls": lambda d: d["classifier"]["params_trainable"],
            "params_train_srg": lambda d: d["surrogate"]["params_trainable"],
            "params_train_exp": lambda d: d["explainer"]["params_trainable"],
            "params_train_fin": lambda d: d["final"]["params_trainable"],
            "gflops_cls": lambda d: d["classifier"]["gflops"],
            "gflops_srg": lambda d: d["surrogate"]["gflops"],
            "gflops_exp": lambda d: d["explainer"]["gflops"],
            "gflops_fin": lambda d: d["final"]["gflops"],
            "inf_tm_cls": lambda d: d["classifier"]["time_avg"],
            "inf_tm_srg": lambda d: d["surrogate"]["time_avg"],
            "inf_tm_exp": lambda d: d["explainer"]["time_avg"],
            "inf_tm_fin": lambda d: d["final"]["time_avg"],
        },
        "train_resources.json": {
            "trn_tm_srg": lambda d: d["srg_tm"]["avg"],
            "inf_tm_exp": lambda d: d["exp_tm"]["avg"],
            "trn_mem_srg": lambda d: d["srg_mem"]["avg"],
            "inf_mem_exp": lambda d: d["exp_mem"]["avg"],
        },
    }

    exps = pathlib.Path(__file__).parent / "../experiments/"
    keys = [p.name for p in exps.iterdir() if p.is_dir()]
    keys = [k for k in keys if k.startswith("1023_")]

    rows: Dict[str, Dict[str, Union[int, float, None]]] = {}
    for key in keys:
        print(f"- {key}")
        rows[key] = get_result(key, rules)

    columns = list(list(rows.values())[0].keys())
    raw_rows: List[List[str]] = []
    raw_rows.append(["key"] + columns)
    for key, row in rows.items():
        r = [key]
        for col in columns:
            item = row.get(col, None)
            item = f"{item}" if item is not None else "-"
            r.append(item)
        raw_rows.append(r)
    with open("grab_results.csv", "w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(",".join(row) + "\n")
    return
