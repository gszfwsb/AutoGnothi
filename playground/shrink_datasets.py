import json
import os
import pathlib
import random
from typing import Dict, List, Union

from ..scripts.resources import load_cfg_dataset
from ..scripts.shell import _load_env
from ..scripts.types import (
    Config_Dataset,
    Config_Dataset_YelpPolarity,
)


def main() -> None:
    # this script reduces the size of the datasets for faster testing
    pick_dataset(
        dataset=Config_Dataset_YelpPolarity(
            kind="yelp_polarity",
            train_size=560000,
            test_size=38000,
            test_seed=2333,
        ),
        ds_test_size=38000,
        ft_model_id="ft_bert_base_yelp",
        out_id="yelp_polarity_mini",
    )
    return


TAKE_ALL = False


def pick_dataset(
    dataset: Config_Dataset, ds_test_size: int, ft_model_id: str, out_id: str
) -> None:
    # load
    root = pathlib.Path(__file__).parent
    model_path = root / f"../experiments/{ft_model_id}/"
    env = _load_env(model_path)
    d_loader = load_cfg_dataset(dataset, env.model_path)

    # pick
    n, pick_n = ds_test_size, 1000
    if TAKE_ALL:
        pick_n = n
    seed = int(os.urandom(4).hex(), 16)
    random.seed(seed)
    keep_ids: List[int] = []
    keep_ids = list(range(n))
    random.shuffle(keep_ids)
    keep_ids = keep_ids[:pick_n]
    keep_ids_set = set(keep_ids)

    # save
    result_j: List[Dict[str, Union[str, int]]] = []
    for i, (_inputs, _targets) in enumerate(d_loader.test(1)):
        if i not in keep_ids_set:
            continue
        result_j.append({"inputs": _inputs[0], "targets": _targets[0]})

    result_p = root / f"../datasets/{out_id}/test.json"
    result_p.parent.mkdir(parents=True, exist_ok=True)
    with open(result_p, "w", encoding="utf-8") as f:
        f.write(json.dumps(result_j, ensure_ascii=False, indent=2) + "\n")
    print(f"(seed {seed}) saved to {result_p}")
    return
