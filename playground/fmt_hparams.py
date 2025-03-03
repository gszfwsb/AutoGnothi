import json
import pathlib
from typing import cast

from ..scripts.types import Config_Dataset_YelpPolarity, ExpConfig
from ..utils.tools import not_null


def main() -> None:
    experiments = pathlib.Path(__file__).parent / "../experiments/"
    for folder in experiments.iterdir():
        if not folder.is_dir():
            continue
        hparams_p = folder / ".hparams.json"
        if not hparams_p.exists():
            continue
        print(folder.name)
        # format conversion
        with open(hparams_p, "r", encoding="utf-8") as f:
            raw = f.read()
            obj = json.loads(raw)
            hparams = ExpConfig.model_validate(obj)
        # modifiers
        _modify_tayp(folder.name, hparams)
        # save file
        new_raw = hparams.model_dump_json(by_alias=True, exclude_unset=True)
        obj = json.loads(new_raw)
        new_raw = json.dumps(obj, indent=2)
        if new_raw.strip() != raw.strip():
            with open(hparams_p, "w", encoding="utf-8") as f:
                f.write(new_raw + "\n")
    return


def _modify_tayp(name: str, hp: ExpConfig) -> None:
    cast(Config_Dataset_YelpPolarity, hp.dataset).train_size = 2048
    cast(Config_Dataset_YelpPolarity, hp.dataset).test_size = 128
    hp.train_classifier.epochs = 0
    hp.train_classifier.ckpt_when = "<=20:%2==0; _:%5==0"
    hp.train_classifier.batch_size = 32
    hp.train_surrogate.epochs = 20
    hp.train_surrogate.ckpt_when = "<=20:%2==0; _:%5==0"
    hp.train_surrogate.batch_size = 32
    hp.train_explainer.epochs = 100
    hp.train_explainer.ckpt_when = "<=20:%2==0; <=50:%5==0; _:%10==0"
    hp.train_explainer.batch_size = 2
    hp.train_explainer.n_mask_samples = 16
    hp.eval_accuracy.batch_size = 32
    hp.eval_accuracy.resolution = 48
    hp.eval_faithfulness.batch_size = 32
    hp.eval_faithfulness.resolution = 16
    key = "autognothi-" + name
    not_null(hp.logger_classifier).wandb_name = f"{key}-cls"
    not_null(hp.logger_surrogate).wandb_name = f"{key}-srg"
    not_null(hp.logger_explainer).wandb_name = f"{key}-exp"
    hp.eval_performance.loops = 1
    return
