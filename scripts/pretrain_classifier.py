import json
import pathlib
from typing import Any, Callable

import torch
import torch.utils.data

from ..recipes.vanilla_bert import VanillaBertMisc
from ..recipes.vanilla_vit import VanillaViTMisc
from ..utils.nnmodel import freeze_model_parameters
from .env import ExpEnv
from .resources import get_recipe, load_epoch_ckpt, load_epoch_model
from .train_all import conv_pretrained_classifier
from .train_classifier import train_classifier


def pretrain_classifier(env: ExpEnv, device: torch.device) -> None:
    env.log("[[[ fine-tune pretrained model ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.training.support_classifier:
        raise ValueError("cannot fine-tune model: classification not supported")

    export_misc: Callable[[Any, pathlib.Path], None] = ...
    if config.net.kind == "vanilla_bert":
        export_misc = _export_misc_bert
        set_model_mode = lambda net, _train: freeze_model_parameters(  # noqa: E731
            net, ..., requires_grad=True
        )
    elif config.net.kind == "vanilla_vit":
        export_misc = _export_misc_vit
        set_model_mode = lambda net, _train: freeze_model_parameters(  # noqa: E731
            net, ..., requires_grad=True
        )
    else:
        raise ValueError(f"unsupported model kind: {config.net.kind}")

    epoch_classifier, _ = load_epoch_ckpt(
        env.model_path, "classifier", config.train_classifier.epochs
    )
    if epoch_classifier is None:
        env.log(":: initializing ft model")
        conv_pretrained_classifier(env)
        epoch_classifier = 0
    if epoch_classifier < config.train_classifier.epochs:
        env.log(f":: training ft model from epoch {epoch_classifier}")

        train_classifier(env, device, set_model_mode=set_model_mode)

    m_misc = m_recipe.load_misc(env.model_path, m_config)
    epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )
    if epoch_classifier < config.train_classifier.epochs:
        raise ValueError("classifier not fully trained")

    dest_path = pathlib.Path(__file__).parent / "../params/" / env.model_path.name
    dest_path.mkdir(parents=True, exist_ok=True)
    with open(dest_path / "model.json", "w", encoding="utf-8") as f:
        j = json.loads(config.net.params.model_dump_json())
        f.write(json.dumps(j, indent=2))
    export_misc(m_misc, dest_path)
    torch.save(m_classifier.state_dict(), dest_path / "model.ckpt")

    env.log("[[[ fine-tuning complete ]]]")
    return


def _export_misc_bert(misc: VanillaBertMisc, path: pathlib.Path) -> None:
    misc.tokenizer.save_pretrained(path)
    return


def _export_misc_vit(misc: VanillaViTMisc, path: pathlib.Path) -> None:
    return
