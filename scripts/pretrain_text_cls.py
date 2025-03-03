import pathlib
import shutil

import torch
import torch.utils.data

from .env import ExpEnv
from .resources import load_epoch_ckpt
from .train_all import conv_pretrained_classifier
from .train_classifier import train_classifier


def pretrain_text_cls(env: ExpEnv, device: torch.device) -> None:
    env.log("[[[ pretrain text classifier ]]]")
    config = env.config
    assert config.net.kind == "vanilla_bert"
    assert config.train_classifier.epochs > 0
    assert config.train_surrogate.epochs == 0
    assert config.train_explainer.epochs == 0

    epoch_classifier, _m_classifier = load_epoch_ckpt(env.model_path, "classifier", 0)
    del _m_classifier
    if epoch_classifier is None:
        conv_pretrained_classifier(env)
    train_classifier(env, device)

    out_name = env.model_path.name
    out_dir = pathlib.Path(__file__).parent / "../params/" / out_name
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    env.log(f"> copying artifacts to {out_dir}...")
    shutil.copytree(env.model_path / "tokenizer/", out_dir / "tokenizer/")
    shutil.copytree(env.model_path / ".hparams.json", out_dir / ".hparams.json")
    shutil.copytree(
        env.model_path / f"classifier-epoch-{config.train_classifier.epochs}.ckpt",
        out_dir / "classifier.ckpt",
    )
    env.log(f"[[[ pretrain text classifier ok: {out_name} ]]]")
    return
