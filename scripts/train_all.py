import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from ..params.loader import load_params
from ..recipes.vanilla_bert import vanilla_bert_recipe
from ..utils.tools import guard_never
from .env import ExpEnv
from .resources import get_recipe, load_epoch_ckpt, load_epoch_model, save_epoch_ckpt
from .train_classifier import train_classifier
from .train_explainer import train_explainer
from .train_surrogate import train_surrogate
from .types import Config_Train


def train_all(env: ExpEnv, device: torch.device) -> None:
    config = env.config

    def _detect_stage() -> int:
        # detect in inverse order
        epoch_final, _ = load_epoch_ckpt(env.model_path, "final", 0)
        if epoch_final is not None:
            return 7
        epoch_explainer, _ = load_epoch_ckpt(
            env.model_path, "explainer", config.train_explainer.epochs
        )
        if epoch_explainer is not None:
            if epoch_explainer == config.train_explainer.epochs:
                return 6
            return 5
        epoch_surrogate, _ = load_epoch_ckpt(
            env.model_path, "surrogate", config.train_surrogate.epochs
        )
        if epoch_surrogate is not None:
            if epoch_surrogate == config.train_surrogate.epochs:
                return 4
            return 3
        epoch_classifier, _ = load_epoch_ckpt(env.model_path, "classifier", 0)
        if epoch_classifier is not None:
            if epoch_classifier == config.train_classifier.epochs:
                return 2
            return 1
        return 0

    stage = _detect_stage()
    env.log(f"[[[ current stage: {stage} / 7 ]]]")
    if stage < 1:
        conv_pretrained_classifier(env)
    if stage < 2:
        with env.fork(lambda ec: ec.logger_classifier) as cl_env:
            train_classifier(cl_env, device)
    if stage < 3:
        conv_classifier_surrogate(env)
    if stage < 4:
        with env.fork(lambda ec: ec.logger_surrogate) as sg_env:
            train_surrogate(sg_env, device)
    if stage < 5:
        conv_surrogate_explainer(env)
    if stage < 6:
        with env.fork(lambda ec: ec.logger_explainer) as ex_env:
            train_explainer(ex_env, device)
    if stage < 7:
        conv_explainer_final(env)
    env.log("[[[ all stages ok ]]]")
    return


def conv_pretrained_classifier(env: ExpEnv) -> None:
    base_path = ...
    _ = vanilla_bert_recipe, guard_never, base_path

    env.log("[[[ loading base params... ]]]")
    base_params, base_misc = load_params(
        env.config.net.base_model, num_labels=env.config.net.params.num_labels
    )

    env.log("[[[ converting base -> classifier {0}... ]]]")
    m_recipe, m_config = get_recipe(env.config)
    m_classifier = m_recipe.conv_pretrained_classifier(m_config, base_params)
    train_classifier = Config_Train(
        epochs=0,
        ckpt_when="_:%1==0",
        lr=0.0,
        batch_size=1,
    )
    save_epoch_ckpt(env.model_path, "classifier", train_classifier, 0, m_classifier)

    if isinstance(base_misc, PreTrainedTokenizerBase):
        env.log("[[[ converting base tokenizer... ]]]")
        tk_path = env.model_path / "tokenizer"
        tk_path.mkdir(parents=True, exist_ok=True)
        base_misc.save_pretrained(tk_path)
    else:
        env.log("[[[ skipped base misc ]]]")

    env.log("[[[ convert base -> classifier {0} ok ]]]")
    return


def conv_classifier_surrogate(env: ExpEnv) -> None:
    env.log("[[[ loading classifier params... ]]]")
    m_recipe, m_config = get_recipe(env.config)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    epoch_classifier, m_classifier = load_epoch_model(env, m_recipe, "classifier")
    if epoch_classifier < env.config.train_classifier.epochs:
        raise ValueError("under-trained classifier")

    env.log(f"[[[ converting classifier {epoch_classifier} -> surrogate {0}... ]]]")
    m_surrogate = m_recipe.conv_classifier_surrogate(m_config, m_misc, m_classifier)
    save_epoch_ckpt(
        env.model_path, "surrogate", env.config.train_surrogate, 0, m_surrogate
    )
    env.log(f"[[[ convert classifier {epoch_classifier} -> surrogate {0} ok ]]]")
    return


def conv_surrogate_explainer(env: ExpEnv) -> None:
    env.log("[[[ loading surrogate params... ]]]")
    m_recipe, m_config = get_recipe(env.config)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    epoch_surrogate, m_surrogate = load_epoch_model(env, m_recipe, "surrogate")
    if epoch_surrogate < env.config.train_surrogate.epochs:
        raise ValueError("under-trained surrogate")

    env.log(f"[[[ converting surrogate {epoch_surrogate} -> explainer {0}... ]]]")
    m_explainer = m_recipe.conv_surrogate_explainer(m_config, m_misc, m_surrogate)
    save_epoch_ckpt(
        env.model_path, "explainer", env.config.train_explainer, 0, m_explainer
    )
    env.log(f"[[[ convert surrogate {epoch_surrogate} -> explainer {0} ok ]]]")
    return


def conv_explainer_final(env: ExpEnv) -> None:
    env.log("[[[ loading all params... ]]]")
    m_recipe, m_config = get_recipe(env.config)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    epoch_classifier, m_classifier = load_epoch_model(env, m_recipe, "classifier")
    epoch_surrogate, m_surrogate = load_epoch_model(env, m_recipe, "surrogate")
    epoch_explainer, m_explainer = load_epoch_model(env, m_recipe, "explainer")
    if epoch_classifier < env.config.train_classifier.epochs:
        raise ValueError("under-trained classifier")
    if epoch_surrogate < env.config.train_surrogate.epochs:
        raise ValueError("under-trained surrogate")
    if epoch_explainer < env.config.train_explainer.epochs:
        raise ValueError("under-trained explainer")

    env.log(f"[[[ converting models -> final {0}... ]]]")
    m_final = m_recipe.conv_explainer_final(
        m_config, m_misc, m_classifier, m_surrogate, m_explainer
    )
    if not _verify_final_coherency(env, torch.device("cpu"), m_final):
        raise ValueError("cannot save final model due to non-coherency")

    train_final = Config_Train(
        epochs=0,
        ckpt_when="_:%1==0",
        lr=0.0,
        batch_size=1,
    )
    save_epoch_ckpt(env.model_path, "final", train_final, 0, m_final)
    env.log(f"[[[ convert models -> final {0} ok ]]]")
    return


def _verify_final_coherency(
    env: ExpEnv,
    device: torch.device,
    m_final: nn.Module,
) -> bool:
    env.log("[[[ verifying final model coherency... ]]]")
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if not m_recipe.measurements.verify_final_coherency:
        env.log("[[[ skipped: net recipe does not support this ]]]")
        return True

    env.log("loading model parameters...")
    _epoch_classifier, m_classifier = load_epoch_model(
        env, m_recipe, "classifier", device=device
    )
    _epoch_surrogate, m_surrogate = load_epoch_model(
        env, m_recipe, "surrogate", device=device
    )
    _epoch_explainer, m_explainer = load_epoch_model(
        env, m_recipe, "explainer", device=device
    )

    env.log("judging...")
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    n_players = m_recipe.n_players(m_config)
    nil_Xs = m_recipe.gen_null(m_config, m_misc, device)
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    m_classifier.eval()
    m_surrogate.eval()
    m_explainer.eval()
    m_final.eval()

    with torch.no_grad():
        _, cls_ref = m_recipe.fw_classifier(m_classifier, nil_Xs, nil_mask)
        srg_ref, _ = m_recipe.fw_surrogate(m_surrogate, nil_Xs, nil_mask)
        surrogate_grand = surrogate_null = srg_ref
        exp_ref, _ = m_recipe.fw_explainer(
            m_explainer, nil_Xs, nil_mask, surrogate_grand, surrogate_null
        )
        cls_out, exp_out = m_recipe.fw_final(m_final, nil_Xs)

    cls_diff = torch.max(torch.abs(cls_ref - cls_out))
    exp_diff = torch.max(torch.abs(exp_ref - exp_out))
    env.log(f"cls_diff: {cls_diff.item()}, exp_diff: {exp_diff.item()}")

    eps = 1e-5
    if cls_diff > eps or exp_diff > eps:
        env.log("[[[ !!! final is not coherent !!! ]]]")
        raise ValueError("final model is not coherent")

    env.log("[[[ verified final model is coherent ]]]")
    return True
