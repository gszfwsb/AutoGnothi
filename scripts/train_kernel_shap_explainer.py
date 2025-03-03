from typing import List, cast

import shap
import torch
import torch.utils.data
from torch import Tensor

from ..models.kernel_shap_bert import KernelShapBertConfig
from .env import ExpEnv
from .resources import get_recipe, load_cfg_dataset, load_epoch_model, save_epoch_ckpt


def train_kernel_shap_explainer(env: ExpEnv, device: torch.device) -> None:
    config = env.config
    m_recipe, m_config = get_recipe(config)
    if (
        not m_recipe.training.support_explainer
        and not m_recipe.training.exp_variant_kernel_shap
    ):
        env.log("[[[ skip: explainer cannot be trained ]]]")
        return

    d_loader = load_cfg_dataset(config.dataset, env.model_path)
    m_misc = m_recipe.load_misc(env.model_path, m_config)
    gen_input = m_recipe.gen_input(m_config, m_misc, device)

    epoch_explainer, m_explainer = load_epoch_model(
        env, m_recipe, "explainer", device=device
    )
    if epoch_explainer >= config.train_explainer.epochs:
        env.log("[[[ explainer already trained ]]]")
        return

    # just load data
    env.log("> loading data...")
    all_Xs: List[Tensor] = []
    for _inputs, _targets in d_loader.train(config.train_explainer.batch_size):
        Xs, _Zs = gen_input(_inputs, _targets)
        all_Xs.append(Xs.cpu())
    Xs = torch.cat(all_Xs)
    env.log(f"> received bulk data: {Xs.shape}")

    # compress
    data_size = 1
    if config.net.kind == "kernel_shap_bert":
        m_config = cast(KernelShapBertConfig, m_config)
        data_size = m_config.kernel_shap_data_size
    else:
        raise ValueError(f"unsupported model: {config.net.kind}")
    Xs_small = torch.from_numpy(shap.kmeans(Xs, data_size).data)
    Xs_small = Xs_small.to(dtype=torch.long, device=device)
    env.log(f"> compressed data: {Xs_small.shape}")

    # apply precomputed parameters
    model_sd = m_explainer.state_dict()
    if config.net.kind == "kernel_shap_bert":
        model_sd["Xs_train"] = Xs_small
    else:
        raise ValueError(f"unsupported model: {config.net.kind}")
    m_explainer.load_state_dict(model_sd)
    env.log("> loaded precomputed parameters")

    # save model ckpt
    saved = save_epoch_ckpt(
        env.model_path,
        "explainer",
        config.train_explainer,
        config.train_explainer.epochs,
        m_explainer,
    )
    if saved:
        env.flush_cfg()
    return
