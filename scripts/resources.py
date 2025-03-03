import pathlib
from typing import (
    Any,
    List,
    Literal,
    Optional,
    OrderedDict,
    Tuple,
    Union,
    cast,
    overload,
)

import torch
import torch.utils.data
from torch import nn

from ..datasets.loader import (
    CvTransformResize,
    CvTransforms,
    DatasetLoader,
    load_imagenette,
    load_nlp_samples,
    load_yelp_polarity,
    load_yelp_polarity_mini,
)
from ..recipes.duo_vanilla_bert import duo_vanilla_bert_recipe
from ..recipes.duo_vanilla_vit import duo_vanilla_vit_recipe
from ..recipes.froyo_bert import froyo_bert_recipe
from ..recipes.froyo_vit import froyo_vit_recipe
from ..recipes.kernel_shap_bert import kernel_shap_bert_recipe
from ..recipes.ltt_bert import ltt_bert_recipe
from ..recipes.ltt_vit import ltt_vit_recipe
from ..recipes.types import (
    ModelRecipe,
    TClassifier,
    TConfig,
    TExplainer,
    TFinal,
    TSurrogate,
)
from ..recipes.vanilla_bert import vanilla_bert_recipe
from ..recipes.vanilla_vit import vanilla_vit_recipe
from ..scripts.types import (
    Config_Dataset,
    Config_Dataset_Kind,
    Config_Train,
    ExpConfig,
)
from ..utils.strings import ranged_modulo_test
from ..utils.tools import guard_never
from .env import ExpEnv


def get_recipe(
    config: ExpConfig,
) -> Tuple[ModelRecipe[TConfig, Any, Any, Any, Any, Any], TConfig]:
    if config.net.kind == "duo_vanilla_bert":
        recipe, cfg = duo_vanilla_bert_recipe(), config.net.params
    elif config.net.kind == "duo_vanilla_vit":
        recipe, cfg = duo_vanilla_vit_recipe(), config.net.params
    elif config.net.kind == "froyo_bert":
        recipe, cfg = froyo_bert_recipe(), config.net.params
    elif config.net.kind == "froyo_vit":
        recipe, cfg = froyo_vit_recipe(), config.net.params
    elif config.net.kind == "kernel_shap_bert":
        recipe, cfg = kernel_shap_bert_recipe(), config.net.params
    elif config.net.kind == "ltt_bert":
        recipe, cfg = ltt_bert_recipe(), config.net.params
    elif config.net.kind == "ltt_vit":
        recipe, cfg = ltt_vit_recipe(), config.net.params
    elif config.net.kind == "vanilla_bert":
        recipe, cfg = vanilla_bert_recipe(), config.net.params
    elif config.net.kind == "vanilla_vit":
        recipe, cfg = vanilla_vit_recipe(), config.net.params
    else:
        guard_never(config.net.kind)
    # keep track of recipe version
    if config.net.version != recipe.version:
        raise ValueError(
            f"mismatch recipe version: (config) {config.net.version} != (code) {recipe.version}"
        )
    return cast(Any, (recipe, cfg))


def load_id_dataset(
    kind: Union[str, Config_Dataset_Kind], img_px_size: Optional[int] = None
) -> DatasetLoader:
    def _default_cv_transforms() -> CvTransforms:
        assert img_px_size is not None
        return CvTransforms(
            resize=CvTransformResize(
                height=img_px_size,
                width=img_px_size,
            )
        )

    if kind == "nlp_samples":
        return load_nlp_samples()
    elif kind == "yelp_polarity_mini":
        return load_yelp_polarity_mini()
    elif kind == "yelp_polarity":
        return load_yelp_polarity(
            train_size=560000,
            test_size=38000,
            test_seed=42,
        )
    elif kind == "imagenette":
        assert img_px_size is not None
        return load_imagenette(
            train_size=9469,
            test_size=3925,
            test_seed=42,
            transforms=_default_cv_transforms(),
        )
    else:
        raise ValueError(f"unknown dataset kind: {kind}")


def load_cfg_dataset(cfg: Config_Dataset, root_dir: pathlib.Path) -> DatasetLoader:
    if cfg.kind == "nlp_samples":
        return load_nlp_samples()
    elif cfg.kind == "yelp_polarity_mini":
        return load_yelp_polarity_mini()
    elif cfg.kind == "yelp_polarity":
        return load_yelp_polarity(
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            test_seed=cfg.test_seed,
        )
    elif cfg.kind == "imagenette":
        return load_imagenette(
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            test_seed=cfg.test_seed,
            transforms=cfg.transforms,
        )
    else:
        guard_never(cfg.kind)


@overload
def load_epoch_ckpt(
    path: pathlib.Path, id: str, max_epochs: int, required: Literal[True]
) -> Tuple[int, OrderedDict[str, torch.Tensor]]: ...
@overload
def load_epoch_ckpt(
    path: pathlib.Path, id: str, max_epochs: int, required: bool = False
) -> Tuple[Optional[int], Optional[OrderedDict[str, torch.Tensor]]]: ...
def load_epoch_ckpt(
    path: pathlib.Path, id: str, max_epochs: int, required: bool = False
) -> Tuple[Optional[int], Optional[OrderedDict[str, torch.Tensor]]]:
    if hasattr(torch.serialization, "add_safe_globals"):
        # newer torch versions have this safeguard for untrusted code
        torch.serialization.add_safe_globals([OrderedDict])  # type: ignore
    files = [i.name for i in path.iterdir()]
    for epoch in range(max_epochs, -1, -1):
        ckpt_name = f"{id}-epoch-{epoch}.ckpt"
        if ckpt_name in files:
            obj = torch.load(
                path / ckpt_name,
                weights_only=False,
                map_location=torch.device("cpu"),
            )
            return epoch, obj
    if required:
        raise FileNotFoundError(f"no checkpoint found for '{id}' under '{path}'")
    return None, None


def get_epoch_ckpts(path: pathlib.Path, id: str, max_epochs: int) -> List[int]:
    epochs: List[int] = []
    for epoch in range(max_epochs + 1):
        ckpt_name = f"{id}-epoch-{epoch}.ckpt"
        if (path / ckpt_name).exists():
            epochs.append(epoch)
    return epochs


def save_epoch_ckpt(
    path: pathlib.Path,
    id: str,
    cfg: Config_Train,
    epoch: int,
    state_dict: Union[nn.Module, OrderedDict[str, torch.Tensor]],
) -> bool:
    """... -> saved?"""

    def _should_save(ep: int) -> bool:
        is_initial_ckpt = ep == 0
        is_intermediate_ckpt = ranged_modulo_test(cfg.ckpt_when)(ep)
        is_final_ckpt = ep == cfg.epochs
        return is_initial_ckpt or is_intermediate_ckpt or is_final_ckpt

    def _to_path(ep: int) -> pathlib.Path:
        ckpt_name = f"{id}-epoch-{ep}.ckpt"
        return path / ckpt_name

    if isinstance(state_dict, nn.Module):
        state_dict = OrderedDict(state_dict.state_dict())

    # https://docs.wandb.ai/guides/runs/rewind
    #     Rewind is in private preview at the moment. We therefore ensure that
    #     with every log we save the state of the model at that point in time.

    this_ckpt = _to_path(epoch)
    if this_ckpt.exists():
        this_ckpt.unlink()
    torch.save(state_dict, this_ckpt)

    # TODO: upload to cloud?

    if not _should_save(epoch - 1):
        last_ckpt = _to_path(epoch - 1)
        if last_ckpt.exists():
            last_ckpt.unlink()

    return True


@overload
def load_epoch_model(
    env: ExpEnv,
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, TExplainer, TFinal],
    section: Literal["classifier"],
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, TClassifier]: ...
@overload
def load_epoch_model(
    env: ExpEnv,
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, TExplainer, TFinal],
    section: Literal["surrogate"],
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, TSurrogate]: ...
@overload
def load_epoch_model(
    env: ExpEnv,
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, TExplainer, TFinal],
    section: Literal["explainer"],
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, TExplainer]: ...
@overload
def load_epoch_model(
    env: ExpEnv,
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, TExplainer, TFinal],
    section: Literal["final"],
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, TFinal]: ...
def load_epoch_model(
    env: ExpEnv,
    m_recipe: ModelRecipe[TConfig, Any, TClassifier, TSurrogate, TExplainer, TFinal],
    section: Literal["classifier", "surrogate", "explainer", "final"],
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, Union[TClassifier, TSurrogate, TExplainer, TFinal]]:
    m_recipe, m_config = get_recipe(env.config)
    if section == "classifier":
        max_epochs, net = env.config.train_classifier.epochs, m_recipe.t_classifier
    elif section == "surrogate":
        max_epochs, net = env.config.train_surrogate.epochs, m_recipe.t_surrogate
    elif section == "explainer":
        max_epochs, net = env.config.train_explainer.epochs, m_recipe.t_explainer
    elif section == "final":
        max_epochs, net = 0, m_recipe.t_final
    else:
        guard_never(section)
    epoch, mpt = load_epoch_ckpt(env.model_path, section, max_epochs, required=True)
    model = net(m_config)
    model.load_state_dict(mpt)
    model = model.to(device=device)
    model.eval()
    return epoch, model
