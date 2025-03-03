import json
import pathlib
from typing import Literal, Optional, Union

import pydantic

from ..datasets.loader import CvTransforms
from ..models.duo_vanilla_bert import DuoVanillaBertConfig
from ..models.duo_vanilla_vit import DuoVanillaViTConfig
from ..models.froyo_bert import FroyoBertConfig
from ..models.froyo_vit import FroyoViTConfig
from ..models.kernel_shap_bert import KernelShapBertConfig
from ..models.ltt_bert import LttBertConfig
from ..models.ltt_vit import LttViTConfig
from ..models.vanilla_bert import VanillaBertConfig
from ..models.vanilla_vit import VanillaViTConfig
from ..utils.strings import flatten_dict

ConfigRelPath = str
"""Relative path follows these rules:
 1. "./..." or "../" always is relative to a certain root directory, which:
      - when loaded from a config file is the model directory (which happened
        to contain that file)
 2. or an absolute path otherwise"""


def resolve_config_rel_path(
    rel_path: ConfigRelPath, root_dir_at: pathlib.Path
) -> pathlib.Path:
    parts = rel_path.replace("\\", "/").split("/")
    if parts and (parts[0] == "." or parts[0] == ".."):
        return root_dir_at.joinpath(rel_path).resolve()
    else:
        return pathlib.Path(rel_path).resolve()


Config_Dataset_Kind = Literal[
    # mini sets
    "nlp_samples",
    # text classification
    "yelp_polarity_mini",
    "yelp_polarity",
    # image classification
    "imagenette",
]


class Config_Dataset_NlpSamples(pydantic.BaseModel):
    kind: Literal["nlp_samples"] = "nlp_samples"


class Config_Dataset_YelpPolarityMini(pydantic.BaseModel):
    kind: Literal["yelp_polarity_mini"] = "yelp_polarity_mini"


class Config_Dataset_YelpPolarity(pydantic.BaseModel):
    kind: Literal["yelp_polarity"] = "yelp_polarity"
    train_size: int  # <= 560000
    test_size: int  # <= 38000
    test_seed: int


class Config_Dataset_ImageNette(pydantic.BaseModel):
    kind: Literal["imagenette"] = "imagenette"
    train_size: int  # <= 9469
    test_size: int  # <= 3925
    test_seed: int
    transforms: CvTransforms


Config_Dataset = Union[
    Config_Dataset_NlpSamples,
    # text classification
    Config_Dataset_YelpPolarityMini,
    Config_Dataset_YelpPolarity,
    # image classification
    Config_Dataset_ImageNette,
]


Config_Net_BaseModel_BertClassifier = Literal[
    # community classifiers
    "bert_tayp",
    # base models
    "prj_bert_mini",
    "prj_bert_small",
    "prj_bert_medium",
    "gg_bert_base",
    "gg_bert_large",
    # locally fine-tuned models
    "ft_bert_base_yelp",
    "ft_bert_large_yelp",
    "ft_bert_medium_yelp",
    "ft_bert_mini_yelp",
    "ft_bert_small_yelp",
]


Config_Net_BaseModel_ViTClassifier = Literal[
    # base models
    "gg_vit_tiny",
    "gg_vit_small",
    "gg_vit_base",
    "gg_vit_large",
    # locally fine-tuned models
    "ft_vit_tiny_imagenette",
    "ft_vit_small_imagenette",
    "ft_vit_base_imagenette",
    "ft_vit_large_imagenette",
]


class Config_Net_DuoVanillaBert(pydantic.BaseModel):
    kind: Literal["duo_vanilla_bert"] = "duo_vanilla_bert"
    version: str
    base_model: Config_Net_BaseModel_BertClassifier
    params: DuoVanillaBertConfig


class Config_Net_DuoVanillaViT(pydantic.BaseModel):
    kind: Literal["duo_vanilla_vit"] = "duo_vanilla_vit"
    version: str
    base_model: Config_Net_BaseModel_ViTClassifier
    params: DuoVanillaViTConfig


class Config_Net_FroyoBert(pydantic.BaseModel):
    kind: Literal["froyo_bert"] = "froyo_bert"
    version: str
    base_model: Config_Net_BaseModel_BertClassifier
    params: FroyoBertConfig


class Config_Net_FroyoViT(pydantic.BaseModel):
    kind: Literal["froyo_vit"] = "froyo_vit"
    version: str
    base_model: Config_Net_BaseModel_ViTClassifier
    params: FroyoViTConfig


class Config_Net_KernelShapBert(pydantic.BaseModel):
    kind: Literal["kernel_shap_bert"] = "kernel_shap_bert"
    version: str
    base_model: Config_Net_BaseModel_BertClassifier
    params: KernelShapBertConfig


class Config_Net_LttBert(pydantic.BaseModel):
    kind: Literal["ltt_bert"] = "ltt_bert"
    version: str
    base_model: Config_Net_BaseModel_BertClassifier
    params: LttBertConfig


class Config_Net_LttViT(pydantic.BaseModel):
    kind: Literal["ltt_vit"] = "ltt_vit"
    version: str
    base_model: Config_Net_BaseModel_ViTClassifier
    params: LttViTConfig


class Config_Net_VanillaBert(pydantic.BaseModel):
    kind: Literal["vanilla_bert"] = "vanilla_bert"
    version: str
    base_model: Config_Net_BaseModel_BertClassifier
    params: VanillaBertConfig


class Config_Net_VanillaViT(pydantic.BaseModel):
    kind: Literal["vanilla_vit"] = "vanilla_vit"
    version: str
    base_model: Config_Net_BaseModel_ViTClassifier
    params: VanillaViTConfig


Config_Net = Union[
    Config_Net_DuoVanillaBert,
    Config_Net_DuoVanillaViT,
    Config_Net_FroyoBert,
    Config_Net_FroyoViT,
    Config_Net_KernelShapBert,
    Config_Net_LttBert,
    Config_Net_LttViT,
    Config_Net_VanillaBert,
    Config_Net_VanillaViT,
]


class Config_Train(pydantic.BaseModel):
    epochs: int  # always resume from last known checkpoint
    ckpt_when: str  # see utils.strings, e.g. `<=5:%2==0; <=10:%3==0`
    lr: float
    batch_size: int
    EXPERIMENTAL_progressive_training: Optional[bool] = None


class Config_Train_Explainer(Config_Train):
    # epochs: int
    # ckpt_when: str
    # lr: float
    # batch_size: int
    # EXPERIMENTAL_progressive_training: Optional[bool] = None
    n_mask_samples: int
    lambda_efficiency: float
    lambda_norm: float


class Config_Eval_Accuracy(pydantic.BaseModel):
    dataset: Optional[Config_Dataset]
    batch_size: int
    resolution: int  # sampling: accuracy on n masked samples


class Config_Eval_Faithfulness(pydantic.BaseModel):
    dataset: Optional[Config_Dataset]
    batch_size: int
    resolution: int  # sampling: n perturbed samples


class Config_Eval_ClsAcc(pydantic.BaseModel):
    dataset: Optional[Config_Dataset]
    on_exp_epochs: Optional[str]  # see utils.strings, e.g. `<=5:%2==0'
    batch_size: int


class Config_Eval_Performance(pydantic.BaseModel):
    dataset: Optional[Config_Dataset]
    loops: int  # duplicate the dataset for n times


class Config_Eval_TrainResources(pydantic.BaseModel):
    dataset: Optional[Config_Dataset]
    batch_size: int
    max_samples: int


class Config_Eval_BranchesCka(pydantic.BaseModel):
    dataset: Optional[Config_Dataset]
    batch_size: int


class Config_Eval_DualTaskSimilarity(pydantic.BaseModel):
    dataset: Optional[Config_Dataset]
    batch_size: int


class Config_Logger(pydantic.BaseModel):
    wandb_enabled: bool
    wandb_project: str
    wandb_name: str
    # THESE ARE SET AUTOMATICALLY UPON UPDATE
    wandb_run_id: Optional[str] = None
    wandb_global_step: Optional[int] = None  # last uploaded step, defaults to 0


class ExpConfig(pydantic.BaseModel):
    schema_version: Optional[str] = pydantic.Field(
        alias="$schema",  # type: ignore
        serialization_alias="$schema",
    )

    seed: int
    dataset: Config_Dataset
    net: Config_Net
    train_classifier: Config_Train
    train_surrogate: Config_Train
    train_explainer: Config_Train_Explainer
    logger_classifier: Optional[Config_Logger] = None
    logger_surrogate: Optional[Config_Logger] = None
    logger_explainer: Optional[Config_Logger] = None
    eval_accuracy: Config_Eval_Accuracy
    eval_faithfulness: Config_Eval_Faithfulness
    eval_cls_acc: Config_Eval_ClsAcc
    eval_performance: Config_Eval_Performance
    eval_train_resources: Config_Eval_TrainResources
    eval_branches_cka: Optional[Config_Eval_BranchesCka] = None
    eval_dual_task_similarity: Optional[Config_Eval_DualTaskSimilarity] = None

    def flatten_dump(self) -> dict:
        ret = json.loads(self.model_dump_json(by_alias=True, exclude_unset=False))
        del ret["logger_classifier"]
        del ret["logger_surrogate"]
        del ret["logger_explainer"]
        return flatten_dict(ret)

    pass


def main() -> None:
    schema = ExpConfig.model_json_schema()
    schema_path = pathlib.Path(__file__).parent / "../experiments/hparams_schema.json"
    print(f"generating schema --> {schema_path.resolve()}")
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(schema, indent=2))
        f.write("\n")
    return
