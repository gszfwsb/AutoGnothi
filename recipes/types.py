import dataclasses
import enum
import pathlib
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pydantic
import torch
from torch import Tensor, nn

TConfig = TypeVar("TConfig", bound=pydantic.BaseModel)

TMisc = TypeVar("TMisc")

TClassifier = TypeVar("TClassifier", bound=nn.Module)

TSurrogate = TypeVar("TSurrogate", bound=nn.Module)

TExplainer = TypeVar("TExplainer", bound=nn.Module)

TFinal = TypeVar("TFinal", bound=nn.Module)


class ModelMode(enum.Enum):
    classifier_eval = "classifier_eval"
    surrogate_eval = "surrogate_eval"
    surrogate_train = "surrogate_train"
    explainer_eval = "explainer_eval"
    explainer_train = "explainer_train"
    pass


@dataclasses.dataclass
class ModelRecipe_Training:
    support_classifier: bool
    support_surrogate: bool
    support_explainer: bool
    exp_variant_duo: bool
    exp_variant_kernel_shap: bool


TAltClassifier = TypeVar("TAltClassifier", bound=nn.Module)

TAltExplainer = TypeVar("TAltExplainer", bound=nn.Module)


@dataclasses.dataclass
class ModelRecipe_Measurements_DualTaskSimilarity(
    Generic[TConfig, TClassifier, TExplainer, TAltClassifier, TAltExplainer]
):
    allow: Literal[True]
    # the measurements are made based on an altered version of the model to
    # retrieve its internal gradients.
    t_alt_classifier: Type[TAltClassifier]
    t_alt_explainer: Type[TAltExplainer]
    conv_alt_models: Callable[
        [TConfig, TClassifier, TExplainer], Tuple[TAltClassifier, TAltExplainer]
    ]
    # inspect gradient input relative to this module :: (Fc, Fe) -> d'Fc, d'Fe
    grad_modules: Callable[[TAltClassifier, TAltExplainer], Tuple[nn.Module, nn.Module]]
    # :: (Fc, Fe, Xs, mask, 'surrogate_grand', 'surrogate_null') -> Ys, shap
    fw_alt: Callable[
        [TAltClassifier, TAltExplainer, Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor],
    ]


@dataclasses.dataclass
class ModelRecipe_Measurements(Generic[TConfig, TClassifier, TExplainer]):
    verify_final_coherency: bool
    allow_accuracy: bool
    allow_faithfulness: bool
    allow_cls_acc: bool
    allow_performance_cls: bool
    allow_performance_srg_exp: bool
    allow_performance_fin: bool
    allow_train_resources: bool
    allow_dual_task_similarity: Union[
        Literal[False],
        ModelRecipe_Measurements_DualTaskSimilarity[
            TConfig, TClassifier, TExplainer, Any, Any
        ],
    ]
    allow_branches_cka: bool


@dataclasses.dataclass
class ModelRecipe(Generic[TConfig, TMisc, TClassifier, TSurrogate, TExplainer, TFinal]):
    id: str
    version: str
    # types & classes
    t_config: Type[TConfig]
    t_classifier: Type[TClassifier]  # (t_config) -> nn.Module
    t_surrogate: Type[TSurrogate]  # (t_config) -> nn.Module
    t_explainer: Type[TExplainer]  # (t_config) -> nn.Module
    t_final: Type[TFinal]  # (t_config) -> nn.Module

    # convert to initial weights
    load_misc: Callable[[pathlib.Path, TConfig], TMisc]
    conv_pretrained_classifier: Callable[[TConfig, Union[nn.Module, Any]], TClassifier]
    conv_classifier_surrogate: Callable[[TConfig, TMisc, TClassifier], TSurrogate]
    conv_surrogate_explainer: Callable[[TConfig, TMisc, TSurrogate], TExplainer]
    conv_explainer_final: Callable[
        [TConfig, TMisc, TClassifier, TSurrogate, TExplainer], TFinal
    ]

    # definitions
    n_players: Callable[[TConfig], int]  # for shapley values
    # (_inputs, _targets) -> (Xs <bs, ...>, Zs <bs>)
    #                            <bs, tks> for text cls
    #                            <bs, n_ch, h, w> for img cls
    gen_input: Callable[
        [TConfig, TMisc, torch.device],
        Callable[[Any, Any], Tuple[Tensor, Tensor]],
    ]
    # -> Xs <1, ...>
    #       <1, tks> for text cls
    #       <1, n_ch, h, w> for img cls
    gen_null: Callable[
        [TConfig, TMisc, torch.device],
        Tensor,
    ]

    # training configs
    training: ModelRecipe_Training

    # forward pass
    # :: (F, Xs,          mask) ->     Ys,     *Ys (original, verbatim)
    #        L. <bs, ...> L. <bs, cls> L. <bs> L. <bs>?
    fw_classifier: Callable[[TClassifier, Tensor, Tensor], Tuple[Tensor, Tensor]]
    # :: (F, Xs, mask) -> Ys, *Ys?
    fw_surrogate: Callable[
        [TSurrogate, Tensor, Tensor], Tuple[Tensor, Optional[Tensor]]
    ]
    # :: (F, Xs, mask, 'surrogate_grand', 'surrogate_null') -> shap, *Ys?
    #                   L. <bs, cls>       L. <1, cls>         L. <bs, cls, players>
    fw_explainer: Callable[
        [TExplainer, Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Optional[Tensor]]
    ]
    # WARNING: we require that classifier & explainer results be both returned.
    #          this function is expected to run only under no_grad().
    # WARNING: since calculation of the explainer relies upon the grand & null
    #          values of the corresponding surrogate, this variant may
    #          internally include a surrogate side branch.
    # Note: here the logits precedes explanations is because it's a more
    #       important part of the inference.
    # :: (F, Xs) -> *Ys, shap
    fw_final: Callable[[TFinal, Tensor], Tuple[Tensor, Tensor]]

    # measurements
    measurements: ModelRecipe_Measurements

    pass
