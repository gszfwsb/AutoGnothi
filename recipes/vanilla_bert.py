import dataclasses
import pathlib
from typing import Any, Callable, List, Optional, Tuple, cast

import torch
from torch import Tensor, nn
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    BertModel,
    PreTrainedTokenizerBase,
)

from ..models.vanilla_bert import (
    VanillaBertClassifier,
    VanillaBertConfig,
    VanillaBertExplainer,
    VanillaBertFinal,
    VanillaBertSurrogate,
)
from ..utils.nnmodel import MergeStateDictRules, New, merge_state_dicts
from .types import ModelRecipe, ModelRecipe_Measurements, ModelRecipe_Training


@dataclasses.dataclass
class VanillaBertMisc:
    tokenizer: PreTrainedTokenizerBase
    pass


VanillaBertRecipe = ModelRecipe[
    VanillaBertConfig,
    VanillaBertMisc,
    VanillaBertClassifier,
    VanillaBertSurrogate,
    VanillaBertExplainer,
    VanillaBertFinal,
]


def vanilla_bert_recipe() -> VanillaBertRecipe:
    return ModelRecipe(
        id="vanilla_bert",
        version="beta.1.01",
        t_config=VanillaBertConfig,
        t_classifier=VanillaBertClassifier,
        t_surrogate=VanillaBertSurrogate,
        t_explainer=VanillaBertExplainer,
        t_final=VanillaBertFinal,
        load_misc=_load_misc,
        conv_pretrained_classifier=_conv_pretrained_classifier,
        conv_classifier_surrogate=_conv_classifier_surrogate,
        conv_surrogate_explainer=_conv_surrogate_explainer,
        conv_explainer_final=_conv_explainer_final,
        n_players=lambda cfg: cfg.max_position_embeddings - 1,
        gen_input=lambda cfg, misc, device: _gen_input(
            max_position_embeddings=cfg.max_position_embeddings,
            tokenizer=misc.tokenizer,
            device=device,
        ),
        gen_null=lambda cfg, misc, device: _gen_null(
            max_position_embeddings=cfg.max_position_embeddings,
            tokenizer=misc.tokenizer,
            device=device,
        ),
        training=ModelRecipe_Training(
            support_classifier=True,
            support_surrogate=True,
            support_explainer=True,
            exp_variant_duo=False,
            exp_variant_kernel_shap=False,
        ),
        fw_classifier=_fw_classifier,
        fw_surrogate=_fw_surrogate,
        fw_explainer=_fw_explainer,
        fw_final=_fw_final,
        measurements=ModelRecipe_Measurements(
            verify_final_coherency=True,
            allow_accuracy=True,
            allow_faithfulness=True,
            allow_cls_acc=True,
            allow_performance_cls=True,
            allow_performance_srg_exp=True,
            allow_performance_fin=True,
            allow_train_resources=True,
            allow_dual_task_similarity=False,
            allow_branches_cka=True,
        ),
    )


def _load_misc(m_path: pathlib.Path, cfg: VanillaBertConfig) -> VanillaBertMisc:
    tokenizer = AutoTokenizer.from_pretrained(m_path / "tokenizer")
    return VanillaBertMisc(
        tokenizer=tokenizer,
    )


def pre_conv_bert(cfg: VanillaBertConfig, model: nn.Module) -> VanillaBertClassifier:
    rules: MergeStateDictRules = {}
    if isinstance(model, VanillaBertClassifier):
        rules = {"{_}": ...}
    elif isinstance(model, BertForSequenceClassification):
        rules = {
            "bert.embeddings.word_embeddings.weight": ...,
            "bert.embeddings.position_embeddings.weight": ...,
            "bert.embeddings.token_type_embeddings.weight": ...,
            "bert.embeddings.LayerNorm.{wb}": ...,
            "bert.encoder.layer.{i}.attention.self.query.{wb}": "bert.encoder.layers.{i}.attention.self.query.{wb}",
            "bert.encoder.layer.{i}.attention.self.key.{wb}": "bert.encoder.layers.{i}.attention.self.key.{wb}",
            "bert.encoder.layer.{i}.attention.self.value.{wb}": "bert.encoder.layers.{i}.attention.self.value.{wb}",
            "bert.encoder.layer.{i}.attention.output.dense.{wb}": "bert.encoder.layers.{i}.attention.output.dense.{wb}",
            "bert.encoder.layer.{i}.attention.output.LayerNorm.{wb}": "bert.encoder.layers.{i}.attention.output.LayerNorm.{wb}",
            "bert.encoder.layer.{i}.intermediate.dense.{wb}": "bert.encoder.layers.{i}.intermediate.dense.{wb}",
            "bert.encoder.layer.{i}.output.dense.{wb}": "bert.encoder.layers.{i}.output.dense.{wb}",
            "bert.encoder.layer.{i}.output.LayerNorm.{wb}": "bert.encoder.layers.{i}.output.LayerNorm.{wb}",
            "bert.pooler.dense.{wb}": "bert_pooler.dense.{wb}",
            "classifier.{wb}": ...,
        }
    elif isinstance(model, BertModel):
        rules = {
            "embeddings.word_embeddings.weight": "bert.embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
            "embeddings.LayerNorm.{wb}": "bert.embeddings.LayerNorm.{wb}",
            "encoder.layer.{i}.attention.self.query.{wb}": "bert.encoder.layers.{i}.attention.self.query.{wb}",
            "encoder.layer.{i}.attention.self.key.{wb}": "bert.encoder.layers.{i}.attention.self.key.{wb}",
            "encoder.layer.{i}.attention.self.value.{wb}": "bert.encoder.layers.{i}.attention.self.value.{wb}",
            "encoder.layer.{i}.attention.output.dense.{wb}": "bert.encoder.layers.{i}.attention.output.dense.{wb}",
            "encoder.layer.{i}.attention.output.LayerNorm.{wb}": "bert.encoder.layers.{i}.attention.output.LayerNorm.{wb}",
            "encoder.layer.{i}.intermediate.dense.{wb}": "bert.encoder.layers.{i}.intermediate.dense.{wb}",
            "encoder.layer.{i}.output.dense.{wb}": "bert.encoder.layers.{i}.output.dense.{wb}",
            "encoder.layer.{i}.output.LayerNorm.{wb}": "bert.encoder.layers.{i}.output.LayerNorm.{wb}",
            "pooler.dense.{wb}": "bert_pooler.dense.{wb}",
            New(): "classifier.{wb}",
        }
    classifier = VanillaBertClassifier(cfg)
    merge_state_dicts((rules, model), into=classifier)
    return classifier


def _conv_pretrained_classifier(
    cfg: VanillaBertConfig, model: nn.Module
) -> VanillaBertClassifier:
    classifier = pre_conv_bert(cfg, model)
    return classifier


def _conv_classifier_surrogate(
    cfg: VanillaBertConfig, _misc: VanillaBertMisc, classifier: VanillaBertClassifier
) -> VanillaBertSurrogate:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": ...,
        "classifier.{_}": ...,  # re-use pretrained knowledge whenever possible
        # "classifier.{_}": None,
        # New(): "classifier.{_}",
    }
    surrogate = VanillaBertSurrogate(cfg)
    merge_state_dicts((rules, classifier), into=surrogate)
    return surrogate


def _conv_surrogate_explainer(
    cfg: VanillaBertConfig,
    _misc: VanillaBertMisc,
    surrogate: VanillaBertSurrogate,
) -> VanillaBertExplainer:
    rules: MergeStateDictRules = {
        "bert.{_}": ...,
        "bert_pooler.{_}": None,
        "classifier.{_}": None,
        New(): "explainer_attn.{i}.attention.self.query.{wb}",
        New(): "explainer_attn.{i}.attention.self.key.{wb}",
        New(): "explainer_attn.{i}.attention.self.value.{wb}",
        New(): "explainer_attn.{i}.attention.output.dense.{wb}",
        New(): "explainer_attn.{i}.attention.output.LayerNorm.{wb}",
        New(): "explainer_attn.{i}.intermediate.dense.{wb}",
        New(): "explainer_attn.{i}.output.dense.{wb}",
        New(): "explainer_attn.{i}.output.LayerNorm.{wb}",
        New(): "explainer_mlp.0.{wb}",
        New(): "explainer_mlp.2.{wb}",
        New(): "explainer_mlp.4.{wb}",
    }
    explainer = VanillaBertExplainer(cfg)
    merge_state_dicts((rules, surrogate), into=explainer)
    return explainer


def _conv_explainer_final(
    cfg: VanillaBertConfig,
    misc: VanillaBertMisc,
    classifier: VanillaBertClassifier,
    surrogate: VanillaBertSurrogate,
    explainer: VanillaBertExplainer,
) -> VanillaBertFinal:
    # we need to replay the surrogate model
    device = classifier.bert.embeddings.word_embeddings.weight.device
    n_players = cfg.max_position_embeddings - 1
    nil_Xs = _gen_null(
        max_position_embeddings=cfg.max_position_embeddings,
        tokenizer=misc.tokenizer,
        device=device,
    )
    nil_mask = torch.ones((1, n_players), dtype=torch.long, device=device)
    surrogate.eval()
    with torch.no_grad():
        surrogate_null, _ = _fw_surrogate(surrogate, nil_Xs, nil_mask)

    rules_cls: MergeStateDictRules = {"{_}": "classifier.{_}"}
    rules_srg: MergeStateDictRules = {"{_}": "surrogate.{_}"}
    rules_exp: MergeStateDictRules = {"{_}": "explainer.{_}"}
    rules_misc: MergeStateDictRules = {"surrogate_null": ...}

    final = VanillaBertFinal(cfg)
    merge_state_dicts(
        (rules_cls, classifier),
        (rules_srg, surrogate),
        (rules_exp, explainer),
        (rules_misc, {"surrogate_null": surrogate_null}),
        into=final,
    )
    return final


def _gen_input(
    max_position_embeddings: int,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> Callable[[Any, Any], Tuple[Tensor, Tensor]]:
    """Avoids masking special tokens, and shap the rest. For example:
      > [CLS, 1, 1, 4, 5, 1, 4, END]
    will receive a mask of:
      > [1, 0/1, 0/1, 0/1, 0/1, 0/1, 0/1, 1]
    With these random positions equivalently distributed."""

    max_length = max_position_embeddings

    def mask_input(raw_xs: List[str], raw_ys: List[int]):
        xs_l: list[Tensor] = []

        for raw_x in raw_xs:
            inputs = tokenizer(
                raw_x,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
            )
            input_ids = cast(Tensor, inputs["input_ids"])
            attention_mask = cast(Tensor, inputs["attention_mask"])
            # length restriction.
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            _ = attention_mask  # unused

            xs_l.append(input_ids)

        xs = torch.cat(xs_l, dim=0).to(device)
        ys = torch.tensor(raw_ys).to(device)
        return xs, ys

    return mask_input


def _gen_null(
    max_position_embeddings: int,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> Tensor:
    inputs = tokenizer(
        "",
        return_tensors="pt",
        padding="max_length",
        max_length=max_position_embeddings,
    )
    input_ids = cast(Tensor, inputs["input_ids"])
    x = input_ids.to(device)
    return x


def _fw_xs_preprocess(xs: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    # normally input masking is not needed
    # __SPECIAL_TOKEN__ = 0  # 0: [PAD] / 103: [MASK] / disable input masking
    device = xs.device
    batch_size, _ = xs.shape
    mask_cls = torch.ones((batch_size, 1), dtype=mask.dtype, device=device)
    mask = torch.cat([mask_cls, mask], dim=1)
    # xs = xs * mask + __SPECIAL_TOKEN__ * (1 - mask)
    token_type_ids = torch.zeros_like(xs, device=device)
    return xs, mask, token_type_ids


def _fw_classifier(
    model: VanillaBertClassifier, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, logits


def _fw_surrogate(
    model: VanillaBertSurrogate, xs: Tensor, mask: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    logits = model(xs, mask, token_type_ids)
    return logits, None


def _fw_explainer(
    model: VanillaBertExplainer,
    xs: Tensor,
    mask: Tensor,
    surrogate_grand: Tensor,
    surrogate_null: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    xs, mask, token_type_ids = _fw_xs_preprocess(xs, mask)
    attr = model(xs, mask, token_type_ids, surrogate_grand, surrogate_null)
    return attr, None


def _fw_final(
    model: VanillaBertFinal,
    xs: Tensor,
) -> Tuple[Tensor, Tensor]:
    device = xs.device
    mask = torch.ones_like(xs, device=device)
    token_type_ids = torch.zeros_like(xs, device=device)
    logits, attr = model(xs, mask, token_type_ids)
    return logits, attr
