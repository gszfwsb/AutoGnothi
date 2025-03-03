import copy
import json
import pathlib
from typing import Any, Literal, Tuple, cast, overload

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertModel,
    BertTokenizerFast,
    PreTrainedTokenizerBase,
    ViTConfig,
    ViTForImageClassification,
)

from ..models.vanilla_bert import VanillaBertClassifier, VanillaBertConfig
from ..models.vanilla_vit import VanillaViTClassifier, VanillaViTConfig
from ..utils.nnmodel import (
    MergeStateDictRules,
    New,
    force_save_model,
    merge_state_dicts,
)


@overload
def load_params(  # type: ignore
    kind: Literal[
        # task-specific language models
        "bert_tayp",
        # base language models
        "prj_bert_mini",
        "prj_bert_small",
        "prj_bert_medium",
        "gg_bert_base",
        "gg_bert_large",
        # fine-tuned language models
        "ft_bert_base_yelp",
        "ft_bert_large_yelp",
        "ft_bert_medium_yelp",
        "ft_bert_mini_yelp",
        "ft_bert_small_yelp",
        # base vision models
        "gg_vit_tiny",
        "gg_vit_small",
        "gg_vit_base",
        "gg_vit_large",
        # fine-tuned vision models
        "ft_vit_tiny_imagenette",
        "ft_vit_small_imagenette",
        "ft_vit_base_imagenette",
        "ft_vit_large_imagenette",
    ],
    num_labels: int = 0,
) -> Tuple[BertForSequenceClassification, PreTrainedTokenizerBase]: ...


def load_params(kind: str, num_labels: int = 0) -> Tuple[nn.Module, Any]:
    assert num_labels != 0  # need specify pretrain labels
    if kind == "bert_tayp":
        root = pathlib.Path(__file__).parent / "bert_tayp"
        if not root.exists() or len(list(root.iterdir())) < 6:
            model_id = "textattack/bert-base-uncased-yelp-polarity"
            print(f"! downloading `{model_id}`...")
            force_save_model(
                id=model_id,
                path=root,
                save_model=BertForSequenceClassification,
                save_tokenizer=BertTokenizerFast,
            )
        model = BertForSequenceClassification.from_pretrained(root)
        model = cast(BertForSequenceClassification, model)
        tokenizer = AutoTokenizer.from_pretrained(
            root, clean_up_tokenization_spaces=True
        )
        assert isinstance(model, BertForSequenceClassification)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return model, tokenizer

    elif (
        False
        or kind == "prj_bert_mini"
        or kind == "prj_bert_small"
        or kind == "prj_bert_medium"
        or kind == "gg_bert_base"
        or kind == "gg_bert_large"
    ):
        root = pathlib.Path(__file__).parent / kind
        if not root.exists() or len(list(root.iterdir())) < 6:
            model_id = {
                "prj_bert_mini": "prajjwal1/bert-mini",
                "prj_bert_small": "prajjwal1/bert-small",
                "prj_bert_medium": "prajjwal1/bert-medium",
                "gg_bert_base": "google-bert/bert-base-uncased",
                "gg_bert_large": "google-bert/bert-large-uncased",
            }[kind]
            print(f"! downloading `{model_id}`...")
            force_save_model(
                id=model_id,
                path=root,
                save_model=BertModel,
                save_tokenizer=BertTokenizerFast,
            )
        # load model, literally
        backbone = BertModel.from_pretrained(root)
        backbone = cast(BertModel, backbone)
        tokenizer = AutoTokenizer.from_pretrained(
            root, clean_up_tokenization_spaces=True
        )
        # convert classifier
        config = copy.deepcopy(backbone.config)
        config.num_labels = num_labels
        model = BertForSequenceClassification(config)
        rules: MergeStateDictRules = {
            "{_}": "bert.{_}",
            New(): "classifier.{wb}",
        }
        merge_state_dicts((rules, backbone), into=model)
        # finish conv
        assert isinstance(model, BertForSequenceClassification)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return model, tokenizer

    elif (
        False
        or kind == "ft_bert_base_yelp"
        or kind == "ft_bert_large_yelp"
        or kind == "ft_bert_medium_yelp"
        or kind == "ft_bert_mini_yelp"
        or kind == "ft_bert_small_yelp"
    ):
        root = pathlib.Path(__file__).parent / kind
        if not root.exists():
            raise ValueError(f"no fine-tuned model `{kind}` found")
        # load self fine-tuned model
        with open(root / "model.json", "r", encoding="utf-8") as f:
            _r = f.read()
            _j = json.loads(_r)
            ft_config = VanillaBertConfig.model_validate(_j)
        ft_ckpt = torch.load(
            root / "model.ckpt", weights_only=True, map_location=torch.device("cpu")
        )
        ft_model = VanillaBertClassifier(ft_config)
        ft_model.load_state_dict(ft_ckpt)
        # convert to unified format
        config = BertConfig(
            vocab_size=ft_config.vocab_size,
            hidden_size=ft_config.hidden_size,
            num_hidden_layers=ft_config.num_hidden_layers,
            num_attention_heads=ft_config.num_attention_heads,
            intermediate_size=ft_config.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=ft_config.hidden_dropout_prob,
            attention_probs_dropout_prob=ft_config.attention_probs_dropout_prob,
            max_position_embeddings=ft_config.max_position_embeddings,
            type_vocab_size=ft_config.type_vocab_size,
            initializer_range=0.02,
            layer_norm_eps=ft_config.layer_norm_eps,
            pad_token_id=ft_config.pad_token_id,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            num_labels=num_labels,
        )
        model = BertForSequenceClassification(config)
        rules: MergeStateDictRules = {
            "bert.embeddings.{_}": ...,
            "bert.encoder.layers.{i}.{_}": "bert.encoder.layer.{i}.{_}",
            "bert_pooler.dense.{wb}": "bert.pooler.dense.{wb}",
            "classifier.{wb}": ...,
        }
        merge_state_dicts((rules, ft_model), into=model)
        # load tokenizer and misc
        tokenizer = AutoTokenizer.from_pretrained(
            root, clean_up_tokenization_spaces=True
        )
        assert isinstance(model, BertForSequenceClassification)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return model, tokenizer

    elif (
        False
        or kind == "gg_vit_tiny"
        or kind == "gg_vit_small"
        or kind == "gg_vit_base"
        or kind == "gg_vit_large"
    ):
        root = pathlib.Path(__file__).parent / kind
        if not root.exists() or len(list(root.iterdir())) < 2:
            model_id = {
                "gg_vit_tiny": "WinKawaks/vit-tiny-patch16-224",
                "gg_vit_small": "WinKawaks/vit-small-patch16-224",
                "gg_vit_base": "google/vit-base-patch16-224",
                "gg_vit_large": "google/vit-large-patch16-224",
            }[kind]
            print(f"! downloading `{model_id}`...")
            force_save_model(
                id=model_id,
                path=root,
                save_model=ViTForImageClassification,
            )
        # load model, literally
        backbone = ViTForImageClassification.from_pretrained(root)
        backbone = cast(ViTForImageClassification, backbone)
        # the backbone is already a classifier
        model = backbone
        assert isinstance(model, ViTForImageClassification)
        return model, None

    elif (
        False
        or kind == "ft_vit_tiny_imagenette"
        or kind == "ft_vit_small_imagenette"
        or kind == "ft_vit_base_imagenette"
        or kind == "ft_vit_large_imagenette"
    ):
        root = pathlib.Path(__file__).parent / kind
        if not root.exists():
            raise ValueError(f"no fine-tuned model `{kind}` found")
        # load self fine-tuned model
        with open(root / "model.json", "r", encoding="utf-8") as f:
            _r = f.read()
            _j = json.loads(_r)
            ft_config = VanillaViTConfig.model_validate(_j)
        ft_ckpt = torch.load(
            root / "model.ckpt", weights_only=True, map_location=torch.device("cpu")
        )
        ft_model = VanillaViTClassifier(ft_config)
        ft_model.load_state_dict(ft_ckpt)
        # convert to unified format
        config = ViTConfig(
            hidden_size=ft_config.hidden_size,
            num_hidden_layers=ft_config.num_hidden_layers,
            num_attention_heads=ft_config.num_attention_heads,
            intermediate_size=ft_config.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=ft_config.hidden_dropout_prob,
            attention_probs_dropout_prob=ft_config.attention_probs_dropout_prob,
            initializer_range=0.02,
            layer_norm_eps=ft_config.layer_norm_eps,
            image_size=ft_config.img_px_size,
            patch_size=ft_config.img_patch_size,
            num_channels=ft_config.img_channels,
            qkv_bias=True,
            encoder_stride=16,
            num_labels=num_labels,
        )
        model = ViTForImageClassification(config)
        rules: MergeStateDictRules = {
            "vit.embeddings.{_}": ...,
            "vit.encoder.layers.{i}.attention.self.query.{wb}": "vit.encoder.layer.{i}.attention.attention.query.{wb}",
            "vit.encoder.layers.{i}.attention.self.key.{wb}": "vit.encoder.layer.{i}.attention.attention.key.{wb}",
            "vit.encoder.layers.{i}.attention.self.value.{wb}": "vit.encoder.layer.{i}.attention.attention.value.{wb}",
            "vit.encoder.layers.{i}.attention.output.dense.{wb}": "vit.encoder.layer.{i}.attention.output.dense.{wb}",
            "vit.encoder.layers.{i}.intermediate.dense.{wb}": "vit.encoder.layer.{i}.intermediate.dense.{wb}",
            "vit.encoder.layers.{i}.output.dense.{wb}": "vit.encoder.layer.{i}.output.dense.{wb}",
            "vit.encoder.layers.{i}.layernorm_before.{wb}": "vit.encoder.layer.{i}.layernorm_before.{wb}",
            "vit.encoder.layers.{i}.layernorm_after.{wb}": "vit.encoder.layer.{i}.layernorm_after.{wb}",
            "vit.layernorm.{wb}": ...,
            "classifier.{wb}": ...,
        }
        merge_state_dicts((rules, ft_model), into=model)
        assert isinstance(model, ViTForImageClassification)
        return model, None

    else:
        # TODO: fine-tuned vision models
        raise ValueError(f"invalid kind: {kind}")


def preload_all_params() -> None:
    load_params("bert_tayp", num_labels=2)
    load_params("prj_bert_mini", num_labels=1)
    load_params("prj_bert_small", num_labels=1)
    load_params("prj_bert_medium", num_labels=1)
    load_params("gg_bert_base", num_labels=1)
    load_params("gg_bert_large", num_labels=1)
    load_params("gg_vit_tiny", num_labels=1)
    load_params("gg_vit_small", num_labels=1)
    load_params("gg_vit_base", num_labels=1)
    load_params("gg_vit_large", num_labels=1)
    return
