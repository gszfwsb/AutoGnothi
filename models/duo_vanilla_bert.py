from typing import List, Tuple, cast

import pydantic
import torch
from torch import Tensor, nn
from transformers.modeling_utils import ModuleUtilsMixin

from ..utils.nnmodel import ObservableModuleMixin
from .shapley import normalize_shapley_explanation
from .vanilla_bert import (
    VanillaBertClassifier,
    VanillaBertConfig,
    VanillaBertLayer,
    VanillaBertModel,
    VanillaBertPooler,
    VanillaBertSurrogate,
)


class DuoVanillaBertConfig(pydantic.BaseModel):
    """Same as Vanilla BERT, except that we train it with dual objectives (both
    classification & explanation)."""

    attention_probs_dropout_prob: float
    explainer_attn_num_layers: int
    explainer_head_hidden_size: int
    explainer_normalize: bool
    hidden_dropout_prob: float
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_labels: int
    pad_token_id: int
    type_vocab_size: int
    vocab_size: int

    @property
    def is_decoder(self) -> bool:
        return False

    def into(self) -> VanillaBertConfig:
        return VanillaBertConfig(
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            explainer_attn_num_layers=self.explainer_attn_num_layers,
            explainer_head_hidden_size=self.explainer_head_hidden_size,
            explainer_normalize=self.explainer_normalize,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_labels=self.num_labels,
            pad_token_id=self.pad_token_id,
            type_vocab_size=self.type_vocab_size,
            vocab_size=self.vocab_size,
        )

    pass


class DuoVanillaBertClassifier(VanillaBertClassifier):
    def __init__(self, config: DuoVanillaBertConfig):
        super().__init__(config.into())

    pass


class DuoVanillaBertSurrogate(VanillaBertSurrogate):
    def __init__(self, config: DuoVanillaBertConfig):
        super().__init__(config.into())

    pass


class DuoVanillaBertExplainer(nn.Module, ModuleUtilsMixin, ObservableModuleMixin):
    def __init__(self, config: DuoVanillaBertConfig):
        nn.Module.__init__(self)
        ModuleUtilsMixin.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.bert = VanillaBertModel(config.into())
        self.bert_pooler = VanillaBertPooler(hidden_size=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        attn_layers: List[nn.Module] = []
        for i_ly in range(config.explainer_attn_num_layers):
            attn_layer = VanillaBertLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_norm_eps=config.layer_norm_eps,
                num_attention_heads=config.num_attention_heads,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            attn_layers.append(attn_layer)
        self.explainer_attn = nn.ModuleList(attn_layers)

        self.explainer_dropout = nn.Dropout(config.hidden_dropout_prob)
        mlp_max_width = int(config.explainer_head_hidden_size)
        mlp_layers: List[nn.Module] = []
        mlp_layers.append(nn.Linear(config.hidden_size, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, config.num_labels))
        self.explainer_mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        inp [ ]: input_ids:       <batch_size, n_tokens>
        inp [ ]: attention_mask:  <batch_size, n_tokens>
        inp [ ]: token_type_ids:  <batch_size, n_tokens> := {1...}
        inp [ ]: surrogate_grand: <batch_size, n_classes>
        inp [ ]: surrogate_null:  <1, n_classes>
        ret [0]: logits:          <batch_size, n_classes>
        ret [1]: output:          <batch_size, n_players := n_tokens - 1, n_classes>
        """

        input_shape = cast(Tuple[int], input_ids.size())
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        self.om_record_features(repr_cls=output, repr_exp=output)
        logits = self.bert_pooler(output)
        logits = self.dropout(logits)
        logits = self.classifier(logits)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        for attn_layer in self.explainer_attn:
            output = attn_layer(output, extended_attention_mask)
        output = self.explainer_dropout(output)
        # (batch_size, all_tokens, num_labels)
        output = self.explainer_mlp(output)
        # (batch_size, num_labels=2, num_tokens - 1)
        # excluding class token.
        if self.config.explainer_normalize:
            output = normalize_shapley_explanation(
                output, surrogate_grand, surrogate_null
            )
        output = output[:, 1:, :].permute(0, 2, 1)
        return logits, output

    pass


class DuoVanillaBertFinal(nn.Module, ObservableModuleMixin):
    def __init__(self, config: DuoVanillaBertConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config
        # self.classifier = VanillaBertClassifier(self.config)
        self.surrogate = VanillaBertSurrogate(config.into())
        self.surrogate_null = nn.Parameter(
            torch.zeros((1, self.config.num_labels)), requires_grad=False
        )
        self.explainer = DuoVanillaBertExplainer(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # logits = self.classifier(input_ids, attention_mask, token_type_ids)
        # om_repr_cls = self.classifier.om_take_observations()
        if self.config.explainer_normalize:
            surrogate_grand = self.surrogate(input_ids, attention_mask, token_type_ids)
            om_repr_srg = self.surrogate.om_take_observations()
        else:
            surrogate_grand = ...
            om_repr_srg = {}
        logits, explainer = self.explainer(
            input_ids,
            attention_mask,
            token_type_ids,
            surrogate_grand,
            self.surrogate_null,
        )
        om_repr_exp = self.explainer.om_take_observations()
        self.om_record_features(
            repr_cls=om_repr_exp.get("repr_cls", None),
            repr_srg=om_repr_srg.get("repr_srg", None),
            repr_exp=om_repr_exp.get("repr_exp", None),
        )
        return logits, explainer

    def om_retain_observations(self, flag: bool = True) -> None:
        ObservableModuleMixin.om_retain_observations(self, flag)
        # self.classifier.om_retain_observations(flag)
        self.surrogate.om_retain_observations(flag)
        self.explainer.om_retain_observations(flag)

    pass
