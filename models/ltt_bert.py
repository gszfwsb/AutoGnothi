from typing import Dict, List, Tuple, cast

import pydantic
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.modeling_utils import ModuleUtilsMixin
from typing_extensions import Self

from ..utils.nnmodel import ObservableModuleMixin, freeze_model_parameters
from .shapley import normalize_shapley_explanation
from .vanilla_bert import (
    VanillaBertConfig,
    VanillaBertEmbeddings,
    VanillaBertLayer,
    VanillaBertPooler,
)


class LttBertConfig(pydantic.BaseModel):
    """LTT = Ladder Transfer Training"""

    attention_probs_dropout_prob: float
    explainer_s_attn_num_layers: int  # side head
    explainer_s_head_hidden_size: int  # side head
    explainer_normalize: bool  # side head
    hidden_dropout_prob: float
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_labels: int
    pad_token_id: int
    s_attn_hidden_size: int  # side attention
    s_attn_intermediate_size: int  # side attention
    type_vocab_size: int
    vocab_size: int

    @property
    def is_decoder(self) -> bool:
        return False

    def into(self) -> VanillaBertConfig:
        return VanillaBertConfig(
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            explainer_attn_num_layers=self.explainer_s_attn_num_layers,
            explainer_head_hidden_size=self.explainer_s_head_hidden_size,
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


class LttBertSurrogate(nn.Module, ObservableModuleMixin):
    def __init__(self, config: LttBertConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.bert = LttBertModel(config=config, num_side_branches=1)
        self.bert_pooler = VanillaBertPooler(hidden_size=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

        self.bert_s_attn_pooler = VanillaBertPooler(
            hidden_size=config.s_attn_hidden_size
        )
        self.s_attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.s_attn_classifier = nn.Linear(config.s_attn_hidden_size, config.num_labels)
        self.s_attn_act = nn.Softmax(dim=-1)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        freeze_model_parameters(self, "bert.embeddings")
        freeze_model_parameters(self, "bert.encoder.layers")
        freeze_model_parameters(self, "bert_pooler")
        freeze_model_parameters(self, "classifier")
        return self

    def ltt_freeze_layers_until(self, layer_id: int) -> None:
        self.bert.encoder.ltt_freeze_layers_until(layer_id)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        output, (srg_output,) = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            side_layer_branches=[0],
        )
        self.om_record_features(repr_cls=output, repr_srg=srg_output)  # no worries
        output = self.bert_pooler(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        logits = self.act(logits)
        srg_output = self.bert_s_attn_pooler(srg_output)
        srg_output = self.s_attn_dropout(srg_output)
        srg_logits = self.s_attn_classifier(srg_output)
        srg_logits = self.s_attn_act(srg_logits)
        return srg_logits, logits

    pass


class LttBertExplainer(nn.Module, ModuleUtilsMixin, ObservableModuleMixin):
    def __init__(self, config: LttBertConfig):
        nn.Module.__init__(self)
        ModuleUtilsMixin.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.bert = LttBertModel(config=config, num_side_branches=1)
        self.bert_pooler = VanillaBertPooler(hidden_size=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

        s_attn_layers: List[nn.Module] = []
        for i_ly in range(config.explainer_s_attn_num_layers):
            s_attn_layer = VanillaBertLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.s_attn_hidden_size,
                intermediate_size=config.s_attn_intermediate_size,
                layer_norm_eps=config.layer_norm_eps,
                num_attention_heads=config.num_attention_heads,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            s_attn_layers.append(s_attn_layer)
        self.s_attn_attention_layers = nn.Sequential(*s_attn_layers)

        self.s_attn_exp_dropout = nn.Dropout(config.hidden_dropout_prob)
        mlp_max_width = int(config.explainer_s_head_hidden_size)
        mlp_layers: List[nn.Module] = []
        mlp_layers.append(nn.Linear(config.s_attn_hidden_size, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, config.num_labels))
        self.s_attn_explainer = nn.Sequential(*mlp_layers)

    def ltt_freeze_layers_until(self, layer_id: int) -> None:
        self.bert.encoder.ltt_freeze_layers_until(layer_id)

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_model_parameters(self, "bert.embeddings")
        freeze_model_parameters(self, "bert.encoder.layers")
        freeze_model_parameters(self, "bert_pooler")
        freeze_model_parameters(self, "classifier")
        return self

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
        ret [0]: exp_output:      <batch_size, n_players := n_tokens - 1, n_classes>
        ret [1]: logits:          <batch_size, n_classes>
        """

        input_shape = cast(Tuple[int], input_ids.size())
        output, (exp_output,) = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            side_layer_branches=[0],
        )
        self.om_record_features(repr_cls=output, repr_exp=exp_output)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        output = self.bert_pooler(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        logits = self.act(logits)

        for attn_layer in self.s_attn_attention_layers:
            exp_output = attn_layer(exp_output, extended_attention_mask)
        exp_output = self.s_attn_exp_dropout(exp_output)
        # <batch_size, n_players, n_classes>
        exp_output = self.s_attn_explainer(exp_output)

        if self.config.explainer_normalize:
            exp_output = normalize_shapley_explanation(
                exp_output, surrogate_grand, surrogate_null
            )
        # <batch_size, n_classes=2, n_players>
        # excluding class token.
        exp_output = exp_output[:, 1:, :].permute(0, 2, 1)

        return exp_output, logits

    pass


class LttBertFinal(nn.Module, ModuleUtilsMixin, ObservableModuleMixin):
    def __init__(self, config: LttBertConfig):
        nn.Module.__init__(self)
        ModuleUtilsMixin.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.bert = LttBertModel(config=config, num_side_branches=2)
        self.bert_pooler = VanillaBertPooler(hidden_size=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

        self.bert_s_attn_pooler = VanillaBertPooler(
            hidden_size=config.s_attn_hidden_size
        )
        self.s_attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.s_attn_classifier = nn.Linear(config.s_attn_hidden_size, config.num_labels)
        self.s_attn_act = nn.Softmax(dim=-1)
        self.surrogate_null = nn.Parameter(
            torch.zeros((1, self.config.num_labels)), requires_grad=False
        )

        s_attn_layers: List[nn.Module] = []
        for i_ly in range(config.explainer_s_attn_num_layers):
            s_attn_layer = VanillaBertLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.s_attn_hidden_size,
                intermediate_size=config.s_attn_intermediate_size,
                layer_norm_eps=config.layer_norm_eps,
                num_attention_heads=config.num_attention_heads,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            s_attn_layers.append(s_attn_layer)
        self.s_attn_attention_layers = nn.Sequential(*s_attn_layers)

        self.s_attn_exp_dropout = nn.Dropout(config.hidden_dropout_prob)
        mlp_max_width = int(config.explainer_s_head_hidden_size)
        mlp_layers: List[nn.Module] = []
        mlp_layers.append(nn.Linear(config.s_attn_hidden_size, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Linear(mlp_max_width, config.num_labels))
        self.s_attn_explainer = nn.Sequential(*mlp_layers)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        inp [ ]: input_ids:       <batch_size, n_tokens>
        inp [ ]: attention_mask:  <batch_size, n_tokens>
        inp [ ]: token_type_ids:  <batch_size, n_tokens> := {1...}
        ret [0]: logits:          <batch_size, n_classes>
        ret [1]: exp_output:      <batch_size, n_players := n_tokens - 1, n_classes>
        """

        input_shape = cast(Tuple[int], input_ids.size())
        if self.config.explainer_normalize:
            output, (srg_output, exp_output) = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                side_layer_branches=[0, 1],
            )
            self.om_record_features(
                repr_cls=output, repr_srg=srg_output, repr_exp=exp_output
            )
        else:
            output, (exp_output,) = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                side_layer_branches=[1],
            )
            srg_output = ...
            self.om_record_features(repr_cls=output, repr_exp=exp_output)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        output = self.bert_pooler(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        logits = self.act(logits)

        if self.config.explainer_normalize:
            srg_output = self.bert_s_attn_pooler(srg_output)
            srg_output = self.s_attn_dropout(srg_output)
            srg_logits = self.s_attn_classifier(srg_output)
            srg_logits = self.s_attn_act(srg_logits)
            surrogate_grand = srg_logits
            surrogate_null = self.surrogate_null
        else:
            surrogate_grand = ...
            surrogate_null = ...

        for attn_layer in self.s_attn_attention_layers:
            exp_output = attn_layer(exp_output, extended_attention_mask)
        exp_output = self.s_attn_exp_dropout(exp_output)
        # <batch_size, n_players, n_classes>
        exp_output = self.s_attn_explainer(exp_output)

        if self.config.explainer_normalize:
            exp_output = normalize_shapley_explanation(
                exp_output, surrogate_grand, surrogate_null
            )
        # <batch_size, n_classes, n_players>
        # excluding class token.
        exp_output = exp_output[:, 1:, :].permute(0, 2, 1)

        return logits, exp_output

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_model_parameters(self, "bert.embeddings")
        freeze_model_parameters(self, "bert.encoder.layers")
        freeze_model_parameters(self, "bert_pooler")
        freeze_model_parameters(self, "classifier")
        return self

    pass


class LttBertModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config: LttBertConfig, num_side_branches: int):
        super().__init__()
        self.config = config
        self.num_side_branches = num_side_branches
        self.embeddings = VanillaBertEmbeddings(
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            layer_norm_eps=config.layer_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            vocab_size=config.vocab_size,
        )
        self.encoder = LttBertMultiEncoder(
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            num_side_branches=self.num_side_branches,
            s_attn_hidden_size=config.s_attn_hidden_size,
            s_attn_intermediate_size=config.s_attn_intermediate_size,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        side_layer_branches: List[int],
    ) -> Tuple[Tensor, List[Tensor]]:
        input_shape = cast(Tuple[int], input_ids.size())
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        sequence_output, side_layer_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            side_layer_branches=side_layer_branches,
        )
        return sequence_output, side_layer_outputs

    pass


class LttBertMultiEncoder(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_attention_heads: int,
        num_hidden_layers: int,
        num_side_branches: int,
        s_attn_hidden_size: int,
        s_attn_intermediate_size: int,
    ):
        super().__init__()
        self.num_layers = num_hidden_layers
        self.num_branches = num_side_branches

        layers: List[nn.Module] = []
        for _ in range(self.num_layers):
            layer = VanillaBertLayer(
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
                num_attention_heads=num_attention_heads,
                repl_norm_1_ident=False,
                repl_norm_2_ident=False,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        s_attn_maps: Dict[str, nn.Module] = {}
        for i_b in range(self.num_branches):
            for i_ly in range(self.num_layers):
                layer = nn.Linear(hidden_size, s_attn_hidden_size)
                s_attn_maps[f"{i_b}_{i_ly}"] = layer
        self.s_attn_maps = nn.ModuleDict(s_attn_maps)

        s_attn_layers: Dict[str, nn.Module] = {}
        for i_b in range(self.num_branches):
            for i_ly in range(self.num_layers):
                layer = VanillaBertLayer(
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    hidden_dropout_prob=hidden_dropout_prob,
                    hidden_size=s_attn_hidden_size,
                    intermediate_size=s_attn_intermediate_size,
                    layer_norm_eps=layer_norm_eps,
                    num_attention_heads=num_attention_heads,
                    repl_norm_1_ident=False,
                    repl_norm_2_ident=False,
                )
                s_attn_layers[f"{i_b}_{i_ly}"] = layer
        self.s_attn_layers = nn.ModuleDict(s_attn_layers)

        # training hacks
        self._ltt_freeze_layer = self.num_layers

    def ltt_freeze_layers_until(self, layer_id: int) -> None:
        layer_id = max(1, min(self.num_layers, layer_id))
        self._ltt_freeze_layer = layer_id
        return

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        side_layer_branches: List[int],
    ) -> Tuple[Tensor, List[Tensor]]:
        side_states: List[Tensor] = []
        for i_b in range(self.num_branches):
            side_state: Tensor = cast(Tensor, None)
            if i_b in side_layer_branches:
                side_state = cast(Tensor, 0.0)  # so initial assignment works
            side_states.append(side_state)

        for i_ly in range(self.num_layers):
            hidden_states = self.layers[i_ly](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            for i_b in range(self.num_branches):
                if i_ly >= self._ltt_freeze_layer:
                    continue  # skip following layers
                if i_b not in side_layer_branches:
                    continue
                s_attn_map = self.s_attn_maps[f"{i_b}_{i_ly}"]
                side_states[i_b] = side_states[i_b] + F.gelu(s_attn_map(hidden_states))
                s_attn_layer = self.s_attn_layers[f"{i_b}_{i_ly}"]
                side_states[i_b] = s_attn_layer(
                    hidden_states=side_states[i_b],
                    attention_mask=attention_mask,
                )

        side_states = [s for s in side_states if s is not None]
        return hidden_states, side_states

    pass
