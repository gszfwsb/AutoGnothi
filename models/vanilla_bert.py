import math
from typing import List, Optional, Tuple, cast

import pydantic
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.pytorch_utils import apply_chunking_to_forward
from typing_extensions import Self

from ..utils.nnmodel import ObservableModuleMixin, freeze_model_parameters
from .shapley import normalize_shapley_explanation


class VanillaBertConfig(pydantic.BaseModel):
    """equiv. `transformers.BertModel`"""

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

    pass


class VanillaBertClassifier(nn.Module, ObservableModuleMixin):
    def __init__(self, config: VanillaBertConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config
        self.bert = VanillaBertModel(config)

        self.bert_pooler = VanillaBertPooler(hidden_size=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_model_parameters(self, "bert")
        freeze_model_parameters(self, "bert_pooler")
        freeze_model_parameters(self, "classifier")
        return self

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        self.om_record_features(repr_cls=output)
        output = self.bert_pooler(output)
        output = self.dropout(output)
        logits = self.classifier(output)
        logits = self.act(logits)
        return logits

    pass


class VanillaBertSurrogate(VanillaBertClassifier):
    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        return self

    pass


class VanillaBertExplainer(nn.Module, ModuleUtilsMixin, ObservableModuleMixin):
    def __init__(self, config: VanillaBertConfig):
        nn.Module.__init__(self)
        ModuleUtilsMixin.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config
        self.bert = VanillaBertModel(config)

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
    ) -> Tensor:
        """
        inp [ ]: input_ids:       <batch_size, n_tokens>
        inp [ ]: attention_mask:  <batch_size, n_tokens>
        inp [ ]: token_type_ids:  <batch_size, n_tokens> := {1...}
        inp [ ]: surrogate_grand: <batch_size, n_classes>
        inp [ ]: surrogate_null:  <1, n_classes>
        ret [0]: output:          <batch_size, n_players := n_tokens - 1, n_classes>
        """

        input_shape = cast(Tuple[int], input_ids.size())
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        self.om_record_features(repr_exp=output)
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
        return output

    pass


class VanillaBertFinal(nn.Module, ObservableModuleMixin):
    def __init__(self, config: VanillaBertConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config
        self.classifier = VanillaBertClassifier(self.config)
        self.surrogate = VanillaBertSurrogate(self.config)
        self.surrogate_null = nn.Parameter(
            torch.zeros((1, self.config.num_labels)), requires_grad=False
        )
        self.explainer = VanillaBertExplainer(self.config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        inp [ ]: input_ids:       <batch_size, n_tokens>
        inp [ ]: attention_mask:  <batch_size, n_tokens> := {1...}
        inp [ ]: token_type_ids:  <batch_size, n_tokens> := {1...}
        ret [0]: logits:          <batch_size, n_classes>
        ret [1]: explanation:     <batch_size, n_players := n_tokens - 1, n_classes>
        """

        logits = self.classifier(input_ids, attention_mask, token_type_ids)
        om_repr_cls = self.classifier.om_take_observations()
        if self.config.explainer_normalize:
            surrogate_grand = self.surrogate(input_ids, attention_mask, token_type_ids)
            om_repr_srg = self.surrogate.om_take_observations()
        else:
            surrogate_grand = ...
            om_repr_srg = {}
        explainer = self.explainer(
            input_ids,
            attention_mask,
            token_type_ids,
            surrogate_grand,
            self.surrogate_null,
        )
        om_repr_exp = self.explainer.om_take_observations()
        self.om_record_features(
            repr_cls=om_repr_cls.get("repr_cls", None),
            repr_srg=om_repr_srg.get("repr_srg", None),
            repr_exp=om_repr_exp.get("repr_exp", None),
        )
        return logits, explainer

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        freeze_model_parameters(self, "classifier")
        return self

    def om_retain_observations(self, flag: bool = True) -> None:
        ObservableModuleMixin.om_retain_observations(self, flag)
        self.classifier.om_retain_observations(flag)
        self.surrogate.om_retain_observations(flag)
        self.explainer.om_retain_observations(flag)

    pass


class VanillaBertModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config: VanillaBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = VanillaBertEmbeddings(
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            layer_norm_eps=config.layer_norm_eps,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            type_vocab_size=config.type_vocab_size,
            vocab_size=config.vocab_size,
        )
        self.encoder = VanillaBertEncoder(
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        input_shape = cast(Tuple[int], input_ids.size())
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        sequence_output = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
        )
        return sequence_output

    pass


class VanillaBertEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_dropout_prob: float,
        hidden_size: int,
        layer_norm_eps: float,
        max_position_embeddings: int,
        pad_token_id: int,
        type_vocab_size: int,
        vocab_size: int,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = "absolute"
        self.position_ids: Tensor  # dummy
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
    ) -> Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    pass


class VanillaBertEncoder(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_attention_heads: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_hidden_layers):
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

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        for _i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs

        return hidden_states

    pass


class VanillaBertLayer(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_attention_heads: int,
        repl_norm_1_ident: bool,
        repl_norm_2_ident: bool,
    ):
        super().__init__()
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = VanillaBertAttention(
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            num_attention_heads=num_attention_heads,
            repl_norm_ident=repl_norm_1_ident,
        )
        self.add_cross_attention = False
        self.intermediate = VanillaBertIntermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.output = VanillaBertOutput(
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            repl_norm_ident=repl_norm_2_ident,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    pass


class VanillaBertAttention(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_size: int,
        layer_norm_eps: float,
        num_attention_heads: int,
        repl_norm_ident: bool,
    ):
        super().__init__()
        self.self = VanillaBertSelfAttention(
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.output = VanillaBertSelfOutput(
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            repl_norm_ident=repl_norm_ident,
        )
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: torch.FloatTensor,
    ) -> Tensor:
        self_output = self.self(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output

    pass


class VanillaBertSelfAttention(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_size: int,
        num_attention_heads: int,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = "absolute"

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tensor:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class VanillaBertSelfOutput(nn.Module):
    def __init__(
        self,
        hidden_dropout_prob: float,
        hidden_size: int,
        layer_norm_eps: float,
        repl_norm_ident: bool,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if not repl_norm_ident:
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.LayerNorm = nn.Identity()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    pass


class VanillaBertIntermediate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    pass


class VanillaBertOutput(nn.Module):
    def __init__(
        self,
        hidden_dropout_prob: float,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        repl_norm_ident: bool,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        if not repl_norm_ident:
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.LayerNorm = nn.Identity()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    pass


class VanillaBertPooler(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    pass
