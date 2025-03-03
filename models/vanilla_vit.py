import math
from typing import List, Optional, Tuple

import pydantic
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import Self

from ..utils.nnmodel import ObservableModuleMixin, freeze_model_parameters
from .shapley import normalize_shapley_explanation


class VanillaViTConfig(pydantic.BaseModel):
    """differs from `transformers.ViTModel`"""

    attention_probs_dropout_prob: float
    explainer_attn_num_layers: int
    explainer_head_hidden_size: int
    explainer_normalize: bool
    hidden_dropout_prob: float
    hidden_size: int
    intermediate_size: int
    layer_norm_eps: float
    num_attention_heads: int
    num_hidden_layers: int
    num_labels: int
    img_channels: int
    img_px_size: int
    img_patch_size: int

    pass


class VanillaViTClassifier(nn.Module, ObservableModuleMixin):
    def __init__(self, config: VanillaViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.vit = VanillaViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.act = nn.Softmax(dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        freeze_model_parameters(self, "vit")
        freeze_model_parameters(self, "classifier")
        return self

    def forward(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        output = self.vit(x, attention_mask)
        self.om_record_features(repr_cls=output)
        logits = self.classifier(output[:, 0, :])
        logits = self.act(logits)
        return logits

    pass


class VanillaViTSurrogate(VanillaViTClassifier):
    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        return self

    pass


class VanillaViTExplainer(nn.Module, ObservableModuleMixin):
    def __init__(self, config: VanillaViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config

        self.vit = VanillaViTModel(config)

        attn_layers: List[nn.Module] = []
        for i_ly in range(config.explainer_attn_num_layers):
            attn_layer = VanillaViTLayer(
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                layer_norm_eps=config.layer_norm_eps,
                repl_norm_1_ident=i_ly == 0,  # first layer needs norm1 be Identity
                repl_norm_2_ident=False,
            )
            attn_layers.append(attn_layer)
        self.explainer_attn = nn.ModuleList(attn_layers)

        mlp_max_width = int(config.explainer_head_hidden_size)
        mlp_list: List[nn.Module] = []
        mlp_list.append(nn.LayerNorm(config.hidden_size))
        mlp_list.append(nn.Linear(config.hidden_size, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, mlp_max_width))
        mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(mlp_max_width, config.num_labels))
        self.explainer_mlp = nn.Sequential(*mlp_list)

    def forward(
        self,
        pixel_values: Tensor,
        attention_mask: Tensor,
        surrogate_grand: Tensor,
        surrogate_null: Tensor,
    ) -> Tensor:
        """
        inp [ ]: pixel_values:    <batch_size, img_channels, img_px_size, img_px_size>
        inp [ ]: attention_mask:  <batch_size, n_tokens = 1 + n_players>
                                               n_players = (img_px_size / img_patch_size)^2
        inp [ ]: surrogate_grand: <batch_size, n_classes>
        inp [ ]: surrogate_null:  <1, n_classes>
        ret [0]: output:          <batch_size, n_players, n_classes>
        """

        output = self.vit(pixel_values, attention_mask)
        self.om_record_features(repr_exp=output)
        for attn_layer in self.explainer_attn:
            output = attn_layer(output, attention_mask)
        # (batch_size, n_tokens, num_labels)
        output = self.explainer_mlp(output)
        # excluding class token.
        if self.config.explainer_normalize:
            output = normalize_shapley_explanation(
                output, surrogate_grand, surrogate_null
            )
        output = output[:, 1:, :].permute(0, 2, 1)
        return output

    pass


class VanillaViTFinal(nn.Module, ObservableModuleMixin):
    def __init__(self, config: VanillaViTConfig):
        nn.Module.__init__(self)
        ObservableModuleMixin.__init__(self)
        self.config = config
        self.classifier = VanillaViTClassifier(config)
        self.surrogate = VanillaViTSurrogate(config)
        self.surrogate_null = nn.Parameter(
            torch.zeros((1, config.num_labels)), requires_grad=False
        )
        self.explainer = VanillaViTExplainer(config)

    def forward(
        self,
        pixel_values: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        logits = self.classifier(pixel_values, attention_mask)
        om_repr_cls = self.classifier.om_take_observations()
        if self.config.explainer_normalize:
            surrogate_grand = self.surrogate(pixel_values, attention_mask)
            om_repr_srg = self.surrogate.om_take_observations()
        else:
            surrogate_grand = ...
            om_repr_srg = {}
        explainer = self.explainer(
            pixel_values, attention_mask, surrogate_grand, self.surrogate_null
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


class VanillaViTModel(nn.Module):
    def __init__(self, config: VanillaViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = VanillaViTEmbeddings(
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            img_px_size=config.img_px_size,
            img_patch_size=config.img_patch_size,
            img_channels=config.img_channels,
        )
        self.encoder = VanillaViTEncoder(
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            layer_norm_eps=config.layer_norm_eps,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor, attention_mask: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.layernorm(hidden_states)
        return hidden_states

    pass


class VanillaViTEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_dropout_prob: float,
        hidden_size: int,
        img_px_size: int,
        img_patch_size: int,
        img_channels: int,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.patch_embeddings = VanillaViTPatchEmbeddings(
            hidden_size=hidden_size,
            img_channels=img_channels,
            img_px_size=img_px_size,
            img_patch_size=img_patch_size,
        )
        n_patches = self.patch_embeddings.n_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, n_patches + 1, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values: <batch_size, img_channels, img_px_size, img_px_size>
        batch_size, _n_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        # dropout & fin
        embeddings = self.dropout(embeddings)
        return embeddings

    pass


class VanillaViTPatchEmbeddings(nn.Module):
    """Converts pixel values (batch_size, n_channels, height, width) to a
    sequence of embeddings (batch_size, n_patches, hidden_size)."""

    def __init__(
        self,
        hidden_size: int,
        img_channels: int,
        img_px_size: int,
        img_patch_size: int,
    ):
        super().__init__()
        n_patches = (img_px_size // img_patch_size) ** 2
        self.img_channels = img_channels
        self.img_px_size = img_px_size
        self.img_patch_size = img_patch_size
        self.n_patches = n_patches
        self.projection = nn.Conv2d(
            img_channels, hidden_size, kernel_size=img_patch_size, stride=img_patch_size
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        _batch_size, n_channels, height, width = pixel_values.shape
        assert n_channels == self.img_channels
        assert height == self.img_px_size and width == self.img_px_size
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class VanillaViTEncoder(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_hidden_layers: int,
        num_attention_heads: int,
    ):
        super().__init__()
        _layers = [
            VanillaViTLayer(
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
                num_attention_heads=num_attention_heads,
                repl_norm_1_ident=False,
                repl_norm_2_ident=False,
            )
            for i in range(num_hidden_layers)
        ]
        self.layers = nn.ModuleList(_layers)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        for _i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
        return hidden_states

    pass


class VanillaViTLayer(nn.Module):
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
        self.attention = VanillaViTAttention(
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.intermediate = VanillaViTIntermediate(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.output = VanillaViTOutput(
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.layernorm_before = (
            nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            if not repl_norm_1_ident
            else nn.Identity()
        )
        self.layernorm_after = (
            nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            if not repl_norm_2_ident
            else nn.Identity()
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        layer_normed = self.layernorm_before(hidden_states)
        attention_output = self.attention(layer_normed, attention_mask)
        # first residual connection
        hidden_states = hidden_states + attention_output
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        # second residual connection
        layer_output = self.output(layer_output, hidden_states)
        return layer_output

    pass


class VanillaViTAttention(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        hidden_dropout_prob: float,
        hidden_size: int,
        num_attention_heads: int,
    ):
        super().__init__()
        self.self = VanillaViTSelfAttention(
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.output = VanillaViTSelfOutput(
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_size=hidden_size,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        hidden_states = self.self(hidden_states, attention_mask)
        hidden_states = self.output(hidden_states)
        return hidden_states

    pass


class VanillaViTSelfAttention(nn.Module):
    def __init__(
        self,
        attention_probs_dropout_prob: float,
        num_attention_heads: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=True)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=True)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # apply mask
        # TODO: verify whether mul is better than add
        attention_mask = attention_mask.reshape((attention_mask.shape[0], 1, 1, -1))
        attention_scores = attention_scores * attention_mask
        # attention_scores = attention_scores + attention_mask

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

    pass


class VanillaViTSelfOutput(nn.Module):
    def __init__(self, hidden_dropout_prob: float, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    pass


class VanillaViTIntermediate(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    pass


class VanillaViTOutput(nn.Module):
    def __init__(
        self,
        hidden_dropout_prob: float,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

    pass


class VanillaViTPooler(nn.Module):
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
