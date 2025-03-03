from typing import Callable, List, Optional, cast

import numpy as np
import pydantic
import rich
import rich.progress
import shap
import torch
from torch import Tensor, nn

from ..utils.nnmodel import freeze_model_parameters
from .vanilla_bert import VanillaBertClassifier, VanillaBertConfig


class KernelShapBertConfig(pydantic.BaseModel):
    """BERT that uses KernelShap for explanation."""

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

    kernel_shap_n_samples: int
    kernel_shap_data_size: int

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


class KernelShapBertClassifier(VanillaBertClassifier):
    def __init__(self, config: KernelShapBertConfig):
        super().__init__(config.into())

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        freeze_model_parameters(self, "bert")
        freeze_model_parameters(self, "bert_pooler")
        freeze_model_parameters(self, "classifier")
        return self

    pass


class KernelShapBertSurrogate(KernelShapBertClassifier):
    pass


class KernelShapBertExplainer(nn.Module):
    def __init__(self, config: KernelShapBertConfig):
        # generally unused model. use final only
        nn.Module.__init__(self)
        self.config = config
        self.Xs_train = nn.Parameter(
            torch.zeros(
                (
                    config.kernel_shap_data_size,  # let this many be kept
                    config.max_position_embeddings,  # <cls> + n_players
                ),
                dtype=torch.long,  # we save tokens
            ),
            requires_grad=False,  # only fp tensors can have grad
        )

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        freeze_model_parameters(self, "Xs_train")
        return self

    pass


class KernelShapBertFinal(nn.Module):
    def __init__(self, config: KernelShapBertConfig):
        nn.Module.__init__(self)
        self.config = config
        self.classifier = KernelShapBertClassifier(config)
        self.explainer = KernelShapBertExplainer(config)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        freeze_model_parameters(self, "classifier")
        freeze_model_parameters(self, "explainer")
        return self

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> Tensor:
        logits = self.classifier(input_ids, attention_mask, token_type_ids)
        return logits

    pass


def kernel_shap_torch(
    fw_classifier: Callable[[Tensor], Tensor],
    Xs_train: Tensor,
    Xs_explain: Tensor,
    n_samples: int,
    batch_size: int,
    silent: bool,
) -> Tensor:
    """KernelSHAP adapter for PyTorch models. We suggest putting this logic in
    recipe files instead of model files.

    inp [0]: fw_classifier: (Xs: <bs, n_players>) -> Ys: <bs>,
             whereas Xs are perturbed inputs, and no masking is done;
    inp [1]: <bs_train, n_players>: clustered training data;
    inp [2]: <bs_explain, n_players>: data to explain;
    inp [3]: n_samples: number of samples to draw in KernelSHAP;
    inp [4]: batch_size: batch size for model inference;
    inp [5]: silent: whether to display progress bar;
    out [ ]: <bs_explain, n_players>: shapley values for each player."""

    device = Xs_explain.device
    progress: Optional[rich.progress.Progress] = None
    task: Optional[rich.progress.TaskID] = None
    # <train, n_players>, <1, n_players>, <n_samples * n_players, n_players>
    run_size = (
        Xs_train.shape[0] + 1 + max(n_samples, Xs_train.shape[0] * Xs_train.shape[1])
    )

    def safe_fw_cls(xs_n: np.ndarray) -> np.ndarray:
        xs = torch.from_numpy(xs_n).to(dtype=torch.long, device=device)
        # return torch.rand((xs.shape[0], 2)).cpu().numpy()
        ys: List[Tensor] = []
        for i in range(0, len(xs), batch_size):
            x = xs[i : i + batch_size]
            ys.append(fw_classifier(x))
            if progress is not None and task is not None:
                progress.update(task, advance=x.shape[0])
        return torch.cat(ys).cpu().numpy()

    def explain() -> Tensor:
        explainer = shap.KernelExplainer(
            model=safe_fw_cls,
            data=Xs_train.cpu().numpy(),
            link="logit",
        )
        num_inputs, *_ = Xs_explain.shape
        outputs: List[Tensor] = []
        for i in range(0, num_inputs):
            row = Xs_explain[i : i + 1]
            shap_values = explainer.shap_values(  # list<ndarray>, for each cls
                row.cpu().numpy(), nsamples=n_samples, silent=True
            )
            # <bs, <cls> + players, cls>
            exp = np.stack(cast(List[np.ndarray], shap_values))
            # <bs, cls, players>
            exp = torch.from_numpy(exp).permute(0, 2, 1)[:, :, 1:]
            outputs.append(exp)
        return torch.cat(outputs).to(device)

    # we need progress
    if not silent:
        with rich.progress.Progress() as _p:
            progress = _p
            task = progress.add_task(
                "KernelSHAP: explaining...", total=batch_size * run_size
            )
            attr = explain()
            progress.stop()
    else:
        attr = explain()
    return attr
