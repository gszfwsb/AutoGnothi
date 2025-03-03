import random
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


def loss_shapley_new(
    batch_size: int,
    n_mask_samples: int,
    n_players: int,
    mask: Tensor,
    v_0: Tensor,
    v_s: Tensor,
    v_1: Tensor,
    phi: Tensor,
) -> Tensor:
    """The really proper loss function:
    https://github.com/suinleelab/vit-shapley/blob/master/vit_shapley/modules/explainer.py
    We do not know why it works like this until this day.

    inp [3]: mask: <batch_size, n_mask_samples, n_players>, first col is
        cls token, order: [b0 s0, b0 s1, b1 s0, b1 s1, ...];
    inp [4]: v_0: <1, n_classes> of the student's evaluation on an empty image;
    inp [5]: v_s: <batch_size * n_mask_samples, n_classes> of the student's
        evaluation on all currently masked images as per `mask`, flattened;
    inp [6]: v_1: <batch_size, n_classes> of the student's evaluation on all
        images (same as v_s) but without masks;
    inp [7]: phi: <batch_size, n_classes, n_players> of attribution per class;
    ret [ ]: loss: scalar value."""

    # <batch_size, n_players>, <batch_size, n_mask_samples, n_players>
    #  -> <batch_size, n_mask_samples, n_classes>
    surrogate_values = v_s
    # <1, n_classes>
    surrogate_null = v_0

    # attribution: <batch_size, n_players, n_classes>
    values_pred = phi.permute(0, 2, 1)
    # <1, n_classes> + <batch_size, n_mask_samples, n_players> @ <batch_size, n_players, n_classes>
    #  -> <batch_size, n_mask_samples, n_classes>
    #  -> <batch_size * n_mask_samples, n_classes>
    values_pred_approx = surrogate_null.reshape(1, 1, -1) + mask.float() @ values_pred
    values_pred_approx = values_pred_approx.reshape(batch_size * n_mask_samples, -1)

    # <batch_size * n_mask_samples, n_classes>, <batch_size * n_mask_samples, n_classes> -> <>
    value_diff = n_players * F.mse_loss(
        input=values_pred_approx, target=surrogate_values, reduction="mean"
    )

    loss = value_diff
    return loss


def mask_shapley_new(n_mask_samples: int, n_players: int) -> Tensor:
    """Also refer to loss_shapley_new. Masked samples must be paired so that
    we can keep the average `0`."""

    paired_mask_samples = True
    if paired_mask_samples:
        assert n_mask_samples % 2 == 0
        n_mask_samples = n_mask_samples // 2

    probs = torch.arange(1, n_players) * (n_players - torch.arange(1, n_players))
    probs = 1 / probs
    probs = probs / probs.sum()

    masks_1 = torch.rand(n_mask_samples, n_players)
    masks_2 = (1 / n_players) * _torch_choice(
        a=torch.arange(n_players - 1), p=probs, size=(n_mask_samples, 1)
    )
    masks = (masks_1 > masks_2).long()

    if paired_mask_samples:  # explainer_train
        masks = torch.stack([masks, 1 - masks], dim=1).reshape(
            n_mask_samples * 2, n_players
        )
    return masks


def normalize_shapley_explanation(pred: Tensor, grand: Tensor, null: Tensor) -> Tensor:
    """Normalization for shapley values.
    inp [ ]: pred:            <batch_size, n_players, n_classes>
    inp [ ]: surrogate_grand: <batch_size, n_classes> (with mask <1...>)
    inp [ ]: surrogate_null:  <1, n_classes> (from nil input)
    ret [0]: pred':           <batch_size, n_players, n_classes>"""

    batch_size, n_players, _ = pred.shape
    grand = grand.unsqueeze(1)  # <batch_size, 1, n_classes>
    null = null.reshape((1, 1, -1)).repeat(batch_size, 1, 1)
    diff = (grand - null) - torch.sum(pred, dim=1).unsqueeze(1)
    return pred + diff / n_players


def loss_logits_kl_divergence(
    ref: Tensor,
    current: Tensor,
) -> Tensor:
    """Try to use KL divergence as a loss function."""

    return F.kl_div(
        input=F.log_softmax(ref, dim=-1),
        target=F.softmax(current, dim=-1),
        reduction="batchmean",
    )


def mask_purely_uniform(batch_size: int, n_features: int) -> Tensor:
    """Generate mask with certain features randomly masked out. The number of
    features masked out is selected uniformly."""

    # this effectively uniforms the number of features masked out
    masks = torch.rand((batch_size, n_features)) > torch.rand((batch_size, 1))
    return masks.long()


def mask_uniform_selective(batch_size: int, n_features: int, n_masked: int) -> Tensor:
    """Generate mask with a fixed number of features masked out."""

    ret: List[List[int]] = []
    for _ in range(batch_size):
        ids = list(range(n_features))
        random.shuffle(ids)
        ids = set(ids[:n_masked])
        ls = [0 if i in ids else 1 for i in range(n_features)]
        ret.append(ls)
    return torch.tensor(ret, dtype=torch.long)


def _torch_choice(a: Tensor, p: Tensor, size: tuple) -> Tensor:
    prefix = torch.cumsum(p, dim=0) - p
    samples = torch.rand(size).reshape(-1, 1)
    position = torch.max((samples >= prefix).sum(dim=1) - 1, torch.tensor(0))
    return a[position].reshape(size)
