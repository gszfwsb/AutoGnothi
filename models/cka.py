from typing import Optional

import torch
from torch import Tensor


def kernel_cka(X: Tensor, Y: Tensor, sigma: Optional[float] = None) -> Tensor:
    # <batch_size, heads, alpha> cka <batch_size, heads, beta>
    bs, *_ = X.shape
    return torch.stack([__kernel_cka(X[i], Y[i], sigma) for i in range(bs)])


def linear_cka(X: Tensor, Y: Tensor) -> Tensor:
    # <batch_size, heads, alpha> cka <batch_size, heads, beta>
    bs, *_ = X.shape
    return torch.stack([__linear_cka(X[i], Y[i]) for i in range(bs)])


def __kernel_cka(X: Tensor, Y: Tensor, sigma: Optional[float] = None):
    # <heads, alpha> cka <heads, beta>
    hsic = __kernel_hsic(X, Y, sigma)
    var1 = torch.sqrt(__kernel_hsic(X, X, sigma))
    var2 = torch.sqrt(__kernel_hsic(Y, Y, sigma))
    return hsic / (var1 * var2)


def __linear_cka(X: Tensor, Y: Tensor):
    # <heads, alpha> cka <heads, beta>
    hsic = __linear_hsic(X, Y)
    var1 = torch.sqrt(__linear_hsic(X, X))
    var2 = torch.sqrt(__linear_hsic(Y, Y))
    return hsic / (var1 * var2)


def __kernel_hsic(X: Tensor, Y: Tensor, sigma: Optional[float]):
    return torch.sum(__centering(__rbf(X, sigma)) * __centering(__rbf(Y, sigma)))


def __linear_hsic(X: Tensor, Y: Tensor):
    lX = X @ X.T
    lY = Y @ Y.T
    return torch.sum(__centering(lX) * __centering(lY))


def __centering(K: Tensor):
    # <heads, heads>
    n, *_ = K.shape
    unit = torch.ones(n, n, device=K.device)
    Id = torch.eye(n, device=K.device)
    H = Id - unit / n
    # HKH are the same with KH, KH is the first centering, H(KH) do the second
    # time, results are the sme with one time centering
    return (H @ K) @ H


def __rbf(X: Tensor, sigma: Optional[float]):
    GX = X @ X.T
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        m_dist = torch.median(KX[KX != 0])
        sig = torch.sqrt(m_dist)
    else:
        sig = torch.tensor(sigma, device=X.device)
    KX *= -0.5 / (sig * sig)
    KX = torch.exp(KX)
    return KX
