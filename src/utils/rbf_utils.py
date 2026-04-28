import math
from typing import Optional

import torch


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def median_heuristic_sigma(
    *embeddings: torch.Tensor,
    normalize: bool = True,
) -> float:
    """
    Median pairwise distance heuristic for RBF bandwidth selection.

    Args:
        embeddings: one or more [N, D] tensors to concatenate before computing distances.
        normalize: whether to L2-normalize before distance computation.
    """
    if not embeddings:
        return 1.0
    Z = torch.cat(embeddings, dim=0)
    if normalize:
        Z = _l2_normalize(Z)
    if Z.size(0) < 2:
        return 1.0
    dists = torch.pdist(Z, p=2)
    if dists.numel() == 0:
        return 1.0
    median_sqdist = torch.median(dists ** 2).item()
    sigma = math.sqrt(max(median_sqdist / 2.0, 1e-12))
    return float(sigma)


def rbf_kernel_matrix(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive for RBF kernel.")
    X = X.float()
    Y = Y.float()
    x_norm = (X ** 2).sum(dim=1, keepdim=True)
    y_norm = (Y ** 2).sum(dim=1, keepdim=True).transpose(0, 1)
    sq_dists = (x_norm + y_norm - 2.0 * X @ Y.transpose(0, 1)).clamp_min(0.0)
    return torch.exp(-sq_dists / (2.0 * sigma ** 2))


def normalize_embeddings(x: torch.Tensor) -> torch.Tensor:
    """Public helper for shared L2 normalization."""
    return _l2_normalize(x)
