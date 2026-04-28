from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

EPS = 1e-12
UNIFORM_TOL = 1e-6


def compute_move_grid(
    steps: int,
    mask_schedule: Optional[Sequence[float]] = None,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    if mask_schedule is None:
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        move_grid = t_grid.clone()
    else:
        schedule = torch.tensor(mask_schedule, device=device, dtype=dtype)
        if schedule.numel() == steps + 1:
            move_grid = schedule
        elif schedule.numel() == steps:
            move_grid = torch.cat([schedule, schedule[-1:].clone()], dim=0)
        else:
            values = schedule.view(1, 1, -1)
            move_grid = F.interpolate(
                values,
                size=steps + 1,
                mode="linear",
                align_corners=True,
            ).view(-1)
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)

    move_grid = move_grid.clamp_min(0.0).clamp_max(1.0)
    delta_move = (move_grid[:-1] - move_grid[1:]).clamp_min(0.0)
    return t_grid, move_grid, delta_move


def get_num_transfer_tokens_uniform(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    mask_num = mask_index.sum(dim=1)  # [B]
    base = (mask_num // steps).to(torch.int64)         # [B]
    rem = (mask_num % steps).to(torch.int64)           # [B]
    num_transfer_tokens = base[:, None].expand(-1, steps).clone()  # [B, steps]
    step_ids = torch.arange(steps, device=mask_index.device, dtype=torch.int64)[None, :]
    num_transfer_tokens += (step_ids < rem[:, None]).to(torch.int64)
    return num_transfer_tokens


def get_num_transfer_tokens_move(
    mask_index: torch.Tensor,
    steps: int,
    move_grid: torch.Tensor,
    *,
    eps: float = EPS,
    uniform_tol: float = UNIFORM_TOL,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if move_grid.numel() != steps + 1:
        raise ValueError(
            f"move_grid must have length steps+1 ({steps + 1}), got {move_grid.numel()}"
        )
    delta_move = (move_grid[:-1] - move_grid[1:]).clamp_min(0.0)
    if delta_move.numel() <= 1:
        return get_num_transfer_tokens_uniform(mask_index, steps)
    if (delta_move.max() - delta_move.min()) <= uniform_tol:
        return get_num_transfer_tokens_uniform(mask_index, steps)
    sum_delta = float(delta_move.sum().item())
    if sum_delta <= eps:
        return get_num_transfer_tokens_uniform(mask_index, steps)

    mask_num = mask_index.sum(dim=1).to(torch.int64)  # [B]
    weights = (delta_move / sum_delta).to(torch.float32)  # [steps]
    raw = mask_num[:, None].to(torch.float32) * weights[None, :]
    counts = torch.floor(raw).to(torch.int64)
    remainder = (mask_num - counts.sum(dim=1)).clamp_min(0)
    max_k = int(remainder.max().item()) if remainder.numel() else 0
    if max_k > 0:
        frac = raw - counts.to(raw.dtype)
        _, top_idx = torch.topk(frac, k=max_k, dim=1)
        rank = torch.arange(max_k, device=mask_index.device, dtype=torch.int64)[None, :]
        take = rank < remainder[:, None]
        counts.scatter_add_(1, top_idx, take.to(torch.int64))
    return counts
