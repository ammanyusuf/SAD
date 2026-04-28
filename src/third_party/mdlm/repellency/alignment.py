"""Alignment strategies for continuation-aware repellency."""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class AlignmentResult:
    x_t: torch.Tensor
    x_0_hat: torch.Tensor
    cont_mask: Optional[torch.Tensor]
    should_apply: bool


class ContinuationAlignmentStrategy:
    def align(self, x_t: torch.Tensor, x_0_hat: torch.Tensor) -> AlignmentResult:
        raise NotImplementedError

    def scatter(self, p_safe: torch.Tensor, cont_mask: Optional[torch.Tensor], template: torch.Tensor) -> torch.Tensor:
        if cont_mask is None:
            return p_safe
        p_safe_full = template.clone()
        for b in range(cont_mask.size(0)):
            positions = cont_mask[b]
            if positions.any():
                slice_len = int(positions.sum().item())
                p_safe_full[b, positions] = p_safe[b, :slice_len]
        return p_safe_full

    @staticmethod
    def _create_continuation_probs(x_0_hat: torch.Tensor, cont_mask: torch.Tensor, length: int) -> torch.Tensor:
        B, _, V = x_0_hat.shape
        fill = 1.0 / float(V)
        p_cont = torch.full(
            (B, length, V),
            fill_value=fill,
            device=x_0_hat.device,
            dtype=x_0_hat.dtype,
        )
        for b in range(B):
            positions = cont_mask[b]
            if positions.any():
                slice_len = min(int(positions.sum().item()), length)
                if slice_len > 0:
                    p_cont[b, :slice_len] = x_0_hat[b, positions][:slice_len]
        return p_cont


class DefaultAlignmentStrategy(ContinuationAlignmentStrategy):
    def __init__(self, continuation_length: Optional[int], mask_index: int, pad_index: Optional[int]):
        self.continuation_length = continuation_length
        self.mask_index = mask_index
        self.pad_token = pad_index if pad_index is not None else mask_index
        self.fill_token = mask_index

    def align(self, x_t: torch.Tensor, x_0_hat: torch.Tensor) -> AlignmentResult:
        if self.continuation_length is None:
            return AlignmentResult(x_t=x_t, x_0_hat=x_0_hat, cont_mask=None, should_apply=True)
        B, L_total = x_t.shape
        L_cont = self.continuation_length
        x_cont = torch.full((B, L_cont), self.fill_token, device=x_t.device, dtype=x_t.dtype)
        cont_mask = torch.zeros((B, L_total), device=x_t.device, dtype=torch.bool)
        slice_len = min(L_total, L_cont)
        if slice_len > 0:
            x_cont[:, :slice_len] = x_t[:, :slice_len]
            cont_mask[:, :slice_len] = True
        p_cont = self._create_continuation_probs(x_0_hat, cont_mask, L_cont)
        return AlignmentResult(x_t=x_cont, x_0_hat=p_cont, cont_mask=cont_mask, should_apply=True)


class NoAlignmentStrategy(ContinuationAlignmentStrategy):
    def align(self, x_t: torch.Tensor, x_0_hat: torch.Tensor) -> AlignmentResult:
        return AlignmentResult(x_t=x_t, x_0_hat=x_0_hat, cont_mask=None, should_apply=True)


class LeftMaskAlignmentStrategy(ContinuationAlignmentStrategy):
    def __init__(self, mask_index: int, pad_index: Optional[int], continuation_length: Optional[int], vocab_size: int):
        self.mask_index = mask_index
        self.pad_index = pad_index
        self.continuation_length = continuation_length
        self.vocab_size = vocab_size
        self.pad_token = pad_index if pad_index is not None else mask_index
        self.fill_token = mask_index
        self._cached_x_t: Optional[torch.Tensor] = None
        self._cached_x_cont: Optional[torch.Tensor] = None
        self._cached_cont_mask: Optional[torch.Tensor] = None

    def _continuation_from_cache(self, x_t: torch.Tensor):
        if self._cached_x_t is not None and torch.equal(self._cached_x_t, x_t):
            return self._cached_x_cont, self._cached_cont_mask
        return None, None

    def _cache(self, x_t: torch.Tensor, x_cont: torch.Tensor, cont_mask: torch.Tensor):
        self._cached_x_t = x_t.detach().clone()
        self._cached_x_cont = x_cont
        self._cached_cont_mask = cont_mask

    def _extract_continuation(self, x_t: torch.Tensor):
        cached_cont, cached_mask = self._continuation_from_cache(x_t)
        if cached_cont is not None and cached_mask is not None:
            return cached_cont, cached_mask

        B, L_total = x_t.shape
        L_cont = self.continuation_length if self.continuation_length is not None else L_total
        x_cont = torch.full((B, L_cont), self.fill_token, device=x_t.device, dtype=x_t.dtype)
        cont_mask = torch.zeros((B, L_total), device=x_t.device, dtype=torch.bool)
        valid_any = False

        for b in range(B):
            mask_positions = (x_t[b] == self.mask_index).nonzero(as_tuple=False)
            if mask_positions.numel() == 0:
                continue
            start = int(mask_positions.min().item())
            end = min(start + L_cont, L_total)
            cont_mask[b, start:end] = True
            slice_len = end - start
            if slice_len > 0:
                x_cont[b, :slice_len] = x_t[b, start:end]
                valid_any = True

        if not valid_any:
            return None, None

        self._cache(x_t, x_cont, cont_mask)
        return x_cont, cont_mask

    def align(self, x_t: torch.Tensor, x_0_hat: torch.Tensor) -> AlignmentResult:
        x_cont, cont_mask = self._extract_continuation(x_t)
        if x_cont is None or cont_mask is None:
            return AlignmentResult(x_t=x_t, x_0_hat=x_0_hat, cont_mask=None, should_apply=False)
        length = self.continuation_length if self.continuation_length is not None else cont_mask.size(1)
        p_cont = self._create_continuation_probs(x_0_hat, cont_mask, length)
        return AlignmentResult(x_t=x_cont, x_0_hat=p_cont, cont_mask=cont_mask, should_apply=True)


def build_alignment_strategy(name: Optional[str], mask_index: int, pad_index: Optional[int],
                             continuation_length: Optional[int], vocab_size: int) -> ContinuationAlignmentStrategy:
    if name is None or name.lower() in {"", "default"}:
        return DefaultAlignmentStrategy(continuation_length, mask_index, pad_index)
    if name.lower() == "left":
        return LeftMaskAlignmentStrategy(mask_index, pad_index, continuation_length, vocab_size)
    if name.lower() == "none":
        return NoAlignmentStrategy()
    raise ValueError(f"Unsupported alignment strategy: {name}")
