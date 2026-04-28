import os
import types

import pytest
import torch
import torch.nn.functional as F

from third_party.mdlm.repellency.safe_denoiser import MaskKernelRepellency
from unsafe_prep.tests.utils import build_test_unsafe_tensor


def _build_repellency(tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="left"):
    proj_path = tmp_path / "proj_refs.pt"
    torch.save(ref_data, proj_path)
    return MaskKernelRepellency(
        ref_data=ref_data,
        embed_fn=lambda x: x,
        forward_fn=None,
        num_timesteps=1,
        max_idx=1,
        beta_min=0.0,
        beta_max=0.0,
        vocab_size=vocab_size,
        mask_index=mask_index,
        pad_index=pad_index,
        cache_proj_ref=True,
        proj_ref_path=str(proj_path),
        alignment_strategy=alignment_strategy,
        scale=1.0,
    )


class CountingEmbed:
    def __init__(self, return_3d: bool = True):
        self.calls = 0
        self.return_3d = return_3d
        self.last_input = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_input = x.clone()
        B, L = x.shape
        if self.return_3d:
            emb = torch.zeros(B, L, 2, device=x.device, dtype=torch.float32)
            emb[..., 0] = x.float()
            emb[..., 1] = 1.0
            return emb
        emb = torch.zeros(B, 2, device=x.device, dtype=torch.float32)
        emb[..., 0] = x.float().mean(dim=-1)
        emb[..., 1] = 1.0
        return emb


def _build_repellency_with_embed(
    tmp_path,
    ref_data,
    vocab_size,
    mask_index,
    pad_index=None,
    alignment_strategy="left",
    use_semantic_gating=True,
    semantic_weight=1.0,
    cache_semantic_ref=False,
    semantic_ref_path=None,
    embed_return_3d=True,
):
    embed = CountingEmbed(return_3d=embed_return_3d)
    proj_path = tmp_path / "proj_embed_refs.pt"
    torch.save(ref_data, proj_path)
    repellor = MaskKernelRepellency(
        ref_data=ref_data,
        embed_fn=embed,
        forward_fn=None,
        num_timesteps=1,
        max_idx=1,
        beta_min=0.0,
        beta_max=0.0,
        vocab_size=vocab_size,
        mask_index=mask_index,
        pad_index=pad_index,
        cache_proj_ref=True,
        proj_ref_path=str(proj_path),
        alignment_strategy=alignment_strategy,
        scale=1.0,
        use_semantic_gating=use_semantic_gating,
        semantic_weight=semantic_weight,
        cache_semantic_ref=cache_semantic_ref,
        semantic_ref_path=str(semantic_ref_path) if semantic_ref_path is not None else None,
    )
    return repellor, embed


def test_default_alignment_anchors_first_mask_when_present(tmp_path):
    vocab_size = 8
    mask_index = 5
    pad_index = 6
    ref_data = torch.tensor([[3, 3, 3, 3, 3]], dtype=torch.long)
    repellor = _build_repellency(tmp_path, ref_data, vocab_size, mask_index, pad_index, alignment_strategy="left")

    prompt = [1, 1, 1, 1, 1, 2]
    continuation = [mask_index] * 4
    x_t = torch.tensor([prompt + continuation], dtype=torch.long)

    logits = torch.zeros(1, len(prompt) + len(continuation), vocab_size)
    logits[:, :, 2] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        flipped = torch.flip(x_0_hat, dims=[-1])
        return flipped, {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.5]))["x_0_hat"]

    first_mask_idx = int((x_t[0] == mask_index).nonzero(as_tuple=False).min().item())
    L_ref = ref_data.size(1)
    aligned_end = min(first_mask_idx + L_ref, x_t.size(1))
    assert torch.allclose(result[:, :first_mask_idx], x_0_hat[:, :first_mask_idx])
    assert not torch.allclose(result[:, first_mask_idx:aligned_end], x_0_hat[:, first_mask_idx:aligned_end])
    assert torch.allclose(result[:, aligned_end:], x_0_hat[:, aligned_end:])


def test_prompt_mask_alignment_path(tmp_path):
    vocab_size = 8
    mask_index = 5
    ref_data = torch.tensor([[2, 2, 2, 2]], dtype=torch.long)
    repellor = _build_repellency(tmp_path, ref_data, vocab_size, mask_index, alignment_strategy="left")

    prompt = [1, 1, 1]
    continuation = [mask_index, mask_index]
    x_t = torch.tensor([prompt + continuation], dtype=torch.long)
    prompt_mask = torch.tensor([[True, True, True, False, False]])

    logits = torch.zeros(1, len(prompt) + len(continuation), vocab_size)
    logits[:, :, 3] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    result = repellor.conditioning_1(
        x_0_hat,
        x_t=x_t,
        move=torch.tensor([0.5]),
        prompt_mask=prompt_mask,
        prompt_width=len(prompt),
    )["x_0_hat"]

    assert result.shape == x_0_hat.shape


def test_no_mask_bypasses_alignment(tmp_path):
    vocab_size = 6
    mask_index = 4
    ref_data = torch.tensor([[2, 2, 2]], dtype=torch.long)
    repellor = _build_repellency(tmp_path, ref_data, vocab_size, mask_index)

    x_t = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    logits = torch.zeros(1, x_t.size(1), vocab_size)
    logits[..., 1] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.2]))["x_0_hat"]

    assert torch.allclose(result, x_0_hat)


def test_first_mask_anchor_with_multiple_segments(tmp_path):
    vocab_size = 7
    mask_index = 5
    ref_data = torch.tensor([[3, 3, 3]], dtype=torch.long)
    repellor = _build_repellency(tmp_path, ref_data, vocab_size, mask_index)

    tokens = [0, mask_index, 1, 1, mask_index, mask_index]
    x_t = torch.tensor([tokens], dtype=torch.long)

    logits = torch.zeros(1, len(tokens), vocab_size)
    logits[:, :, 2] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.3]))["x_0_hat"]

    unchanged_prefix = result[:, :1]
    assert torch.allclose(unchanged_prefix, x_0_hat[:, :1])
    assert not torch.allclose(result[:, 1:4], x_0_hat[:, 1:4])


def test_empirical_denoiser_sees_only_continuation(tmp_path):
    vocab_size = 10
    mask_index = 9
    pad_index = 8
    ref_data = torch.tensor([[3, 3, 3, 3]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index, alignment_strategy="left"
    )

    prompt = [1, 1, 1]
    continuation = [mask_index] * 5
    x_t = torch.tensor([prompt + continuation], dtype=torch.long)

    logits = torch.zeros(1, len(prompt) + len(continuation), vocab_size)
    logits[..., 2] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    seen = {}

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        seen["x_t"] = x_t.clone()
        seen["x_0_hat"] = x_0_hat.clone()
        return x_0_hat, {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.5]))

    assert "x_t" in seen and "x_0_hat" in seen
    first_mask_idx = int((x_t[0] == mask_index).nonzero(as_tuple=False).min().item())
    assert first_mask_idx >= len(prompt)
    expected_len = ref_data.size(1)
    expected_slice = x_t[:, first_mask_idx:first_mask_idx + expected_len]
    assert seen["x_t"].shape[1] == expected_len
    assert torch.equal(seen["x_t"], expected_slice)
    assert seen["x_0_hat"].shape[:2] == seen["x_t"].shape


def test_truncates_to_ref_length(tmp_path):
    vocab_size = 11
    mask_index = 9
    pad_index = 10
    ref_data = torch.tensor([[4, 4]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index, alignment_strategy="left"
    )

    prompt = [0, 0]
    continuation = [mask_index, mask_index, mask_index, mask_index]
    x_t = torch.tensor([prompt + continuation], dtype=torch.long)
    L_prompt = len(prompt)
    L_total = x_t.size(1)

    logits = torch.zeros(1, L_total, vocab_size)
    logits[:, :L_prompt, 1] = 3.0
    logits[:, L_prompt:, 2] = 4.0
    x_0_hat = torch.softmax(logits, dim=-1)

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        flipped = torch.flip(x_0_hat, dims=[-1])
        return flipped, {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.5]))["x_0_hat"]

    assert torch.allclose(result[:, :L_prompt], x_0_hat[:, :L_prompt])
    L_ref = ref_data.size(1)
    cont_start = L_prompt
    cont_mid = cont_start + L_ref
    assert not torch.allclose(result[:, cont_start:cont_mid], x_0_hat[:, cont_start:cont_mid])
    assert torch.allclose(result[:, cont_mid:], x_0_hat[:, cont_mid:])


def test_mixed_batch_only_rows_with_mask_change(tmp_path):
    vocab_size = 8
    mask_index = 5
    pad_index = 6
    ref_data = torch.tensor([[3, 3, 3]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index, alignment_strategy="left"
    )

    x_t = torch.tensor([
        [1, 1, mask_index, mask_index],
        [2, 2, 2, 2],
    ], dtype=torch.long)
    logits = torch.zeros(2, x_t.size(1), vocab_size)
    logits[..., 4] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        return torch.flip(x_0_hat, dims=[-1]), {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    result = repellor.conditioning_1(
        x_0_hat, x_t=x_t, move=torch.tensor([0.3, 0.3])
    )["x_0_hat"]

    assert torch.allclose(result[1], x_0_hat[1])
    first_mask_idx = int((x_t[0] == mask_index).nonzero(as_tuple=False).min().item())
    assert torch.allclose(result[0, :first_mask_idx], x_0_hat[0, :first_mask_idx])
    assert not torch.allclose(result[0, first_mask_idx:], x_0_hat[0, first_mask_idx:])


def test_default_alignment_passes_leading_window(tmp_path):
    vocab_size = 9
    mask_index = 6
    pad_index = 7
    ref_data = torch.tensor([[4, 4, 4]], dtype=torch.long)
    repellor = _build_repellency(tmp_path, ref_data, vocab_size, mask_index, pad_index, alignment_strategy="left")

    x_t = torch.tensor([[9, 8, mask_index, mask_index, mask_index]], dtype=torch.long)
    logits = torch.zeros(1, x_t.size(1), vocab_size)
    logits[..., 1] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    seen = {}

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        seen["x_t"] = x_t.clone()
        seen["x_0_hat"] = x_0_hat.clone()
        return x_0_hat, {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.4]))

    first_mask_idx = int((x_t[0] == mask_index).nonzero(as_tuple=False).min().item())
    L_ref = ref_data.size(1)
    aligned_end = min(first_mask_idx + L_ref, x_t.size(1))
    expected_slice = x_t[:, first_mask_idx:aligned_end]
    assert seen["x_t"].shape[1] == expected_slice.shape[1]
    assert torch.equal(seen["x_t"], expected_slice)
    assert seen["x_0_hat"].shape[:2] == seen["x_t"].shape


def test_alignment_strategy_none_uses_full_sequence(tmp_path):
    vocab_size = 9
    mask_index = 7
    pad_index = 8
    ref_data = torch.tensor([[3, 3, 3, 3]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index, alignment_strategy="none"
    )

    x_t = torch.tensor([[0, 0, mask_index, mask_index, 1]], dtype=torch.long)
    logits = torch.zeros(1, x_t.size(1), vocab_size)
    logits[..., 2] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    seen = {}

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        seen["x_t"] = x_t.clone()
        seen["x_0_hat"] = x_0_hat.clone()
        return x_0_hat, {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.4]))["x_0_hat"]

    assert result.shape == x_0_hat.shape
    assert torch.allclose(result, x_0_hat)
    assert "x_t" not in seen
    assert "x_0_hat" not in seen


def test_scale_zero_disables_conditioning(tmp_path):
    vocab_size = 6
    mask_index = 4
    ref_data = torch.tensor([[1, 1, 1]], dtype=torch.long)
    repellor = _build_repellency(tmp_path, ref_data, vocab_size, mask_index, alignment_strategy="left")
    repellor.scale = 0.0
    x_t = torch.tensor([[mask_index, mask_index, mask_index]], dtype=torch.long)
    logits = torch.zeros(1, x_t.size(1), vocab_size)
    logits[..., 2] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        return torch.flip(x_0_hat, dims=[-1]), {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)
    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.4]))["x_0_hat"]
    assert torch.allclose(result, x_0_hat)


def test_conditioning_preserves_shape_and_dtype(tmp_path):
    vocab_size = 7
    mask_index = 5
    pad_index = 6
    ref_data = torch.tensor([[3, 3, 3]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index, alignment_strategy="left"
    )

    x_t = torch.tensor([[1, mask_index, mask_index]], dtype=torch.long)
    logits = torch.zeros(1, x_t.size(1), vocab_size, dtype=torch.float32)
    logits[..., 1] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.3]))["x_0_hat"]

    assert result.shape == x_0_hat.shape
    assert result.dtype == x_0_hat.dtype


def test_histogram_matches_uniform_reference_counts(tmp_path):
    vocab_size = 6
    mask_index = 4
    ref_data = torch.tensor([[1], [1], [2], [2], [2]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="none"
    )

    xt = torch.tensor([[mask_index]], dtype=torch.long)

    def fake_logqt(self, xt_arg, move_arg):
        B = xt_arg.size(0)
        N = self.proj_refs.size(0)
        zeros = torch.zeros(B, N, device=xt_arg.device)
        return zeros, torch.ones_like(zeros)

    repellor._log_qt_mask_kernel = types.MethodType(fake_logqt, repellor)

    p_unsafe, _ = repellor._unsafe_posterior(xt, move=torch.tensor([0.5]))
    expected = torch.zeros(vocab_size, dtype=p_unsafe.dtype)
    expected[1] = 2.0 / 5.0
    expected[2] = 3.0 / 5.0
    assert torch.allclose(p_unsafe[0, 0], expected, atol=1e-6)


def test_pad_tokens_are_ignored_in_histograms(tmp_path):
    vocab_size = 6
    mask_index = 5
    pad_index = 0
    ref_data = torch.tensor(
        [
            [pad_index, 1],
            [2, 1],
            [3, 2],
        ],
        dtype=torch.long,
    )
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=pad_index, alignment_strategy="none"
    )
    xt = torch.tensor([[mask_index, mask_index]], dtype=torch.long)
    move = torch.tensor([0.3])
    p_unsafe, _ = repellor._unsafe_posterior(xt, move, x_0_hat=None)
    expected = torch.zeros(vocab_size)
    expected[2] = 1.0
    expected[3] = 1.0
    expected = expected / expected.sum()
    assert torch.allclose(p_unsafe[0, 0], expected.to(p_unsafe.dtype), atol=1e-6)


def test_unsafe_posterior_normalized(tmp_path):
    vocab_size = 8
    mask_index = 5
    ref_data = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    repellor = _build_repellency(tmp_path, ref_data, vocab_size, mask_index, alignment_strategy="none")
    x_t = torch.tensor([[1, mask_index, mask_index, 4]], dtype=torch.long)
    move = torch.tensor([0.4])
    p_unsafe, _ = repellor._unsafe_posterior(x_t, move, x_0_hat=None)
    assert torch.allclose(p_unsafe.sum(dim=-1), torch.ones_like(p_unsafe.sum(dim=-1)))
    assert torch.isfinite(p_unsafe).all()


def test_unmasked_positions_are_delta_copies(tmp_path):
    vocab_size = 7
    mask_index = 5
    ref_data = torch.tensor([[1, 2, 3]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="none"
    )

    xt = torch.tensor([[0, mask_index, 2]], dtype=torch.long)

    def fake_logqt(self, xt_arg, move_arg):
        B = xt_arg.size(0)
        N = self.proj_refs.size(0)
        zeros = torch.zeros(B, N, device=xt_arg.device)
        return zeros, torch.ones_like(zeros)

    repellor._log_qt_mask_kernel = types.MethodType(fake_logqt, repellor)

    p_unsafe, _ = repellor._unsafe_posterior(xt, move=torch.tensor([0.4]))
    for pos in [0, 2]:
        token = xt[0, pos].item()
        expected = F.one_hot(
            torch.tensor(token), num_classes=vocab_size
        ).to(p_unsafe.dtype)
        assert torch.allclose(p_unsafe[0, pos], expected, atol=1e-6)


def test_mask_kernel_prefers_closer_reference(tmp_path):
    vocab_size = 6
    mask_index = 4
    ref_data = torch.tensor([[1, 2, 3], [1, 9, 3]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="none"
    )

    xt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logqt, _ = repellor._log_qt_mask_kernel(xt, move=torch.tensor([0.3]))
    assert logqt[0, 0] > logqt[0, 1]
    logqt_norm = (logqt - logqt.mean(dim=-1, keepdim=True)) / logqt.std(dim=-1, keepdim=True).clamp_min(1e-4)
    w = torch.softmax(logqt_norm, dim=-1)
    assert w[0, 0] > w[0, 1]



def test_safe_denoiser_fixed_point_when_p_equals_p_unsafe(tmp_path):
    vocab_size = 5
    mask_index = 3
    ref_data = torch.tensor([[1, 1]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="none"
    )

    x_t = torch.tensor([[mask_index, mask_index]], dtype=torch.long)
    logits = torch.tensor([[[0.0, 1.0, 2.0, -1.0, 0.5], [0.3, -0.2, 1.5, 0.0, 0.0]]], dtype=torch.float32)
    x_0_hat = torch.softmax(logits, dim=-1)

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        return x_0_hat.clone(), {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.5]))["x_0_hat"]
    assert torch.allclose(result, x_0_hat, atol=1e-6)


def test_repellency_decreases_unsafe_token_probability(tmp_path):
    vocab_size = 3
    mask_index = 2
    ref_data = torch.tensor([[0, 0]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="left"
    )
    repellor.eta = 2.0

    x_t = torch.tensor([[mask_index]], dtype=torch.long)
    x_0_hat = torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float32)
    p_unsafe = torch.tensor([[[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]], dtype=torch.float32)

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        return p_unsafe.clone(), {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.4]))["x_0_hat"]
    before = x_0_hat[0, 0]
    after = result[0, 0]
    assert after[2] < before[2]
    assert (after[0] > before[0]) or (after[1] > before[1])


def test_conditioning_output_normalized(tmp_path):
    vocab_size = 6
    mask_index = 4
    ref_data = torch.tensor([[1, 2, 3]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="left"
    )
    x_t = torch.tensor([[mask_index, mask_index, mask_index]], dtype=torch.long)
    logits = torch.randn(1, x_t.size(1), vocab_size)
    x_0_hat = torch.softmax(logits, dim=-1)
    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.4]))["x_0_hat"]
    assert torch.allclose(result.sum(dim=-1), torch.ones_like(result.sum(dim=-1)))
    assert torch.isfinite(result).all()


def test_alignment_and_repellency_only_on_continuation(tmp_path):
    vocab_size = 4
    mask_index = 3
    ref_data = torch.tensor([[1, 1]], dtype=torch.long)
    repellor = _build_repellency(
        tmp_path, ref_data, vocab_size, mask_index, pad_index=None, alignment_strategy="left"
    )
    repellor.scale = 1.5

    prompt = [0, 0]
    continuation = [mask_index, mask_index]
    x_t = torch.tensor([prompt + continuation], dtype=torch.long)
    L_prompt = len(prompt)
    L_total = len(prompt) + len(continuation)

    logits = torch.zeros(1, L_total, vocab_size, dtype=torch.float32)
    logits[:, :L_prompt, 0] = 3.0
    logits[:, L_prompt:, 1] = 3.0
    x_0_hat = torch.softmax(logits, dim=-1)

    def fake_empirical(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        B, L_cont, V = x_0_hat.shape
        unsafe = torch.zeros(B, L_cont, V, dtype=x_0_hat.dtype, device=x_0_hat.device)
        unsafe[..., 1] = 1.0
        return unsafe, {}

    repellor.empirical_denoiser = types.MethodType(fake_empirical, repellor)

    result = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.5]))["x_0_hat"]

    assert torch.allclose(result[:, :L_prompt], x_0_hat[:, :L_prompt], atol=1e-6)
    before_cont = x_0_hat[:, L_prompt:, :]
    after_cont = result[:, L_prompt:, :]
    assert torch.all(after_cont[..., 1] < before_cont[..., 1])


def test_alignment_semantic_scores_use_continuation(tmp_path):
    vocab_size = 6
    mask_index = 4
    ref_data = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.long)
    recording_embed = CountingEmbed(return_3d=True)
    proj_path = tmp_path / "proj_record.pt"
    torch.save(ref_data, proj_path)
    repellor = MaskKernelRepellency(
        ref_data=ref_data,
        embed_fn=recording_embed,
        forward_fn=None,
        num_timesteps=1,
        max_idx=1,
        beta_min=0.0,
        beta_max=0.0,
        vocab_size=vocab_size,
        mask_index=mask_index,
        pad_index=None,
        cache_proj_ref=True,
        proj_ref_path=str(proj_path),
        alignment_strategy="left",
        scale=1.0,
        use_semantic_gating=True,
        semantic_weight=1.0,
    )
    x_t = torch.tensor([[0, 0, mask_index, mask_index, mask_index]], dtype=torch.long)
    move = torch.tensor([0.5])
    logits = torch.zeros(1, x_t.size(1), vocab_size)
    logits[..., 2] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)
    repellor.conditioning_1(x_0_hat, x_t=x_t, move=move)
    assert recording_embed.last_input is not None
    observed = recording_embed.last_input
    arg_tokens = x_0_hat.argmax(dim=-1)
    first_mask_idx = int((x_t[0] == mask_index).nonzero(as_tuple=False).min().item())
    L_ref = ref_data.size(1)
    aligned_end = min(first_mask_idx + L_ref, x_t.size(1))
    expected = arg_tokens[:, first_mask_idx:aligned_end]
    assert torch.equal(observed, expected)


def test_repellency_handles_refs_from_unsafe_pipeline(tmp_path):
    ref_data, mask_index, pad_index, _, _ = build_test_unsafe_tensor(
        ["unsafe continuation", "continuation unsafe"],
        max_length=5,
    )
    vocab_size = 16
    repellor = _build_repellency(
        tmp_path,
        ref_data=ref_data,
        vocab_size=vocab_size,
        mask_index=mask_index,
        pad_index=pad_index,
        alignment_strategy="left",
    )
    x_t = torch.full((1, ref_data.size(1)), mask_index, dtype=torch.long)
    logits = torch.zeros(1, ref_data.size(1), vocab_size)
    logits[..., 3] = 1.0
    x_0_hat = torch.softmax(logits, dim=-1)
    output = repellor.conditioning_1(x_0_hat, x_t=x_t, move=torch.tensor([0.5]))["x_0_hat"]
    assert output.shape == x_0_hat.shape


def test_encode_semantic_state_ignores_mask_and_pad(tmp_path):
    vocab_size = 10
    mask_index = 7
    pad_index = 8
    ref_data = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    repellor, _ = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        pad_index=pad_index,
        alignment_strategy="none",
    )
    x_t = torch.tensor([[1, mask_index, pad_index, 5]], dtype=torch.long)
    z = repellor._encode_semantic_state(x_t, x_0_hat=None)
    expected = torch.tensor([[(1.0 + 5.0) / 2.0, 1.0]], dtype=torch.float32)
    assert z.shape == (1, 2)
    assert torch.allclose(z, expected, atol=1e-6)


def test_encode_semantic_state_accepts_2d_embeddings(tmp_path):
    vocab_size = 10
    mask_index = 7
    ref_data = torch.tensor([[1, 2, 3]], dtype=torch.long)
    repellor, _ = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        embed_return_3d=False,
    )
    x_t = torch.tensor([[2, 4, 6]], dtype=torch.long)
    z = repellor._encode_semantic_state(x_t, x_0_hat=None)
    assert z.shape == (1, 2)
    expected_mean = x_t.float().mean(dim=-1, keepdim=True)
    expected = torch.cat([expected_mean, torch.ones_like(expected_mean)], dim=-1)
    assert torch.allclose(z, expected, atol=1e-6)


def test_semantic_scores_rank_closer_reference_higher(tmp_path):
    vocab_size = 20
    mask_index = 7
    ref_data = torch.tensor([
        [1, 1, 2, 1],
        [9, 10, 11, 10],
    ], dtype=torch.long)
    repellor, _ = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=True,
        semantic_weight=1.0,
    )
    x_t = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    repellor._compute_semantic_ref_embeddings()
    z_t, ref_embs = repellor._semantic_scores_for_refs(x_t, x_0_hat=None)
    rbf_logits = repellor._semantic_rbf_logits(z_t, ref_embs)
    assert rbf_logits.shape == (1, 2)
    assert rbf_logits[0, 0] > rbf_logits[0, 1]


def test_semantic_weight_zero_matches_disabled(tmp_path):
    vocab_size = 12
    mask_index = 4
    ref_data = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    repellor_zero, _ = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=True,
        semantic_weight=0.0,
    )
    repellor_off, _ = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=False,
    )
    x_t = torch.tensor([[mask_index, mask_index, mask_index]], dtype=torch.long)
    move = torch.tensor([0.2])
    assert repellor_zero._semantic_scores_for_refs(x_t, x_0_hat=None) is not None
    p_zero, _ = repellor_zero._unsafe_posterior(x_t, move, x_0_hat=None)
    p_off, _ = repellor_off._unsafe_posterior(x_t, move, x_0_hat=None)
    assert torch.allclose(p_zero, p_off)


def test_semantic_gating_changes_reference_weights(tmp_path, monkeypatch):
    vocab_size = 15
    mask_index = 9
    ref_data = torch.tensor([
        [1, 2, 3],
        [8, 9, 10],
        [1, 2, 4],
    ], dtype=torch.long)
    repellor, _ = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=True,
        semantic_weight=2.0,
    )
    x_t = torch.tensor([[1, 2, 3]], dtype=torch.long)
    move = torch.tensor([0.5])

    def fake_logqt(self, xt_arg, move_arg):
        B = xt_arg.size(0)
        N = self.proj_refs.size(0)
        zeros = torch.zeros(B, N, device=xt_arg.device)
        return zeros, torch.ones_like(zeros)

    monkeypatch.setattr(
        repellor,
        "_log_qt_mask_kernel",
        types.MethodType(fake_logqt, repellor),
    )

    repellor._unsafe_posterior(x_t, move, x_0_hat=None)
    logqt, _ = repellor._log_qt_mask_kernel(x_t, move)
    z_t, ref_embs = repellor._semantic_scores_for_refs(x_t, x_0_hat=None)
    semantic_logits = repellor._semantic_rbf_logits(z_t, ref_embs)
    logqt_norm = (logqt - logqt.mean(dim=-1, keepdim=True)) / logqt.std(dim=-1, keepdim=True).clamp_min(1e-4)
    log_weights = logqt_norm + repellor.semantic_weight * semantic_logits / repellor.semantic_temp
    w = torch.softmax(log_weights, dim=-1)
    assert w[0, 0] > w[0, 1]
    assert w[0, 2] > w[0, 1]


def test_semantic_ref_embeddings_cached_in_memory(tmp_path):
    vocab_size = 10
    mask_index = 7
    ref_data = torch.tensor([[1, 2, 3]], dtype=torch.long)
    repellor, embed = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=True,
        semantic_weight=1.0,
        cache_semantic_ref=False,
    )
    x_t = torch.tensor([[1, 2, 3]], dtype=torch.long)
    _ = repellor._semantic_scores_for_refs(x_t, x_0_hat=None)
    calls_after_first = embed.calls
    assert calls_after_first >= 2
    _ = repellor._semantic_scores_for_refs(x_t, x_0_hat=None)
    calls_after_second = embed.calls
    assert calls_after_second == calls_after_first + 1
    assert repellor.semantic_ref_embeddings is not None


def test_semantic_ref_disk_caching(tmp_path):
    vocab_size = 12
    mask_index = 7
    ref_data = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    semantic_ref_path = tmp_path / "semantic_refs.pt"
    repellor1, embed1 = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=True,
        semantic_weight=1.0,
        cache_semantic_ref=True,
        semantic_ref_path=semantic_ref_path,
    )
    x_t = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    _ = repellor1._semantic_scores_for_refs(x_t, x_0_hat=None)
    assert os.path.exists(semantic_ref_path)

    repellor2, embed2 = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=True,
        semantic_weight=1.0,
        cache_semantic_ref=True,
        semantic_ref_path=semantic_ref_path,
    )
    embed2.calls = 0
    _ = repellor2._semantic_scores_for_refs(x_t, x_0_hat=None)
    assert embed2.calls <= 2


def test_semantic_gating_changes_unsafe_posterior(tmp_path, monkeypatch):
    vocab_size = 6
    mask_index = 4
    ref_data = torch.tensor([[1, 1, 1], [3, 3, 3]], dtype=torch.long)
    repellor, _ = _build_repellency_with_embed(
        tmp_path,
        ref_data,
        vocab_size,
        mask_index,
        alignment_strategy="none",
        use_semantic_gating=True,
        semantic_weight=2.0,
    )
    repellor.scale = 1.0
    x_t = torch.tensor([[mask_index, mask_index, mask_index]], dtype=torch.long)
    move = torch.tensor([0.3])

    def fake_logqt(self, xt_arg, move_arg):
        B = xt_arg.size(0)
        N = self.proj_refs.size(0)
        zeros = torch.zeros(B, N, device=xt_arg.device)
        return zeros, torch.ones_like(zeros)

    monkeypatch.setattr(
        repellor,
        "_log_qt_mask_kernel",
        types.MethodType(fake_logqt, repellor),
    )

    def fake_semantic(self, x_t, x_0_hat=None, **kwargs):
        z_t = torch.zeros((x_t.size(0), 2), device=x_t.device, dtype=torch.float32)
        ref_embs = torch.zeros((2, 2), device=x_t.device, dtype=torch.float32)
        return z_t, ref_embs

    def fake_rbf(self, z_t, ref_embs):
        return torch.tensor([[2.0, -2.0]], device=z_t.device, dtype=torch.float32)

    monkeypatch.setattr(
        repellor,
        "_semantic_scores_for_refs",
        types.MethodType(fake_semantic, repellor),
    )
    monkeypatch.setattr(
        repellor,
        "_semantic_rbf_logits",
        types.MethodType(fake_rbf, repellor),
    )

    p_sem, _ = repellor._unsafe_posterior(x_t, move, x_0_hat=None)
    repellor.use_semantic_gating = False
    p_no_sem, _ = repellor._unsafe_posterior(x_t, move, x_0_hat=None)
    assert not torch.allclose(p_sem, p_no_sem)
