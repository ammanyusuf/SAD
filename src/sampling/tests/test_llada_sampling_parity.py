import torch

from sampling.llada_engine import llada_generate
from third_party.LLaDA.generate import generate as official_generate
from third_party.mdlm.repellency.safe_denoiser import MaskKernelRepellency
from unsafe_prep.tests.utils import build_test_unsafe_tensor


class _DummyModel:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def __call__(self, x, attention_mask=None):
        batch, length = x.shape
        logits = torch.full((batch, length, self.vocab_size), -10.0, dtype=torch.float32, device=x.device)
        logits[..., 0] = 10.0
        return type("Output", (), {"logits": logits})


def _build_repellency(tmp_path, vocab_size: int, mask_id: int):
    texts = [
        "alpha beta",
        "beta gamma",
        "gamma alpha",
        "alpha gamma",
    ]
    ref_data, _, _, _, _ = build_test_unsafe_tensor(texts)
    proj_path = tmp_path / "proj_refs.pt"
    torch.save(ref_data, proj_path)
    return MaskKernelRepellency(
        ref_data=ref_data,
        embed_fn=lambda x: x,
        forward_fn=None,
        num_timesteps=4,
        max_idx=4,
        beta_min=0.0,
        beta_max=0.0,
        vocab_size=vocab_size,
        mask_index=mask_id,
        pad_index=None,
        cache_proj_ref=True,
        proj_ref_path=str(proj_path),
        alignment_strategy="left",
        scale=0.0,
        eta=0.0,
    )


def test_llada_matches_official_generate_no_safety():
    vocab_size = 8
    mask_id = 7
    model = _DummyModel(vocab_size)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    steps = 4
    gen_length = 4
    block_length = 4

    out_official = official_generate(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
    )

    out_local, _ = llada_generate(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        effective_vocab=vocab_size,
        repellency=None,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        sampling_mode="pure_diffusion",
        transfer_schedule="uniform",
    )

    assert torch.equal(out_official, out_local)


def test_llada_repellency_zero_scale_preserves_outputs(tmp_path):
    vocab_size = 8
    mask_id = 7
    model = _DummyModel(vocab_size)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    steps = 4
    gen_length = 4
    block_length = 4

    base, _ = llada_generate(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        effective_vocab=vocab_size,
        repellency=None,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        sampling_mode="pure_diffusion",
        transfer_schedule="uniform",
    )

    repellor = _build_repellency(tmp_path, vocab_size=vocab_size, mask_id=mask_id)
    with_repellency, _ = llada_generate(
        model=model,
        prompt=prompt,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        effective_vocab=vocab_size,
        repellency=repellor,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
        sampling_mode="pure_diffusion",
        transfer_schedule="uniform",
    )

    assert torch.equal(base, with_repellency)
