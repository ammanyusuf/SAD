import torch

from sampling.safe_hooks import build_dream_repellency_hook
from sampling.sample_text import SafetySettings
from third_party.DiffuGuard.utility.generate_function_dream import generate_dream_hidden


class _DummyDreamTokenizer:
    def __init__(self, vocab_size: int, mask_token_id: int) -> None:
        self.vocab = {
            "<|pad|>": 0,
            "<unk>": 1,
            "<|eos|>": 2,
            "<|bos|>": 3,
            "<|mask|>": mask_token_id,
            "Sure": 4,
            "Sorry": 5,
        }
        self.pad_token = "<|pad|>"
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2
        self.bos_token_id = 3
        self.mask_token = "<|mask|>"
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

    def __call__(self, text: str, add_special_tokens: bool = False):
        return type("TokOut", (), {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)})

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = [tok for tok in str(text).split() if tok]
        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def add_special_tokens(self, mapping):
        if "pad_token" in mapping:
            tok = mapping["pad_token"]
            if tok not in self.vocab:
                self.vocab[tok] = max(self.vocab.values()) + 1
            self.pad_token = tok
            self.pad_token_id = self.vocab[tok]


class _DummyDreamModel:
    def __init__(self, vocab_size: int, mask_token_id: int):
        self.config = type("Cfg", (), {"vocab_size": vocab_size})
        self.generation_config = type(
            "GenCfg",
            (),
            {
                "mask_token_id": mask_token_id,
                "pad_token_id": 0,
                "eos_token_id": 2,
                "bos_token_id": 3,
                "clone": lambda self_: self_,
            },
        )()
        self.device = torch.device("cpu")

    def diffusion_generate(
        self,
        inputs=None,
        input_ids=None,
        attention_mask=None,
        generation_config=None,
        max_new_tokens=None,
        steps=None,
        temperature=None,
        top_p=None,
        output_history=False,
        return_dict_in_generate=True,
        generation_logits_hook_func=None,
        generation_tokens_hook_func=None,
        generation_hidden_hook_func=None,
    ):
        x = input_ids if input_ids is not None else inputs
        if x is None:
            x = torch.empty((1, 0), dtype=torch.long)
        x = x.clone()
        max_new_tokens = int(max_new_tokens or 0)
        if max_new_tokens > 0:
            tail = torch.full(
                (x.shape[0], max_new_tokens),
                int(self.generation_config.mask_token_id),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, tail], dim=1)
        steps = int(steps or 1)

        for i in range(steps):
            logits = torch.full(
                (x.shape[0], x.shape[1], self.config.vocab_size),
                -10.0,
                dtype=torch.float32,
                device=x.device,
            )
            logits[..., 0] = 10.0 + float(i)
            if generation_logits_hook_func is not None:
                logits = generation_logits_hook_func(i, x, logits)
            mask_positions = (x == self.generation_config.mask_token_id)
            if mask_positions.any():
                first_mask = mask_positions.float().argmax(dim=1)
                for b in range(x.shape[0]):
                    pos = int(first_mask[b].item())
                    if mask_positions[b, pos]:
                        token = int(torch.argmax(logits[b, pos]).item())
                        x[b, pos] = token
            if generation_tokens_hook_func is not None:
                x = generation_tokens_hook_func(i, x, logits)
            if generation_hidden_hook_func is not None:
                h_last = torch.zeros((x.shape[0], x.shape[1], 4), dtype=torch.float32, device=x.device)
                generation_hidden_hook_func(i, x, h_last)

        if return_dict_in_generate:
            return type("Out", (), {"sequences": x})
        return x


def test_dream_hidden_matches_native_generate_no_safety():
    vocab_size = 8
    mask_id = 7
    model = _DummyDreamModel(vocab_size, mask_id)
    tokenizer = _DummyDreamTokenizer(vocab_size, mask_id)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    steps = 4
    gen_length = 4

    out_native = model.diffusion_generate(
        input_ids=prompt,
        attention_mask=attention_mask,
        max_new_tokens=gen_length,
        steps=steps,
        temperature=0.0,
        return_dict_in_generate=True,
    ).sequences

    out_hidden, _ = generate_dream_hidden(
        model=model,
        tokenizer=tokenizer,
        input_ids=prompt,
        attention_mask=attention_mask,
        gen_length=gen_length,
        steps=steps,
        block_length=gen_length,
        temperature=0.0,
        top_p=None,
        sp_threshold=1e9,
        refinement_steps=0,
        remask_ratio=0.0,
        correct_only_first_block=True,
        fill_all_masks=False,
        mask_id=mask_id,
        attack_method="none",
        initial_mask_in_prompt=(prompt == mask_id),
    )

    assert torch.equal(out_native, out_hidden)


def test_dream_repellency_zero_scale_preserves_outputs(tmp_path):
    vocab_size = 8
    mask_id = 7
    model = _DummyDreamModel(vocab_size, mask_id)
    tokenizer = _DummyDreamTokenizer(vocab_size, mask_id)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    steps = 4
    gen_length = 4

    base, _ = generate_dream_hidden(
        model=model,
        tokenizer=tokenizer,
        input_ids=prompt,
        attention_mask=attention_mask,
        gen_length=gen_length,
        steps=steps,
        block_length=gen_length,
        temperature=0.0,
        top_p=None,
        sp_threshold=1e9,
        refinement_steps=0,
        remask_ratio=0.0,
        correct_only_first_block=True,
        fill_all_masks=False,
        mask_id=mask_id,
        attack_method="none",
        initial_mask_in_prompt=(prompt == mask_id),
    )

    ref_data = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long)
    unsafe_path = tmp_path / "unsafe_refs.pt"
    torch.save(ref_data, unsafe_path)

    safety = SafetySettings(
        enabled=True,
        eta=0.0,
        scale=0.0,
        unsafe_artifacts=unsafe_path,
    )

    logits_hook = build_dream_repellency_hook(
        tokenizer,
        safety,
        device=torch.device("cpu"),
        mask_token_id=mask_id,
        attention_mask=attention_mask,
        prompt_width=prompt.shape[1],
        total_steps=steps,
        vocab_size=vocab_size,
    )

    with_repellency, _ = generate_dream_hidden(
        model=model,
        tokenizer=tokenizer,
        input_ids=prompt,
        attention_mask=attention_mask,
        gen_length=gen_length,
        steps=steps,
        block_length=gen_length,
        temperature=0.0,
        top_p=None,
        sp_threshold=1e9,
        refinement_steps=0,
        remask_ratio=0.0,
        correct_only_first_block=True,
        fill_all_masks=False,
        mask_id=mask_id,
        attack_method="none",
        initial_mask_in_prompt=(prompt == mask_id),
        logits_hook=logits_hook,
    )

    assert torch.equal(base, with_repellency)
