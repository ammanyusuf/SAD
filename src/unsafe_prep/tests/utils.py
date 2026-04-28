from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

from unsafe_prep.schemas import RawUnsafeRecord, TokenizedUnsafeRecord
from unsafe_prep.utils import tokenize_record


@dataclass
class ToyTokenizer:
    """Minimal tokenizer for unit tests without HF downloads."""

    vocab: dict
    pad_token_id: int = 0
    mask_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    unk_token_id: int = 4

    pad_token: str = "[PAD]"
    mask_token: str = "[MASK]"
    sep_token: str = "[SEP]"
    eos_token: str = "[SEP]"

    def __init__(self) -> None:
        base_vocab = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "<unk>": 4,
        }
        base_vocab.update({
            "alpha": 5,
            "beta": 6,
            "gamma": 7,
            "delta": 8,
            "prompt": 9,
            "prefix": 10,
            "unsafe": 11,
            "continuation": 12,
            "mask": 13,
        })
        self.vocab = base_vocab

    def encode(self, text: str, add_special_tokens: bool = True,
               max_length: int | None = None, truncation: bool = False) -> List[int]:
        tokens = [tok.strip().lower() for tok in text.split() if tok.strip()]
        ids = [self.vocab.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)


def build_test_unsafe_tensor(
    texts: Sequence[str],
    max_length: int = 6,
    append_prompt: bool = False,
) -> Tuple[torch.LongTensor, int, int, ToyTokenizer, List[TokenizedUnsafeRecord]]:
    """Tokenize sample texts into the format used by unsafe artifacts."""
    tokenizer = ToyTokenizer()
    pad_id = tokenizer.pad_token_id
    mask_index = tokenizer.mask_token_id
    records = []
    for idx, text in enumerate(texts):
        meta = {}
        if append_prompt:
            meta = {"prompt_text": f"prompt {idx}"}
        records.append(
            RawUnsafeRecord(
                source="unit-test",
                category="demo",
                answer_text=text,
                toxicity_score=None,
                meta=meta,
            )
        )
    tokenized: List[TokenizedUnsafeRecord] = []
    for record in records:
        tokenized.append(
            tokenize_record(
                tokenizer=tokenizer,
                pad_id=pad_id,
                mask_index=mask_index,
                max_length=max_length,
                record=record,
                append_prompt=append_prompt,
            )
        )
    tensor = torch.tensor([rec.input_ids for rec in tokenized], dtype=torch.long)
    return tensor, mask_index, pad_id, tokenizer, tokenized
