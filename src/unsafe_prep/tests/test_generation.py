import json
import torch

from unsafe_prep.pipeline import ShardWriter
from unsafe_prep.schemas import RawUnsafeRecord, TokenizedUnsafeRecord
from unsafe_prep.tests.utils import ToyTokenizer, build_test_unsafe_tensor
from unsafe_prep.utils import tokenize_record


def test_build_test_unsafe_tensor_matches_expected():
    tensor, mask_index, pad_index, tokenizer, records = build_test_unsafe_tensor(
        ["alpha beta", "gamma"],
        max_length=5,
    )
    assert mask_index == tokenizer.mask_token_id
    assert pad_index == tokenizer.pad_token_id
    expected = torch.tensor(
        [
            [tokenizer.cls_token_id, tokenizer.vocab["alpha"], tokenizer.vocab["beta"], tokenizer.sep_token_id, pad_index],
            [tokenizer.cls_token_id, tokenizer.vocab["gamma"], tokenizer.sep_token_id, pad_index, pad_index],
        ],
        dtype=torch.long,
    )
    assert torch.equal(tensor, expected)
    assert all(isinstance(rec, TokenizedUnsafeRecord) for rec in records)


def test_tokenize_record_appends_prompt_text():
    tokenizer = ToyTokenizer()
    record = RawUnsafeRecord(
        source="unit",
        category="demo",
        answer_text="alpha",
        meta={"prompt_text": "prompt prefix"},
    )
    tokenized = tokenize_record(
        tokenizer=tokenizer,
        pad_id=tokenizer.pad_token_id,
        mask_index=tokenizer.mask_token_id,
        max_length=6,
        record=record,
        append_prompt=True,
    )
    # Expect CLS prompt alpha SEP PAD PAD
    assert tokenized.length == 5
    expected_ids = torch.tensor(
        [
            tokenizer.cls_token_id,
            tokenizer.vocab["prompt"],
            tokenizer.vocab["prefix"],
            tokenizer.vocab["alpha"],
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ],
        dtype=torch.long,
    )
    assert torch.equal(torch.tensor(tokenized.input_ids), expected_ids)


def test_shard_writer_emits_tensor(tmp_path):
    tokenizer = ToyTokenizer()
    records = [
        tokenize_record(tokenizer, tokenizer.pad_token_id, tokenizer.mask_token_id, 5, RawUnsafeRecord("src", "cat", "alpha")),
        tokenize_record(tokenizer, tokenizer.pad_token_id, tokenizer.mask_token_id, 5, RawUnsafeRecord("src", "cat", "beta")),
    ]
    writer = ShardWriter(tmp_path, shard_size=2, dry_run=False, overwrite=True)
    for rec in records:
        writer.add(rec, category_tokens=("demo",))
    writer._flush()
    shard_path = tmp_path / "shard-00000.pt"
    assert shard_path.exists()
    payload = torch.load(shard_path)
    assert torch.equal(payload["input_ids"], torch.tensor([rec.input_ids for rec in records], dtype=torch.long))
    stats_path = shard_path.with_suffix(".stats.json")
    assert stats_path.exists()
    meta = json.loads(stats_path.read_text())
    assert meta["count"] == 2
