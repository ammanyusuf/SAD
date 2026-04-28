#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Base"
DEFAULT_MASK_ID = 126336


@dataclass
class GenParams:
    set_name: int = 7
    line_index: int = 0
    index: int = 0
    tail_len: int = 100
    steps: int = 18
    alg: str = "greddy"  # 按你示例拼写
    gumbel_temperature: float = 1.0
    cfg_scale: float = 0.0
    eps: float = 0.001
    logp_threshold: float = math.log(0.001)  # -6.907755...


def find_span_token_positions_with_offsets(
    tokenizer,
    text: str,
    span_start: int,
    span_end: int,
) -> List[int]:
    """用 fast tokenizer 的 offsets_mapping，把 [span_start, span_end) 覆盖到的 token 索引找出来"""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        raise RuntimeError("Tokenizer does not provide offset_mapping (need a fast tokenizer).")

    mask_positions = [
        i for i, (a, b) in enumerate(offsets)
        if not (b <= span_start or a >= span_end)
    ]
    if not mask_positions:
        raise RuntimeError("No tokens overlap the target span; check span offsets.")
    return mask_positions


def find_email_token_positions(
    tokenizer,
    text: str,
    email: str,
    token_ids: List[int],
) -> List[int]:
    """
    目标：只 mask 你拼到末尾的那个 email，所以用 rfind 定位最后一次出现的位置。
    优先 offsets_mapping；不行就 fallback 子序列匹配（从右往左找最后一次匹配）。
    """
    start_char = text.rfind(email)
    if start_char < 0:
        raise ValueError(f"Email substring not found in text: {email}")
    end_char = start_char + len(email)

    # 1) offsets_mapping（更稳）
    try:
        return find_span_token_positions_with_offsets(tokenizer, text, start_char, end_char)
    except Exception:
        pass

    # 2) fallback：token id 子序列匹配（找最后一次）
    email_ids = tokenizer(email, add_special_tokens=False)["input_ids"]

    def find_sublist_last(haystack: List[int], needle: List[int]) -> Optional[int]:
        if not needle or len(needle) > len(haystack):
            return None
        for i in range(len(haystack) - len(needle), -1, -1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return None

    pos0 = find_sublist_last(token_ids, email_ids)
    if pos0 is None:
        raise RuntimeError(
            "Failed to locate email token-id sequence in token_ids. "
            "Try using a fast tokenizer (use_fast=True) / trust_remote_code=True."
        )
    return list(range(pos0, pos0 + len(email_ids)))


def convert_one(
    data: Dict[str, Any],
    model_name: str,
    mask_id: int,
    params: GenParams,
    seed: int = 88737,
) -> Dict[str, Any]:
    """
    关键：把 email 内容接到 context_text 的最后：
    text = context_text + email
    mask_positions 对应末尾 email 的 token 索引
    """
    email = data["email"]
    context_text = data["context_text"]

    # ✅ 改这里：email 拼到末尾（紧贴，不加分隔符）
    text = f"{context_text}{email}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    enc = tokenizer(text, add_special_tokens=False)
    token_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    mask_positions = find_email_token_positions(tokenizer, text, email, token_ids)

    masked_token_ids = [token_ids[i] for i in mask_positions]
    masked_tokens = [tokens[i] for i in mask_positions]

    masked_input_ids = list(token_ids)
    for i in mask_positions:
        masked_input_ids[i] = mask_id

    mask_ratio = len(mask_positions) / max(1, len(token_ids))

    traj = {
        "traj_id": 0,
        "seed": seed,
        "logp": 0.0,         # 占位
        "p": 1.0,            # 占位
        "p_percent": 100.0,  # 占位
        "mask_positions": mask_positions,
        "masked_token_ids": masked_token_ids,
        "masked_tokens": masked_tokens,
        "masked_input_ids": masked_input_ids,
        "prompt_len": len(token_ids),
        "steps": params.steps,
        "alg": params.alg,
        "gumbel_temperature": params.gumbel_temperature,
        "cfg_scale": params.cfg_scale,
        "eps": params.eps,
    }

    out = {
        "line_index": params.line_index,
        "index": params.index,
        "set_name": params.set_name,
        "text": text,
        "tail_len": params.tail_len,
        "mask_ratio": mask_ratio,
        "mask_id": mask_id,
        "prompt_len": len(token_ids),
        "steps": params.steps,
        "alg": params.alg,
        "gumbel_temperature": params.gumbel_temperature,
        "cfg_scale": params.cfg_scale,
        "eps": params.eps,
        "logp_threshold": params.logp_threshold,
        "token_ids": token_ids,
        "tokens": tokens,
        "good_trajectories": [traj],  # 只要一个 trajectory
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=str, required=True, help="input json file: {'email':..., 'context_text':...}")
    ap.add_argument("--output", "-o", type=str, required=True, help="output json file")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    ap.add_argument("--set-name", type=int, default=7)
    ap.add_argument("--line-index", type=int, default=0)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--tail-len", type=int, default=100)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--seed", type=int, default=88737)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = GenParams(
        set_name=args.set_name,
        line_index=args.line_index,
        index=args.index,
        tail_len=args.tail_len,
        steps=args.steps,
    )

    out = convert_one(
        data=data,
        model_name=args.model,
        mask_id=args.mask_id,
        params=params,
        seed=args.seed,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"OK: wrote {args.output}")


if __name__ == "__main__":
    main()
