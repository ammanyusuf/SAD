#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file

from lit_gpt.model import GPT, Config


# -----------------------------
# Utils
# -----------------------------

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_state_dict_local(ckpt_path: str) -> Dict[str, Any]:
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        return load_file(str(p))

    ckpt = torch.load(str(p), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def apply_temperature_topk(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if temperature != 1.0:
        logits = logits / temperature

    if torch.isnan(logits).any():
        logits = torch.where(torch.isnan(logits), -torch.inf, logits)

    if top_k is not None and top_k > 0:
        top_k = min(int(top_k), logits.shape[-1])
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, -torch.inf, logits)

    return logits


def encode_text(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def add_timestamp(path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def prefix_match_len(gen: List[int], gt: List[int]) -> int:
    n = min(len(gen), len(gt))
    m = 0
    for i in range(n):
        if gen[i] != gt[i]:
            break
        m += 1
    return m


# -----------------------------
# Core: KV-cache batched generation
# -----------------------------

@torch.no_grad()
def generate_batch_kvcache(
    model: GPT,
    prompt_tokens: List[int],
    gen_len: int,
    batch_size: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """
    Generate for a batch of size B (B = batch_size) with the same prompt.
    Returns generated ids [B, gen_len].
    """
    if gen_len <= 0:
        raise ValueError("gen_len must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if len(prompt_tokens) < 1:
        raise ValueError("prompt_tokens must be non-empty")

    model.reset_cache()

    prompt = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, P]
    x = prompt.repeat(batch_size, 1)  # [B, P]
    P = x.size(1)

    max_seq_length = P + gen_len
    input_pos = torch.arange(P, device=device, dtype=torch.long)

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
        logits = model(x, max_seq_length=max_seq_length, input_pos=input_pos)  # [B, P, V]

    generated = []
    cur_pos = P
    for _ in range(gen_len):
        next_logits = logits[:, -1, :]  # [B, V]
        next_logits = apply_temperature_topk(next_logits, temperature=temperature, top_k=top_k)
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        generated.append(next_token.squeeze(1))               # [B]

        input_pos = torch.tensor([cur_pos], device=device, dtype=torch.long)
        cur_pos += 1

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            logits = model(next_token, max_seq_length=max_seq_length, input_pos=input_pos)  # [B, 1, V]

    return torch.stack(generated, dim=1)  # [B, gen_len]


# -----------------------------
# Main
# -----------------------------

def main():
    setup_logger()

    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--samples_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)

    # model
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--lit_model_name", type=str, default="1028")
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    # generation config
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_k", type=int, default=40)

    # NEW: user-controlled batching + trials
    parser.add_argument("--batch_size", type=int, default=64, help="micro-batch size (trials per forward)")
    parser.add_argument("--n_trials", type=int, default=1000, help="total trials per sample")

    # slicing config
    parser.add_argument("--prompt_len", type=int, default=80)
    parser.add_argument("--gen_len", type=int, default=20)

    # truncate prompt from the LEFT if needed to fit block_size with gen_len reserved
    parser.add_argument("--truncate_prompt_to_fit", action="store_true")

    # optional: limit number of samples processed
    parser.add_argument("--max_samples", type=int, default=10**18)

    args = parser.parse_args()

    if not args.samples_path.exists():
        raise FileNotFoundError(f"not found: {args.samples_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path = add_timestamp(args.output_path)

    device = torch.device(args.device)
    use_amp = (device.type == "cuda")

    # ---- load model ----
    model_name = f"Diff_LLaMA_{args.lit_model_name}M"
    config = Config.from_name(model_name)
    model = GPT(config).to(device)

    state_dict = load_state_dict_local(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys: %d", len(missing))
    if unexpected:
        logging.warning("Unexpected keys: %d", len(unexpected))

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    # Align buffers dtype to param dtype
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    block_size = getattr(config, "block_size", None)
    if block_size is None:
        logging.warning("config.block_size is None; will not enforce context limit.")
    else:
        logging.info("Model block_size=%d", int(block_size))

    # ---- run ----
    n_total = 0
    n_used = 0
    n_skipped_short = 0
    n_skipped_fit = 0

    with args.samples_path.open("r", encoding="utf-8") as fin, args.output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_total += 1
            if n_used >= args.max_samples:
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", None)
            if not isinstance(text, str) or not text:
                continue

            tokens = encode_text(tokenizer, text)
            need = args.prompt_len + args.gen_len
            if len(tokens) < need:
                n_skipped_short += 1
                continue

            prompt_tokens = tokens[:args.prompt_len]
            gt_tokens = tokens[args.prompt_len:args.prompt_len + args.gen_len]
            gt_list = [int(x) for x in gt_tokens]

            # enforce fit: prompt_len + gen_len <= block_size
            if block_size is not None:
                max_prompt = int(block_size) - args.gen_len
                if max_prompt < 1:
                    raise ValueError(f"block_size({block_size}) too small for gen_len={args.gen_len}")
                if len(prompt_tokens) > max_prompt:
                    if not args.truncate_prompt_to_fit:
                        n_skipped_fit += 1
                        continue
                    prompt_tokens = prompt_tokens[-max_prompt:]

            # per-sample accumulators
            hit_count = 0
            hist: Dict[str, int] = {}

            # run trials in micro-batches
            remaining = int(args.n_trials)
            while remaining > 0:
                cur_b = min(int(args.batch_size), remaining)

                gen = generate_batch_kvcache(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    gen_len=args.gen_len,
                    batch_size=cur_b,
                    temperature=args.gen_temperature,
                    top_k=args.gen_top_k,
                    device=device,
                    use_amp=use_amp,
                )  # [cur_b, gen_len]

                gen_cpu = gen.detach().to("cpu")
                for i in range(cur_b):
                    gen_list = gen_cpu[i].tolist()
                    m = prefix_match_len(gen_list, gt_list)  # 0..gen_len
                    k = str(m)
                    hist[k] = hist.get(k, 0) + 1
                    if m == args.gen_len:
                        hit_count += 1

                remaining -= cur_b

            hit_prob = hit_count / float(args.n_trials)

            if "ar" not in obj or not isinstance(obj["ar"], dict) or obj["ar"] is None:
                obj["ar"] = {}

            obj["ar"]["gen_eval"] = {
                "prompt_len_tokens": args.prompt_len,
                "gen_len_tokens": args.gen_len,
                "n_trials": args.n_trials,
                "batch_size": args.batch_size,
                "temperature": args.gen_temperature,
                "top_k": args.gen_top_k,
                "hit_count": hit_count,
                "hit_prob": hit_prob,
                "hit_token_hist": hist,  # {"15":5,"17":20,...} per sample
            }

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_used += 1

            if n_used % 10 == 0:
                logging.info(
                    "used=%d | total_lines=%d | skipped_short=%d | skipped_fit=%d",
                    n_used, n_total, n_skipped_short, n_skipped_fit
                )

    logging.info("DONE")
    logging.info("total_lines=%d", n_total)
    logging.info("used=%d", n_used)
    logging.info("skipped_short(<prompt+gen)=%d", n_skipped_short)
    logging.info("skipped_fit(prompt too long and no truncate)=%d", n_skipped_fit)
    logging.info("output=%s", str(args.output_path))


if __name__ == "__main__":
    main()
