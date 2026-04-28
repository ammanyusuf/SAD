#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import logging
import math
import random
import statistics
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file

from lit_gpt.model import GPT, Config


# -----------------------------
# Utilities (mostly unchanged)
# -----------------------------

def apply_temperature_topk(logits: torch.Tensor, temperature: float, top_k: int):
    """
    Exactly same behavior as original:
      - divide by temperature
      - replace NaN with -inf
      - if top_k > 0: keep only top_k logits, rest -> -inf
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if temperature != 1.0:
        logits = logits / temperature

    # Robustness: NaN -> -inf
    if torch.isnan(logits).any():
        logits = torch.where(torch.isnan(logits), -torch.inf, logits)

    if top_k is not None and top_k > 0:
        top_k = min(int(top_k), logits.shape[-1])
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        cutoff = values[..., -1, None]          # the kth largest value
        logits = torch.where(logits < cutoff, -torch.inf, logits)

    return logits


def setup_logger(log_path: Optional[Path]):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def parse_float_list(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def add_timestamp_to_path(path: Path, ts: Optional[str] = None) -> Path:
    if ts is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def add_run_tags_to_path(path: Path, ts: str, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}_{ts}{path.suffix}")


def load_samples(samples_path: Path, max_samples: int, shuffle: bool, seed: int) -> List[Dict[str, Any]]:
    samples = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples[:max_samples]


def load_state_dict_local(ckpt_path: str) -> Dict[str, Any]:
    """
    Same compatibility logic as original:
      - .safetensors -> load_file
      - .pth/.pt     -> torch.load, and try to locate subkeys
    """
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        return load_file(str(p))

    ckpt = torch.load(str(p), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def summarize(values: List[float]) -> Dict[str, Any]:
    """
    Same summary logic:
      - only count finite values
      - mean/stdev/min/max on finite subset
    """
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"count": 0, "mean": float("nan"), "stdev": 0.0, "min": float("nan"), "max": float("nan")}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "stdev": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def compute_np_stats(log_sums: List[float], p_targets: List[float], n_targets: List[int]) -> Dict[str, Any]:
    """
    Same as original:
      - convert each log_sum to prob via exp
      - invalid (non-finite) => prob=0 but still counted in total_count
      - p_hat = average(prob) over *all samples*
      - compute:
          n_for_p[p] : how many tries to reach overall success p
          p_for_n[n] : overall success after n tries
    """
    if not log_sums:
        return {"p_hat": float("nan"), "processable_count": 0, "n_for_p": {}, "p_for_n": {}}

    total_count = len(log_sums)
    probs = []
    processable_count = 0
    for v in log_sums:
        if math.isfinite(v):
            probs.append(math.exp(v))
            processable_count += 1
        else:
            probs.append(0.0)

    p_hat = sum(probs) / total_count

    n_for_p = {}
    for p in p_targets:
        if not math.isfinite(p_hat):
            n_for_p[str(p)] = None
            continue
        denom = math.log1p(-p_hat)
        if denom == 0:
            n_for_p[str(p)] = None
        else:
            n_for_p[str(p)] = int(math.ceil(math.log(1 - p) / denom))

    p_for_n = {}
    for n in n_targets:
        if not math.isfinite(p_hat):
            p_for_n[str(n)] = float("nan")
            continue
        p_for_n[str(n)] = -math.expm1(n * math.log1p(-p_hat))

    return {"p_hat": p_hat, "processable_count": processable_count, "n_for_p": n_for_p, "p_for_n": p_for_n}


# ---------------------------------------------
# Core: teacher-forcing suffix LL (variable len)
# ---------------------------------------------

@torch.no_grad()
def ar_prefix_suffix_loglikelihood_varlen_batch_from_logits(
    logits: torch.Tensor,          # [B, L, V]  model(x) output
    input_ids: torch.Tensor,       # [B, L]     token ids (prefix+suffix)
    prefix_lens: List[int],        # len(prefix_tokens) per sample
    suffix_lens: List[int],        # len(suffix_tokens) per sample
    temperature: float = 1.0,
    top_k: int = 0,
) -> Tuple[List[float], List[float], List[float], List[int], List[bool]]:
    """
    Variable-length version of your original batch logic.
    We keep the SAME semantics:
      1) target = suffix tokens
      2) logits slice = positions that predict those suffix tokens (shift by 1)
      3) coverage check: after temperature, target must be in top-k at every suffix position
      4) final dist: temperature + top-k truncation + log_softmax
      5) if invalid: set outputs NaN

    Returns:
      ll_sum_list        : list[float] length B
      ll_per_token_list  : list[float] length B
      greedy_correct_list: list[float] length B   (1 if all suffix tokens greedy-match, else 0)
      suffix_len_list    : list[int]   length B
      invalid_list       : list[bool]  length B
    """
    device = logits.device
    B, L, V = logits.shape

    ll_sum_list: List[float] = []
    ll_tok_list: List[float] = []
    greedy_list: List[float] = []
    suffix_len_list: List[int] = []
    invalid_list: List[bool] = []

    nan = float("nan")

    for i in range(B):
        prefix_len = int(prefix_lens[i])
        suffix_len = int(suffix_lens[i])
        suffix_len_list.append(suffix_len)

        # Basic sanity:
        # We need at least 1 token in prefix so that logits at (prefix_len-1) exists.
        if prefix_len < 1 or suffix_len <= 0:
            ll_sum_list.append(nan)
            ll_tok_list.append(nan)
            greedy_list.append(nan)
            invalid_list.append(True)
            continue

        # This script batches by total L, but each sample may effectively use only its own suffix range.
        # Ensure we won't slice out of bounds:
        # last suffix token sits at position (prefix_len + suffix_len - 1) in input_ids,
        # which is predicted by logits position (prefix_len + suffix_len - 2).
        end_pos_exclusive = prefix_len + suffix_len
        if end_pos_exclusive > L:
            ll_sum_list.append(nan)
            ll_tok_list.append(nan)
            greedy_list.append(nan)
            invalid_list.append(True)
            continue

        # ---- targets (suffix tokens) ----
        target = input_ids[i, prefix_len:end_pos_exclusive]                      # [suffix_len]

        # ---- logits that predict target tokens ----
        # logits_t has shape [suffix_len, V]
        # positions: prefix_len-1 ... (prefix_len+suffix_len-2) inclusive
        logits_t = logits[i, prefix_len - 1:end_pos_exclusive - 1, :]            # [suffix_len, V]

        # ---- coverage check (same as original) ----
        invalid = torch.zeros((), dtype=torch.bool, device=device)

        logits_check = logits_t
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if temperature != 1.0:
            logits_check = logits_check / temperature
        if torch.isnan(logits_check).any():
            logits_check = torch.where(torch.isnan(logits_check), -torch.inf, logits_check)

        if top_k is not None and top_k > 0:
            k = min(int(top_k), V)
            _, topk_idx = torch.topk(logits_check, k=k, dim=-1)                  # [suffix_len, k]
            covered = (topk_idx == target.unsqueeze(-1)).any(dim=-1)             # [suffix_len]
            invalid = ~covered.all()                                             # scalar

        # ---- final distribution: temperature + topk truncation + renorm ----
        logits_final = apply_temperature_topk(logits_t, temperature=temperature, top_k=top_k)
        log_probs = F.log_softmax(logits_final, dim=-1)                          # [suffix_len, V]
        tgt_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)        # [suffix_len]

        # Any -inf / nan => invalid (same as original)
        invalid = invalid | (~torch.isfinite(tgt_logp).all())

        if bool(invalid.item()):
            ll_sum_list.append(nan)
            ll_tok_list.append(nan)
            greedy_list.append(nan)
            invalid_list.append(True)
            continue

        ll_sum = tgt_logp.sum().float().item()
        ll_per_token = ll_sum / float(suffix_len)

        pred = torch.argmax(logits_final, dim=-1)                                # [suffix_len]
        greedy_correct = float((pred == target).all().item())                    # 1.0 or 0.0

        ll_sum_list.append(ll_sum)
        ll_tok_list.append(ll_per_token)
        greedy_list.append(greedy_correct)
        invalid_list.append(False)

    return ll_sum_list, ll_tok_list, greedy_list, suffix_len_list, invalid_list


# -----------------------------
# Tokenization helper
# -----------------------------

def encode_text(tokenizer, text: str) -> List[int]:
    # We keep it explicit: no special tokens, same as typical LM continuation eval.
    return tokenizer.encode(text, add_special_tokens=False)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples_path", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # ===== AR model ckpt =====
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--lit_model_name", type=str, default="1028")
    parser.add_argument("--tokenizer_name", type=str, required=True)

    # temperature / topk (align with diffusion naming)
    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_k", type=int, default=40)

    # NP stats targets (same defaults you used before)
    parser.add_argument("--p_targets", type=str, default="0.1,0.5,0.9,0.99")
    parser.add_argument("--n_targets", type=str, default="1,10,100")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)

    # Tokenization boundary behavior:
    # - separate     : suffix_tokens = encode(email)
    # - concat_split : suffix_tokens = encode(context_text + email)[len(encode(context_text)):]
    parser.add_argument("--suffix_tokenization_mode", type=str, default="separate",
                        choices=["separate", "concat_split"])

    # If total length exceeds model block_size:
    # - truncate prefix from the LEFT to fit, keeping suffix intact (recommended for continuation eval)
    parser.add_argument("--truncate_prefix_to_fit", action="store_true")

    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--log_path", type=Path, default=None)
    parser.add_argument("--debug_print_first_k", type=int, default=0)

    args = parser.parse_args()
    args.p_targets = parse_float_list(args.p_targets)
    args.n_targets = parse_int_list(args.n_targets)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    if args.log_path is not None:
        args.log_path = add_timestamp_to_path(args.log_path, run_ts)
    if args.output_path is not None:
        args.output_path = add_run_tags_to_path(
            args.output_path,
            run_ts,
            f"ar_email_topk{args.gen_top_k}_T{args.gen_temperature}",
        )

    setup_logger(args.log_path)
    logging.info("Starting AR email-suffix evaluation (teacher forcing, keep original topk logic)")
    logging.info("Args: %s", vars(args))

    if not args.samples_path.exists():
        raise FileNotFoundError(f"samples jsonl not found: {args.samples_path}")

    samples = load_samples(args.samples_path, args.max_samples, args.shuffle, args.seed)
    logging.info("Loaded %d samples from %s", len(samples), args.samples_path)

    device = torch.device(args.device)

    # ===== load AR model =====
    model_name = f"Diff_LLaMA_{args.lit_model_name}M"
    config = Config.from_name(model_name)

    model = GPT(config).to(device)
    state_dict = load_state_dict_local(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys when loading ckpt: %d", len(missing))
    if unexpected:
        logging.warning("Unexpected keys when loading ckpt: %d", len(unexpected))

    # dtype: cuda -> bf16; cpu -> fp32 (more stable)
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    # Align floating buffers dtype to param dtype (same trick as your original)
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # ===== output jsonl =====
    results_f = None
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_f = open(args.output_path, "w", encoding="utf-8")

    # -------------------------------------------------------
    # Prepare + tokenize samples
    # -------------------------------------------------------
    # We'll build items of:
    #   (global_idx, context_text, suffix_text, prefix_tokens, suffix_tokens, input_tokens, prefix_len, suffix_len)
    valid_items: List[Dict[str, Any]] = []
    printed_debug = 0

    for idx, sample in enumerate(samples):
        context_text = sample.get("context_text", None)

        # --- changed: suffix field autodetect (email -> phone_number) ---
        email = sample.get("email", None)
        phone_number = sample.get("phone_number", None)

        if email is not None:
            suffix_text = email
            suffix_field = "email"
        elif phone_number is not None:
            suffix_text = phone_number
            suffix_field = "phone_number"
        else:
            suffix_text = None
            suffix_field = None
        # ---------------------------------------------------------------

        if context_text is None or suffix_text is None:
            logging.warning("Sample %d missing context_text or (email/phone_number), skipping", idx)
            continue

        # 1) tokenize prefix from context_text
        prefix_tokens = encode_text(tokenizer, context_text)

        # Ensure prefix_len >= 1, otherwise we cannot index (prefix_len-1) for logits slice.
        if len(prefix_tokens) < 1:
            # Minimal fallback: insert BOS if available; otherwise skip
            bos = getattr(tokenizer, "bos_token_id", None)
            if bos is None:
                logging.warning("Sample %d has empty prefix and tokenizer has no BOS, skipping", idx)
                continue
            logging.warning("Sample %d has empty prefix; prepending BOS token id=%s", idx, str(bos))
            prefix_tokens = [int(bos)]

        # 2) tokenize suffix from suffix_text
        if args.suffix_tokenization_mode == "separate":
            suffix_tokens = encode_text(tokenizer, suffix_text)
        else:
            # More faithful continuation tokenization:
            # encode(context_text + suffix_text) then slice out the continuation part
            full_tokens = encode_text(tokenizer, context_text + suffix_text)
            suffix_tokens = full_tokens[len(prefix_tokens):]

        if len(suffix_tokens) < 1:
            logging.warning("Sample %d suffix tokenized to empty, skipping", idx)
            continue

        # 3) build input = prefix + suffix (teacher forcing)
        input_tokens = prefix_tokens + suffix_tokens

        # 4) handle block_size limit if needed
        block_size = getattr(config, "block_size", None)
        if block_size is not None and len(input_tokens) > int(block_size):
            if not args.truncate_prefix_to_fit:
                logging.warning(
                    "Sample %d length=%d exceeds block_size=%d; skipping (use --truncate_prefix_to_fit to keep suffix)",
                    idx, len(input_tokens), int(block_size)
                )
                continue

            # Keep suffix intact, truncate prefix from the LEFT to fit.
            max_prefix = int(block_size) - len(suffix_tokens)
            if max_prefix < 1:
                logging.warning(
                    "Sample %d suffix_len=%d alone exceeds/equals block_size=%d; skipping",
                    idx, len(suffix_tokens), int(block_size)
                )
                continue

            prefix_tokens = prefix_tokens[-max_prefix:]
            input_tokens = prefix_tokens + suffix_tokens

        prefix_len = len(prefix_tokens)
        suffix_len = len(suffix_tokens)

        # Debug prints (optional)
        if args.debug_print_first_k > 0 and printed_debug < args.debug_print_first_k:
            logging.info("DEBUG Sample %d | prefix_len=%d suffix_len=%d | suffix_field=%s", idx, prefix_len, suffix_len, suffix_field)
            logging.info("DEBUG context_text=%r", context_text)
            logging.info("DEBUG suffix_text=%r", suffix_text)
            printed_debug += 1

        valid_items.append({
            "index": idx,
            "context_text": context_text,
            "suffix_text": suffix_text,
            "suffix_field": suffix_field,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "input_tokens": input_tokens,
        })

    logging.info("Processable samples: %d / %d", len(valid_items), len(samples))
    if not valid_items:
        logging.info("No valid samples. Exit.")
        if results_f is not None:
            results_f.close()
        return

    # -------------------------------------------------------
    # Bucket by total length L to avoid padding
    # -------------------------------------------------------
    buckets: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in valid_items:
        L = len(item["input_tokens"])
        buckets[L].append(item)

    logging.info("Bucketing by total length L: %d buckets", len(buckets))

    # -------------------------------------------------------
    # Evaluation accumulators
    # -------------------------------------------------------
    all_ll_sum: List[float] = []
    all_ll_tok: List[float] = []
    all_greedy: List[float] = []
    all_invalid: List[int] = []
    all_suffix_lens: List[int] = []

    use_amp = (device.type == "cuda")

    # -------------------------------------------------------
    # Run each bucket in batches (one forward per batch)
    # -------------------------------------------------------
    for L, items in sorted(buckets.items(), key=lambda x: x[0]):
        logging.info("Processing bucket L=%d with %d samples", L, len(items))

        bs = max(1, int(args.batch_size))
        for start in range(0, len(items), bs):
            chunk = items[start:start + bs]
            B = len(chunk)

            # Build [B, L] tensor (same L within bucket)
            x_list = [it["input_tokens"] for it in chunk]
            x = torch.tensor(x_list, dtype=torch.long, device=device)  # [B, L]

            prefix_lens = [it["prefix_len"] for it in chunk]
            suffix_lens = [it["suffix_len"] for it in chunk]

            # One forward per batch
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                logits = model(x)  # [B, L, V]

            # Compute per-sample suffix LL with SAME logic as before (topk coverage + truncation)
            ll_sums, ll_toks, greedys, suffix_len_list, invalid_flags = (
                ar_prefix_suffix_loglikelihood_varlen_batch_from_logits(
                    logits=logits,
                    input_ids=x,
                    prefix_lens=prefix_lens,
                    suffix_lens=suffix_lens,
                    temperature=args.gen_temperature,
                    top_k=args.gen_top_k,
                )
            )

            # Record outputs
            for i, it in enumerate(chunk):
                idx = it["index"]
                context_text = it["context_text"]
                suffix_text = it["suffix_text"]
                suffix_field = it["suffix_field"]

                ll_sum = ll_sums[i]
                ll_tok = ll_toks[i]
                greedy = greedys[i]
                suffix_len_i = suffix_len_list[i]
                invalid_topk = bool(invalid_flags[i])

                # Per-sample NP stats (list length = 1), kept for alignment with your old outputs
                ar_np = compute_np_stats([ll_sum], args.p_targets, args.n_targets)

                all_ll_sum.append(ll_sum)
                all_ll_tok.append(ll_tok)
                all_greedy.append(greedy)
                all_suffix_lens.append(suffix_len_i)
                all_invalid.append(1 if invalid_topk else 0)

                logging.info(
                    "Sample %d | prefix_len=%d suffix_len=%d | suffix_field=%s | T=%.3f top_k=%d | "
                    "ll_sum=%s | ll/token=%s | greedy_correct=%s | invalid_topk=%s | p_hat=%s",
                    idx,
                    int(prefix_lens[i]),
                    int(suffix_len_i),
                    suffix_field,
                    args.gen_temperature,
                    args.gen_top_k,
                    ll_sum,
                    ll_tok,
                    greedy,
                    invalid_topk,
                    ar_np["p_hat"],
                )
                logging.info("Sample %d | n_for_p=%s | p_for_n=%s", idx, ar_np["n_for_p"], ar_np["p_for_n"])

                if results_f is not None:
                    out = {
                        "index": idx,
                        "prefix_text": context_text,   # context_text is prefix
                        "suffix_text": suffix_text,    # email or phone_number (autodetected)
                        "suffix_field": suffix_field,  # <-- added for debugging/traceability
                        "tokenization_mode": args.suffix_tokenization_mode,
                        "ar": {
                            "ll_sum": ll_sum,
                            "ll_per_token": ll_tok,
                            "greedy_correct": greedy,
                            "invalid_topk": invalid_topk,
                            "temperature": args.gen_temperature,
                            "top_k": args.gen_top_k,
                            "suffix_len_tokens": suffix_len_i,

                            # Keep NP stats fields identical in spirit to your old jsonl
                            "processable_count": ar_np["processable_count"],
                            "p_hat": ar_np["p_hat"],
                            "n_for_p": ar_np["n_for_p"],
                            "p_for_n": ar_np["p_for_n"],
                        },
                    }
                    results_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if results_f is not None:
        results_f.close()

    # -------------------------------------------------------
    # Summary (global)
    # -------------------------------------------------------
    ll_sum_s = summarize(all_ll_sum)
    ll_tok_s = summarize(all_ll_tok)

    # invalid rate: fraction of samples that became invalid due to top-k coverage or -inf logp
    invalid_rate = sum(all_invalid) / max(1, len(all_invalid))

    # greedy correctness mean over finite values
    greedy_finite = [v for v in all_greedy if math.isfinite(v)]
    greedy_mean = statistics.fmean(greedy_finite) if greedy_finite else float("nan")

    suffix_len_mean = statistics.fmean(all_suffix_lens) if all_suffix_lens else float("nan")

    # Rate-level NP stats over all samples (same logic as your original summary)
    np_all = compute_np_stats(all_ll_sum, args.p_targets, args.n_targets)

    logging.info(
        "GLOBAL summary | samples=%d | suffix_len_mean=%.2f | invalid_topk_rate=%.4f | "
        "greedy_correct_mean=%.4f | "
        "ll_sum: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f | "
        "ll/token: mean=%.4f std=%.4f",
        len(all_ll_sum),
        suffix_len_mean,
        invalid_rate,
        greedy_mean,
        ll_sum_s["count"],
        ll_sum_s["mean"],
        ll_sum_s["stdev"],
        ll_sum_s["min"],
        ll_sum_s["max"],
        ll_tok_s["mean"],
        ll_tok_s["stdev"],
    )
    logging.info(
        "GLOBAL np | processable=%d/%d | p_hat=%s | n_for_p=%s | p_for_n=%s",
        np_all["processable_count"],
        len(all_ll_sum),
        np_all["p_hat"],
        np_all["n_for_p"],
        np_all["p_for_n"],
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
