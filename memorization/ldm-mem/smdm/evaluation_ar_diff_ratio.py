import argparse
import json
import logging
import math
import random
import statistics
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file

from lit_gpt.model import GPT, Config

TAIL_LEN = 30


def apply_temperature_topk(logits, temperature: float, top_k: int):
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


def parse_float_list(value: str):
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str):
    return [int(x) for x in value.split(",") if x.strip()]


def add_timestamp_to_path(path: Path, ts: Optional[str] = None) -> Path:
    if ts is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def add_run_tags_to_path(path: Path, ts: str, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}_{ts}{path.suffix}")


def load_samples(samples_path: Path, max_samples: int, shuffle: bool, seed: int):
    samples = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples[:max_samples]


def load_state_dict_local(ckpt_path: str):
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        return load_file(str(p))

    ckpt = torch.load(str(p), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def summarize(values: List[float]) -> Dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
            "stdev": 0.0,
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "stdev": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def compute_np_stats(log_sums, p_targets, n_targets):
    if not log_sums:
        return {
            "p_hat": float("nan"),
            "processable_count": 0,
            "n_for_p": {},
            "p_for_n": {},
        }
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

    return {
        "p_hat": p_hat,
        "processable_count": processable_count,
        "n_for_p": n_for_p,
        "p_for_n": p_for_n,
    }


@torch.no_grad()
def ar_prefix_suffix_loglikelihood_batch_from_logits(
    logits: torch.Tensor,   # [B, L, V]  (model(x) output)
    tails: torch.Tensor,    # [B, L]     (tail token ids)
    mask_rate: float,
    temperature: float = 1.0,
    top_k: int = 0,
):
    """
    Batch version:
      - One forward gives logits [B, L, V]
      - For each mask_rate, slice + (optional) topk/temperature, compute LL for suffix.
    Returns per-sample lists:
      ll_sum_list, ll_tok_list, greedy_correct_list, suffix_len, invalid_flags_list
    """
    device = logits.device
    B, L, V = logits.shape

    suffix_len = int(L * mask_rate)
    if suffix_len <= 0:
        raise ValueError(f"mask_rate={mask_rate} too small for tail_len={L} (suffix_len=0)")
    if suffix_len >= L:
        suffix_len = L - 1
    prefix_len = L - suffix_len
    assert prefix_len >= 1

    target = tails[:, prefix_len:]  # [B, suffix_len]
    logits_t = logits[:, prefix_len - 1 : L - 1, :]  # [B, suffix_len, V]

    # ---- coverage check: target must be inside top_k after temperature (before truncation) ----
    invalid = torch.zeros(B, dtype=torch.bool, device=device)

    logits_check = logits_t
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature != 1.0:
        logits_check = logits_check / temperature
    if torch.isnan(logits_check).any():
        logits_check = torch.where(torch.isnan(logits_check), -torch.inf, logits_check)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), V)
        _, topk_idx = torch.topk(logits_check, k=k, dim=-1)  # [B, suffix_len, k]
        covered = (topk_idx == target.unsqueeze(-1)).any(dim=-1)  # [B, suffix_len]
        invalid = ~covered.all(dim=-1)

    # ---- final distribution: temperature + topk truncation + renorm ----
    logits_final = apply_temperature_topk(logits_t, temperature=temperature, top_k=top_k)
    log_probs = F.log_softmax(logits_final, dim=-1)
    tgt_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, suffix_len]

    invalid = invalid | (~torch.isfinite(tgt_logp).all(dim=-1))

    ll_sum = tgt_logp.sum(dim=-1).float()  # [B]
    ll_per_token = ll_sum / float(suffix_len)

    pred = torch.argmax(logits_final, dim=-1)  # [B, suffix_len]
    greedy_correct = (pred == target).all(dim=-1).float()

    # Set invalid samples to NaN to match previous behavior
    nan = float("nan")
    ll_sum = ll_sum.masked_fill(invalid, nan)
    ll_per_token = ll_per_token.masked_fill(invalid, nan)
    greedy_correct = greedy_correct.masked_fill(invalid, nan)

    return (
        ll_sum.tolist(),
        ll_per_token.tolist(),
        greedy_correct.tolist(),
        suffix_len,
        invalid.tolist(),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_path",
        type=Path,
        default=Path(
            "smdm/data/unique_bin_samples.jsonl"
        ),
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # ===== AR 模型 ckpt =====
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="smdm/workdir/scaling_debug/arm-1028M-100.0/final.pth",
    )
    parser.add_argument("--lit_model_name", type=str, default="1028")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    )


    parser.add_argument("--mask_rates", type=str, default="0.05,0.1,0.15,0.2,0.3,0.5")
    parser.add_argument("--tail_len", type=int, default=30)


    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_k", type=int, default=40)


    parser.add_argument("--p_targets", type=str, default="0.1,0.5,0.9,0.99")
    parser.add_argument("--n_targets", type=str, default="1,10,100")

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--log_path", type=Path, default=None)
    parser.add_argument("--debug_print_first_k", type=int, default=0)

    args = parser.parse_args()
    args.mask_rates = parse_float_list(args.mask_rates)
    args.p_targets = parse_float_list(args.p_targets)
    args.n_targets = parse_int_list(args.n_targets)

    global TAIL_LEN
    TAIL_LEN = args.tail_len

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    if args.log_path is not None:
        args.log_path = add_timestamp_to_path(args.log_path, run_ts)
    if args.output_path is not None:
        args.output_path = add_run_tags_to_path(
            args.output_path, run_ts, f"ar_{args.gen_top_k}_T{args.gen_temperature}"
        )

    setup_logger(args.log_path)
    logging.info("Starting AR baseline evaluation (BATCHED; one forward per batch)")
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

    # dtype: cuda -> bf16; cpu -> fp32
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()


    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    # ===== output =====
    results_f = None
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_f = open(args.output_path, "w", encoding="utf-8")

    by_rate = {
        f"{r:g}": {"ll_sum": [], "ll_tok": [], "greedy": [], "suffix_len": [], "invalid": []}
        for r in args.mask_rates
    }

    # ===== prepare valid samples =====
    valid: List[Tuple[int, Dict[str, Any]]] = []
    for idx, sample in enumerate(samples):
        token_ids = sample.get("tokens", [])
        if not token_ids:
            logging.warning("Sample %d missing tokens, skipping", idx)
            continue
        if len(token_ids) < args.tail_len:
            logging.warning(
                "Sample %d too short token_len=%d < tail_len=%d, skipping",
                idx,
                len(token_ids),
                args.tail_len,
            )
            continue
        valid.append((idx, sample))

    logging.info("Processable samples: %d / %d", len(valid), len(samples))

    # ===== eval (batched) =====
    printed_debug = 0
    bs = max(1, int(args.batch_size))

    for start in range(0, len(valid), bs):
        chunk = valid[start : start + bs]

        tails_list: List[List[int]] = []
        meta: List[Tuple[int, str, str]] = []  # (idx, set_name, tail_text)

        for idx, sample in chunk:
            set_name = sample.get("set_name", "")
            token_ids = sample.get("tokens", [])
            tail_tokens = token_ids[-args.tail_len:]
            tails_list.append(tail_tokens)

            tail_text = decode_tokens(tokenizer, tail_tokens)
            meta.append((idx, set_name, tail_text))

            if args.debug_print_first_k > 0 and printed_debug < args.debug_print_first_k:
                logging.info("Sample %d | set=%s | tail_text=%s", idx, set_name, tail_text)
                printed_debug += 1

        x = torch.tensor(tails_list, dtype=torch.long, device=device)  # [B, L]

        use_amp = (device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            logits = model(x)  # [B, L, V] ✅ one forward per batch

        for r in args.mask_rates:
            try:
                ll_sums, ll_toks, greedys, suffix_len, invalid_flags = (
                    ar_prefix_suffix_loglikelihood_batch_from_logits(
                        logits=logits,
                        tails=x,
                        mask_rate=r,
                        temperature=args.gen_temperature,
                        top_k=args.gen_top_k,
                    )
                )
            except Exception as e:
                logging.exception("Batch AR eval failed on mask_rate=%s: %s", r, str(e))
                # fallback: mark all as invalid for this batch/rate
                suffix_len = int(args.tail_len * r)
                ll_sums = [float("nan")] * len(meta)
                ll_toks = [float("nan")] * len(meta)
                greedys = [float("nan")] * len(meta)
                invalid_flags = [True] * len(meta)

            key = f"{r:g}"
            for i, (idx, set_name, tail_text) in enumerate(meta):
                ll_sum = ll_sums[i]
                ll_tok = ll_toks[i]
                greedy = greedys[i]
                invalid_topk = bool(invalid_flags[i])

                # 对齐 diffusion：对单条 log_sum 也跑一遍 np（列表长度=1）
                ar_np = compute_np_stats([ll_sum], args.p_targets, args.n_targets)

                by_rate[key]["ll_sum"].append(ll_sum)
                by_rate[key]["ll_tok"].append(ll_tok)
                by_rate[key]["greedy"].append(greedy)
                by_rate[key]["suffix_len"].append(suffix_len)
                by_rate[key]["invalid"].append(1 if invalid_topk else 0)

                logging.info(
                    "Sample %d | Mask %.3f | T=%.3f top_k=%d | suffix_len=%d | ll_sum=%s | ll/token=%s | invalid_topk=%s | p_hat=%s",
                    idx,
                    r,
                    args.gen_temperature,
                    args.gen_top_k,
                    suffix_len,
                    ll_sum,
                    ll_tok,
                    invalid_topk,
                    ar_np["p_hat"],
                )
                logging.info("Sample %d | Mask %.3f | n_for_p=%s | p_for_n=%s", idx, r, ar_np["n_for_p"], ar_np["p_for_n"])

                if results_f is not None:
                    out = {
                        "index": idx,
                        "set_name": set_name,
                        "mask_ratio": r,
                        "text": tail_text,  #
                        "ar": {
                            "ll_sum": ll_sum,
                            "ll_per_token": ll_tok,
                            "greedy_correct": greedy,
                            "invalid_topk": invalid_topk,
                            "temperature": args.gen_temperature,
                            "top_k": args.gen_top_k,
                            "processable_count": ar_np["processable_count"],
                            "p_hat": ar_np["p_hat"],
                            "n_for_p": ar_np["n_for_p"],
                            "p_for_n": ar_np["p_for_n"],
                        },
                    }
                    results_f.write(json.dumps(out) + "\n")

    if results_f is not None:
        results_f.close()

    # ===== summary =====
    for key, stat in by_rate.items():
        ll_sum_s = summarize(stat["ll_sum"])
        ll_tok_s = summarize(stat["ll_tok"])
        invalid_rate = sum(stat["invalid"]) / max(1, len(stat["invalid"]))

        suffix_len_mean = (
            statistics.fmean([x for x in stat["suffix_len"] if x is not None])
            if stat["suffix_len"]
            else float("nan")
        )

        # ✅ per-rate 的 NP（建议保留）
        np_rate = compute_np_stats(stat["ll_sum"], args.p_targets, args.n_targets)

        logging.info(
            "Mask %s summary | suffix_len~%.2f | invalid_topk_rate=%.4f | "
            "ll_sum: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f | "
            "ll/token: mean=%.4f std=%.4f",
            key,
            suffix_len_mean,
            invalid_rate,
            ll_sum_s["count"],
            ll_sum_s["mean"],
            ll_sum_s["stdev"],
            ll_sum_s["min"],
            ll_sum_s["max"],
            ll_tok_s["mean"],
            ll_tok_s["stdev"],
        )
        logging.info(
            "Mask %s np | processable=%d/%d | p_hat=%s | n_for_p=%s | p_for_n=%s",
            key,
            np_rate["processable_count"],
            len(stat["ll_sum"]),
            np_rate["p_hat"],
            np_rate["n_for_p"],
            np_rate["p_for_n"],
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
