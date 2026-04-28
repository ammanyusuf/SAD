#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import random
import statistics
import time
from contextlib import nullcontext
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file 

from lit_gpt.diffmodel import TransEncoder, Config


# =========================
# Logging utils
# =========================
def add_timestamp_to_path(path: Path, ts: Optional[str] = None) -> Path:
    if ts is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def add_run_tags_to_path(path: Path, ts: str, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}_{ts}{path.suffix}")


def add_range_to_path(path: Optional[Path], start: int, end: Optional[int]) -> Optional[Path]:
    if path is None:
        return None
    s = f"s{start}"
    e = f"e{end}" if end is not None else "eend"
    return path.with_name(f"{path.stem}_{s}_{e}{path.suffix}")


def setup_logger(log_path: Optional[Path]):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


# =========================
# Sample loading
# =========================
def load_jsonl(samples_path: Path, max_samples: int, shuffle: bool, seed: int) -> List[dict]:
    samples: List[dict] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples[:max_samples]


# =========================
# Local ckpt loading (优先 safetensors + 兼容 pth)
# =========================
def extract_state_dict_from_ckpt(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        for k in ("model_state_dict", "model_state"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    raise TypeError(f"Unrecognized ckpt format: {type(ckpt_obj)}")


def clean_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(sd.keys())
    has_module = any(k.startswith("module.") for k in keys)
    has_forward_module = any(k.startswith("_forward_module.") for k in keys)
    has_model = any(k.startswith("model.") for k in keys)
    has_diff_model = any(k.startswith("diff_model.") for k in keys)

    def strip(k: str) -> str:
        if has_forward_module and k.startswith("_forward_module."):
            k = k[len("_forward_module.") :]
        if has_module and k.startswith("module."):
            k = k[len("module.") :]
        if has_model and k.startswith("model."):
            k = k[len("model.") :]
        if has_diff_model and k.startswith("diff_model."):
            k = k[len("diff_model.") :]
        return k

    return {strip(k): v for k, v in sd.items()}


def load_state_dict_local(ckpt_path: str) -> Dict[str, torch.Tensor]:
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        sd = load_file(str(p))
        return clean_state_dict_keys(sd)

    ckpt = torch.load(str(p), map_location="cpu")
    sd = extract_state_dict_from_ckpt(ckpt)
    sd = clean_state_dict_keys(sd)
    return sd


def infer_vocab_size(model: torch.nn.Module, config: Any) -> Optional[int]:
    for attr in ("vocab_size", "padded_vocab_size"):
        v = getattr(config, attr, None)
        if isinstance(v, int) and v > 0:
            return int(v)

    v = getattr(model, "vocab_size", None)
    if isinstance(v, int) and v > 0:
        return int(v)

    for path in (
        ("transformer", "wte"),
        ("model", "embed_tokens"),
        ("tok_embeddings",),
        ("embedding",),
        ("emb",),
    ):
        cur = model
        ok = True
        for p in path:
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and hasattr(cur, "weight") and torch.is_tensor(cur.weight):
            return int(cur.weight.shape[0])
    return None


def load_local_transencoder_and_tokenizer(
    lit_model_name: str,
    tokenizer_name: str,
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.nn.Module, AutoTokenizer, Optional[int], Any]:
    model_name = f"Diff_LLaMA_{lit_model_name}M"
    logging.info("Loading local TransEncoder config: %s", model_name)
    config = Config.from_name(model_name)

    logging.info("Instantiating TransEncoder...")
    model = TransEncoder(config).to(device)

    logging.info("Loading checkpoint (local): %s", ckpt_path)
    state_dict = load_state_dict_local(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info("load_state_dict(strict=False): missing=%d unexpected=%d", len(missing), len(unexpected))
    if unexpected:
        logging.warning("Unexpected keys head: %s", unexpected[:20])
    if missing:
        logging.warning("Missing keys head: %s", missing[:20])

    model = model.to(device=device, dtype=dtype)
    model.eval()

    # ✅ buffer dtype 对齐
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    vocab_size = infer_vocab_size(model, config)
    return model, tokenizer, vocab_size, config


# ============================================================
# 1-step generation + scoring (MC trajectories)
# ============================================================
def sample_gumbel_argmax(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("gumbel_temperature must be > 0")
    # logits: [..., V]
    # gumbel: -log(-log(U))
    u = torch.rand_like(logits, dtype=torch.float32)
    u = u.clamp_(1e-12, 1.0 - 1e-12)
    g = -torch.log(-torch.log(u))
    # argmax(logits/temperature + g) 等价于 argmax(logits + temperature*g)
    scores = logits.to(torch.float32) + float(temperature) * g
    return torch.argmax(scores, dim=-1)


@torch.inference_mode()
def generate_and_score_batch_step1(
    model: torch.nn.Module,
    gt_tokens: torch.Tensor,   # [B, L]
    mask_id: int,
    mask_count: int,
    prompt_len_for_cfg: int,
    alg: str,                  
    temperature: float,
    cfg_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert gt_tokens.dim() == 2
    B, L = gt_tokens.shape
    if mask_count <= 0 or mask_count > L:
        raise ValueError(f"invalid mask_count={mask_count} for L={L}")

    # ---- random mask positions across full text, per-trajectory independent ----
    # idx: [B, mask_count]
    r = torch.rand((B, L), device=device)
    _, idx = torch.topk(r, k=mask_count, dim=1, largest=True, sorted=False)

    x = gt_tokens.clone()
    x.scatter_(1, idx, mask_id)

    use_amp = (device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

    with amp_ctx:
        if cfg_scale > 0.0:
            un_x = x.clone()
            if prompt_len_for_cfg > 0:
                un_x[:, :prompt_len_for_cfg] = mask_id
            x_ = torch.cat([x, un_x], dim=0)     # [2B, L]
            logits_all = model(x_)               # [2B, L, V]
            logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
            logits_full = logits_u + (cfg_scale + 1.0) * (logits_c - logits_u)  # [B, L, V]
        else:
            logits_full = model(x)               # [B, L, V]

    # mask logits： [B, mask_count, V]
    b_ar = torch.arange(B, device=device).unsqueeze(1).expand(B, mask_count)
    logits_sel = logits_full[b_ar, idx, :]  # [B, mask_count, V]

    # gumbel： [B, mask_count]
    sampled = sample_gumbel_argmax(logits_sel, temperature=temperature)

    # GT token： [B, mask_count]
    gt_sel = gt_tokens.gather(1, idx)

    correct = (sampled == gt_sel).to(torch.int64)          # [B, mask_count]
    correct_per_row = correct.sum(dim=1)                   # [B]
    exact_flags = (correct_per_row == mask_count)          # [B]
    return correct_per_row, exact_flags


def eval_sample_step1_mc(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    mask_id: int,
    mask_ratio: float,
    tail_len: int,
    seed: int,
    total_traj: int,
    traj_batch_size: int,
    alg: str,
    temperature: float,
    cfg_scale: float,
) -> Dict[str, Any]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = token_ids[-tail_len:]
    seq_len = len(token_ids)
    if seq_len < tail_len:
        raise ValueError(f"sequence too short after tail cut: {seq_len} < tail_len={tail_len}")

    mask_count = int(seq_len * mask_ratio)
    if mask_count <= 0 or mask_count > seq_len:
        raise ValueError(f"invalid mask_ratio={mask_ratio} for seq_len={seq_len}")

    prompt_len = seq_len - mask_count
    device = next(model.parameters()).device

    gt_1 = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

    exact_hits = 0
    hist_mask_correct = torch.zeros((mask_count + 1,), dtype=torch.int64, device="cpu")
    for start in range(0, total_traj, traj_batch_size):
        cur_bs = min(traj_batch_size, total_traj - start)

        chunk_seed = int(seed + start * 1337 + mask_count * 17 + seq_len)
        torch.manual_seed(chunk_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(chunk_seed)

        gt = gt_1.repeat(cur_bs, 1)  # [B, L]

        correct_per_row, exact_flags = generate_and_score_batch_step1(
            model=model,
            gt_tokens=gt,
            mask_id=mask_id,
            mask_count=mask_count,
            prompt_len_for_cfg=prompt_len,
            alg=alg,
            temperature=temperature,
            cfg_scale=cfg_scale,
            device=device,
        )

        exact_hits += int(exact_flags.sum().detach().cpu().item())

        # histogram update
        cc_cpu = correct_per_row.detach().cpu()
        bc = torch.bincount(cc_cpu, minlength=mask_count + 1)
        hist_mask_correct += bc

        if (start // traj_batch_size) % 50 == 0:
            mean_cc = float(cc_cpu.to(torch.float32).mean().item())
            max_cc = int(cc_cpu.max().item())
            min_cc = int(cc_cpu.min().item())
            logging.info(
                "[traj_chunk %d:%d] bs=%d | correct_mask_tokens mean=%.3f min=%d max=%d | exact_hits_so_far=%d",
                start, start + cur_bs, cur_bs, mean_cc, min_cc, max_cc, exact_hits,
            )

    hit_rate = exact_hits / float(total_traj)

    hist_total_correct = {str(i + prompt_len): int(hist_mask_correct[i].item()) for i in range(mask_count + 1)}
    hist_mask_correct_dict = {str(i): int(hist_mask_correct[i].item()) for i in range(mask_count + 1)}
    # E[correct_mask]
    mean_correct_mask = sum(i * int(hist_mask_correct[i].item()) for i in range(mask_count + 1)) / float(total_traj)

    return {
        "seq_len": int(seq_len),
        "mask_count": int(mask_count),
        "prompt_len": int(prompt_len),
        "total_traj": int(total_traj),
        "exact_hits": int(exact_hits),
        "exact_hit_rate": float(hit_rate),
        "mean_correct_mask_tokens": float(mean_correct_mask),
        "hist_correct_mask_tokens": hist_mask_correct_dict,
        "hist_correct_total_tokens": hist_total_correct,
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=10_000_000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=88666)

    # local model
    parser.add_argument("--lit_model_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    # diffusion/mask params
    parser.add_argument("--diff_mask_id", type=int, default=126336)
    parser.add_argument("--tail_len", type=int, default=100)
    parser.add_argument("--default_mask_ratio", type=float, default=0.2)
    parser.add_argument("--use_sample_mask_ratio", action="store_true", default=True)

    # generation eval params
    parser.add_argument("--steps", type=int, default=1, help="固定为1（1步生成），仅做校验用")
    parser.add_argument("--total_traj", type=int, default=100000, help="trajectories per sample")
    parser.add_argument("--traj_batch_size", type=int, default=64, help="micro-batch size for trajectories (VRAM)")

    # sampler params
    parser.add_argument("--alg", type=str, default="origin", choices=["origin", "greddy"])
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--gumbel_temperature", type=float, default=1.0)

    # device/dtype
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # output/log
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--log_path", type=Path, default=None)

    args = parser.parse_args()

    if int(args.steps) != 1:
        raise ValueError("reset 1。")

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    if args.log_path is not None and args.log_path.name.endswith(".log"):
        args.log_path = add_timestamp_to_path(args.log_path, ts=run_ts)
    if args.output_path is not None and args.output_path.name.endswith(".jsonl"):
        args.output_path = add_run_tags_to_path(args.output_path, run_ts, tag="gen_step1_hit_rate")

    args.output_path = add_range_to_path(args.output_path, int(args.start_index), args.end_index)
    args.log_path = add_range_to_path(args.log_path, int(args.start_index), args.end_index)

    setup_logger(args.log_path)
    logging.info("Starting diffusion 1-step generation hit-rate eval (trajectory MC)")
    logging.info("Args: %s", vars(args))

    if not args.input_path.exists():
        raise FileNotFoundError(f"input jsonl not found: {args.input_path}")

    samples_all = load_jsonl(args.input_path, args.max_samples, args.shuffle, args.seed)
    start_i = max(0, int(args.start_index))
    end_i = int(args.end_index) if args.end_index is not None else len(samples_all)
    end_i = min(len(samples_all), max(start_i, end_i))
    samples = samples_all[start_i:end_i]
    logging.info("Loaded %d lines (total_loaded=%d, range=[%d:%d])", len(samples), len(samples_all), start_i, end_i)

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    model, tokenizer, vocab_size, _config = load_local_transencoder_and_tokenizer(
        lit_model_name=args.lit_model_name,
        tokenizer_name=args.tokenizer_name,
        ckpt_path=args.ckpt_path,
        device=device,
        dtype=dtype,
    )
    logging.info("Model ready. vocab_size(inferred)=%s", str(vocab_size))

    results_f = None
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_f = open(args.output_path, "w", encoding="utf-8")

    # global accumulators
    global_exact_hits = 0
    global_total_traj = 0
    global_hist_mask_correct: Counter = Counter()
    global_mask_count_ref: Optional[int] = None  
    global_hist_by_mask_count: Dict[int, Counter] = {}

    for line_idx, sample in enumerate(samples, start=start_i):
        sample_index = sample.get("index", line_idx)
        set_name = sample.get("set_name", sample.get("setname", ""))

        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            logging.warning("Line %d (sample_index=%s) missing/empty text, skipping", line_idx, str(sample_index))
            continue

        mask_ratio = args.default_mask_ratio
        if args.use_sample_mask_ratio and ("mask_ratio" in sample):
            try:
                mask_ratio = float(sample["mask_ratio"])
            except Exception:
                mask_ratio = args.default_mask_ratio


        run_seed = int(args.seed + int(sample_index) * 1_000_000 + 17)

        logging.info(
            "Line %d | sample_index=%s | set=%s | mask_ratio=%.4f | steps=1 | total_traj=%d | alg=%s",
            line_idx, str(sample_index), str(set_name), mask_ratio, int(args.total_traj), args.alg
        )

        t0 = time.time()
        try:
            stats = eval_sample_step1_mc(
                model=model,
                tokenizer=tokenizer,
                text=text,
                mask_id=int(args.diff_mask_id),
                mask_ratio=float(mask_ratio),
                tail_len=int(args.tail_len),
                seed=run_seed,
                total_traj=int(args.total_traj),
                traj_batch_size=int(args.traj_batch_size),
                alg=str(args.alg),
                temperature=float(args.gumbel_temperature),
                cfg_scale=float(args.cfg_scale),
            )

            dt = time.time() - t0
            logging.info(
                "DONE line %d | exact_hit_rate=%.6f (%d/%d) | mask_count=%d | mean_correct_mask=%.3f | time=%.2fs",
                line_idx,
                stats["exact_hit_rate"],
                stats["exact_hits"],
                stats["total_traj"],
                stats["mask_count"],
                stats["mean_correct_mask_tokens"],
                dt,
            )

            # update global
            global_exact_hits += int(stats["exact_hits"])
            global_total_traj += int(stats["total_traj"])

            mc = int(stats["mask_count"])
            hist_local = stats["hist_correct_mask_tokens"]  # dict str->int
            if mc not in global_hist_by_mask_count:
                global_hist_by_mask_count[mc] = Counter()
            global_hist_by_mask_count[mc].update({int(k): int(v) for k, v in hist_local.items()})


            if global_mask_count_ref is None:
                global_mask_count_ref = mc
            if mc == global_mask_count_ref:
                global_hist_mask_correct.update({int(k): int(v) for k, v in hist_local.items()})

            out = {
                "line_index": int(line_idx),
                "index": sample_index,
                "set_name": set_name,
                "mask_ratio": float(mask_ratio),
                "steps": 1,
                "alg": args.alg,
                "cfg_scale": float(args.cfg_scale),
                "gumbel_temperature": float(args.gumbel_temperature),
                "tail_len": int(args.tail_len),
                "diff_mask_id": int(args.diff_mask_id),
                "total_traj": int(args.total_traj),
                "traj_batch_size": int(args.traj_batch_size),
                "result": stats,
            }

        except Exception as e:
            logging.exception("Failed line %d (sample_index=%s): %s", line_idx, str(sample_index), str(e))
            out = {
                "line_index": int(line_idx),
                "index": sample_index,
                "set_name": set_name,
                "mask_ratio": float(mask_ratio),
                "steps": 1,
                "alg": args.alg,
                "cfg_scale": float(args.cfg_scale),
                "gumbel_temperature": float(args.gumbel_temperature),
                "tail_len": int(args.tail_len),
                "diff_mask_id": int(args.diff_mask_id),
                "total_traj": int(args.total_traj),
                "traj_batch_size": int(args.traj_batch_size),
                "error": str(e),
            }

        if results_f is not None:
            results_f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if results_f is not None:
        results_f.close()

    # global summary
    if global_total_traj > 0:
        global_rate = global_exact_hits / float(global_total_traj)
        logging.info("===========================================")
        logging.info("GLOBAL exact hit rate: %.8f (%d/%d)", global_rate, global_exact_hits, global_total_traj)

        if global_hist_mask_correct:
            logging.info("GLOBAL hist_correct_mask_tokens (only for mask_count=%s):", str(global_mask_count_ref))
            keys_sorted = sorted(global_hist_mask_correct.keys())
            for k in keys_sorted:
                logging.info("  correct_mask_tokens=%d : %d", int(k), int(global_hist_mask_correct[k]))


        logging.info("GLOBAL hist grouped by mask_count:")
        for mc in sorted(global_hist_by_mask_count.keys()):
            c = global_hist_by_mask_count[mc]
            total = sum(c.values())
            exact = c.get(mc, 0)
            rate = exact / float(total) if total > 0 else float("nan")
            logging.info("  mask_count=%d | total_traj=%d | exact=%d | exact_rate=%.8f", mc, total, exact, rate)

    logging.info("Done.")


if __name__ == "__main__":
    main()
