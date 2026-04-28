
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file


# =========================
# Logger
# =========================
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


def _flush_logs():
    logger = logging.getLogger()
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass


# =========================
# Data loader (jsonl sequential + start/end line)
# =========================
def load_jsonl_samples(
    samples_path: Path,
    start_line: int = 1,
    end_line: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    start_line / end_line are 1-based, end_line is inclusive.
    """
    if start_line < 1:
        raise ValueError("--start_line must be >= 1 (1-based line index).")
    if end_line is not None and end_line < start_line:
        raise ValueError("--end_line must be >= start_line (inclusive).")

    samples: List[Dict[str, Any]] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if ln < start_line:
                continue
            if end_line is not None and ln > end_line:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_no"] = ln
            samples.append(obj)
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


# =========================
# Tokenizer helpers (match free_new behavior)
# =========================
def ensure_tokenizer_has_mask_id(tokenizer, mask_id: int, mask_token: str = "[MASK]") -> None:
    """
    Ensure tokenizer length >= mask_id+1.
    If tokenizer is shorter, add additional special tokens so that mask_token lands at id == mask_id.
    """
    cur_len = len(tokenizer)
    desired = int(mask_id) + 1
    if cur_len >= desired:
        return

    need = desired - cur_len
    # Add placeholders first, then [MASK] last so it ends up at exactly mask_id.
    pad_tokens = [f"[UNUSED_{i}]" for i in range(need - 1)]
    tokens_to_add = pad_tokens + [mask_token]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    new_len = len(tokenizer)
    logging.info("Tokenizer resized: %d -> %d (mask_id=%d)", cur_len, new_len, int(mask_id))


# =========================
# Checkpoint helpers (local lit_gpt model)
# =========================
def load_state_dict_local(ckpt_path: str) -> Dict[str, torch.Tensor]:
    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        return load_file(str(p))
    ckpt = torch.load(str(p), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # already a state_dict-like dict
    raise ValueError(f"Unrecognized checkpoint format at: {ckpt_path}")


def load_local_diff_model(
    lit_model_name: str,
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """
    Build local lit_gpt diffusion model (TransEncoder) and load weights.
    """
    model_name = f"Diff_LLaMA_{lit_model_name}M"
    config = Config.from_name(model_name)

    model = TransEncoder(config).to(device)
    sd = load_state_dict_local(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    logging.info("load_state_dict(strict=False): missing=%d unexpected=%d", len(missing), len(unexpected))
    if unexpected:
        logging.warning("Unexpected keys head: %s", unexpected[:20])
    if missing:
        logging.warning("Missing keys head: %s", missing[:20])

    # Move to dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Fix potential dtype mismatch for buffers (e.g., fused rotary buffers)
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    return model


# =========================
# Sampling utils (same behavior as free_new)
# =========================
def apply_temperature_topk(logits, temperature: float, top_k: int):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature != 1.0:
        logits = logits / temperature
    if torch.isnan(logits).any():
        logits = torch.where(torch.isnan(logits), -torch.inf, logits)
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, -torch.inf, logits)
    return logits


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        r = int(remainder[i].item())
        if r > 0:
            num_transfer_tokens[i, :r] += 1
    return num_transfer_tokens


def sample_token_from_log_probs(log_probs, torch_gen):
    probs = torch.exp(log_probs)
    if not torch.isfinite(probs).all():
        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        valid = torch.isfinite(log_probs)
        if valid.any():
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            pick = torch.randint(0, valid_idx.numel(), (1,), generator=torch_gen, device=log_probs.device)
            return int(valid_idx[pick].item())
        return int(torch.randint(0, log_probs.shape[0], (1,), generator=torch_gen, device=log_probs.device).item())
    probs = probs / total
    return int(torch.multinomial(probs, 1, generator=torch_gen).item())


def sample_with_random_order_free(
    model,
    x,
    mask_id,
    steps,
    torch_gen,
    temperature=1.0,
    top_k=0,
):
    """
    随机顺序解码，自由生成（不回填 ground truth）
    model(x) -> logits [bs, seq, vocab]
    """
    if steps <= 0:
        return x
    mask_index = x == mask_id
    if not mask_index.any():
        return x
    max_mask = int(mask_index.sum(dim=1).max().item())
    steps = min(steps, max_mask)
    num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)

    device = x.device
    model_dtype = next(model.parameters()).dtype
    use_amp = (device.type == "cuda") and (model_dtype in (torch.float16, torch.bfloat16))

    for i in range(steps):
        mask_index = x == mask_id
        if not mask_index.any():
            break

        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=model_dtype, enabled=use_amp):
                    logits = model(x)
            else:
                logits = model(x)

        logits = apply_temperature_topk(logits, temperature=temperature, top_k=top_k)
        log_probs = F.log_softmax(logits, dim=-1)

        rand_scores = torch.rand(x.shape, generator=torch_gen, device=device)
        rand_scores = torch.where(mask_index, rand_scores, -torch.inf)

        for b in range(x.shape[0]):
            k = int(num_transfer_tokens[b, i].item())
            if k <= 0:
                continue
            _, select_index = torch.topk(rand_scores[b], k=k)
            for pos in select_index.tolist():
                sampled_token = sample_token_from_log_probs(log_probs[b, pos], torch_gen)
                x[b, pos] = sampled_token

    return x


def build_random_mask_positions(seq_len, mask_count, cur_bs, torch_gen, device):
    scores = torch.rand(cur_bs, seq_len, generator=torch_gen, device=device)
    _, mask_pos = torch.topk(scores, k=mask_count, dim=1)
    return mask_pos


def count_hits(gt_tokens, gen_tokens, mask_positions):
    hits = 0
    for p in mask_positions:
        if gen_tokens[p] == gt_tokens[p]:
            hits += 1
    return hits


# =========================
# Core evaluation (same outputs as free_new)
# =========================
def diffusion_free_generate(
    model,
    tail_tokens: List[int],
    mask_id: int,
    mask_ratio: float,
    mc_samples: int,
    mc_batch_size: int,
    seed: int,
    temperature: float,
    top_k: int,
    diffusion_mode: str = "both",  # "both" | "gen1" | "genstep"
    progress_interval: int = 1,
):
    """
    diffusion_mode:
      - "both": run gen1 and genstep
      - "gen1": run only one-step generation
      - "genstep": run only multi-step generation (steps=mask_count)
    Outputs aligned to free_new.py.
    """
    if diffusion_mode not in ("both", "gen1", "genstep"):
        raise ValueError(f"Unknown diffusion_mode={diffusion_mode}, expected one of: both/gen1/genstep")

    run_gen1 = diffusion_mode in ("both", "gen1")
    run_steps = diffusion_mode in ("both", "genstep")

    seq_len = len(tail_tokens)
    mask_count = int(seq_len * float(mask_ratio))
    if mask_count <= 0 or mask_count > seq_len:
        raise ValueError(f"invalid mask_ratio={mask_ratio} for seq_len={seq_len}")

    device = next(model.parameters()).device
    base = torch.tensor(tail_tokens, dtype=torch.long, device=device).unsqueeze(0)

    gen1_hits = [] if run_gen1 else None
    gen_steps_hits = [] if run_steps else None

    mask_gen = torch.Generator(device=device)
    mask_gen.manual_seed(seed + 100)

    gen1_gen = None
    gen_steps_gen = None
    if run_gen1:
        gen1_gen = torch.Generator(device=device)
        gen1_gen.manual_seed(seed + 200)
    if run_steps:
        gen_steps_gen = torch.Generator(device=device)
        gen_steps_gen.manual_seed(seed + 300)

    total_batches = (mc_samples + mc_batch_size - 1) // mc_batch_size
    t0 = time.time()

    with torch.no_grad():
        for bi, start in enumerate(range(0, mc_samples, mc_batch_size), start=1):
            cur_bs = min(mc_batch_size, mc_samples - start)

            if (bi % progress_interval == 0) or (bi == 1):
                elapsed = time.time() - t0
                done_so_far = start
                pct = 100.0 * done_so_far / mc_samples if mc_samples > 0 else 0.0
                logging.info(
                    "[diffusion] batch %d/%d START | done=%d/%d (%.2f%%) | elapsed=%.1fs",
                    bi, total_batches, done_so_far, mc_samples, pct, elapsed
                )
                _flush_logs()

            batch = base.repeat(cur_bs, 1)

            mask_pos = build_random_mask_positions(seq_len, mask_count, cur_bs, mask_gen, device)
            mask = torch.zeros(cur_bs, seq_len, dtype=torch.bool, device=device)
            mask.scatter_(1, mask_pos, True)

            masked_batch = batch.clone()
            masked_batch[mask] = int(mask_id)

            gen1_tokens = None
            gen_steps_tokens = None

            if run_gen1:
                gen1_tokens = sample_with_random_order_free(
                    model, masked_batch.clone(),
                    mask_id=int(mask_id),
                    steps=1,
                    torch_gen=gen1_gen,
                    temperature=temperature,
                    top_k=top_k,
                )

            if run_steps:
                gen_steps_tokens = sample_with_random_order_free(
                    model, masked_batch.clone(),
                    mask_id=int(mask_id),
                    steps=mask_count,
                    torch_gen=gen_steps_gen,
                    temperature=temperature,
                    top_k=top_k,
                )

            for row in range(cur_bs):
                positions = mask_pos[row].detach().cpu().tolist()

                if run_gen1:
                    gen1_tail = gen1_tokens[row].detach().cpu().tolist()
                    gen1_hits.append(count_hits(tail_tokens, gen1_tail, positions))

                if run_steps:
                    gen_steps_tail = gen_steps_tokens[row].detach().cpu().tolist()
                    gen_steps_hits.append(count_hits(tail_tokens, gen_steps_tail, positions))

            done = start + cur_bs
            if (bi % progress_interval == 0) or (done >= mc_samples):
                elapsed = time.time() - t0
                it_s = done / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * done / mc_samples if mc_samples > 0 else 0.0
                logging.info(
                    "[diffusion] batch %d/%d END   | done=%d/%d (%.2f%%) | %.2f it/s",
                    bi, total_batches, done, mc_samples, pct, it_s
                )
                _flush_logs()

    out = {
        "seq_len": seq_len,
        "mask_count": mask_count,
        "mc_samples": mc_samples,
        "diffusion_mode": diffusion_mode,
    }

    if run_steps:
        total_steps_hits = int(sum(gen_steps_hits))
        hit_rate = total_steps_hits / float(mask_count * mc_samples)
        all_correct_rate = sum(h == mask_count for h in gen_steps_hits) / float(mc_samples)
        out.update({
            "gen_steps_hits": gen_steps_hits,
            "hit_rate": hit_rate,
            "all_correct_rate": all_correct_rate,
        })

    if run_gen1:
        total_gen1_hits = int(sum(gen1_hits))
        gen1_hit_rate = total_gen1_hits / float(mask_count * mc_samples)
        gen1_all_correct_rate = sum(h == mask_count for h in gen1_hits) / float(mc_samples)
        out.update({
            "gen1_hits": gen1_hits,
            "gen1_hit_rate": gen1_hit_rate,
            "gen1_all_correct_rate": gen1_all_correct_rate,
        })

    return out


def prefix_suffix_free_generate(
    model,
    tail_tokens: List[int],
    mask_id: int,
    prompt_len: int,
    mc_samples: int,
    mc_batch_size: int,
    seed: int,
    temperature: float,
    top_k: int,
):
    """
    Prefix visible, suffix masked (positions prompt_len..end)
    Record hit count each run + hit_rate
    Outputs aligned to free_new.py.
    """
    seq_len = len(tail_tokens)
    if seq_len < prompt_len:
        raise ValueError(f"sequence too short: {seq_len} < prompt_len={prompt_len}")

    mask_positions = list(range(prompt_len, seq_len))
    mask_count = len(mask_positions)

    device = next(model.parameters()).device
    base = torch.tensor(tail_tokens, dtype=torch.long, device=device).unsqueeze(0)

    hits: List[int] = []

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 500)

    with torch.no_grad():
        for start in range(0, mc_samples, mc_batch_size):
            cur_bs = min(mc_batch_size, mc_samples - start)
            batch = base.repeat(cur_bs, 1)

            mask = torch.zeros(cur_bs, seq_len, dtype=torch.bool, device=device)
            mask[:, prompt_len:] = True

            masked_batch = batch.clone()
            masked_batch[mask] = int(mask_id)

            gen_tokens = sample_with_random_order_free(
                model,
                masked_batch.clone(),
                mask_id=int(mask_id),
                steps=mask_count,
                torch_gen=gen,
                temperature=temperature,
                top_k=top_k,
            )

            for row in range(cur_bs):
                gen_tail = gen_tokens[row].detach().cpu().tolist()
                gen_mask = gen_tail[prompt_len:]
                gt_mask = tail_tokens[prompt_len:]
                hit_count = sum(1 for a, b in zip(gen_mask, gt_mask) if a == b)
                hits.append(hit_count)

    total_hits = int(sum(hits))
    hit_rate = total_hits / float(mask_count * mc_samples)

    return {
        "seq_len": seq_len,
        "mask_count": mask_count,
        "mc_samples": mc_samples,
        "hits": hits,
        "hit_rate": hit_rate,
    }


# =========================
# Main (CLI aligned to free_new + local model args)
# =========================
def parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp16", "float16"]:
        return torch.float16
    if s in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s} (use bf16/fp16/fp32)")


def main():
    parser = argparse.ArgumentParser()

    # Data (same as free_new)
    parser.add_argument("--samples_path", type=Path, required=True, help="Your new jsonl file")
    parser.add_argument("--start_line", type=int, default=1, help="1-based start line (inclusive)")
    parser.add_argument("--end_line", type=int, default=None, help="1-based end line (inclusive)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap after slicing start/end")

    # Tokenizer name (keep free_new arg name)
    parser.add_argument("--hf_model_name", type=str, required=True, help="HF tokenizer name/path")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable if tokenizer needs it")

    # Local lit_gpt diffusion model (extra arg, but model loading remains local)
    parser.add_argument(
        "--lit_model_name",
        type=str,
        default="1028",
        help="Local lit_gpt model size tag, used as Diff_LLaMA_{lit_model_name}M",
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="Local diffusion model checkpoint (.pth/.pt/.safetensors)")

    # Generation (same defaults as free_new; keep mask_id=32000)
    parser.add_argument("--diff_mask_id", type=int, default=32000)
    parser.add_argument("--tail_len", type=int, default=100)
    parser.add_argument("--mc_samples", type=int, default=10000)
    parser.add_argument("--mc_batch_size", type=int, default=512)

    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_k", type=int, default=40)

    # NEW: control diffusion generation mode (same as free_new)
    parser.add_argument(
        "--diffusion_mode",
        type=str,
        default="both",
        choices=["both", "gen1", "genstep"],
        help="Diffusion generation mode: both (gen1+genstep), gen1 (one-step only), genstep (multi-step only)",
    )

    parser.add_argument("--run_prefix_suffix", action="store_true")
    parser.add_argument("--prompt_len", type=int, default=50)

    # Runtime (same as free_new)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16/fp16/fp32")

    # Output (same as free_new)
    parser.add_argument("--output_path", type=Path, required=True, help="Write jsonl results here")
    parser.add_argument("--log_path", type=Path, default=None)

    args = parser.parse_args()

    setup_logger(args.log_path)
    logging.info("Args: %s", vars(args))

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    # Load tokenizer (HF), ensure mask_id exists
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    #ensure_tokenizer_has_mask_id(tokenizer, mask_id=args.diff_mask_id, mask_token="[MASK]")

    # Load local diffusion model (lit_gpt)
    model = load_local_diff_model(
        lit_model_name=args.lit_model_name,
        ckpt_path=args.ckpt_path,
        device=device,
        dtype=dtype,
    )
    logging.info("Local diffusion model loaded: Diff_LLaMA_%sM | dtype=%s | device=%s", args.lit_model_name, str(dtype), str(device))

    # Basic sanity check for mask_id range (best-effort)
    try:
        # Many models expose embedding tables differently; this is a best-effort probe.
        # If your TransEncoder exposes vocab size elsewhere, adjust here.
        pass
    except Exception:
        pass

    # Load data
    if not args.samples_path.exists():
        raise FileNotFoundError(f"samples_path not found: {args.samples_path}")
    samples = load_jsonl_samples(
        args.samples_path,
        start_line=args.start_line,
        end_line=args.end_line,
        max_samples=args.max_samples,
    )
    logging.info(
        "Loaded %d lines from %s (start=%s end=%s)",
        len(samples), args.samples_path, args.start_line, args.end_line
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for si, sample in enumerate(samples):
            line_no = sample.get("_line_no", None)
            index = sample.get("index", line_no if line_no is not None else si)
            set_name = sample.get("set_name", None)
            mask_ratio = float(sample.get("mask_ratio", 0.15))

            # Align with free_new: prefer "text"; fallback to "tokens" (best-effort)
            text = sample.get("text", None)
            token_ids: Optional[List[int]] = None

            if isinstance(text, str) and len(text.strip()) > 0:
                token_ids = tokenizer(text, add_special_tokens=False).input_ids
            else:
                # fallback: if user jsonl still provides tokens
                toks = sample.get("tokens", None)
                if isinstance(toks, list) and len(toks) > 0 and all(isinstance(t, int) for t in toks):
                    token_ids = toks
                else:
                    logging.warning("Skip empty sample. index=%s line=%s (no valid text/tokens)", str(index), str(line_no))
                    continue

            if len(token_ids) < args.tail_len:
                logging.warning(
                    "Skip too short sample. index=%s tokens=%d < tail_len=%d",
                    str(index), len(token_ids), args.tail_len
                )
                continue

            tail_tokens = token_ids[-args.tail_len:]

            logging.info(
                "Sample index=%s line=%s set=%s tokens=%d mask_ratio=%.4f",
                str(index), str(line_no), str(set_name), len(token_ids), mask_ratio
            )

            # Run diffusion eval
            t0 = time.time()
            diff_stats = diffusion_free_generate(
                model=model,
                tail_tokens=tail_tokens,
                mask_id=args.diff_mask_id,
                mask_ratio=mask_ratio,
                mc_samples=args.mc_samples,
                mc_batch_size=args.mc_batch_size,
                seed=1234 + int(si),
                temperature=args.gen_temperature,
                top_k=args.gen_top_k,
                diffusion_mode=args.diffusion_mode,
            )

            if args.diffusion_mode in ("both", "genstep"):
                logging.info(
                    "Diffusion done in %.2fs | hit_rate=%.6f",
                    time.time() - t0, diff_stats["hit_rate"]
                )
            else:
                logging.info(
                    "Diffusion done in %.2fs | gen1_hit_rate=%.6f",
                    time.time() - t0, diff_stats["gen1_hit_rate"]
                )

            # Write output (only include fields that were actually computed)
            diffusion_out = {
                "mask_count": diff_stats["mask_count"],
                "mc_samples": diff_stats["mc_samples"],
                "diffusion_mode": diff_stats["diffusion_mode"],
            }
            if "gen1_hits" in diff_stats:
                diffusion_out.update({
                    "gen1_hits": diff_stats["gen1_hits"],
                    "gen1_hit_rate": diff_stats["gen1_hit_rate"],
                    "gen1_all_correct_rate": diff_stats.get("gen1_all_correct_rate", None),
                })
            if "gen_steps_hits" in diff_stats:
                diffusion_out.update({
                    "gen_steps_hits": diff_stats["gen_steps_hits"],
                    "hit_rate": diff_stats["hit_rate"],
                    "all_correct_rate": diff_stats.get("all_correct_rate", None),
                })

            out_obj = {
                "index": index,
                "set_name": set_name,
                "line_no": line_no,
                "mask_ratio": mask_ratio,
                "tail_len": args.tail_len,
                "diffusion": diffusion_out,
            }

            # Optional prefix-suffix eval
            if args.run_prefix_suffix:
                t1 = time.time()
                ps_stats = prefix_suffix_free_generate(
                    model=model,
                    tail_tokens=tail_tokens,
                    mask_id=args.diff_mask_id,
                    prompt_len=args.prompt_len,
                    mc_samples=args.mc_samples,
                    mc_batch_size=args.mc_batch_size,
                    seed=5678 + int(si),
                    temperature=args.gen_temperature,
                    top_k=args.gen_top_k,
                )
                logging.info("Prefix-suffix done in %.2fs | hit_rate=%.6f", time.time() - t1, ps_stats["hit_rate"])
                out_obj["prefix_suffix"] = ps_stats

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            out_f.flush()

    logging.info("All done. Results saved to: %s", str(args.output_path))


if __name__ == "__main__":
    main()
