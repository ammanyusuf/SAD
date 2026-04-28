import argparse
import json
import logging
import math
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from lit_gpt.diffmodel import TransEncoder, Config
from safetensors.torch import load_file

TAIL_LEN = 30


def add_timestamp_to_path(path: Path, ts: Optional[str] = None) -> Path:
    if ts is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def add_run_tags_to_path(path: Path, ts: str, mc_samples: int) -> Path:
    return path.with_name(f"{path.stem}_{mc_samples}_{ts}{path.suffix}")


def setup_logger(log_path: Optional[Path]):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def load_samples(
    samples_path: Path,
    max_samples: int,
    shuffle: bool,
    seed: int,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
):
    """
    Load samples from a JSONL file with optional 1-based inclusive line slicing:
      - start_line=None means start from line 1
      - end_line=None means read to EOF
    Shuffling and max_samples truncation are applied after slicing.
    """
    if start_line is not None and start_line <= 0:
        raise ValueError(f"start_line must be >= 1, got {start_line}")
    if end_line is not None and end_line <= 0:
        raise ValueError(f"end_line must be >= 1, got {end_line}")
    if start_line is not None and end_line is not None and start_line > end_line:
        raise ValueError(f"start_line ({start_line}) > end_line ({end_line})")

    s = start_line if start_line is not None else 1
    e = end_line  # None => EOF

    samples = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if lineno < s:
                continue
            if e is not None and lineno > e:
                break
            if line.strip():
                samples.append(json.loads(line))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples[:max_samples]


def decode_tokens_with_mask(tokenizer, token_ids, mask_id, mask_token="[MASK]"):
    pieces = []
    for tid in token_ids:
        if tid == mask_id:
            pieces.append(mask_token)
        else:
            pieces.append(tokenizer.decode([tid], skip_special_tokens=False))
    return "".join(pieces)


def decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def parse_float_list(value: str):
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str):
    return [int(x) for x in value.split(",") if x.strip()]


def summarize_finite(values):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {
            "mean": float("nan"),
            "stdev": 0.0,
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "mean": statistics.fmean(finite),
        "stdev": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
        "count": len(finite),
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


def load_state_dict_local(ckpt_path: str):
    """
    Load weights from either .safetensors or PyTorch checkpoint formats (.pth/.pt).
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


def _rand(shape, *, device, dtype, generator: Optional[torch.Generator] = None):
    return torch.rand(shape, device=device, dtype=dtype, generator=generator)


def add_gumbel_noise(logits: torch.Tensor, temperature: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    logits = logits.to(torch.float64)
    noise = _rand(logits.shape, device=logits.device, dtype=torch.float64, generator=generator)
    noise = noise.clamp_min(1e-12)
    gumbel_noise = (-torch.log(noise)) ** float(temperature)
    return logits.exp() / gumbel_noise


def gt_logprob_under_gumbel_sampling(logits_fp64: torch.Tensor, gt_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Under Gumbel-max sampling, the induced categorical distribution is softmax(logits/temperature).
    This returns the log-probability of the ground-truth token under that distribution.
    """
    if temperature <= 0:
        raise ValueError("gumbel_temperature must be > 0")
    scaled = logits_fp64 / float(temperature)
    log_probs = F.log_softmax(scaled, dim=-1)
    return log_probs.gather(1, gt_ids.unsqueeze(1)).squeeze(1)


def recover_logprob_from_masked_batch(
    model: torch.nn.Module,
    x: torch.Tensor,            # [B, L] masked input, using mask_id for masked tokens
    gt: torch.Tensor,           # [B, L] ground-truth token ids
    mask_id: int,
    steps: int,
    alg: str,                   # "origin" | "greddy"
    temperature: float,
    eps: float,
    cfg_scale: float = 0.0,
    prompt_len: int = 0,
    torch_gen: Optional[torch.Generator] = None,
    debug_log_rank: bool = False,
    debug_log_rank_max_positions: int = 5,
    log_steps: bool = False,
    log_steps_max_samples: int = 1,
    log_steps_label: str = "",
):
    """
    Iteratively "recover" masked tokens and accumulate the ground-truth log-probabilities for filled positions.

    Returns:
      x_out: [B, L] (teacher-forced reconstruction)
      total_log: [B] float64
      invalid_mask: [B] bool
    """
    if steps <= 0:
        invalid = torch.zeros((x.shape[0],), device=x.device, dtype=torch.bool)
        return x, torch.zeros((x.shape[0],), device=x.device, dtype=torch.float64), invalid

    B, L = x.shape
    device = x.device

    # Compute the initial timestep from the current masked fraction.
    with torch.no_grad():
        num_mask_row = (x == mask_id).sum(dim=1).to(torch.int64)  # [B]
        mask_count_float = num_mask_row.to(torch.float64).mean().item()
        p0 = float(mask_count_float) / float(L) if L > 0 else 0.0
        t0 = (p0 - eps) / (1.0 - eps) if (1.0 - eps) != 0 else 1.0
        if not math.isfinite(t0):
            t0 = 1.0
        t0 = max(min(t0, 1.0), eps)

    timesteps = torch.linspace(t0, eps, steps + 1, device=device)
    total_log = torch.zeros((B,), device=device, dtype=torch.float64)
    invalid_mask = torch.zeros((B,), device=device, dtype=torch.bool)

    use_amp = (device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == mask_id)  # [B, L]
            if int(mask_index.sum().item()) == 0:
                break

            with amp_ctx:
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if prompt_len > 0:
                        un_x[:, :prompt_len] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)  # [2B, L]
                    logits_all = model(x_)            # [2B, L, V]
                    logits_c, logits_u = torch.chunk(logits_all, 2, dim=0)
                    logits_full = logits_u + (cfg_scale + 1.0) * (logits_c - logits_u)  # [B, L, V]
                else:
                    logits_full = model(x)            # [B, L, V]

            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1.0 - (s / t).item() if i < steps - 1 else 1.0
                r = _rand((B, L), device=device, dtype=torch.float32, generator=torch_gen)
                transfer_pos_mask = (r < p_transfer) & mask_index  # [B, L]

                b_idx, pos_idx = transfer_pos_mask.nonzero(as_tuple=True)
                if b_idx.numel() > 0:
                    sel_logits = logits_full[b_idx, pos_idx, :].to(torch.float64)  # [N, V]
                    sel_gt = gt[b_idx, pos_idx]                                     # [N]
                    sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)

                    finite = torch.isfinite(sel_lp)
                    if not finite.all():
                        bad_b = b_idx[~finite]
                        invalid_mask[bad_b] = True
                        sel_lp = torch.where(finite, sel_lp, torch.zeros_like(sel_lp))

                    total_log.index_add_(0, b_idx, sel_lp)

                    good = finite
                    if good.any():
                        x[b_idx[good], pos_idx[good]] = sel_gt[good]

                    if log_steps:
                        for j in range(min(B, log_steps_max_samples)):
                            if j == 0:
                                sel_pos_j = pos_idx[b_idx == j].detach().cpu().tolist()
                                sel_tok_j = gt[j, sel_pos_j].detach().cpu().tolist() if sel_pos_j else []
                                logging.info(
                                    "%s step=%d origin transfer_count=%d pos=%s token_ids=%s",
                                    log_steps_label, i, len(sel_pos_j), sel_pos_j[:50], sel_tok_j[:50],
                                )

                    if debug_log_rank:
                        for j in range(min(B, log_steps_max_samples)):
                            sel_pos_j = pos_idx[b_idx == j]
                            if sel_pos_j.numel() == 0:
                                continue
                            sel_pos_j = sel_pos_j[: max(0, debug_log_rank_max_positions)]
                            ranks = []
                            for p_ in sel_pos_j.tolist():
                                tid = int(gt[j, p_].item())
                                gt_logit = logits_full[j, p_, tid]
                                rank = int((logits_full[j, p_] > gt_logit).sum().item()) + 1
                                ranks.append(rank)
                            logging.info(
                                "%s step=%d origin ranks(sample=%d) pos=%s ranks=%s",
                                log_steps_label, i, j, sel_pos_j.detach().cpu().tolist(), ranks,
                            )

            elif alg == "greddy":
                logits_masked = logits_full[mask_index]  # [Nmask_total, V]
                logits_with_noise = add_gumbel_noise(logits_masked, temperature=temperature, generator=torch_gen)
                x0_masked = torch.argmax(logits_with_noise, dim=-1)  # [Nmask_total]

                logits_masked_fp64 = logits_masked.to(torch.float64)
                p = F.softmax(logits_masked_fp64, dim=-1)
                confidence_masked = torch.gather(p, dim=-1, index=x0_masked.unsqueeze(-1)).squeeze(-1)

                confidence_full = torch.full((B, L), float("-inf"), device=device, dtype=torch.float64)
                confidence_full[mask_index] = confidence_masked

                num_mask = mask_index.sum(dim=1).to(torch.int64)
                if i < steps - 1:
                    frac = (1.0 - (s / t).item())
                    k = torch.floor(num_mask.to(torch.float64) * float(frac)).to(torch.int64)
                else:
                    k = num_mask

                max_k = int(k.max().item())
                if max_k <= 0:
                    continue

                _, top_pos = torch.topk(confidence_full, k=max_k, dim=1)

                ar = torch.arange(max_k, device=device).unsqueeze(0)
                take = ar < k.unsqueeze(1)

                b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, max_k)
                sel_b = b_idx[take]
                sel_pos = top_pos[take]

                if sel_b.numel() == 0:
                    continue

                sel_logits = logits_full[sel_b, sel_pos, :].to(torch.float64)
                sel_gt = gt[sel_b, sel_pos]
                sel_lp = gt_logprob_under_gumbel_sampling(sel_logits, sel_gt, temperature=temperature)

                finite = torch.isfinite(sel_lp)
                if not finite.all():
                    bad_b = sel_b[~finite]
                    invalid_mask[bad_b] = True
                    sel_lp = torch.where(finite, sel_lp, torch.zeros_like(sel_lp))

                total_log.index_add_(0, sel_b, sel_lp)

                good = finite
                if good.any():
                    x[sel_b[good], sel_pos[good]] = sel_gt[good]

                if log_steps:
                    for j in range(min(B, log_steps_max_samples)):
                        pos_j = sel_pos[sel_b == j].detach().cpu().tolist()
                        tok_j = gt[j, pos_j].detach().cpu().tolist() if pos_j else []
                        logging.info(
                            "%s step=%d greddy transfer_count=%d pos=%s token_ids=%s",
                            log_steps_label, i, len(pos_j), pos_j[:50], tok_j[:50],
                        )

                if debug_log_rank:
                    for j in range(min(B, log_steps_max_samples)):
                        pos_j = sel_pos[sel_b == j]
                        if pos_j.numel() == 0:
                            continue
                        pos_j = pos_j[: max(0, debug_log_rank_max_positions)]
                        ranks = []
                        for p_ in pos_j.tolist():
                            tid = int(gt[j, p_].item())
                            gt_logit = logits_full[j, p_, tid]
                            rank = int((logits_full[j, p_] > gt_logit).sum().item()) + 1
                            ranks.append(rank)
                        logging.info(
                            "%s step=%d greddy ranks(sample=%d) pos=%s ranks=%s",
                            log_steps_label, i, j, pos_j.detach().cpu().tolist(), ranks,
                        )

            else:
                raise NotImplementedError(f"Unknown alg={alg}")

    total_log = torch.where(invalid_mask, torch.full_like(total_log, float("nan")), total_log)
    return x, total_log, invalid_mask


def diffusion_loglikelihood(
    model,
    tokenizer,
    token_ids,
    mask_id: int,
    mask_ratio: float,
    mc_samples: int,
    mc_batch_size: int,
    seed: int,
    debug_mc_samples: int,
    debug_log_masks: bool,
    debug_log_masked_text: bool,
    debug_masked_text_mc_samples: int,
    debug_log_shapes: bool,
    debug_log_rank: bool,
    debug_log_rank_max_positions: int,
    gen_eachstep: bool,
    gen_temperature: float,
    gen_top_k: int,  # retained for CLI compatibility (not used)
    recover_alg: str,
    recover_eps: float,
    recover_cfg_scale: float,
    only_gen1: bool,
    p_targets,
    n_targets,
    log_generated_text: bool,
    log_generated_text_mc_samples: int,
):
    token_ids = token_ids[-TAIL_LEN:]
    seq_len = len(token_ids)
    if seq_len < TAIL_LEN:
        raise ValueError(f"sequence too short: {seq_len}")

    device = next(model.parameters()).device

    base = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    mask_count = int(seq_len * mask_ratio)
    if mask_count <= 0 or mask_count > seq_len:
        raise ValueError(f"invalid mask_ratio={mask_ratio} for seq_len={seq_len}")

    gen_steps = mask_count if gen_eachstep else 1

    if gen_top_k and gen_top_k > 0:
        logging.warning("gen_top_k=%d is ignored (full softmax is used).", gen_top_k)

    masked_ll_log_sums = []
    gen1_total_logs = []
    gen_steps_total_logs = []
    gen1_invalid_mask = []
    gen_steps_invalid_mask = []

    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    recover_gen = torch.Generator(device=device)
    recover_gen.manual_seed(seed + 1000)

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        mc_index = 0
        gen_text_index = 0

        for start in range(0, mc_samples, mc_batch_size):
            cur_bs = min(mc_batch_size, mc_samples - start)
            batch = base.repeat(cur_bs, 1)  # GT [B, L]

            # Independent masks per trajectory, with exactly mask_count positions per row.
            scores = torch.rand(cur_bs, seq_len, generator=torch_gen, device=device)
            _, mask_pos = torch.topk(scores, k=mask_count, dim=1)
            mask = torch.zeros(cur_bs, seq_len, dtype=torch.bool, device=device)
            mask.scatter_(1, mask_pos, True)

            masked_batch = batch.clone()
            masked_batch[mask] = mask_id

            # Masked-token log-likelihood under the model.
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                logits = model(masked_batch)
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, batch.unsqueeze(-1)).squeeze(-1)
            batch_log_sums = (gathered * mask).sum(dim=1)
            masked_ll_log_sums.extend(batch_log_sums.detach().cpu().tolist())

            prompt_len = seq_len - mask_count

            # Single-step recovery (teacher-forced).
            gen1_tokens, gen1_logs, gen1_invalid = recover_logprob_from_masked_batch(
                model=model,
                x=masked_batch.clone(),
                gt=batch,
                mask_id=mask_id,
                steps=1,
                alg=recover_alg,
                temperature=gen_temperature,
                eps=recover_eps,
                cfg_scale=recover_cfg_scale,
                prompt_len=prompt_len,
                torch_gen=recover_gen,
                debug_log_rank=debug_log_rank,
                debug_log_rank_max_positions=debug_log_rank_max_positions,
                log_steps=log_generated_text,
                log_steps_max_samples=log_generated_text_mc_samples,
                log_steps_label="Diffusion Recover Gen1",
            )
            gen1_total_logs.extend(gen1_logs.detach().cpu().tolist())
            gen1_invalid_mask.extend(gen1_invalid.detach().cpu().tolist())

            # Multi-step recovery, optionally skipped.
            if not only_gen1:
                gen_steps_tokens, gen_steps_logs, gen_steps_invalid = recover_logprob_from_masked_batch(
                    model=model,
                    x=masked_batch.clone(),
                    gt=batch,
                    mask_id=mask_id,
                    steps=gen_steps,
                    alg=recover_alg,
                    temperature=gen_temperature,
                    eps=recover_eps,
                    cfg_scale=recover_cfg_scale,
                    prompt_len=prompt_len,
                    torch_gen=recover_gen,
                    debug_log_rank=debug_log_rank,
                    debug_log_rank_max_positions=debug_log_rank_max_positions,
                    log_steps=log_generated_text,
                    log_steps_max_samples=log_generated_text_mc_samples,
                    log_steps_label="Diffusion Recover GenSteps",
                )
                gen_steps_total_logs.extend(gen_steps_logs.detach().cpu().tolist())
                gen_steps_invalid_mask.extend(gen_steps_invalid.detach().cpu().tolist())
            else:
                gen_steps_tokens = None

            # Teacher forcing reconstructs GT; text logging is for diagnostics only.
            if log_generated_text:
                for row in range(cur_bs):
                    if gen_text_index >= log_generated_text_mc_samples:
                        break
                    gen1_text = decode_tokens(tokenizer, gen1_tokens[row].detach().cpu().tolist())
                    logging.info("Recover Gen1 text=%s", gen1_text)

                    if (not only_gen1) and gen_steps_tokens is not None:
                        gen_steps_text = decode_tokens(tokenizer, gen_steps_tokens[row].detach().cpu().tolist())
                        logging.info("Recover GenSteps text=%s", gen_steps_text)

                    gen_text_index += 1

            if debug_mc_samples > 0:
                for row in range(cur_bs):
                    if mc_index >= debug_mc_samples:
                        break
                    if debug_log_masks:
                        logging.info(
                            "Diffusion MC[%d] mask_pos=%s",
                            mc_index,
                            mask_pos[row].detach().cpu().tolist(),
                        )
                    if debug_log_masked_text and mc_index < debug_masked_text_mc_samples:
                        masked_tokens = masked_batch[row].detach().cpu().tolist()
                        masked_text = decode_tokens_with_mask(tokenizer, masked_tokens, mask_id)
                        logging.info(
                            "Diffusion MC[%d] masked_input_text=%s",
                            mc_index,
                            masked_text,
                        )
                    logging.info(
                        "Diffusion MC[%d] masked_ll_log_sum=%.4f",
                        mc_index,
                        batch_log_sums[row].item(),
                    )
                    mc_index += 1

    masked_stats = summarize_finite(masked_ll_log_sums)
    gen1_stats = summarize_finite(gen1_total_logs)
    gen_steps_stats = summarize_finite(gen_steps_total_logs)

    gen1_np = compute_np_stats(gen1_total_logs, p_targets, n_targets)
    gen_steps_np = compute_np_stats(gen_steps_total_logs, p_targets, n_targets)

    return {
        "log_sums": masked_ll_log_sums,
        "mean": masked_stats["mean"],
        "stdev": masked_stats["stdev"],
        "min": masked_stats["min"],
        "max": masked_stats["max"],
        "gen1_log_sums": gen1_total_logs,
        "gen1_hit_log_mean": gen1_stats["mean"],
        "gen1_hit_log_stdev": gen1_stats["stdev"],
        "gen1_hit_log_min": gen1_stats["min"],
        "gen1_hit_log_max": gen1_stats["max"],
        "gen1_processable_count": gen1_np["processable_count"],
        "gen1_p_hat": gen1_np["p_hat"],
        "gen1_n_for_p": gen1_np["n_for_p"],
        "gen1_p_for_n": gen1_np["p_for_n"],
        "gen_steps_log_sums": gen_steps_total_logs,
        "gen_steps_hit_log_mean": gen_steps_stats["mean"],
        "gen_steps_hit_log_stdev": gen_steps_stats["stdev"],
        "gen_steps_hit_log_min": gen_steps_stats["min"],
        "gen_steps_hit_log_max": gen_steps_stats["max"],
        "gen_steps_processable_count": gen_steps_np["processable_count"],
        "gen_steps_p_hat": gen_steps_np["p_hat"],
        "gen_steps_n_for_p": gen_steps_np["n_for_p"],
        "gen_steps_p_for_n": gen_steps_np["p_for_n"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_path",
        type=Path,
        default=Path("smdm/data/unique_bin_samples.jsonl"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("."))
    parser.add_argument("--dataset_name", type=str, default="")

    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--start_line", type=int, default=None, help="1-based start line (inclusive)")
    parser.add_argument("--end_line", type=int, default=None, help="1-based end line (inclusive)")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="smdm/workdir/scaling_debug/mdm-1028M-100.0/final.pth",
    )
    parser.add_argument("--lit_model_name", type=str, default="1028")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    )

    parser.add_argument("--diff_mask_id", type=int, default=32000)
    parser.add_argument("--diff_mc_samples", type=int, default=128)
    parser.add_argument("--diff_mc_batch_size", type=int, default=16)
    parser.add_argument("--mask_rates", type=str, default="0.05,0.1,0.2,0.3,0.5")

    parser.add_argument("--debug_mc_samples", type=int, default=3)
    parser.add_argument("--debug_log_masks", action="store_true")
    parser.add_argument("--debug_log_tokens", action="store_true")
    parser.add_argument("--debug_log_ground_truth", action="store_true")
    parser.add_argument("--debug_log_masked_text", action="store_true")
    parser.add_argument("--debug_masked_text_mc_samples", type=int, default=1)

    parser.add_argument("--gen_eachstep", action="store_true")
    parser.add_argument("--tail_len", type=int, default=30)

    parser.add_argument("--log_generated_text", action="store_true")
    parser.add_argument("--log_generated_text_mc_samples", type=int, default=1)

    parser.add_argument("--debug_log_shapes", action="store_true")
    parser.add_argument("--debug_log_rank", action="store_true")
    parser.add_argument("--debug_log_rank_max_positions", type=int, default=5)

    parser.add_argument("--gen_temperature", type=float, default=1.0)
    parser.add_argument("--gen_top_k", type=int, default=40)

    parser.add_argument("--recover_alg", type=str, default="origin", choices=["origin", "greddy"])
    parser.add_argument("--recover_eps", type=float, default=1e-3)
    parser.add_argument("--recover_cfg_scale", type=float, default=0.0)

    parser.add_argument("--only_gen1", action="store_true")

    parser.add_argument("--p_targets", type=str, default="0.1,0.5,0.9,0.99")
    parser.add_argument("--n_targets", type=str, default="1,10,100")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--log_path", type=Path, default=None)
    args = parser.parse_args()

    args.mask_rates = parse_float_list(args.mask_rates)
    args.p_targets = parse_float_list(args.p_targets)
    args.n_targets = parse_int_list(args.n_targets)

    run_ts_file = time.strftime("%Y%m%d_%H%M%S")
    if args.log_path is not None and args.log_path.name == "memo_run_mask_rates.log":
        args.log_path = add_timestamp_to_path(args.log_path, ts=run_ts_file)
    if args.output_path is not None and args.output_path.name == "memo_results_mask_rates.jsonl":
        args.output_path = add_run_tags_to_path(args.output_path, run_ts_file, args.diff_mc_samples)

    global TAIL_LEN
    TAIL_LEN = args.tail_len

    setup_logger(args.log_path)
    logging.info("Starting memorization evaluation")
    logging.info("Args: %s", vars(args))

    samples_path = args.samples_path
    if not samples_path.exists():
        if args.dataset_name:
            fallback = args.data_root / args.dataset_name / "samples.jsonl"
            if fallback.exists():
                samples_path = fallback
            else:
                raise FileNotFoundError(f"samples jsonl not found: {samples_path} or {fallback}")
        else:
            raise FileNotFoundError(f"samples jsonl not found: {samples_path}")

    samples = load_samples(
        samples_path,
        args.max_samples,
        args.shuffle,
        args.seed,
        start_line=args.start_line,
        end_line=args.end_line,
    )
    logging.info(
        "Loaded %d samples from %s (start_line=%s, end_line=%s)",
        len(samples),
        samples_path,
        str(args.start_line),
        str(args.end_line),
    )

    device = torch.device(args.device)

    model_name = f"Diff_LLaMA_{args.lit_model_name}M"
    config = Config.from_name(model_name)

    model = TransEncoder(config).to(device)
    state_dict = load_state_dict_local(args.ckpt_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])
    if missing:
        logging.warning("Missing keys when loading ckpt: %d", len(missing))
    if unexpected:
        logging.warning("Unexpected keys when loading ckpt: %d", len(unexpected))

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Align buffer dtypes with parameter dtype.
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    _vocab_size = getattr(model, "vocab_size", None) or getattr(config, "vocab_size", None)

    results_f = None
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_f = open(args.output_path, "w", encoding="utf-8")

    by_rate = {f"{rate:g}": {"masked": [], "gen1": [], "gen_steps": []} for rate in args.mask_rates}

    for idx, sample in enumerate(samples):
        set_name = sample.get("set_name", "")
        token_ids = sample.get("tokens", [])
        if not token_ids:
            logging.warning("Sample %d missing tokens, skipping", idx)
            continue

        logging.info("Sample %d | set=%s | token_len=%d", idx, set_name, len(token_ids))
        if args.debug_log_ground_truth:
            text = sample.get("text", "")
            logging.info("Sample %d | ground_truth_text=%s", idx, text)
        if args.debug_log_tokens:
            logging.info("Sample %d | tail_tokens=%s", idx, token_ids[-TAIL_LEN:])

        for mask_ratio in args.mask_rates:
            start_time = time.time()
            diff_stats = diffusion_loglikelihood(
                model=model,
                tokenizer=tokenizer,
                token_ids=token_ids,
                mask_id=args.diff_mask_id,
                mask_ratio=mask_ratio,
                mc_samples=args.diff_mc_samples,
                mc_batch_size=args.diff_mc_batch_size,
                seed=args.seed + idx,
                debug_mc_samples=args.debug_mc_samples,
                debug_log_masks=args.debug_log_masks,
                debug_log_masked_text=args.debug_log_masked_text,
                debug_masked_text_mc_samples=args.debug_masked_text_mc_samples,
                debug_log_shapes=args.debug_log_shapes,
                debug_log_rank=args.debug_log_rank,
                debug_log_rank_max_positions=args.debug_log_rank_max_positions,
                gen_eachstep=args.gen_eachstep,
                gen_temperature=args.gen_temperature,
                gen_top_k=args.gen_top_k,
                recover_alg=args.recover_alg,
                recover_eps=args.recover_eps,
                recover_cfg_scale=args.recover_cfg_scale,
                only_gen1=args.only_gen1,
                p_targets=args.p_targets,
                n_targets=args.n_targets,
                log_generated_text=args.log_generated_text,
                log_generated_text_mc_samples=args.log_generated_text_mc_samples,
            )

            rate_key = f"{mask_ratio:g}"
            by_rate[rate_key]["masked"].append(diff_stats["mean"])
            by_rate[rate_key]["gen1"].append(diff_stats["gen1_hit_log_mean"])
            by_rate[rate_key]["gen_steps"].append(diff_stats["gen_steps_hit_log_mean"])

            logging.info(
                "Mask %.3f | masked LL mean=%.4f std=%.4f min=%.4f max=%.4f mc=%d",
                mask_ratio,
                diff_stats["mean"],
                diff_stats["stdev"],
                diff_stats["min"],
                diff_stats["max"],
                args.diff_mc_samples,
            )

            logging.info(
                "Mask %.3f | Recover-Gen1 total_log mean=%.4f std=%.4f min=%.4f max=%.4f mc=%d alg=%s temp=%.3f eps=%.2e",
                mask_ratio,
                diff_stats["gen1_hit_log_mean"],
                diff_stats["gen1_hit_log_stdev"],
                diff_stats["gen1_hit_log_min"],
                diff_stats["gen1_hit_log_max"],
                args.diff_mc_samples,
                args.recover_alg,
                args.gen_temperature,
                args.recover_eps,
            )

            if not args.only_gen1:
                logging.info(
                    "Mask %.3f | Recover-GenSteps total_log mean=%.4f std=%.4f min=%.4f max=%.4f mc=%d steps=%s",
                    mask_ratio,
                    diff_stats["gen_steps_hit_log_mean"],
                    diff_stats["gen_steps_hit_log_stdev"],
                    diff_stats["gen_steps_hit_log_min"],
                    diff_stats["gen_steps_hit_log_max"],
                    args.diff_mc_samples,
                    ("mask_count" if args.gen_eachstep else "1"),
                )

            logging.info(
                "Mask %.3f | Gen1 total_log first10=%s",
                mask_ratio,
                [round(x, 4) for x in diff_stats["gen1_log_sums"][:10]],
            )

            if not args.only_gen1:
                logging.info(
                    "Mask %.3f | GenSteps total_log first10=%s",
                    mask_ratio,
                    [round(x, 4) for x in diff_stats["gen_steps_log_sums"][:10]],
                )

            logging.info("Mask %.3f time=%.2fs", mask_ratio, time.time() - start_time)

            if results_f is not None:
                tail_text = decode_tokens(tokenizer, token_ids[-TAIL_LEN:])
                out = {
                    "index": idx,
                    "set_name": set_name,
                    "mask_ratio": mask_ratio,
                    "text": tail_text,
                    "diffusion": {
                        "masked_ll_mean": diff_stats["mean"],
                        "masked_ll_std": diff_stats["stdev"],
                        "masked_ll_min": diff_stats["min"],
                        "masked_ll_max": diff_stats["max"],
                        "recover_alg": args.recover_alg,
                        "recover_eps": args.recover_eps,
                        "recover_cfg_scale": args.recover_cfg_scale,
                        "gen_temperature": args.gen_temperature,
                        "only_gen1": bool(args.only_gen1),
                        "gen1_total_log_mean": diff_stats["gen1_hit_log_mean"],
                        "gen1_total_log_std": diff_stats["gen1_hit_log_stdev"],
                        "gen1_total_log_min": diff_stats["gen1_hit_log_min"],
                        "gen1_total_log_max": diff_stats["gen1_hit_log_max"],
                        "gen1_processable_count": diff_stats["gen1_processable_count"],
                        "gen1_p_hat": diff_stats["gen1_p_hat"],
                        "gen1_n_for_p": diff_stats["gen1_n_for_p"],
                        "gen1_p_for_n": diff_stats["gen1_p_for_n"],
                        "gen_steps_total_log_mean": diff_stats["gen_steps_hit_log_mean"],
                        "gen_steps_total_log_std": diff_stats["gen_steps_hit_log_stdev"],
                        "gen_steps_total_log_min": diff_stats["gen_steps_hit_log_min"],
                        "gen_steps_total_log_max": diff_stats["gen_steps_hit_log_max"],
                        "gen_steps_processable_count": diff_stats["gen_steps_processable_count"],
                        "gen_steps_p_hat": diff_stats["gen_steps_p_hat"],
                        "gen_steps_n_for_p": diff_stats["gen_steps_n_for_p"],
                        "gen_steps_p_for_n": diff_stats["gen_steps_p_for_n"],
                    },
                }
                results_f.write(json.dumps(out) + "\n")

    if results_f is not None:
        results_f.close()

    for rate_key, stats in by_rate.items():
        masked_clean = [v for v in stats["masked"] if math.isfinite(v)]
        gen1_clean = [v for v in stats["gen1"] if math.isfinite(v)]
        gen_steps_clean = [v for v in stats["gen_steps"] if math.isfinite(v)]

        if masked_clean:
            logging.info(
                "Mask %s summary | masked n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
                rate_key,
                len(masked_clean),
                statistics.fmean(masked_clean),
                statistics.pstdev(masked_clean) if len(masked_clean) > 1 else 0.0,
                min(masked_clean),
                max(masked_clean),
            )

        if gen1_clean:
            logging.info(
                "Mask %s summary | Recover-Gen1 total_log n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
                rate_key,
                len(gen1_clean),
                statistics.fmean(gen1_clean),
                statistics.pstdev(gen1_clean) if len(gen1_clean) > 1 else 0.0,
                min(gen1_clean),
                max(gen1_clean),
            )

        if (not args.only_gen1) and gen_steps_clean:
            logging.info(
                "Mask %s summary | Recover-GenSteps total_log n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
                rate_key,
                len(gen_steps_clean),
                statistics.fmean(gen_steps_clean),
                statistics.pstdev(gen_steps_clean) if len(gen_steps_clean) > 1 else 0.0,
                min(gen_steps_clean),
                max(gen_steps_clean),
            )

    logging.info("Done.")


if __name__ == "__main__":
    main()
