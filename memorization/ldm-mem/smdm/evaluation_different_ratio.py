import argparse
import json
import logging
import math
import random
import statistics
import time
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

    p = Path(ckpt_path)
    if p.suffix == ".safetensors":
        return load_file(str(p))
    # fallback: .pth/.pt
    ckpt = torch.load(str(p), map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def sample_with_random_order(
    model,
    x,
    mask_id,
    steps,
    target_batch,
    torch_gen,
    tokenizer=None,
    log_steps=False,
    log_steps_max_samples=1,
    log_steps_label="",
    log_shapes=False,
    log_rank=False,
    log_rank_max_positions=5,
    temperature=1.0,
    top_k=0,
):

    if steps <= 0:
        return x, None, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    mask_index = x == mask_id
    if not mask_index.any():
        return x, None, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    max_mask = int(mask_index.sum(dim=1).max().item())
    steps = min(steps, max_mask)

    num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)
    log_sums = torch.zeros(x.shape[0], device=x.device)
    invalid_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    for i in range(steps):
        mask_index = x == mask_id
        if not mask_index.any():
            break
        logits = model(x)

        logits = apply_temperature_topk(logits, temperature=temperature, top_k=top_k)
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(-1, target_batch.unsqueeze(-1)).squeeze(-1)

        rand_scores = torch.rand(x.shape, generator=torch_gen, device=x.device)
        rand_scores = torch.where(mask_index, rand_scores, -torch.inf)

        transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        for j in range(x.shape[0]):
            k = int(num_transfer_tokens[j, i].item())
            if k <= 0:
                continue
            _, select_index = torch.topk(rand_scores[j], k=k)
            transfer_index[j, select_index] = True

            if log_shapes and j < log_steps_max_samples:
                logging.info(
                    "%s step=%d shapes: x=%s mask_index=%s logits=%s log_probs=%s gathered=%s target=%s transfer=%s",
                    log_steps_label,
                    i,
                    tuple(x.shape),
                    tuple(mask_index.shape),
                    tuple(logits.shape),
                    tuple(log_probs.shape),
                    tuple(gathered.shape),
                    tuple(target_batch.shape),
                    tuple(transfer_index.shape),
                )

            if log_steps and j < log_steps_max_samples:
                positions = select_index.detach().cpu().tolist()
                token_ids = target_batch[j, select_index].detach().cpu().tolist()
                token_texts = []
                if tokenizer is not None:
                    token_texts = [
                        tokenizer.decode([tid], skip_special_tokens=False)
                        for tid in token_ids
                    ]

                token_ranks = []
                if log_rank:
                    if log_rank_max_positions > 0:
                        positions = positions[:log_rank_max_positions]
                        token_ids = token_ids[:log_rank_max_positions]
                        if token_texts:
                            token_texts = token_texts[:log_rank_max_positions]
                    for pos, tid in zip(positions, token_ids):
                        gt_logit = logits[j, pos, tid]
                        rank = int((logits[j, pos] > gt_logit).sum().item()) + 1
                        token_ranks.append(rank)

                logging.info(
                    "%s step=%d pos=%s token_ids=%s token_texts=%s token_ranks=%s",
                    log_steps_label,
                    i,
                    positions,
                    token_ids,
                    token_texts,
                    token_ranks,
                )

        selected_log_probs = torch.where(
            transfer_index,
            gathered,
            torch.zeros_like(gathered),
        )

        invalid = transfer_index & (~torch.isfinite(gathered))
        if invalid.any():
            step_zero = invalid.any(dim=1)
            if step_zero.any():
                invalid_mask |= step_zero
                log_sums = torch.where(
                    invalid_mask, torch.full_like(log_sums, float("nan")), log_sums
                )
            selected_log_probs = torch.where(
                invalid, torch.zeros_like(selected_log_probs), selected_log_probs
            )

        active = (~invalid_mask).unsqueeze(1)
        selected_log_probs = selected_log_probs * active
        log_sums += selected_log_probs.sum(dim=1)

        transfer_index = transfer_index & (~invalid_mask).unsqueeze(1)
        x = torch.where(transfer_index, target_batch, x)

    return x, log_sums, invalid_mask


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
    gen_top_k: int,
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

    log_sums = []
    gen1_log_sums = []
    gen_steps_log_sums = []
    gen1_invalid_mask = []
    gen_steps_invalid_mask = []
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        mc_index = 0
        gen_text_index = 0
        gen_order = torch.Generator(device=device)
        gen_order.manual_seed(seed + 1000)

        for start in range(0, mc_samples, mc_batch_size):
            cur_bs = min(mc_batch_size, mc_samples - start)
            batch = base.repeat(cur_bs, 1)

            scores = torch.rand(cur_bs, seq_len, generator=torch_gen, device=device)
            _, mask_pos = torch.topk(scores, k=mask_count, dim=1)
            mask = torch.zeros(cur_bs, seq_len, dtype=torch.bool, device=device)
            mask.scatter_(1, mask_pos, True)

            masked_batch = batch.clone()
            masked_batch[mask] = mask_id


            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                logits = model(masked_batch)

            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, batch.unsqueeze(-1)).squeeze(-1)
            batch_log_sums = (gathered * mask).sum(dim=1)
            log_sums.extend(batch_log_sums.detach().cpu().tolist())

            gen1_tokens, gen1_ll, gen1_invalid = sample_with_random_order(
                model,
                masked_batch.clone(),
                mask_id,
                steps=1,
                target_batch=batch,
                torch_gen=gen_order,
                tokenizer=tokenizer,
                log_steps=log_generated_text,
                log_steps_max_samples=log_generated_text_mc_samples,
                log_steps_label="Diffusion Gen1",
                log_shapes=debug_log_shapes,
                log_rank=debug_log_rank,
                log_rank_max_positions=debug_log_rank_max_positions,
                temperature=gen_temperature,
                top_k=gen_top_k,
            )
            if gen1_ll is not None:
                gen1_log_sums.extend(gen1_ll.detach().cpu().tolist())
                gen1_invalid_mask.extend(gen1_invalid.detach().cpu().tolist())

            gen_steps_tokens, gen_steps_ll, gen_steps_invalid = sample_with_random_order(
                model,
                masked_batch.clone(),
                mask_id,
                steps=gen_steps,
                target_batch=batch,
                torch_gen=gen_order,
                tokenizer=tokenizer,
                log_steps=log_generated_text,
                log_steps_max_samples=log_generated_text_mc_samples,
                log_steps_label="Diffusion GenSteps",
                log_shapes=debug_log_shapes,
                log_rank=debug_log_rank,
                log_rank_max_positions=debug_log_rank_max_positions,
                temperature=gen_temperature,
                top_k=gen_top_k,
            )
            if gen_steps_ll is not None:
                gen_steps_log_sums.extend(gen_steps_ll.detach().cpu().tolist())
                gen_steps_invalid_mask.extend(gen_steps_invalid.detach().cpu().tolist())

            if log_generated_text:
                for row in range(cur_bs):
                    if gen_text_index >= log_generated_text_mc_samples:
                        break
                    gen1_text = decode_tokens(
                        tokenizer, gen1_tokens[row].detach().cpu().tolist()
                    )
                    gen_steps_text = decode_tokens(
                        tokenizer, gen_steps_tokens[row].detach().cpu().tolist()
                    )
                    logging.info("Diffusion Gen1 text=%s", gen1_text)
                    logging.info("Diffusion GenSteps text=%s", gen_steps_text)
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
                        masked_text = decode_tokens_with_mask(
                            tokenizer, masked_tokens, mask_id
                        )
                        logging.info(
                            "Diffusion MC[%d] masked_input_text=%s",
                            mc_index,
                            masked_text,
                        )
                    logging.info(
                        "Diffusion MC[%d] log_sum=%.4f",
                        mc_index,
                        batch_log_sums[row].item(),
                    )
                    mc_index += 1

    masked_stats = summarize_finite(log_sums)
    gen1_stats = summarize_finite(gen1_log_sums)
    gen_steps_stats = summarize_finite(gen_steps_log_sums)

    gen1_np = compute_np_stats(gen1_log_sums, p_targets, n_targets)
    gen_steps_np = compute_np_stats(gen_steps_log_sums, p_targets, n_targets)

    return {
        "log_sums": log_sums,
        "mean": masked_stats["mean"],
        "stdev": masked_stats["stdev"],
        "min": masked_stats["min"],
        "max": masked_stats["max"],
        "gen1_log_sums": gen1_log_sums,
        "gen1_hit_log_mean": gen1_stats["mean"],
        "gen1_hit_log_stdev": gen1_stats["stdev"],
        "gen1_hit_log_min": gen1_stats["min"],
        "gen1_hit_log_max": gen1_stats["max"],
        "gen1_processable_count": gen1_np["processable_count"],
        "gen1_p_hat": gen1_np["p_hat"],
        "gen1_n_for_p": gen1_np["n_for_p"],
        "gen1_p_for_n": gen1_np["p_for_n"],
        "gen_steps_log_sums": gen_steps_log_sums,
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
        default=Path(
            "smdm/data/unique_bin_samples.jsonl"
        ),
    )
    parser.add_argument("--data_root", type=Path, default=Path("."))
    parser.add_argument("--dataset_name", type=str, default="")

    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)


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

    parser.add_argument("--diff_mask_id", type=int, default=126336)
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
        args.output_path = add_run_tags_to_path(
            args.output_path, run_ts_file, args.diff_mc_samples
        )

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

    samples = load_samples(samples_path, args.max_samples, args.shuffle, args.seed)
    logging.info("Loaded %d samples from %s", len(samples), samples_path)

    device = torch.device(args.device)

    # ===== 本地模型加载（按参考代码方式） =====
    model_name = f"Diff_LLaMA_{args.lit_model_name}M"
    config = Config.from_name(model_name)

    model = TransEncoder(config).to(device)
    state_dict = load_state_dict_local(args.ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys when loading ckpt: %d", len(missing))
    if unexpected:
        logging.warning("Unexpected keys when loading ckpt: %d", len(unexpected))
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    param_dtype = next(model.parameters()).dtype
    for name, buf in model.named_buffers():
        if torch.is_floating_point(buf) and buf.dtype != param_dtype:
            model._buffers[name] = buf.to(dtype=param_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    vocab_size = getattr(model, "vocab_size", None) or getattr(config, "vocab_size", None)
    if vocab_size is not None and args.diff_mask_id > int(vocab_size):
        raise ValueError(
            f"diff_mask_id={args.diff_mask_id} > vocab_size={vocab_size}. "
            f"Please fix mask id or vocab."
        )

    results_f = None
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        results_f = open(args.output_path, "w", encoding="utf-8")

    by_rate = {
        f"{rate:g}": {"masked": [], "gen1": [], "gen_steps": []} for rate in args.mask_rates
    }

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
            logging.info(
                "Sample %d | tail_tokens=%s",
                idx,
                token_ids[-TAIL_LEN:],
            )

        for mask_ratio in args.mask_rates:
            start_time = time.time()
            diff_stats = diffusion_loglikelihood(
                model,
                tokenizer,
                token_ids,
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
                "Mask %.3f | Gen1 hit-log mean=%.4f std=%.4f min=%.4f max=%.4f mc=%d",
                mask_ratio,
                diff_stats["gen1_hit_log_mean"],
                diff_stats["gen1_hit_log_stdev"],
                diff_stats["gen1_hit_log_min"],
                diff_stats["gen1_hit_log_max"],
                args.diff_mc_samples,
            )
            logging.info(
                "Mask %.3f | GenSteps hit-log mean=%.4f std=%.4f min=%.4f max=%.4f mc=%d",
                mask_ratio,
                diff_stats["gen_steps_hit_log_mean"],
                diff_stats["gen_steps_hit_log_stdev"],
                diff_stats["gen_steps_hit_log_min"],
                diff_stats["gen_steps_hit_log_max"],
                args.diff_mc_samples,
            )
            logging.info(
                "Mask %.3f | Gen1 LL first10=%s",
                mask_ratio,
                [round(x, 4) for x in diff_stats["gen1_log_sums"][:10]],
            )
            logging.info(
                "Mask %.3f | GenSteps LL first10=%s",
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
                        "gen1_hit_log_mean": diff_stats["gen1_hit_log_mean"],
                        "gen1_hit_log_std": diff_stats["gen1_hit_log_stdev"],
                        "gen1_hit_log_min": diff_stats["gen1_hit_log_min"],
                        "gen1_hit_log_max": diff_stats["gen1_hit_log_max"],
                        "gen1_processable_count": diff_stats["gen1_processable_count"],
                        "gen1_p_hat": diff_stats["gen1_p_hat"],
                        "gen1_n_for_p": diff_stats["gen1_n_for_p"],
                        "gen1_p_for_n": diff_stats["gen1_p_for_n"],
                        "gen_steps_hit_log_mean": diff_stats["gen_steps_hit_log_mean"],
                        "gen_steps_hit_log_std": diff_stats["gen_steps_hit_log_stdev"],
                        "gen_steps_hit_log_min": diff_stats["gen_steps_hit_log_min"],
                        "gen_steps_hit_log_max": diff_stats["gen_steps_hit_log_max"],
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
                "Mask %s summary | Gen1 hit-log n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
                rate_key,
                len(gen1_clean),
                statistics.fmean(gen1_clean),
                statistics.pstdev(gen1_clean) if len(gen1_clean) > 1 else 0.0,
                min(gen1_clean),
                max(gen1_clean),
            )
        if gen_steps_clean:
            logging.info(
                "Mask %s summary | GenSteps hit-log n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
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
