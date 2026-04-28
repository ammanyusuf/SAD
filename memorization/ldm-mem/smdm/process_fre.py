#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
from lit_gpt import Tokenizer


# -------------------------
# Audit utils (same core logic)
# -------------------------

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSON parse error: {path} line {line_no}: {e}")


def compute_set_name(iter_num: int, max_iter: int) -> int:
    n_sets = 10
    bin_size = max_iter // n_sets
    if bin_size <= 0:
        return 0
    if iter_num <= 0:
        return 0
    return min(n_sets - 1, (iter_num - 1) // bin_size)


def scan_max_iter(audit_dir: Path, pattern: str) -> int:
    files = sorted(audit_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No audit files matched: {audit_dir}/{pattern}")
    mx = -1
    for fp in files:
        for rec in iter_jsonl(fp):
            it = rec.get("iter_num", None)
            if it is None:
                continue
            try:
                it = int(it)
            except Exception:
                continue
            if it > mx:
                mx = it
    return mx


def collect_unique_bins(
    audit_dir: Path,
    pattern: str,
    unique_by_basename: bool,
    max_iter: Optional[int],  # None means no cap, but we still need max for set_name
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    bin_map: Dict[str, Dict[str, Any]] = {}
    files = sorted(audit_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No audit files matched: {audit_dir}/{pattern}")

    kept, dropped, missing_iter = 0, 0, 0
    last_kept_iter = None

    # if max_iter is None, we will not drop by iter_num
    effective_max_for_set = max_iter if (max_iter is not None and max_iter > 0) else 1

    for fp in files:
        for rec in iter_jsonl(fp):
            it = rec.get("iter_num")
            if it is None:
                missing_iter += 1
                continue

            try:
                it = int(it)
            except Exception:
                missing_iter += 1
                continue

            if max_iter is not None and max_iter > 0 and it > max_iter:
                dropped += 1
                continue

            kept += 1
            last_kept_iter = it if last_kept_iter is None else max(last_kept_iter, it)

            bins = rec.get("bins", [])
            if not isinstance(bins, list):
                continue

            set_name = compute_set_name(it, effective_max_for_set)

            for b in bins:
                if not isinstance(b, str):
                    continue
                bin_id = os.path.basename(b) if unique_by_basename else b
                if bin_id not in bin_map:
                    bin_map[bin_id] = {"path": b, "set_name": set_name}

    stats = {
        "audit_files": len(files),
        "kept_lines": kept,
        "dropped_lines_iter_gt_max": dropped,
        "skipped_lines_missing_iter": missing_iter,
        "last_kept_iter": last_kept_iter or -1,
        "unique_bins": len(bin_map),
    }
    return bin_map, stats


# -------------------------
# dtype handling (same as your previous code)
# -------------------------

def guess_dtype_and_length(bin_path: Path, vocab_size: int) -> Tuple[np.dtype, int]:
    size = bin_path.stat().st_size
    candidates = [np.uint16, np.uint32]
    best = None

    for dt in candidates:
        itemsize = np.dtype(dt).itemsize
        if size % itemsize != 0:
            continue
        length = size // itemsize
        if length <= 0:
            continue

        mm = np.memmap(bin_path, dtype=dt, mode="r")
        sample = np.asarray(mm[: min(4096, length)], dtype=np.int64)
        del mm

        mx = sample.max(initial=-1)
        mn = sample.min(initial=0)

        score = 0
        if mn >= 0:
            score += 1
        if 0 <= mx < vocab_size:
            score += 3
        elif mx < vocab_size * 2:
            score += 1
        else:
            score -= 2

        # slightly prefer uint16
        if dt == np.uint16:
            score += 1

        cand = (dt, length, score)
        if best is None or cand[2] > best[2]:
            best = cand

    if best is None:
        raise RuntimeError(f"Cannot guess dtype for {bin_path}")
    return best[0], best[1]


def resolve_dtype_and_length(bin_path: Path, vocab_size: int, dtype_mode: str) -> Tuple[np.dtype, int]:
    if dtype_mode == "auto":
        return guess_dtype_and_length(bin_path, vocab_size)

    if dtype_mode == "uint16":
        dt = np.uint16
    elif dtype_mode == "uint32":
        dt = np.uint32
    else:
        raise RuntimeError(f"Unknown dtype mode: {dtype_mode}")

    size = bin_path.stat().st_size
    itemsize = np.dtype(dt).itemsize
    if size % itemsize != 0:
        raise RuntimeError(f"{bin_path} size not divisible by {dt}")

    length = size // itemsize
    if length <= 0:
        raise RuntimeError(f"Empty bin: {bin_path}")

    return dt, length


# -------------------------
# Frequency from bins (counts -> Laplace smoothed fre_dis)
# -------------------------

def count_tokens_in_bin(
    bin_path: Path,
    vocab_size: int,
    dtype_mode: str,
    chunk_tokens: int,
) -> Tuple[np.ndarray, int, int]:
    dt, length = resolve_dtype_and_length(bin_path, vocab_size, dtype_mode)
    mm = np.memmap(bin_path, dtype=dt, mode="r")

    counts = np.zeros(vocab_size, dtype=np.uint64)
    seen = 0
    bad = 0

    pos = 0
    while pos < length:
        end = min(length, pos + chunk_tokens)
        chunk = np.asarray(mm[pos:end], dtype=np.int64)

        mask = (chunk >= 0) & (chunk < vocab_size)
        if not np.all(mask):
            bad += int((~mask).sum())
            chunk = chunk[mask]

        if chunk.size:
            counts += np.bincount(chunk, minlength=vocab_size).astype(np.uint64)

        seen += (end - pos)
        pos = end

    del mm
    return counts, seen, bad


def laplace_smooth(counts: np.ndarray) -> np.ndarray:
    V = counts.size
    denom = counts.sum(dtype=np.uint64) + np.uint64(V)
    return (counts.astype(np.float64) + 1.0) / float(denom)


def token_to_word(tokenizer: Tokenizer, tid: int) -> str:
    # decode single token to text piece
    return tokenizer.decode(np.asarray([tid], dtype=np.int64))


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--pattern", default="audit_rank*.jsonl")
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--out_pt", required=True, help="output torch .pt (counts + fre_dis + meta)")
    ap.add_argument("--out_csv", required=True, help="output csv with rank,token,word,frequent")

    ap.add_argument("--max_iter", type=int, default=0,
                    help="0 means auto(max iter in audit); negative means no cap; positive means cap at that iter")
    ap.add_argument("--unique_by_fullpath", action="store_true")
    ap.add_argument("--skip_missing", action="store_true")

    ap.add_argument("--dtype", choices=["auto", "uint16", "uint32"], default="auto")
    ap.add_argument("--chunk_tokens", type=int, default=2_000_000)

    args = ap.parse_args()

    audit_dir = Path(args.audit_dir)
    tokenizer = Tokenizer(Path(args.tokenizer_path))
    vocab_size = tokenizer.vocab_size

    # max_iter handling:
    #   >0: cap
    #   =0: auto(max in audit) and cap == auto (effectively full)
    #   <0: no cap
    if args.max_iter == 0:
        mx = scan_max_iter(audit_dir, args.pattern)
        if mx <= 0:
            # still proceed without cap
            max_iter_cap = None
            effective_for_set = 1
        else:
            max_iter_cap = mx
            effective_for_set = mx
    elif args.max_iter < 0:
        max_iter_cap = None
        # still want a reasonable set_name; use auto max if possible
        mx = scan_max_iter(audit_dir, args.pattern)
        effective_for_set = mx if mx > 0 else 1
    else:
        max_iter_cap = int(args.max_iter)
        effective_for_set = max_iter_cap

    # collect bins
    bin_map, stats = collect_unique_bins(
        audit_dir=audit_dir,
        pattern=args.pattern,
        unique_by_basename=not args.unique_by_fullpath,
        max_iter=max_iter_cap if max_iter_cap is not None else effective_for_set,  # for set_name + optional cap
    )

    # if "no cap" mode, we should not drop by iter, so redo with max_iter=None but keep set_name max
    if args.max_iter < 0:
        # set_name uses effective_for_set; cap is None
        bin_map, stats = collect_unique_bins(
            audit_dir=audit_dir,
            pattern=args.pattern,
            unique_by_basename=not args.unique_by_fullpath,
            max_iter=None,
        )
        # overwrite set_name max usage by recomputing set_name is optional; most people don't care.
        # We keep it simple here.

    print("[INFO] audit stats:", stats)
    print(f"[INFO] vocab_size={vocab_size} dtype_mode={args.dtype}")

    total_counts = np.zeros(vocab_size, dtype=np.uint64)
    total_seen = 0
    total_bad = 0
    missing = 0

    for bin_id, info in sorted(bin_map.items()):
        p = Path(info["path"])
        if not p.exists():
            missing += 1
            if args.skip_missing:
                continue
            raise FileNotFoundError(p)

        counts, seen, bad = count_tokens_in_bin(
            p, vocab_size=vocab_size, dtype_mode=args.dtype, chunk_tokens=int(args.chunk_tokens)
        )
        total_counts += counts
        total_seen += seen
        total_bad += bad

    # Laplace smoothing (exactly your logic)
    fre_dis = laplace_smooth(total_counts)

    # eps rule for "unseen tokens" (explicitly set to min freq == last place)
    min_freq = float(fre_dis.min())
    fre_dis = fre_dis.copy()
    fre_dis[total_counts == 0] = min_freq

    present = int((total_counts > 0).sum())
    print(f"[INFO] scanned_tokens={total_seen} bad_tokens={total_bad} missing_bins={missing} present_tokens={present}/{vocab_size}")
    print(f"[INFO] min_freq={min_freq:g} max_freq={float(fre_dis.max()):g}")

    # Save .pt
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "counts": torch.from_numpy(total_counts.astype(np.int64)),
            "fre_dis": torch.from_numpy(fre_dis.astype(np.float32)),
            "meta": {
                "vocab_size": vocab_size,
                "audit_dir": str(audit_dir.resolve()),
                "pattern": args.pattern,
                "max_iter_arg": args.max_iter,
                "unique_bins": stats["unique_bins"],
                "missing_bins": missing,
                "scanned_tokens": total_seen,
                "bad_tokens": total_bad,
                "dtype_mode": args.dtype,
                "chunk_tokens": int(args.chunk_tokens),
                "laplace": "(1+count)/(sum(count)+V)",
                "eps_policy": "unseen(count==0) set to min_freq (same as last rank)",
            },
        },
        out_pt,
    )
    print(f"[DONE] wrote pt -> {out_pt.resolve()}")

    # Build CSV rows: rank, token, word, frequent
    # Sort by frequent desc, tie-break by token id asc (stable/reproducible)
    token_ids = np.arange(vocab_size, dtype=np.int64)
    order = np.lexsort((token_ids, -fre_dis))  # primary: -fre_dis, secondary: token_id
    # lexsort sorts by last key first, so we pass (secondary, primary) correctly:
    # Actually lexsort uses keys in order, last key is primary.
    # Here last key is -fre_dis => primary, first key token_ids => secondary.

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token", "word", "frequent"])
        for r, tid in enumerate(order, start=1):
            word = token_to_word(tokenizer, int(tid))
            w.writerow([r, int(tid), word, f"{float(fre_dis[tid]):.10g}"])

    print(f"[DONE] wrote csv -> {out_csv.resolve()}")


if __name__ == "__main__":
    main()
