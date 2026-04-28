#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from lit_gpt import Tokenizer


def build_cld3_detector():
    try:
        import cld3  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'pycld3' (Google CLD3 binding). "
            "Install with: pip install pycld3\n"
            f"Import error: {e}"
        )

    def detect(text: str) -> Tuple[bool, float, str]:
        res = cld3.get_language(text)
        if res is None:
            return (False, 0.0, "")
        lang = getattr(res, "language", "") or ""
        prob = float(getattr(res, "probability", 0.0) or 0.0)
        ok = (lang == "en") and (prob >= 0.90)
        return (ok, prob, lang)

    return detect


def load_audit_bin_use_counts(audit_path: Path) -> Dict[str, int]:
    """
    Read audit jsonl and count how many times each bin is referenced (use count).
    Output: { "train_enron_001.bin": 73, ... }
    """
    if not audit_path.exists():
        raise FileNotFoundError(f"audit_jsonl not found: {audit_path}")

    counts: Dict[str, int] = {}
    bad_lines = 0
    total_lines = 0

    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_lines += 1
                continue

            bins = obj.get("bins", None)
            if not isinstance(bins, list):
                continue

            for b in bins:
                try:
                    name = Path(str(b)).name
                except Exception:
                    continue
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1

    if bad_lines > 0:
        print(f"[audit][warn] bad json lines skipped: {bad_lines}/{total_lines}")

    if not counts:
        raise RuntimeError(f"[audit] no bins found in audit file: {audit_path}")

    return counts


def parse_dtype(dtype_str: str) -> np.dtype:
    s = str(dtype_str).strip().lower()
    if s in ("uint16", "u16", "np.uint16"):
        return np.uint16
    if s in ("uint32", "u32", "np.uint32"):
        return np.uint32
    raise ValueError(f"Unsupported dtype: {dtype_str}. Use uint16 or uint32.")


def bin_length_tokens(bin_path: Path, dt: np.dtype) -> int:
    size = bin_path.stat().st_size
    itemsize = np.dtype(dt).itemsize
    if size % itemsize != 0:
        raise RuntimeError(
            f"File size not divisible by dtype itemsize: {bin_path} "
            f"size={size} itemsize={itemsize} dtype={dt}"
        )
    length = size // itemsize
    if length <= 0:
        raise RuntimeError(f"Empty or invalid bin: {bin_path} length={length}")
    return int(length)


def sample_non_overlapping_windows_fixed_dtype(
    bin_path: Path,
    vocab_size: int,
    rng: np.random.Generator,
    dt: np.dtype,
    window: int = 200,
    n_windows: int = 4,
    max_bad_ratio: float = 0.50,  # maximum allowed ratio of out-of-range tokens
) -> List[Tuple[int, List[int]]]:
    """
    Sample n_windows non-overlapping windows from a bin file, each with `window` tokens.
    Uses a FIXED dtype (no guessing).

    Returns: [(start, tokens_list), ...]
    """
    length = bin_length_tokens(bin_path, dt)
    need = window * n_windows
    if length < need:
        raise RuntimeError(f"Bin too short (<{need} tokens): {bin_path} length={length}")

    # Choose a start so that [start, start+need) fits, then split into n_windows segments => non-overlapping by design
    start = int(rng.integers(0, length - need + 1))

    mm = np.memmap(bin_path, dtype=dt, mode="r")
    chunk = np.asarray(mm[start:start + need], dtype=np.int64)
    del mm

    # Lightweight validation: too many out-of-range tokens usually means wrong dtype or wrong file format
    bad = int(((chunk < 0) | (chunk >= vocab_size)).sum())
    if bad > int(need * max_bad_ratio):
        raise RuntimeError(
            f"Too many out-of-range tokens ({bad}/{need}) for {bin_path}. "
            f"dtype={dt}, vocab_size={vocab_size}. "
            f"Likely wrong dtype OR file is not a raw token-id array."
        )

    out: List[Tuple[int, List[int]]] = []
    for i in range(n_windows):
        s = start + i * window
        toks = chunk[i * window:(i + 1) * window].tolist()
        out.append((s, toks))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Sample windows ONLY from bins that appeared in audit jsonl. "
            "For each sampled window: decode with Tokenizer, CLD3 filter (en>=0.90), "
            "write jsonl. setname = bin use count in audit.\n\n"
            "This version uses FIXED dtype (no guessing)."
        )
    )
    ap.add_argument("--dataset_dir", type=str, required=True, help="Directory containing *.bin files")
    ap.add_argument("--pattern", type=str, default="*.bin", help="Glob pattern for bin files inside dataset_dir")
    ap.add_argument("--audit_jsonl", type=str, required=True, help="Training audit output audit*.jsonl (or merged)")
    ap.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer checkpoint path")
    ap.add_argument("--out", type=str, required=True, help="Output jsonl path")
    ap.add_argument("--num_bins", type=int, default=144, help="How many bin files to randomly sample (only from audited bins)")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed (reproducible)")
    ap.add_argument("--no_bin_field", action="store_true", help="Do not include the 'bin' field in output")
    ap.add_argument("--window", type=int, default=500, help="Tokens per window")
    ap.add_argument("--n_windows", type=int, default=100, help="Windows per bin (non-overlapping)")
    ap.add_argument("--dtype", type=str, default="uint16", help="Fixed dtype for reading bin: uint16 or uint32 (no guessing)")
    ap.add_argument("--max_bad_ratio", type=float, default=0.50, help="Max allowed out-of-range token ratio before treating as wrong dtype/format")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    tok_path = Path(args.tokenizer_path)
    out_path = Path(args.out)
    audit_path = Path(args.audit_jsonl)
    dt = parse_dtype(args.dtype)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer_path not found: {tok_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"audit_jsonl not found: {audit_path}")

    # 1) Load audit counts
    bin_use_count = load_audit_bin_use_counts(audit_path)
    audited_names = set(bin_use_count.keys())
    print(f"[audit] unique bins in audit: {len(audited_names)}")

    # 2) CLD3 detector
    cld3_detect = build_cld3_detector()

    # 3) Tokenizer + vocab size
    tokenizer = Tokenizer(tok_path)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        try:
            vocab_size = len(tokenizer)  # type: ignore
        except Exception:
            raise RuntimeError("Cannot determine vocab_size from Tokenizer. Please adapt code to set vocab_size.")
    vocab_size = int(vocab_size)

    # 4) Collect candidate bin paths from dataset_dir, strictly limited to audited bins
    all_paths = sorted(dataset_dir.glob(args.pattern))
    path_by_name: Dict[str, Path] = {}
    for p in all_paths:
        if p.is_file() and p.suffix == ".bin":
            if p.name in audited_names:
                if p.name not in path_by_name:
                    path_by_name[p.name] = p

    bins = sorted(path_by_name.values(), key=lambda x: x.name)
    if not bins:
        raise RuntimeError(
            f"No .bin files under {dataset_dir} matched pattern={args.pattern} "
            f"AND appeared in audit."
        )

    # 5) Diagnostics: bins mentioned in audit but missing from dataset_dir
    missing = sorted(audited_names.difference(set(path_by_name.keys())))
    if missing:
        print(f"[audit][warn] bins mentioned in audit but not found in dataset_dir: {len(missing)}")
        print("  e.g. first 20:", missing[:20])

    print(f"[OK] candidate bins (audit ∩ dataset_dir): {len(bins)}")
    print(f"[OK] fixed dtype: {dt} (itemsize={np.dtype(dt).itemsize})")

    # 6) Sample bins
    rng_pick = np.random.default_rng(args.seed)
    rng_shuffle = np.random.default_rng(args.seed + 99991)

    k = min(int(args.num_bins), len(bins))
    chosen_idx = rng_pick.choice(len(bins), size=k, replace=False)
    chosen = [bins[i] for i in chosen_idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    total_windows = 0
    kept_windows = 0
    filtered_non_en = 0
    skipped_short = 0
    skipped_bad_format_or_dtype = 0

    # 7) Extract windows, decode, CLD3 filter
    for p in chosen:
        if p.name not in audited_names:
            raise RuntimeError(f"Internal error: selected bin not in audit: {p.name}")

        try:
            windows = sample_non_overlapping_windows_fixed_dtype(
                p,
                vocab_size=vocab_size,
                rng=rng_pick,
                dt=dt,
                window=int(args.window),
                n_windows=int(args.n_windows),
                max_bad_ratio=float(args.max_bad_ratio),
            )
        except RuntimeError as e:
            msg = str(e)
            if "Bin too short" in msg:
                skipped_short += 1
                continue
            if "Too many out-of-range tokens" in msg or "not divisible by dtype" in msg:
                skipped_bad_format_or_dtype += 1
                continue
            raise

        use_cnt = int(bin_use_count.get(p.name, 0))

        for start, toks in windows:
            total_windows += 1
            text = tokenizer.decode(np.asarray(toks, dtype=np.int64))

            ok_en, prob, lang = cld3_detect(text)
            if not ok_en:
                filtered_non_en += 1
                continue

            rec: Dict[str, Any] = {
                "text": text,
                "tokens": toks,
                "setname": use_cnt,          # setname = audit use count
                "start": int(start),
                "dtype": str(np.dtype(dt)),  # helpful for tracing which dtype was used
            }
            if not args.no_bin_field:
                rec["bin"] = str(p.name)

            # Strictly ensure every record comes from a bin referenced in the audit
            if ("bin" in rec) and (Path(str(rec["bin"])).name not in audited_names):
                raise RuntimeError(f"Record bin not in audit: {rec.get('bin')}")

            records.append(rec)
            kept_windows += 1

    # 8) Final shuffle + write
    rng_shuffle.shuffle(records)

    with out_path.open("w", encoding="utf-8") as w:
        for rec in records:
            if not args.no_bin_field:
                bn = Path(str(rec.get("bin", ""))).name
                if bn and (bn not in audited_names):
                    raise RuntimeError(f"Output contains a record from non-audit bin: {bn}")
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] dataset_dir: {dataset_dir}")
    print(f"[OK] pattern: {args.pattern}")
    print(f"[OK] audit_jsonl: {audit_path}")
    print(f"[OK] audited_bins_total: {len(audited_names)}")
    print(f"[OK] bins_found_in_dataset_dir: {len(bins)}")
    print(f"[OK] sampled_bins: {k}")
    print(f"[OK] skipped_short_bins: {skipped_short}")
    print(f"[OK] skipped_bad_format_or_dtype_bins: {skipped_bad_format_or_dtype}")
    print(f"[OK] total_windows({args.n_windows} per bin): {total_windows}")
    print(f"[OK] filtered_by_cld3_non_en_or_prob_lt_0.90: {filtered_non_en}")
    print(f"[DONE] wrote {len(records)} records -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
