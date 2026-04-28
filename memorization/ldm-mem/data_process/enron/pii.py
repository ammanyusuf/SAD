#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-source audit bins scan .bin tokens to extract PII:
- email (regex)
- phone number (regex)

Outputs 2 files (JSONL):
  <out>.email.pre100.jsonl
  <out>.phone.pre100.jsonl

Format (pre100): keep ONLY preceding N tokens (default 100)
  - email: extra constraint: the pre100 context must NOT contain ANY email
  - phone: extra constraint: the pre100 context must NOT contain ANY phone (same PHONE regex)

Global unique caps:
  email: 3000 (default)
  phone: 2000 (default)

Per-bin caps:
  each type defaults to 1000.

IMPORTANT BIN-EXIT LOGIC (kept):
- Move to NEXT BIN only when ALL enabled types are "done" for this bin:
    done(type) := (per-bin cap reached) OR (global cap reached) OR (type disabled)
- Stop the whole program only when ALL enabled types reached their global cap.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple, Optional, Set, List
from contextlib import ExitStack

import numpy as np
from lit_gpt import Tokenizer


# -------------------------
# JSONL utilities
# -------------------------

def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"JSON parse error: {path} line {line_no}: {e}")


def collect_bins_from_audit_dir(
    audit_dir: Path,
    pattern: str,
    max_iter: int,
    key_by_basename: bool,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Scan audit JSONL files and build a mapping:
      bin_id -> bin_path_string (as found in audit record)
    The bin_id is either basename or full path depending on key_by_basename.
    """
    files = sorted(audit_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No audit files matched: {audit_dir}/{pattern}")

    bin_map: Dict[str, str] = {}
    kept_lines, dropped_lines, missing_iter, bad_bins_field = 0, 0, 0, 0
    total_bins_seen, total_bin_strings = 0, 0

    for fp in files:
        for rec in iter_jsonl(fp):
            it = rec.get("iter_num")
            if it is None:
                missing_iter += 1
                continue
            it = int(it)
            if max_iter > 0 and it > max_iter:
                dropped_lines += 1
                continue

            kept_lines += 1
            bins = rec.get("bins", [])
            if not isinstance(bins, list):
                bad_bins_field += 1
                continue

            for b in bins:
                total_bins_seen += 1
                if not isinstance(b, str):
                    continue
                total_bin_strings += 1
                if not b.endswith(".bin"):
                    continue

                bin_id = os.path.basename(b) if key_by_basename else b
                if bin_id not in bin_map:
                    bin_map[bin_id] = b

    stats = {
        "audit_dir": str(audit_dir),
        "audit_files": len(files),
        "kept_lines": kept_lines,
        "dropped_lines_iter_gt_max": dropped_lines,
        "skipped_lines_missing_iter": missing_iter,
        "bad_bins_field": bad_bins_field,
        "total_bins_seen": total_bins_seen,
        "total_bin_strings": total_bin_strings,
        "unique_bin_count": len(bin_map),
    }
    return bin_map, stats


# -------------------------
# dtype handling
# -------------------------

def guess_dtype_and_length(bin_path: Path, vocab_size: int) -> Tuple[np.dtype, int]:
    """
    Guess whether the .bin file is uint16 or uint32 by checking divisibility and a sample's range.
    """
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

        mx = int(sample.max(initial=-1))
        mn = int(sample.min(initial=0))

        score = 0
        if mn >= 0:
            score += 1
        if 0 <= mx < vocab_size:
            score += 3
        elif mx < vocab_size * 2:
            score += 1
        else:
            score -= 2

        if dt == np.uint16:
            score += 1

        cand = (dt, length, score)
        if best is None or cand[2] > best[2]:
            best = cand

    if best is None:
        raise RuntimeError(f"Cannot guess dtype for {bin_path}")
    return best[0], best[1]


def resolve_dtype_and_length(bin_path: Path, vocab_size: int, dtype_mode: str) -> Tuple[np.dtype, int]:
    """
    Resolve dtype and token length for the .bin file.
    dtype_mode: auto / uint16 / uint32
    """
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
# PII regex / normalization
# -------------------------
EMAIL_RE = re.compile("^([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})$")
PHONE_RE = re.compile("[0-9][0-9][0-9][-.()][0-9][0-9][0-9][-.()][0-9][0-9][0-9][0-9]")

def normalize_phone(s: str) -> str:
    """Keep digits only for a phone number."""
    return "".join(ch for ch in s if ch.isdigit())


# -------------------------
# Token alignment helpers
# -------------------------

def _decode_len(tokenizer: Tokenizer, toks: np.ndarray) -> int:
    """Decode tokens and return decoded text length."""
    return len(tokenizer.decode(np.asarray(toks, dtype=np.int64)))

def charpos_to_token_offset_cached(
    tokenizer: Tokenizer,
    toks_chunk: np.ndarray,
    char_pos: int,
    cache: Dict[int, int],
    full_text_len: Optional[int] = None,
) -> int:
    """
    Find token offset whose decoded text length is <= char_pos (approx start token).
    Uses a cache of prefix decoded lengths for binary search.
    """
    n = int(toks_chunk.shape[0])
    if 0 not in cache:
        cache[0] = 0
    if full_text_len is not None:
        cache[n] = full_text_len

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if mid not in cache:
            cache[mid] = _decode_len(tokenizer, toks_chunk[:mid])
        if cache[mid] <= char_pos:
            lo = mid + 1
        else:
            hi = mid

    return max(0, lo - 1)

def charpos_to_token_boundary_cached(
    tokenizer: Tokenizer,
    toks_chunk: np.ndarray,
    char_pos: int,
    cache: Dict[int, int],
    full_text_len: Optional[int] = None,
) -> int:
    """
    Find token boundary whose decoded text length is >= char_pos (approx end token boundary).
    Uses a cache of prefix decoded lengths for binary search.
    """
    n = int(toks_chunk.shape[0])
    if 0 not in cache:
        cache[0] = 0
    if full_text_len is not None:
        cache[n] = full_text_len

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if mid not in cache:
            cache[mid] = _decode_len(tokenizer, toks_chunk[:mid])
        if cache[mid] < char_pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


# -------------------------
# Chunk processing: find PII (email / phone)
# -------------------------

def process_chunk_find_pii(
    mm: np.memmap,
    base: int,
    end: int,
    length: int,
    tokenizer: Tokenizer,
    context_tokens: int,
    need_email: bool,
    need_phone: bool,
    seen_pos: set,
) -> Iterator[Dict[str, Any]]:
    """
    Decode a token chunk and search for email/phone using regex.
    For each hit, emit a record with:
      - pii span token positions
      - pre-context tokens/text (N tokens before)
    Applies "no same-type in context" constraints:
      - email context must not contain another email
      - phone context must not contain another phone
    """
    toks_chunk = np.asarray(mm[base:end], dtype=np.int64)
    text = tokenizer.decode(toks_chunk)
    if not text:
        return

    full_len = len(text)
    cache: Dict[int, int] = {0: 0, int(toks_chunk.shape[0]): full_len}

    def emit_common(
        pii_type: str,
        pii_value: str,
        span_start_tok_in_chunk: int,
        span_end_tok_in_chunk: int,
    ) -> Optional[Dict[str, Any]]:
        """Build a common record for a detected entity."""
        g_start = base + int(span_start_tok_in_chunk)
        g_end = base + int(span_end_tok_in_chunk)
        if g_end <= g_start:
            g_end = min(length, g_start + 1)

        key = (pii_type, int(g_start), int(g_end), pii_value)
        if key in seen_pos:
            return None
        seen_pos.add(key)

        # Pre-context (N tokens before the hit)
        ctx_start = max(0, g_start - context_tokens)
        ctx_toks = np.asarray(mm[ctx_start:g_start], dtype=np.int64)
        ctx_text = tokenizer.decode(ctx_toks) if ctx_toks.size else ""

        # Extra constraint: context must not contain same-type PII
        if pii_type == "email":
            if EMAIL_RE.search(ctx_text) is not None:
                return None
        if pii_type == "phone":
            if PHONE_RE.search(ctx_text) is not None:
                return None

        return {
            "pii_type": pii_type,
            "pii_value": pii_value,
            "pii_span_token_start": int(g_start),
            "pii_span_token_end": int(g_end),
            "context_token_start": int(ctx_start),
            "context_token_end": int(g_start),
            "context_len_tokens": int(g_start - ctx_start),
            "context_tokens": ctx_toks.tolist(),
            "context_text": ctx_text,
        }

    # Email detection
    if need_email:
        for m in EMAIL_RE.finditer(text):
            email = m.group(0)

            start_off = charpos_to_token_offset_cached(tokenizer, toks_chunk, m.start(), cache, full_text_len=full_len)
            end_bound = charpos_to_token_boundary_cached(tokenizer, toks_chunk, m.end(), cache, full_text_len=full_len)
            if end_bound <= start_off:
                end_bound = min(int(toks_chunk.shape[0]), start_off + 1)

            rec = emit_common("email", email, start_off, end_bound)
            if rec is None:
                continue

            rec["email"] = email
            rec["email_start_token"] = int(base + start_off)
            rec["email_end_token"] = int(base + end_bound)
            yield rec

    # Phone detection
    if need_phone:
        for m in PHONE_RE.finditer(text):
            raw_phone = m.group(0)
            phone_norm = normalize_phone(raw_phone)
            if not phone_norm:
                continue

            start_off = charpos_to_token_offset_cached(tokenizer, toks_chunk, m.start(), cache, full_text_len=full_len)
            end_bound = charpos_to_token_boundary_cached(tokenizer, toks_chunk, m.end(), cache, full_text_len=full_len)
            if end_bound <= start_off:
                end_bound = min(int(toks_chunk.shape[0]), start_off + 1)

            rec = emit_common("phone", raw_phone, start_off, end_bound)
            if rec is None:
                continue

            rec["phone_number"] = raw_phone
            rec["phone_number_norm"] = phone_norm
            rec["phone_start_token"] = int(base + start_off)
            rec["phone_end_token"] = int(base + end_bound)
            yield rec


# -------------------------
# Scan a single bin file (random / sequential)
# -------------------------

def scan_bin_random(
    bin_path: Path,
    tokenizer: Tokenizer,
    vocab_size: int,
    dtype_mode: str,
    chunk_tokens: int,
    overlap_tokens: int,
    context_tokens: int,
    max_bad_ratio: float,
    rng: np.random.Generator,
    need_email: bool,
    need_phone: bool,
    log_every_sec: float,
    max_chunk_trials: int,
) -> Iterator[Dict[str, Any]]:
    """
    Randomly sample chunk positions within a bin to find PII faster when you don't need full coverage.
    """
    dt, length = resolve_dtype_and_length(bin_path, vocab_size, dtype_mode)
    mm = np.memmap(bin_path, dtype=dt, mode="r")
    try:
        step = chunk_tokens - overlap_tokens
        if step <= 0:
            raise RuntimeError("chunk_tokens must be > overlap_tokens")

        n_bases = (length + step - 1) // step
        n_bases = max(1, int(n_bases))

        visited = set()
        seen_pos = set()
        last_log = time.time()
        hits = {"email": 0, "phone": 0}
        trials = 0

        while trials < max_chunk_trials and len(visited) < n_bases:
            trials += 1
            base_idx = int(rng.integers(0, n_bases))
            if base_idx in visited:
                continue
            visited.add(base_idx)

            base = base_idx * step
            end = min(length, base + chunk_tokens)

            toks_chunk = np.asarray(mm[base:end], dtype=np.int64)
            bad = int(np.sum((toks_chunk < 0) | (toks_chunk >= vocab_size)))
            if bad / max(1, toks_chunk.shape[0]) > max_bad_ratio:
                continue

            for rec in process_chunk_find_pii(
                mm=mm,
                base=base,
                end=end,
                length=length,
                tokenizer=tokenizer,
                context_tokens=context_tokens,
                need_email=need_email,
                need_phone=need_phone,
                seen_pos=seen_pos,
            ):
                t = rec.get("pii_type", "")
                if t in hits:
                    hits[t] += 1
                rec["bin_path"] = str(bin_path)
                yield rec

            now = time.time()
            if now - last_log >= log_every_sec:
                print(
                    f"[SCAN-RAND] bin={bin_path.name} visited_chunks={len(visited)}/{n_bases} trials={trials} "
                    f"hits=(email:{hits['email']},phone:{hits['phone']})",
                    file=sys.stderr,
                    flush=True,
                )
                last_log = now
    finally:
        del mm


def scan_bin_sequential(
    bin_path: Path,
    tokenizer: Tokenizer,
    vocab_size: int,
    dtype_mode: str,
    chunk_tokens: int,
    overlap_tokens: int,
    context_tokens: int,
    max_bad_ratio: float,
    need_email: bool,
    need_phone: bool,
    log_every_sec: float,
) -> Iterator[Dict[str, Any]]:
    """
    Sequentially scan all chunks in a bin to maximize coverage.
    """
    dt, length = resolve_dtype_and_length(bin_path, vocab_size, dtype_mode)
    mm = np.memmap(bin_path, dtype=dt, mode="r")
    try:
        step = chunk_tokens - overlap_tokens
        if step <= 0:
            raise RuntimeError("chunk_tokens must be > overlap_tokens")

        seen_pos = set()
        last_log = time.time()
        hits = {"email": 0, "phone": 0}

        for base in range(0, length, step):
            now = time.time()
            if now - last_log >= log_every_sec:
                pct = 100.0 * base / max(1, length)
                print(
                    f"[SCAN-SEQ] bin={bin_path.name} base={base}/{length} ({pct:.1f}%) "
                    f"hits=(email:{hits['email']},phone:{hits['phone']})",
                    file=sys.stderr,
                    flush=True,
                )
                last_log = now

            end = min(length, base + chunk_tokens)
            toks_chunk = np.asarray(mm[base:end], dtype=np.int64)
            bad = int(np.sum((toks_chunk < 0) | (toks_chunk >= vocab_size)))
            if bad / max(1, toks_chunk.shape[0]) > max_bad_ratio:
                continue

            for rec in process_chunk_find_pii(
                mm=mm,
                base=base,
                end=end,
                length=length,
                tokenizer=tokenizer,
                context_tokens=context_tokens,
                need_email=need_email,
                need_phone=need_phone,
                seen_pos=seen_pos,
            ):
                t = rec.get("pii_type", "")
                if t in hits:
                    hits[t] += 1
                rec["bin_path"] = str(bin_path)
                yield rec
    finally:
        del mm


# -------------------------
# Path helper
# -------------------------

def pick_existing_path_from_many(paths: List[Optional[str]]) -> Optional[Path]:
    """Return the first existing path from a list of string paths."""
    for p in paths:
        if not p:
            continue
        pp = Path(p)
        if pp.exists():
            return pp
    return None


# -------------------------
# Output record builders (pre-only)
# -------------------------

def to_email_pre(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an internal record to an email pre-context output record."""
    return {
        "email": rec["email"],
        "email_start_token": rec["email_start_token"],
        "context_token_start": rec["context_token_start"],
        "context_token_end": rec["context_token_end"],
        "context_len_tokens": rec["context_len_tokens"],
        "context_tokens": rec["context_tokens"],
        "context_text": rec["context_text"],
        "bin_path": rec["bin_path"],
        "bin_id": rec.get("bin_id"),
        "source_bin_path": rec.get("source_bin_path"),
        "audit_dir": rec.get("audit_dir"),
    }

def to_phone_pre(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an internal record to a phone pre-context output record."""
    return {
        "phone_number": rec["phone_number"],
        "phone_number_norm": rec["phone_number_norm"],
        "phone_start_token": rec["phone_start_token"],
        "context_token_start": rec["context_token_start"],
        "context_token_end": rec["context_token_end"],
        "context_len_tokens": rec["context_len_tokens"],
        "context_tokens": rec["context_tokens"],
        "context_text": rec["context_text"],
        "bin_path": rec["bin_path"],
        "bin_id": rec.get("bin_id"),
        "source_bin_path": rec.get("source_bin_path"),
        "audit_dir": rec.get("audit_dir"),
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--audit_dir",
        required=True,
        help="Single audit directory (no multi-source intersection).",
    )
    ap.add_argument("--pattern", default="audit*.jsonl", help="Audit file glob pattern.")
    ap.add_argument("--tokenizer_path", required=True, help="Path to lit-gpt tokenizer.")

    # --out is a base prefix, NOT a single file
    ap.add_argument("--out", default="unique_pii_1src", help="Output base/prefix (will create 2 JSONL files).")

    ap.add_argument("--max_iter", type=int, default=0, help="If >0, keep only records with iter_num <= max_iter.")
    ap.add_argument("--key_mode", choices=["basename", "fullpath"], default="basename", help="How to key bins.")
    ap.add_argument("--dtype", choices=["auto", "uint16", "uint32"], default="auto", help="Bin dtype mode.")

    ap.add_argument("--chunk_tokens", type=int, default=4096, help="Tokens to decode per chunk.")
    ap.add_argument("--overlap_tokens", type=int, default=256, help="Chunk overlap in tokens.")
    ap.add_argument("--context_tokens", type=int, default=100, help="Pre-context tokens to keep (default 100).")
    ap.add_argument("--max_bad_ratio", type=float, default=0.25, help="Skip chunk if bad-token ratio exceeds this.")

    # logging
    ap.add_argument("--log_bins_every", type=int, default=1, help="Log overall progress every N bins.")
    ap.add_argument("--log_scan_every_sec", type=float, default=5.0, help="Log scan progress every N seconds.")

    # enable/disable types
    ap.add_argument("--no_email", action="store_true", help="Disable email extraction.")
    ap.add_argument("--no_phone", action="store_true", help="Disable phone extraction.")

    # global unique caps
    ap.add_argument("--max_unique_emails", type=int, default=3000, help="Global unique email cap (0 = unlimited).")
    ap.add_argument("--max_unique_phones", type=int, default=2000, help="Global unique phone cap (0 = unlimited).")

    # per-bin caps
    ap.add_argument("--max_per_bin_email", type=int, default=1000, help="Per-bin unique email cap (0 = unlimited).")
    ap.add_argument("--max_per_bin_phone", type=int, default=1000, help="Per-bin unique phone cap (0 = unlimited).")

    # randomness
    ap.add_argument("--seed", type=int, default=1234, help="Random seed.")
    ap.add_argument("--scan_mode", choices=["random", "sequential"], default="random", help="Scan mode within bin.")
    ap.add_argument("--max_chunk_trials", type=int, default=20000, help="Max trials per bin in random scan mode.")

    args = ap.parse_args()

    extract_email = not args.no_email
    extract_phone = not args.no_phone

    if not (extract_email or extract_phone):
        raise RuntimeError("All PII types are disabled (--no_email --no_phone). Nothing to do.")

    audit_dir = Path(args.audit_dir)
    key_by_basename = (args.key_mode == "basename")

    tokenizer = Tokenizer(Path(args.tokenizer_path))
    vocab_size = int(tokenizer.vocab_size)

    rng_bins = np.random.default_rng(args.seed)
    rng_scan = np.random.default_rng(args.seed + 99991)

    # Collect bins
    print("[INFO] collecting bins from audit ...", file=sys.stderr, flush=True)
    bins_map, stats = collect_bins_from_audit_dir(audit_dir, args.pattern, args.max_iter, key_by_basename)
    bin_ids = sorted(bins_map.keys())

    print("========== BINS (single-source) ==========")
    print(f"audit_dir: {audit_dir}")
    print(f"pattern:   {args.pattern}")
    print(f"key_mode:  {args.key_mode}")
    print(f"unique_bins: {len(bin_ids)}")
    for x in bin_ids:
        print(x)
    print("==========================================\n")

    # Shuffle bins
    rng_bins.shuffle(bin_ids)
    print(f"[INFO] shuffled {len(bin_ids)} bins with seed={args.seed}", file=sys.stderr, flush=True)

    # Resolve output paths
    out_base = Path(args.out)
    if out_base.suffix == ".jsonl":
        out_base = out_base.with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    pre_tag = f"pre{args.context_tokens}"
    paths = {
        "email_pre": out_base.parent / f"{out_base.name}.email.{pre_tag}.jsonl",
        "phone_pre": out_base.parent / f"{out_base.name}.phone.{pre_tag}.jsonl",
    }

    print("[INFO] output files:", file=sys.stderr, flush=True)
    if extract_email:
        print(f"  - email_pre: {paths['email_pre']}", file=sys.stderr, flush=True)
    if extract_phone:
        print(f"  - phone_pre: {paths['phone_pre']}", file=sys.stderr, flush=True)

    # Global uniqueness sets
    seen_emails: Set[str] = set()
    seen_phones: Set[str] = set()  # normalized digits

    missing = 0
    t0 = time.time()
    total_written = 0

    def all_global_limits_reached() -> bool:
        """
        Stop everything only when ALL enabled types reached their global caps.
        If a cap is 0 (unlimited), that type is never considered reached.
        """
        ok_email = (not extract_email) or (args.max_unique_emails > 0 and len(seen_emails) >= args.max_unique_emails)
        ok_phone = (not extract_phone) or (args.max_unique_phones > 0 and len(seen_phones) >= args.max_unique_phones)

        if extract_email and args.max_unique_emails == 0:
            ok_email = False
        if extract_phone and args.max_unique_phones == 0:
            ok_phone = False

        return ok_email and ok_phone

    def bin_done(wrote_email: int, wrote_phone: int) -> bool:
        """
        Move to NEXT BIN only when ALL enabled types are done for this bin.
        done(type) := (per-bin cap reached) OR (global cap reached) OR (type disabled).
        If per-bin cap is 0 => never reached.
        If global cap is 0 => never reached.
        """
        done_email = (not extract_email) or \
            (args.max_per_bin_email > 0 and wrote_email >= args.max_per_bin_email) or \
            (args.max_unique_emails > 0 and len(seen_emails) >= args.max_unique_emails)

        done_phone = (not extract_phone) or \
            (args.max_per_bin_phone > 0 and wrote_phone >= args.max_per_bin_phone) or \
            (args.max_unique_phones > 0 and len(seen_phones) >= args.max_unique_phones)

        # If both per-bin and global are unlimited (0), then never "done" via caps
        if extract_email and args.max_per_bin_email == 0 and args.max_unique_emails == 0:
            done_email = False
        if extract_phone and args.max_per_bin_phone == 0 and args.max_unique_phones == 0:
            done_phone = False

        return done_email and done_phone

    print(
        f"[INFO] start scanning {len(bin_ids)} bins ... "
        f"(scan_mode={args.scan_mode}, limits=(email:{args.max_unique_emails},phone:{args.max_unique_phones}), "
        f"per_bin=(email:{args.max_per_bin_email},phone:{args.max_per_bin_phone}), "
        f"context={args.context_tokens})",
        file=sys.stderr,
        flush=True,
    )

    with ExitStack() as stack:
        f_email_pre = stack.enter_context(paths["email_pre"].open("w", encoding="utf-8")) if extract_email else None
        f_phone_pre = stack.enter_context(paths["phone_pre"].open("w", encoding="utf-8")) if extract_phone else None

        for i, bin_id in enumerate(bin_ids, 1):
            if i == 1 or (args.log_bins_every > 0 and i % args.log_bins_every == 0):
                elapsed = time.time() - t0
                print(
                    f"[PROGRESS] bins={i}/{len(bin_ids)} "
                    f"unique=(email:{len(seen_emails)},phone:{len(seen_phones)}) "
                    f"missing={missing} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

            if all_global_limits_reached():
                print("[INFO] reached ALL enabled global limits, stopping.", file=sys.stderr, flush=True)
                break

            source_bin_path = bins_map.get(bin_id)
            p = pick_existing_path_from_many([source_bin_path])
            if p is None:
                missing += 1
                print(f"[WARN] missing bin file for id={bin_id} path={source_bin_path} (skipped)", file=sys.stderr, flush=True)
                continue

            print(f"[INFO] scanning bin {i}/{len(bin_ids)}: {p}", file=sys.stderr, flush=True)

            wrote_email = wrote_phone = 0

            # If a type already hit the global cap, tell scanner not to extract it
            need_email = extract_email and not (args.max_unique_emails > 0 and len(seen_emails) >= args.max_unique_emails)
            need_phone = extract_phone and not (args.max_unique_phones > 0 and len(seen_phones) >= args.max_unique_phones)

            iterator = None
            try:
                if args.scan_mode == "random":
                    iterator = scan_bin_random(
                        bin_path=p,
                        tokenizer=tokenizer,
                        vocab_size=vocab_size,
                        dtype_mode=args.dtype,
                        chunk_tokens=args.chunk_tokens,
                        overlap_tokens=args.overlap_tokens,
                        context_tokens=args.context_tokens,
                        max_bad_ratio=args.max_bad_ratio,
                        rng=rng_scan,
                        need_email=need_email,
                        need_phone=need_phone,
                        log_every_sec=args.log_scan_every_sec,
                        max_chunk_trials=args.max_chunk_trials,
                    )
                else:
                    iterator = scan_bin_sequential(
                        bin_path=p,
                        tokenizer=tokenizer,
                        vocab_size=vocab_size,
                        dtype_mode=args.dtype,
                        chunk_tokens=args.chunk_tokens,
                        overlap_tokens=args.overlap_tokens,
                        context_tokens=args.context_tokens,
                        max_bad_ratio=args.max_bad_ratio,
                        need_email=need_email,
                        need_phone=need_phone,
                        log_every_sec=args.log_scan_every_sec,
                    )

                for rec in iterator:
                    # Correct behavior: only break a bin when ALL enabled types are done
                    if bin_done(wrote_email, wrote_phone):
                        print(
                            f"[INFO] bin done -> next bin. wrote=(email:{wrote_email},phone:{wrote_phone}) "
                            f"global=(email:{len(seen_emails)},phone:{len(seen_phones)})",
                            file=sys.stderr,
                            flush=True,
                        )
                        break

                    t = rec.get("pii_type")
                    if t not in ("email", "phone"):
                        continue

                    # Attach provenance
                    rec["bin_id"] = bin_id
                    rec["source_bin_path"] = source_bin_path
                    rec["audit_dir"] = str(audit_dir)

                    if t == "email" and extract_email:
                        if args.max_per_bin_email > 0 and wrote_email >= args.max_per_bin_email:
                            continue
                        email = rec.get("email")
                        if not email:
                            continue
                        if email in seen_emails:
                            continue
                        if args.max_unique_emails > 0 and len(seen_emails) >= args.max_unique_emails:
                            continue

                        seen_emails.add(email)
                        wrote_email += 1

                        f_email_pre.write(json.dumps(to_email_pre(rec), ensure_ascii=False) + "\n")

                    elif t == "phone" and extract_phone:
                        if args.max_per_bin_phone > 0 and wrote_phone >= args.max_per_bin_phone:
                            continue
                        phone_norm = rec.get("phone_number_norm") or normalize_phone(rec.get("pii_value", ""))
                        if not phone_norm:
                            continue
                        if phone_norm in seen_phones:
                            continue
                        if args.max_unique_phones > 0 and len(seen_phones) >= args.max_unique_phones:
                            continue

                        seen_phones.add(phone_norm)
                        wrote_phone += 1

                        f_phone_pre.write(json.dumps(to_phone_pre(rec), ensure_ascii=False) + "\n")

                    total_written += 1
                    if total_written % 50 == 0:
                        if f_email_pre:
                            f_email_pre.flush()
                        if f_phone_pre:
                            f_phone_pre.flush()

                    # After a write, we might have reached a per-bin/global cap
                    if bin_done(wrote_email, wrote_phone):
                        print(
                            f"[INFO] bin done after write -> next bin. wrote=(email:{wrote_email},phone:{wrote_phone})",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
            finally:
                if iterator is not None and hasattr(iterator, "close"):
                    try:
                        iterator.close()
                    except Exception:
                        pass

            print(
                f"[INFO] finished bin {p.name}: new_unique_from_bin=(email:{wrote_email},phone:{wrote_phone}) "
                f"global_unique=(email:{len(seen_emails)},phone:{len(seen_phones)})",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"[DONE] email={len(seen_emails)} phone={len(seen_phones)} | elapsed {elapsed:.1f}s")
    print("[DONE] outputs:")
    if extract_email:
        print(f"  - email_pre: {paths['email_pre'].resolve()}")
    if extract_phone:
        print(f"  - phone_pre: {paths['phone_pre'].resolve()}")

    if missing:
        print(f"[WARN] {missing} bins had no existing file on disk (skipped).", file=sys.stderr, flush=True)

    print("\n[STATS]", json.dumps(stats, ensure_ascii=False), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
