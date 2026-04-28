#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union


def warn(message: str) -> None:
    print(f"[flatten_harmbench_results] {message}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten HarmBench results JSON into JSONL of RawUnsafeRecord-like dicts.",
    )
    parser.add_argument("--results-root", required=True, help="Root directory of results JSON files.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--glob", default="**/*.json", help="Glob pattern under results root.")
    parser.add_argument("--require-unsafe-label", action="store_true")
    parser.add_argument(
        "--unsafe-label-source",
        default="label",
        choices=("label", "advbench_label", "either"),
        help="Label source to use when filtering unsafe records.",
    )
    parser.add_argument(
        "--unsafe-label-values",
        type=int,
        nargs="+",
        default=[1],
        help="Unsafe label values to include when filtering.",
    )
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_result_files(root: Path, pattern: str) -> List[Path]:
    files: List[Path] = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        if path.name == "index.json":
            continue
        files.append(path)
    return sorted(files)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _get_text(item: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _normalize_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value_str = value.strip()
        if not value_str:
            return None
        if value_str.isdigit():
            return int(value_str)
        try:
            return int(float(value_str))
        except ValueError:
            return None
    return None


def main() -> int:
    args = parse_args()
    root = Path(args.results_root).expanduser()
    if not root.exists():
        warn(f"Results root does not exist: {root}")
        return 1

    out_path = Path(args.out).expanduser()
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("w", encoding="utf-8")
    else:
        out_handle = None

    unsafe_values = set(args.unsafe_label_values)

    files_scanned = 0
    files_parsed_ok = 0
    items_seen = 0
    items_written = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_missing_label = 0
    skipped_parse = 0
    skipped_exception = 0

    category_counts: Counter[str] = Counter()
    seen_hashes: Set[str] = set()

    files = iter_result_files(root, args.glob)
    for file_path in files:
        files_scanned += 1
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warn(f"Failed to parse JSON: {file_path} ({exc})")
            skipped_exception += 1
            continue

        if not isinstance(payload, dict):
            warn(f"Skipping non-mapping JSON: {file_path}")
            skipped_parse += 1
            continue

        if any(not isinstance(entries, list) for entries in payload.values()):
            warn(f"Skipping unexpected JSON structure: {file_path}")
            skipped_parse += 1
            continue

        files_parsed_ok += 1
        rel_path = _relative_path(file_path, root)

        for category_key, entries in payload.items():
            if not isinstance(entries, list):
                continue
            for idx, item in enumerate(entries):
                items_seen += 1
                if not isinstance(item, dict):
                    skipped_parse += 1
                    continue

                prompt_text = _get_text(item, ("test_case", "prompt", "instruction"))
                completion_text = _get_text(
                    item,
                    ("generation", "completion", "response", "text"),
                )
                completion_text = completion_text.strip()
                if not completion_text:
                    skipped_empty += 1
                    continue
                if len(completion_text) < args.min_chars:
                    skipped_short += 1
                    continue

                label = _normalize_label(item.get("label"))
                advbench_label = _normalize_label(item.get("advbench_label"))

                if args.require_unsafe_label:
                    include = False
                    if args.unsafe_label_source == "label":
                        if label is None:
                            skipped_missing_label += 1
                            continue
                        include = label in unsafe_values
                    elif args.unsafe_label_source == "advbench_label":
                        if advbench_label is None:
                            skipped_missing_label += 1
                            continue
                        include = advbench_label in unsafe_values
                    else:
                        if label is None and advbench_label is None:
                            skipped_missing_label += 1
                            continue
                        include = (label in unsafe_values) or (advbench_label in unsafe_values)
                    if not include:
                        continue

                category = str(category_key)
                record_id = f"{rel_path}::{category}::{idx}"

                if args.dedupe:
                    digest = hashlib.sha1(
                        f"{category}\n{completion_text}".encode("utf-8")
                    ).hexdigest()
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)

                record = {
                    "source": "harmbench",
                    "category": category,
                    "answer_text": completion_text,
                    "toxicity_score": None,
                    "label": label,
                    "advbench_label": advbench_label,
                    "meta": {
                        "record_id": record_id,
                        "file": rel_path,
                        "test_case": prompt_text,
                        "label": label,
                        "advbench_label": advbench_label,
                        "raw_item_keys": sorted(item.keys()),
                    },
                }

                if out_handle is not None:
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

                items_written += 1
                category_counts[category] += 1

                if args.log_every and items_written % args.log_every == 0:
                    print(
                        f"Wrote {items_written} records (scanned {files_scanned} files, seen {items_seen} items)",
                        file=sys.stderr,
                    )

                if args.max_records is not None and items_written >= args.max_records:
                    break
            if args.max_records is not None and items_written >= args.max_records:
                break
        if args.max_records is not None and items_written >= args.max_records:
            break

    if out_handle is not None:
        out_handle.close()

    print("\nSummary:")
    print(f"  files_scanned: {files_scanned}")
    print(f"  files_parsed_ok: {files_parsed_ok}")
    print(f"  items_seen: {items_seen}")
    print(f"  items_written: {items_written}")
    print(f"  skipped_empty: {skipped_empty}")
    print(f"  skipped_short: {skipped_short}")
    print(f"  skipped_missing_label: {skipped_missing_label}")
    print(f"  skipped_parse: {skipped_parse}")
    print(f"  skipped_exception: {skipped_exception}")

    if category_counts:
        print("\nTop categories:")
        for category, count in category_counts.most_common(20):
            print(f"  {category}: {count}")

    if not args.dry_run and items_written == 0:
        warn("No records written.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
