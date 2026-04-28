#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _normalize_key(key: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in key).strip("_")


def _find_column(header: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {_normalize_key(col): col for col in header}
    for candidate in candidates:
        hit = normalized.get(_normalize_key(candidate))
        if hit:
            return hit
    return None


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("CSV header missing; cannot map columns.")
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HarmBench behaviors CSV to DIJA eval JSON.",
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--allow-missing-id",
        action="store_true",
        help="If BehaviorID is missing, generate sequential ids.",
    )
    args = parser.parse_args()

    rows = _read_rows(args.input_csv)
    if not rows:
        raise SystemExit("No rows found in CSV.")

    header = rows[0].keys()
    id_col = _find_column(header, ["BehaviorID", "behavior_id", "id"])
    behavior_col = _find_column(header, ["Behavior", "behavior", "goal", "prompt"])
    refined_col = _find_column(header, ["Refined_behavior", "refined_behavior", "refined", "refined_goal"])

    if behavior_col is None:
        raise SystemExit(
            "Could not find a Behavior column. Expected one of: Behavior, behavior, goal, prompt."
        )

    output: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        behavior = row.get(behavior_col, "").strip()
        if not behavior:
            continue
        if id_col:
            behavior_id = row.get(id_col, "").strip()
        elif args.allow_missing_id:
            behavior_id = str(idx)
        else:
            raise SystemExit(
                "BehaviorID column not found. Use --allow-missing-id to auto-generate."
            )
        item: Dict[str, str] = {
            "BehaviorID": behavior_id,
            "Behavior": behavior,
        }
        if refined_col:
            refined = row.get(refined_col, "").strip()
            if refined:
                item["Refined_behavior"] = refined
        output.append(item)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(output)} rows -> {args.output_json}")


if __name__ == "__main__":
    main()
