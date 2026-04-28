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


def _read_rows(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit("CSV header missing; cannot map columns.")
            return [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit("JSON must be a list of objects.")
        return [{k: (str(v) if v is not None else "").strip() for k, v in item.items()} for item in data]
    raise SystemExit("Unsupported input type. Use .csv or .json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JailbreakBench behaviors to DiffuGuard JSON schema.",
    )
    parser.add_argument("--input", required=True, type=Path, help="JBB behaviors CSV or JSON.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument(
        "--refined-from-goal",
        action="store_true",
        help="Populate refined prompt from goal (default).",
    )
    parser.add_argument(
        "--refined-empty",
        action="store_true",
        help="Leave refined prompt empty.",
    )
    args = parser.parse_args()

    if args.refined_from_goal and args.refined_empty:
        raise SystemExit("Choose only one: --refined-from-goal or --refined-empty.")

    rows = _read_rows(args.input)
    if not rows:
        raise SystemExit("No rows found in input.")

    header = rows[0].keys()
    goal_col = _find_column(header, ["Goal", "goal", "prompt", "vanilla prompt"])
    behavior_col = _find_column(header, ["Behavior", "behavior"])
    target_col = _find_column(header, ["Target", "target"])
    category_col = _find_column(header, ["Category", "category"])
    source_col = _find_column(header, ["Source", "source"])

    if goal_col is None:
        raise SystemExit("Missing Goal column (case-insensitive).")

    refined_from_goal = True
    if args.refined_empty:
        refined_from_goal = False
    if args.refined_from_goal:
        refined_from_goal = True

    output: List[Dict[str, str]] = []
    for row in rows:
        goal = row.get(goal_col, "").strip()
        if not goal:
            continue
        refined = goal if refined_from_goal else ""
        item: Dict[str, str] = {
            "vanilla prompt": goal,
            "refined prompt": refined,
        }
        if behavior_col:
            item["Behavior"] = row.get(behavior_col, "").strip()
        if target_col:
            item["target"] = row.get(target_col, "").strip()
        if category_col:
            item["category"] = row.get(category_col, "").strip()
        if source_col:
            item["source"] = row.get(source_col, "").strip()
        output.append(item)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(output)} rows -> {args.output_json}")


if __name__ == "__main__":
    main()
