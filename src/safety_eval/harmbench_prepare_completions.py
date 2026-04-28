import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_json_objects(path: Path) -> Iterable[Dict]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise TypeError(f"Expected object entries in {path}, found {type(obj).__name__}")
                yield obj
    except json.JSONDecodeError:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    raise TypeError(f"Entry {idx} in {path} is not an object")
                yield item
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for idx, item in enumerate(data["data"]):
                if not isinstance(item, dict):
                    raise TypeError(f"Entry data[{idx}] in {path} is not an object")
                yield item
        else:
            raise


def _load_test_cases(path: Path, text_key: str) -> Dict[str, List[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping at {path}, found {type(raw).__name__}")

    cases: Dict[str, List[str]] = {}
    for behavior_id, entries in raw.items():
        behavior_cases: List[str] = []
        if isinstance(entries, list):
            for idx, entry in enumerate(entries):
                if isinstance(entry, str):
                    behavior_cases.append(entry)
                elif isinstance(entry, dict):
                    if text_key not in entry:
                        raise KeyError(f"Missing key '{text_key}' for {behavior_id}[{idx}] in {path}")
                    value = entry[text_key]
                    if not isinstance(value, str):
                        raise TypeError(
                            f"Expected string at '{behavior_id}[{idx}].{text_key}' in {path}, found {type(value).__name__}"
                        )
                    behavior_cases.append(value)
                else:
                    raise TypeError(
                        f"Unsupported entry type {type(entry).__name__} for {behavior_id}[{idx}] in {path}"
                    )
        elif isinstance(entries, dict):
            if text_key not in entries:
                raise KeyError(f"Missing key '{text_key}' for {behavior_id} in {path}")
            value = entries[text_key]
            if not isinstance(value, str):
                raise TypeError(
                    f"Expected string at '{behavior_id}.{text_key}' in {path}, found {type(value).__name__}"
                )
            behavior_cases.append(value)
        else:
            raise TypeError(f"Unsupported value type {type(entries).__name__} for {behavior_id} in {path}")
        cases[str(behavior_id)] = behavior_cases
    if not cases:
        raise SystemExit(f"No test cases loaded from {path}.")
    return cases


def _parse_prompt_id(prompt_id: str, delimiter: str) -> Tuple[str, int]:
    if delimiter not in prompt_id:
        return prompt_id, 0
    behavior_id, _, suffix = prompt_id.partition(delimiter)
    try:
        index = int(suffix)
    except ValueError:
        raise ValueError(f"Prompt id '{prompt_id}' does not encode an integer index after delimiter '{delimiter}'.")
    return behavior_id, index


def convert_generations(
    generations_path: Path,
    test_cases_path: Path,
    output_path: Path,
    text_key: str,
    prompt_id_key: str,
    delimiter: str,
    test_case_key: str,
) -> None:
    cases = _load_test_cases(test_cases_path, text_key=test_case_key)
    groups: Dict[str, Dict[int, Dict[str, str]]] = defaultdict(dict)

    for idx, record in enumerate(_load_json_objects(generations_path), start=1):
        if prompt_id_key not in record:
            raise KeyError(f"Missing key '{prompt_id_key}' in {generations_path} entry {idx}")
        if text_key not in record:
            raise KeyError(f"Missing key '{text_key}' in {generations_path} entry {idx}")
        prompt_id = str(record[prompt_id_key])
        generation = record[text_key]
        if not isinstance(generation, str):
            raise TypeError(
                f"Expected string generation at {generations_path} entry {idx}, found {type(generation).__name__}"
            )
        behavior_id, case_idx = _parse_prompt_id(prompt_id, delimiter=delimiter)
        groups[behavior_id][case_idx] = {
            "generation": generation,
        }

    missing = []
    output: Dict[str, List[Dict[str, str]]] = {}

    for behavior_id, behavior_cases in cases.items():
        completions = groups.get(behavior_id, {})
        behavior_output: List[Dict[str, str]] = []
        for case_idx, test_case in enumerate(behavior_cases):
            completion = completions.get(case_idx)
            if completion is None:
                missing.append(f"{behavior_id}{delimiter}{case_idx}")
                continue
            behavior_output.append(
                {
                    "test_case": test_case,
                    "generation": completion["generation"],
                }
            )
        if behavior_output:
            output[behavior_id] = behavior_output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    summary = {
        "generations": str(generations_path),
        "test_cases": str(test_cases_path),
        "output": str(output_path),
        "behaviors": len(output),
        "total_generations": sum(len(v) for v in output.values()),
        "missing_prompt_ids": missing,
    }
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MDLM HarmBench generations to the official completions.json format."
    )
    parser.add_argument("--generations", required=True, help="Path to MDLM generations (JSONL or JSON).")
    parser.add_argument("--test-cases", required=True, help="Path to HarmBench test_cases.json.")
    parser.add_argument("--output", required=True, help="Where to write the completions.json file.")
    parser.add_argument(
        "--text-key",
        default="completion",
        help="Key in the generations file that contains the generated text.",
    )
    parser.add_argument(
        "--prompt-id-key",
        default="prompt_id",
        help="Key in the generations file containing the prompt identifier.",
    )
    parser.add_argument(
        "--delimiter",
        default=":",
        help="Delimiter separating behavior_id and test case index in prompt ids.",
    )
    parser.add_argument(
        "--test-case-key",
        default="test_case",
        help="Key in the test_cases file to read when entries are dictionaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_generations(
        generations_path=Path(args.generations).expanduser().resolve(),
        test_cases_path=Path(args.test_cases).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        text_key=args.text_key,
        prompt_id_key=args.prompt_id_key,
        delimiter=args.delimiter,
        test_case_key=args.test_case_key,
    )


if __name__ == "__main__":
    main()
