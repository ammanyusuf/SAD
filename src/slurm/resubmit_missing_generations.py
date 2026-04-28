#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _parse_env_spec(line: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for token in shlex.split(line.strip()):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        env[key] = value
    return env


def _env_key(env: Dict[str, str]) -> Optional[Tuple[str, str]]:
    slug = env.get("EXPERIMENT_SLUG")
    run_id = env.get("RUN_ID")
    if not slug or not run_id:
        return None
    return slug, run_id


def _iter_envlists(spec_root: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(spec_root.glob(pattern))


def _filter_envlists_by_run_prefix(envlists: Sequence[Path], run_prefix: str) -> List[Path]:
    """Return envlists that contain the run prefix at least once."""
    if not run_prefix:
        return list(envlists)
    candidates: List[Path] = []
    needle = run_prefix.strip()
    for envlist in envlists:
        try:
            text = envlist.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle in text:
            candidates.append(envlist)
    # Fall back to all envlists if no candidates were detected.
    return candidates or list(envlists)


def _load_env_index(envlists: Sequence[Path]) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """Index env specs by (slug, run_id) -> (spec_line, envlist_name)."""
    index: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for envlist in envlists:
        try:
            lines = envlist.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            env = _parse_env_spec(line)
            key = _env_key(env)
            if key is not None:
                # Keep the first occurrence; duplicates should be identical.
                index.setdefault(key, (line.strip(), envlist.name))
                continue
            out_dir = env.get("OUTPUT_DIR")
            if out_dir:
                index.setdefault((out_dir, out_dir), (line.strip(), envlist.name))
    return index


def _load_env_index_by_run_id(envlists: Sequence[Path]) -> Dict[str, Tuple[str, str]]:
    """Index env specs by run_id only -> (spec_line, envlist_name)."""
    index: Dict[str, Tuple[str, str]] = {}
    for envlist in envlists:
        try:
            lines = envlist.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            env = _parse_env_spec(line)
            run_id = env.get("RUN_ID")
            if run_id:
                index.setdefault(run_id, (line.strip(), envlist.name))
                continue
            out_dir = env.get("OUTPUT_DIR")
            if out_dir:
                index.setdefault(out_dir, (line.strip(), envlist.name))
    return index


def _has_generations(run_dir: Path) -> bool:
    for path in run_dir.rglob("generations.jsonl"):
        try:
            if path.is_file() and path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


@dataclass(frozen=True)
class RunDirRef:
    slug: str
    run_id: str
    path: Path

    @property
    def key(self) -> Tuple[str, str]:
        return self.slug, self.run_id

    @property
    def display(self) -> str:
        return f"{self.slug}/{self.run_id}"


def _discover_run_dirs(
    results_root: Path,
    slug_contains: Optional[str],
    run_prefix: str,
) -> List[RunDirRef]:
    refs: List[RunDirRef] = []
    # If slug_contains looks like a subpath, search there to avoid deep rglob.
    search_root = results_root
    if slug_contains:
        candidate = results_root / slug_contains
        if candidate.exists():
            search_root = candidate
    # Expect structure: <results_root>/<slug>/<run_id>; discover run dirs directly.
    for run_dir in search_root.rglob(f"{run_prefix}*"):
        if not run_dir.is_dir():
            continue
        try:
            slug_rel = run_dir.parent.relative_to(results_root)
        except ValueError:
            continue
        slug = "/".join(slug_rel.parts)
        if slug_contains and slug_contains not in slug:
            continue
        refs.append(RunDirRef(slug=slug, run_id=run_dir.name, path=run_dir))
    return sorted(refs, key=lambda r: (r.slug, r.run_id))


def _write_resubmit_envlist(
    spec_root: Path,
    specs: Sequence[str],
    *,
    prefix: str = "resubmit_missing",
) -> Path:
    spec_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = spec_root / f"{prefix}_{timestamp}.envlist"
    out_path.write_text("\n".join(specs) + "\n", encoding="utf-8")
    return out_path


def _chunk_list(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for idx in range(0, len(items), chunk_size):
        yield list(items[idx : idx + chunk_size])


def _build_sbatch_command(
    *,
    account: Optional[str],
    nodes: Optional[int],
    wall_time: str,
    mem: str,
    trillium: bool,
    array_expr: str,
    gpus_per_node: str,
    gen_script: Path,
    export_env: Optional[Dict[str, str]] = None,
) -> List[str]:
    export_arg: List[str] = []
    if export_env:
        # Some clusters do not propagate shell-prefixed env vars into sbatch jobs.
        # Use --export to guarantee REPO_ROOT / RESULTS_ROOT / CONFIG_BATCH_FILE
        # (and optional checkpoint/tokenizer paths) are available on the compute node.
        export_tokens = ["ALL"] + [f"{k}={v}" for k, v in export_env.items()]
        export_arg = [f"--export={','.join(export_tokens)}"]
    return [
        "sbatch",
        "--parsable",
        *export_arg,
        *(["--account", account] if account else []),
        *(["--nodes", str(nodes)] if nodes else []),
        f"--time={wall_time}",
        *([] if trillium else [f"--mem={mem}"]),
        f"--array={array_expr}",
        f"--gpus-per-node={gpus_per_node}",
        str(gen_script),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find runs missing generations.jsonl and produce a filtered envlist "
            "plus an sbatch command that resubmits only those runs."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("RESULTS_ROOT", "~/scratch/results")).expanduser(),
        help="Root where results are staged (default: $RESULTS_ROOT or ~/scratch/results).",
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=None,
        help="Directory containing gen_batch_*.envlist (default: <repo-root>/.slurm_specs).",
    )
    parser.add_argument(
        "--envlist-pattern",
        default="gen_batch_*.envlist",
        help="Glob pattern under spec root for envlists.",
    )
    parser.add_argument(
        "--slug-contains",
        default=None,
        help="Only scan slugs containing this substring (e.g., prompt_pipeline/realtoxicity_prompts).",
    )
    parser.add_argument(
        "--run-prefix",
        default="sbatch_prompt-",
        help="Only consider run directories starting with this prefix.",
    )
    parser.add_argument(
        "--array-range",
        default="0-0",
        help="Array range for resubmission (default 0-0).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root to set in suggested sbatch command.",
    )
    parser.add_argument(
        "--gen-script",
        type=Path,
        default=Path("src/slurm/generate_array.sh"),
        help="Path to generate_array.sh (relative to repo root unless absolute).",
    )
    parser.add_argument("--account", default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--time", default="0-08:00")
    parser.add_argument("--mem", default="48G")
    parser.add_argument("--gpus-per-node", default="a100:1")
    parser.add_argument("--trillium", action="store_true")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=1,
        help="Split missing generations into batches of this size (one sbatch command per batch).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report missing runs; do not write a filtered envlist.",
    )
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    spec_root = (args.spec_root or (repo_root / ".slurm_specs")).expanduser().resolve()
    gen_script = args.gen_script if args.gen_script.is_absolute() else (repo_root / args.gen_script)
    gen_script = gen_script.resolve()

    if not results_root.exists():
        raise SystemExit(f"Results root not found: {results_root}")
    if not spec_root.exists():
        raise SystemExit(f"Spec root not found: {spec_root}")
    if not gen_script.exists():
        raise SystemExit(f"Generation script not found: {gen_script}")

    all_envlists = list(_iter_envlists(spec_root, args.envlist_pattern))
    if not all_envlists:
        raise SystemExit(f"No envlists matched '{args.envlist_pattern}' under {spec_root}")
    candidate_envlists = _filter_envlists_by_run_prefix(all_envlists, args.run_prefix)
    env_index = _load_env_index(candidate_envlists)
    env_index_by_run_id = _load_env_index_by_run_id(candidate_envlists)
    if not env_index and not env_index_by_run_id:
        raise SystemExit("No env specs could be indexed from the candidate envlists.")

    run_refs = _discover_run_dirs(results_root, args.slug_contains, args.run_prefix)
    if not run_refs:
        raise SystemExit("No matching run directories were discovered under results root.")

    missing: List[RunDirRef] = [ref for ref in run_refs if not _has_generations(ref.path)]

    print(f"Scanned runs: {len(run_refs)}")
    print(f"Missing generations.jsonl: {len(missing)}")
    print(f"Envlist candidates: {len(candidate_envlists)}")
    if not missing:
        return

    specs: List[str] = []
    unmatched: List[RunDirRef] = []
    matched_envlists: set[str] = set()
    run_to_envlist: Dict[Tuple[str, str], str] = {}
    for ref in missing:
        spec_entry = env_index.get(ref.key)
        if not spec_entry:
            spec_entry = env_index_by_run_id.get(ref.run_id)
        if not spec_entry:
            spec_entry = env_index_by_run_id.get(str(ref.path))
        if not spec_entry:
            unmatched.append(ref)
            continue
        spec_line, envlist_name = spec_entry
        specs.append(spec_line)
        matched_envlists.add(envlist_name)
        run_to_envlist[ref.key] = envlist_name

    for ref in missing:
        envlist_name = run_to_envlist.get(ref.key)
        if envlist_name:
            print(f"- {ref.display}: env_found ({envlist_name})")
        else:
            print(f"- {ref.display}: env_missing")

    if matched_envlists:
        envlist_list = ", ".join(sorted(matched_envlists))
        print(f"\nMatched envlists: {envlist_list}")

    if unmatched:
        print("\nWarning: some missing runs were not found in envlists; they will not be resubmitted automatically.")

    if args.dry_run or not specs:
        return

    batch_size = max(1, args.gen_batch_size)
    batches = list(_chunk_list(specs, batch_size))

    print(f"\nWriting {len(batches)} envlist batch(es) with up to {batch_size} run(s) each.")
    for batch_idx, batch in enumerate(batches):
        filtered_envlist = _write_resubmit_envlist(
            spec_root,
            batch,
            prefix=f"resubmit_missing_batch{batch_idx}",
        )
        print(f"\nWrote filtered envlist: {filtered_envlist}")

        export_env: Dict[str, str] = {
            "REPO_ROOT": str(repo_root),
            "RESULTS_ROOT": str(results_root),
            "CONFIG_BATCH_FILE": str(filtered_envlist),
        }
        if args.checkpoint_path:
            export_env["CHECKPOINT_PATH"] = args.checkpoint_path
        if args.tokenizer_path:
            export_env["TOKENIZER_PATH"] = args.tokenizer_path

        cmd = _build_sbatch_command(
            account=args.account,
            nodes=args.nodes,
            wall_time=args.time,
            mem=args.mem,
            trillium=args.trillium,
            array_expr=args.array_range,
            gpus_per_node=args.gpus_per_node,
            gen_script=gen_script,
            export_env=export_env,
        )
        env_parts = [
            f"REPO_ROOT={shlex.quote(str(repo_root))}",
            f"RESULTS_ROOT={shlex.quote(str(results_root))}",
            f"CONFIG_BATCH_FILE={shlex.quote(str(filtered_envlist))}",
        ]
        if args.checkpoint_path:
            env_parts.append(f"CHECKPOINT_PATH={shlex.quote(args.checkpoint_path)}")
        if args.tokenizer_path:
            env_parts.append(f"TOKENIZER_PATH={shlex.quote(args.tokenizer_path)}")
        env_prefix = " ".join(env_parts)
        print("\nSuggested resubmission command:")
        print(f"{env_prefix} {' '.join(shlex.quote(part) for part in cmd)}")
    print("\nNote: --export is now included to ensure REPO_ROOT/RESULTS_ROOT propagate on clusters like Trillium.")


if __name__ == "__main__":
    main()
