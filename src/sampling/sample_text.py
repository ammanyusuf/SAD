from __future__ import annotations

import time
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerBase

from third_party.mdlm import dataloader, utils
from third_party.mdlm.diffusion import Diffusion
from third_party.mdlm.main import _load_from_checkpoint
from unsafe_prep import utils as unsafe_utils

LOGGER = logging.getLogger(__name__)


def _default_unsafe_artifact_root() -> Path:
    output_root = os.getenv("UNSAFE_OUTPUT_ROOT")
    if output_root:
        return Path(output_root).expanduser()
    scratch = os.getenv("SCRATCH")
    base = Path(scratch).expanduser() if scratch else Path.home()
    return base / "safe-text-diffusion" / "artifacts" / "unsafe_artifacts"


def _infer_tokenizer_family(tokenizer_name_or_path: Optional[str]) -> str:
    if not tokenizer_name_or_path:
        return "unknown"
    lowered = tokenizer_name_or_path.lower()
    for family in ("llada", "dream", "mmada", "mdlm"):
        if family in lowered:
            return family
    return "unknown"


def _parse_unsafe_artifact_name(name: str) -> Optional[Dict[str, object]]:
    tokens = [token for token in name.split("-") if token]
    if len(tokens) < 2:
        return None
    suffix = None
    sample_token = tokens[-1]
    if not sample_token.isdigit() and len(tokens) >= 3:
        maybe_sample = tokens[-2]
        if maybe_sample.isdigit() or maybe_sample.lower() in {"all", "full"}:
            suffix = sample_token.lower()
            sample_token = maybe_sample
            tokens = tokens[:-1]
    dataset_tokens = tokens[:-1]
    if not dataset_tokens:
        return None
    dataset = "-".join(dataset_tokens)
    sample_size: Optional[int]
    take_all = False
    if sample_token.lower() in {"all", "full"}:
        sample_size = None
        take_all = True
    elif sample_token.isdigit():
        sample_size = int(sample_token)
    else:
        return None
    return {
        "dataset": dataset,
        "sample_size": sample_size,
        "take_all": take_all,
        "suffix": suffix,
    }


def _derive_artifact_root_and_name(settings: "SafetySettings") -> Tuple[Optional[Path], Optional[str]]:
    artifact_name = settings.unsafe_artifact_name
    artifact_root = Path(settings.unsafe_artifact_root) if settings.unsafe_artifact_root else None
    if settings.unsafe_artifacts:
        artifacts_path = Path(settings.unsafe_artifacts)
        if artifact_root is None:
            if artifacts_path.suffix == ".pt" or artifacts_path.name.startswith("shard-"):
                artifact_root = artifacts_path.parent.parent if artifacts_path.parent.name else artifacts_path.parent
            else:
                artifact_root = artifacts_path.parent
        if artifact_name is None:
            if artifacts_path.name == "unsafe_reference.pt" or artifacts_path.name.startswith("shard-"):
                artifact_name = artifacts_path.parent.name
            elif artifacts_path.suffix:
                artifact_name = artifacts_path.stem
            else:
                artifact_name = artifacts_path.name
    if artifact_root is None:
        artifact_root = _default_unsafe_artifact_root()
    return artifact_root, artifact_name


def _build_missing_unsafe_artifact(
    settings: "SafetySettings",
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    tokenizer_name_or_path: Optional[str] = None,
) -> Optional[Path]:
    from unsafe_prep.pipeline import DatasetSelection, UnsafePrepConfig, build_unsafe_artifacts, load_config

    artifact_root, artifact_name = _derive_artifact_root_and_name(settings)
    if artifact_root is None or not artifact_name:
        LOGGER.warning("Unsafe artifact auto-build requested, but no artifact name was provided.")
        return None

    tokenizer_name = tokenizer_name_or_path
    if tokenizer_name is None and tokenizer is not None:
        tokenizer_name = getattr(tokenizer, "name_or_path", None)
    if not tokenizer_name:
        tokenizer_name = settings.tokenizer_name_or_path
    if not tokenizer_name:
        LOGGER.warning("Unsafe artifact auto-build requested, but tokenizer name/path is unknown.")
        return None

    parsed = _parse_unsafe_artifact_name(artifact_name)
    if not parsed:
        LOGGER.warning("Unsafe artifact name '%s' does not match the expected naming scheme.", artifact_name)
        return None

    dataset = cast(str, parsed["dataset"])
    sample_size = cast(Optional[int], parsed["sample_size"])
    take_all = cast(bool, parsed["take_all"])
    suffix = cast(Optional[str], parsed["suffix"])
    tokenizer_family = _infer_tokenizer_family(str(tokenizer_name))
    if suffix and tokenizer_family not in {"unknown", suffix}:
        LOGGER.warning(
            "Unsafe artifact suffix '%s' does not match tokenizer family '%s'; building with '%s'.",
            suffix,
            tokenizer_family,
            tokenizer_family,
        )
    elif suffix is None and tokenizer_family not in {"unknown", "mdlm"}:
        LOGGER.warning(
            "Unsafe artifact name '%s' has no family suffix; assuming mdlm defaults (tokenizer='%s').",
            artifact_name,
            tokenizer_name,
        )

    selection: Optional[DatasetSelection] = None
    repo_root = Path(__file__).resolve().parents[2]
    candidate_configs = []
    unsafe_config_env = os.getenv("UNSAFE_CONFIG")
    if unsafe_config_env:
        candidate_configs.append(Path(unsafe_config_env))
    candidate_configs.extend(
        [
            repo_root / "configs" / "unsafe_prep" / "unsafe_prompt_sweep.yaml",
            repo_root / "configs" / "unsafe_prep" / "unsafe_prompt_sweep_legacy.yaml",
        ]
    )
    for cfg_path in candidate_configs:
        resolved_path = cfg_path
        if not resolved_path.is_absolute():
            resolved_path = (repo_root / cfg_path).resolve()
        if not resolved_path.exists():
            continue
        try:
            cfg = load_config(resolved_path)
        except Exception:
            continue
        for entry in cfg.datasets:
            if entry.output_name == artifact_name or entry.output_name == f"{artifact_name}":
                selection = entry
                break
        if selection is not None:
            break

    if selection is None:
        harmbench_aliases = {"harmbench_json", "harmbench_flat", "harmbench_jsonl"}
        if dataset not in {"beavertails", "real-toxicity-prompts", "toxigen", *harmbench_aliases}:
            LOGGER.warning("Unsupported unsafe dataset '%s' for auto-build.", dataset)
            return None
        if dataset in harmbench_aliases:
            harmbench_jsonl = os.getenv("HARM_BENCH_JSONL") or os.getenv("HARM_BENCH_FLAT_JSONL")
            if not harmbench_jsonl:
                scratch = os.getenv("SCRATCH")
                if scratch:
                    candidate = Path(scratch) / "data" / "harmbench_results_initial_release" / "harmbench_flat.jsonl"
                    if candidate.exists():
                        harmbench_jsonl = str(candidate)
            if not harmbench_jsonl:
                LOGGER.warning(
                    "HarmBench JSONL path not set; set HARM_BENCH_JSONL (or HARM_BENCH_FLAT_JSONL)."
                )
                return None
            selection_kwargs = {
                "source": "harmbench_jsonl",
                "split": "train",
                "data_files": {"train": harmbench_jsonl},
                "unsafe_label_values": [1],
                "sample_size": sample_size,
                "take_all": take_all,
                "output_name": artifact_name,
            }
            selection = DatasetSelection(**selection_kwargs)
            max_length = int(os.getenv("UNSAFE_MAX_LENGTH", "1024"))
            shard_size = int(os.getenv("UNSAFE_SHARD_SIZE", "1024"))
            seed = int(os.getenv("UNSAFE_SEED", "1"))
            output_dir = str(artifact_root)
            config = UnsafePrepConfig(
                tokenizer_name_or_path=str(tokenizer_name),
                max_length=max_length,
                shard_size=shard_size,
                seed=seed,
                output_dir=output_dir,
                datasets=[selection],
            )
            LOGGER.info(
                "Auto-building unsafe artifact '%s' (dataset=%s, sample_size=%s, tokenizer=%s) into %s",
                artifact_name,
                "harmbench_jsonl",
                "all" if take_all else sample_size,
                tokenizer_name,
                output_dir,
            )
            build_unsafe_artifacts(
                config=config,
                output_root=artifact_root,
                include=None,
                exclude=None,
                dry_run=False,
                overwrite=False,
            )
            return artifact_root
        selection_kwargs: Dict[str, object] = {
            "source": dataset,
            "split": "330k_train" if dataset == "beavertails" else "train",
            "sample_size": sample_size,
            "take_all": take_all,
            "output_name": artifact_name,
        }
        if dataset == "real-toxicity-prompts":
            selection_kwargs["toxicity_threshold"] = 0.5
        if dataset == "toxigen":
            selection_kwargs["config_name"] = "train"
            selection_kwargs["roberta_threshold"] = 0.8
        selection = DatasetSelection(**selection_kwargs)
    else:
        selection = DatasetSelection(**{**selection.__dict__, "output_name": artifact_name})

    max_length = int(os.getenv("UNSAFE_MAX_LENGTH", "1024"))
    shard_size = int(os.getenv("UNSAFE_SHARD_SIZE", "1024"))
    seed = int(os.getenv("UNSAFE_SEED", "1"))
    output_dir = str(artifact_root)
    config = UnsafePrepConfig(
        tokenizer_name_or_path=str(tokenizer_name),
        max_length=max_length,
        shard_size=shard_size,
        seed=seed,
        output_dir=output_dir,
        datasets=[selection],
    )

    LOGGER.info(
        "Auto-building unsafe artifact '%s' (dataset=%s, sample_size=%s, tokenizer=%s) into %s",
        artifact_name,
        dataset,
        "all" if take_all else sample_size,
        tokenizer_name,
        output_dir,
    )
    build_unsafe_artifacts(config=config, output_root=artifact_root, include=None, exclude=None, dry_run=False, overwrite=False)
    return artifact_root


def _prepare_unsafe_artifacts(
    settings: SafetySettings,
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    tokenizer_name_or_path: Optional[str] = None,
) -> Optional[Path]:
    if not settings.enabled:
        return None
    if settings.unsafe_artifacts and Path(settings.unsafe_artifacts).exists():
        return settings.unsafe_artifacts
    artifact_root = Path(settings.unsafe_artifact_root) if settings.unsafe_artifact_root else None
    if artifact_root is None:
        artifact_root = _default_unsafe_artifact_root()
        settings.unsafe_artifact_root = artifact_root

    try:
        entry = unsafe_utils.find_unsafe_artifact(artifact_root, settings.unsafe_artifact_name)
    except Exception as exc:
        if not settings.auto_build_unsafe_artifacts:
            raise
        LOGGER.warning("Unsafe artifact lookup failed (%s); attempting auto-build.", exc)
        built_root = _build_missing_unsafe_artifact(
            settings,
            tokenizer=tokenizer,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
        if built_root is None:
            raise
        entry = unsafe_utils.find_unsafe_artifact(Path(built_root), settings.unsafe_artifact_name)

    artifact_dir = Path(entry.get("path") or artifact_root)
    storage = entry.get("storage") or {}
    tensor_path = unsafe_utils.materialize_artifact(artifact_dir, storage, overwrite=False)
    settings.unsafe_artifacts = tensor_path
    settings.artifact_stats = {
        "artifact": entry.get("name") or artifact_dir.name,
        "count": entry.get("count"),
        "mean_length": entry.get("mean_length"),
        "std_length": entry.get("std_length"),
        "path": str(artifact_dir),
        "storage": storage,
    }
    return tensor_path


def _get_safety_value(safety_obj: object, key: str, default=None):
    if isinstance(safety_obj, DictConfig):
        return safety_obj.get(key, default)
    return getattr(safety_obj, key, default)


def resolve_eta_config(safety_cfg: object) -> tuple[float, bool]:
    scale = _get_safety_value(safety_cfg, "scale", None)
    eta = _get_safety_value(safety_cfg, "eta", None)
    eta_from_scale = False
    if eta is None:
        if scale is not None:
            eta = scale
            eta_from_scale = True
            LOGGER.error(
                "safety.scale is deprecated; using it as eta. Please switch to safety.eta."
            )
        else:
            eta = 0.0
            LOGGER.warning("safety.eta is not set; using 0.0.")
    elif scale is not None and scale != eta:
        LOGGER.warning(
            "safety.scale=%s is ignored because safety.eta=%s is set.",
            scale,
            eta,
        )
    return float(eta), eta_from_scale


@dataclass
class PromptRecord:
    prompt_id: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetySettings:
    enabled: bool = False
    eta: Optional[float] = None
    weight_mode: str = "eta_beta_hat"
    beta_hat_mode: str = "mc_mean"
    beta_hat_clip_min: Optional[float] = None
    beta_hat_clip_max: Optional[float] = None
    schedule_mode: str = "hard_window"
    eta_from_scale: bool = False
    scale: Optional[float] = None
    unsafe_artifacts: Optional[Path] = None
    unsafe_artifact_root: Optional[Path] = None
    unsafe_artifact_name: Optional[str] = None
    unsafe_prototypes: Optional[Path] = None
    critical_steps: Optional[Any] = None
    t_start: Optional[int] = None
    t_end: Optional[int] = None
    use_semantic_gating: bool = False
    semantic_weight: float = 0.0
    semantic_temp: float = 1.0
    semantic_sigma: Optional[float] = None
    cache_semantic_ref: bool = False
    semantic_ref_path: Optional[Path] = None
    semantic_checkpoint: Optional[Path] = None
    semantic_embed_attr: Optional[str] = None
    artifact_stats: Dict[str, Any] = field(default_factory=dict)
    auto_build_unsafe_artifacts: bool = False
    tokenizer_name_or_path: Optional[str] = None


@dataclass
class GenerationSettings:
    max_new_tokens: int
    prefix_length: int
    sampling_steps: int
    batch_size: int
    seed: int
    sampling_mode: str = "pure_diffusion"
    add_bos: bool = False
    add_eos: bool = False
    unconditional_samples: int = 0
    auto_batch: bool = False
    auto_batch_target_pct: float = 0.0
    max_auto_batch_size: Optional[int] = None
    auto_batch_warmup_prompts: int = 64
    precision: str = "bf16"
    block_length: Optional[int] = None
    temperature: float = 0.0
    transfer_schedule: Optional[str] = None

    @property
    def sequence_length(self) -> int:
        return self.prefix_length + self.max_new_tokens


@dataclass
class ModelSettings:
    model_name: str
    checkpoint_path: Path
    tokenizer_name: str
    precision: str = "bf16"
    variant: Optional[str] = None


@dataclass
class GenerationResult:
    prompt_id: str
    prompt: str
    completion: str
    full_text: str
    token_ids: List[int]
    prompt_length: int
    prompt_mask: List[int]
    metadata: Dict[str, Any]


@dataclass
class GenerationRun:
    results: List[GenerationResult]
    timings: Dict[str, float]
    resolved_config: Dict[str, Any]


@dataclass(frozen=True)
class StopTokens:
    pad_id: Optional[int]
    eos_id: Optional[int]
    eot_id: Optional[int]
    eom_id: Optional[int]
    start_header_id: Optional[int]
    end_header_id: Optional[int]
    bos_id: Optional[int]
    stop_sequences: Tuple[Tuple[int, ...], ...]

    @property
    def stop_ids(self) -> Set[int]:
        return {
            tok
            for tok in (
                self.pad_id,
                self.eos_id,
                self.eot_id,
                self.eom_id,
                self.start_header_id,
                self.end_header_id,
            )
            if tok is not None
        }

    @property
    def guidance_ignore_ids(self) -> List[int]:
        return [
            tok
            for tok in (
                self.pad_id,
                self.eos_id,
                self.eot_id,
                self.start_header_id,
                self.end_header_id,
                self.bos_id,
            )
            if tok is not None
        ]


def _compose_sampling_config(
    model: ModelSettings,
    generation: GenerationSettings,
    safety: SafetySettings,
) -> DictConfig:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "third_party" / "mdlm" / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir), job_name="mdlm_sampling"):
        cfg = compose(config_name="config", overrides=["mode=sample_eval"])
    OmegaConf.set_struct(cfg, False)
    cfg.seed = generation.seed
    cfg.eval.checkpoint_path = str(model.checkpoint_path)
    cfg.eval.generate_samples = True
    cfg.eval.compute_generative_perplexity = False
    cfg.data.tokenizer_name_or_path = model.tokenizer_name
    cfg.loader.batch_size = generation.batch_size
    cfg.loader.eval_batch_size = generation.batch_size
    cfg.loader.global_batch_size = generation.batch_size
    cfg.loader.eval_global_batch_size = generation.batch_size
    cfg.loader.num_workers = 0
    cfg.model.length = generation.sequence_length
    cfg.sampling.steps = generation.sampling_steps
    cfg.sampling.num_sample_batches = 1
    cfg.trainer.num_nodes = 1
    cfg.trainer.devices = 1
    cfg.trainer.accumulate_grad_batches = 1
    eta, eta_from_scale = resolve_eta_config(safety)
    cfg.safety.eta = eta
    cfg.safety.weight_mode = safety.weight_mode
    cfg.safety.beta_hat_mode = safety.beta_hat_mode
    cfg.safety.beta_hat_clip_min = safety.beta_hat_clip_min
    cfg.safety.beta_hat_clip_max = safety.beta_hat_clip_max
    cfg.safety.schedule_mode = safety.schedule_mode
    cfg.safety.enabled = bool(safety.enabled and (safety.unsafe_artifacts or safety.unsafe_prototypes))
    if cfg.safety.enabled and safety.unsafe_artifacts:
        cfg.safety.unsafe_path = str(safety.unsafe_artifacts)
    elif cfg.safety.enabled:
        # Prototype-only path: clear unsafe_path so MDLM does not load defaults.
        cfg.safety.unsafe_path = None
    if safety.unsafe_prototypes:
        cfg.safety.unsafe_prototypes_path = str(safety.unsafe_prototypes)
    if safety.critical_steps is not None:
        cfg.safety.critical_steps = safety.critical_steps
    if safety.t_start is not None:
        cfg.safety.t_start = safety.t_start
    if safety.t_end is not None:
        cfg.safety.t_end = safety.t_end
    cfg.safety.use_semantic_gating = safety.use_semantic_gating
    cfg.safety.semantic_weight = safety.semantic_weight
    cfg.safety.semantic_temp = safety.semantic_temp
    cfg.safety.semantic_sigma = safety.semantic_sigma
    cfg.safety.cache_semantic_ref = safety.cache_semantic_ref
    cfg.safety.semantic_ref_path = str(safety.semantic_ref_path) if safety.semantic_ref_path else cfg.safety.get("semantic_ref_path", None)
    cfg.safety.semantic_checkpoint = str(safety.semantic_checkpoint) if safety.semantic_checkpoint else cfg.safety.get("semantic_checkpoint", None)
    cfg.safety.semantic_embed_attr = safety.semantic_embed_attr
    OmegaConf.set_struct(cfg, True)
    return cfg


def _apply_batch_size(cfg: DictConfig, batch_size: int) -> None:
    cfg.loader.batch_size = batch_size
    cfg.loader.eval_batch_size = batch_size
    cfg.loader.global_batch_size = batch_size
    cfg.loader.eval_global_batch_size = batch_size


def _chunk(seq: Sequence[PromptRecord], size: int) -> Iterable[Sequence[PromptRecord]]:
    if size <= 0:
        raise ValueError(f"Chunk size must be positive (got {size}).")
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _convert_optional_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id in (None, tokenizer.unk_token_id, -1):
        return None
    if isinstance(token_id, list):
        return None
    return int(token_id)


def _resolve_stop_tokens(tokenizer: PreTrainedTokenizerBase) -> StopTokens:
    stop_sequences: List[Tuple[int, ...]] = []
    for marker in ("<|eot_id|>", "<|eom_id|>", "<|start_header_id|>", "<|end_header_id|>"):
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if encoded:
            stop_sequences.append(tuple(encoded))

    return StopTokens(
        pad_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
        eot_id=_convert_optional_token_id(tokenizer, "<|eot_id|>"),
        eom_id=_convert_optional_token_id(tokenizer, "<|eom_id|>"),
        start_header_id=_convert_optional_token_id(tokenizer, "<|start_header_id|>"),
        end_header_id=_convert_optional_token_id(tokenizer, "<|end_header_id|>"),
        bos_id=getattr(tokenizer, "bos_token_id", None),
        stop_sequences=tuple(stop_sequences),
    )




def _build_initial_state(
    tokenizer,
    prompts: Sequence[PromptRecord],
    max_length: int,
    max_prompt_tokens: Optional[int],
    add_bos: bool,
    add_eos: bool,
    mask_token_id: int,
) -> Tuple[torch.LongTensor, List[int], List[bool]]:
    if mask_token_id is None:
        raise ValueError("Prompt conditioning requires a valid mask token id.")
    init_tokens = torch.full(
        (len(prompts), max_length),
        fill_value=mask_token_id,
        dtype=torch.long,
    )
    prompt_lengths: List[int] = []
    truncated_flags: List[bool] = []
    for row, record in enumerate(prompts):
        token_ids: List[int] = tokenizer.encode(record.prompt, add_special_tokens=False)
        original_len = len(token_ids)
        if max_prompt_tokens is not None and max_prompt_tokens > 0:
            token_ids = token_ids[:max_prompt_tokens]
        if add_bos and tokenizer.bos_token_id is not None:
            token_ids = [tokenizer.bos_token_id] + token_ids
        if add_eos and tokenizer.eos_token_id is not None:
            token_ids = token_ids + [tokenizer.eos_token_id]
        prompt_truncated = False
        if len(token_ids) >= max_length:
            token_ids = token_ids[: max_length - 1]
            prompt_truncated = True
        if not prompt_truncated and max_prompt_tokens and original_len > len(token_ids):
            prompt_truncated = True
        init_tokens[row, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
        prompt_lengths.append(len(token_ids))
        truncated_flags.append(prompt_truncated)
    return init_tokens, prompt_lengths, truncated_flags


def _strip_completion_tokens(
    tokens: Sequence[int],
    prompt_length: int,
    stop_ids: Set[int],
    mask_id: Optional[int],
    stop_sequences: Sequence[Sequence[int]] = (),
) -> Tuple[List[int], Optional[int], Optional[int]]:
    trimmed: List[int] = []
    stop_index: Optional[int] = None
    stop_token: Optional[int] = None
    completion = list(tokens[prompt_length:])

    # Find earliest stop position from single-token ids.
    single_stop_pos: Optional[int] = None
    for offset, token in enumerate(completion):
        if token in stop_ids:
            single_stop_pos = prompt_length + offset
            break

    # Find earliest stop position from multi-token sequences.
    seq_stop_pos: Optional[int] = None
    for seq in stop_sequences or []:
        if not seq:
            continue
        seq_len = len(seq)
        for offset in range(0, len(completion) - seq_len + 1):
            if completion[offset : offset + seq_len] == list(seq):
                candidate_pos = prompt_length + offset
                if seq_stop_pos is None or candidate_pos < seq_stop_pos:
                    seq_stop_pos = candidate_pos
                break

    # Choose the earliest stop index among candidates.
    candidates = [pos for pos in (single_stop_pos, seq_stop_pos) if pos is not None]
    if candidates:
        stop_index = min(candidates)
        stop_token = tokens[stop_index]

    # Build trimmed tokens up to the stop point.
    end_index = stop_index if stop_index is not None else len(tokens)
    for idx in range(prompt_length, end_index):
        token = tokens[idx]
        if mask_id is not None and token == mask_id:
            continue
        trimmed.append(token)
    return trimmed, stop_index, stop_token


def _truncate_at_turn_boundary(
    completion_tokens: List[int],
    tokenizer: PreTrainedTokenizerBase,
) -> List[int]:
    """
    Second pass robust truncation for chat-tuned models (e.g. LLaDA) that might emit turn boundaries.
    """
    markers = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>"]
    if getattr(tokenizer, "eos_token", None):
        markers.append(tokenizer.eos_token)
    
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
        
    stop_seqs = []

    for marker in markers:
        tid = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)
        
        # Also try encoding to see if it breaks into multiple tokens
        # We use add_special_tokens=False to just get the marker's ids
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if encoded:
            if len(encoded) == 1:
                stop_ids.add(encoded[0])
            else:
                stop_seqs.append(list(encoded))

    # Also add the specific pair (eot, start_header) if resolved
    eot_enc = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    start_enc = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    if eot_enc and start_enc:
        stop_seqs.append(list(eot_enc + start_enc))

    cutoff = len(completion_tokens)

    # 1. Truncate at earliest single stop token
    for idx, token in enumerate(completion_tokens):
        if token in stop_ids:
            cutoff = min(cutoff, idx)
            break

    # 2. Check for sequences
    # Only need to scan up to current cutoff
    for seq in stop_seqs:
        n = len(seq)
        if n == 0: continue
        limit = min(len(completion_tokens), cutoff) - n + 1
        for idx in range(limit):
            if completion_tokens[idx : idx + n] == seq:
                cutoff = min(cutoff, idx)
                break

    return completion_tokens[:cutoff]

def _assert_no_extra_turn_tokens(
    completion_tokens: Sequence[int],
    decoded_completion: str,
    prompt_length: int,
    tokens: Sequence[int],
    stop_tokens: StopTokens,
    logger: logging.Logger,
) -> None:
    control_ids = {
        "start_header_id": stop_tokens.start_header_id,
        "eot_id": stop_tokens.eot_id,
        "eom_id": stop_tokens.eom_id,
        "end_header_id": stop_tokens.end_header_id,
    }
    for name, token_id in control_ids.items():
        if token_id is None:
            continue
        if token_id in completion_tokens or f"<|{name}|>" in decoded_completion:
            stop_pos = next(
                (idx for idx in range(prompt_length, len(tokens)) if tokens[idx] == token_id),
                None,
            )
            window_start = max(prompt_length, (stop_pos or prompt_length) - 5)
            window_end = min(len(tokens), (stop_pos or prompt_length) + 6)
            window = tokens[window_start:window_end]
            logger.error(
                "Detected %s inside completion (prompt_length=%d, stop_pos=%s, window=%s)",
                name,
                prompt_length,
                stop_pos,
                window,
            )
            raise AssertionError(f"Detected {name} inside completion.")


def _format_metadata(
    base: Dict[str, Any],
    shard_metadata: Dict[str, Any],
    cfg: DictConfig,
    generation: GenerationSettings,
) -> Dict[str, Any]:
    unsafe_artifacts = (
        str(cfg.safety.unsafe_path)
        if cfg.safety.enabled and cfg.safety.get("unsafe_path")
        else None
    )
    metadata = {
        **base,
        **shard_metadata,
        "safe_sampling_enabled": cfg.safety.enabled,
        "safety_eta": cfg.safety.get("eta"),
        "safety_scale": cfg.safety.get("scale"),
        "safety_weight_mode": cfg.safety.get("weight_mode"),
        "safety_beta_hat_mode": cfg.safety.get("beta_hat_mode"),
        "safety_schedule_mode": cfg.safety.get("schedule_mode"),
        "unsafe_artifacts": unsafe_artifacts,
        "sequence_length": cfg.model.length,
        "sampling_steps": cfg.sampling.steps,
        "batch_size": generation.batch_size,
        "seed": generation.seed,
        "precision": generation.precision,
    }
    return metadata


class GenerationEngine:
    """Encapsulates model preparation, auto-batching, and sampling logic."""

    def __init__(
        self,
        prompts: Optional[Sequence[PromptRecord]],
        model: ModelSettings,
        generation: GenerationSettings,
        safety: SafetySettings,
        shard_metadata: Dict[str, Any],
    ) -> None:
        self.prompts: List[PromptRecord] = list(prompts) if prompts else []
        self.model_settings = model
        self.generation_settings = generation
        self.safety_settings = safety
        self.shard_metadata = shard_metadata
        self.logger = utils.get_logger(self.__class__.__name__)
        self._ensure_safety_artifacts()
        self.cfg = _compose_sampling_config(model=model, generation=generation, safety=safety)
        self.tokenizer = None
        self.model_instance: Optional[Diffusion] = None
        self.stop_tokens: Optional[StopTokens] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.auto_info: Dict[str, Any] = {"applied": False}
        self.resolved_cfg: Dict[str, Any] = {}

    def run(self) -> GenerationRun:
        start_time = time.perf_counter()
        if not self.prompts and self.generation_settings.unconditional_samples <= 0:
            resolved_cfg = cast(Dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True))
            timings = {
                "load_seconds": 0.0,
                "generation_seconds": 0.0,
                "total_seconds": 0.0,
                "peak_vram_bytes": 0,
                "auto_batch": {"applied": False},
            }
            return GenerationRun(results=[], timings=timings, resolved_config=resolved_cfg)

        load_start = time.perf_counter()
        self._prepare_model()
        load_seconds = time.perf_counter() - load_start

        prefix_len = self.generation_settings.prefix_length
        prompt_cap = prefix_len if prefix_len > 0 else None
        if self.prompts:
            self.auto_info = self._auto_batch_prompts(prompt_cap)

        self.resolved_cfg = cast(Dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True))
        self._reset_peak_stats()

        generation_start = time.perf_counter()
        results: List[GenerationResult] = []
        results.extend(self._generate_conditioned(prompt_cap))
        results.extend(self._generate_unconditional())
        generation_seconds = time.perf_counter() - generation_start
        total_seconds = time.perf_counter() - start_time

        timings = {
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "peak_vram_bytes": self._capture_peak_vram(),
            "auto_batch": self.auto_info,
        }
        return GenerationRun(results=results, timings=timings, resolved_config=self.resolved_cfg)

    def _ensure_safety_artifacts(self) -> None:
        if not self.safety_settings.enabled:
            return
        if self.safety_settings.unsafe_artifacts is None and self.safety_settings.unsafe_prototypes:
            proto_path = Path(self.safety_settings.unsafe_prototypes)
            if not proto_path.exists():
                self.logger.warning(
                    "Safety prototypes missing at %s; disabling safety.",
                    proto_path,
                )
                self.safety_settings.enabled = False
            else:
                self.logger.info("Using clustered unsafe prototypes from %s", proto_path)
            return
        unsafe_path = _prepare_unsafe_artifacts(self.safety_settings)
        if unsafe_path is None:
            self.logger.warning(
                "Safety requested but unsafe artifacts could not be resolved; disabling safety."
            )
            self.logger.warning(
                "Hint: set safety.auto_build_unsafe_artifacts=true in configs/config.yaml to auto-build."
            )
            self.safety_settings.enabled = False
        else:
            self.logger.info(
                "Loaded unsafe artifact '%s' (%s records, mean length %.2f).",
                self.safety_settings.artifact_stats.get("artifact", Path(unsafe_path).parent.name),
                self.safety_settings.artifact_stats.get("count", "?"),
                self.safety_settings.artifact_stats.get("mean_length", 0.0) or 0.0,
            )

    def _prepare_model(self) -> None:
        L.seed_everything(self.generation_settings.seed)
        self.tokenizer = dataloader.get_tokenizer(self.cfg)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            self.stop_tokens = _resolve_stop_tokens(self.tokenizer)
        self.model_instance = _load_from_checkpoint(config=self.cfg, tokenizer=self.tokenizer)
        self.model_instance.to(self.device)
        self.model_instance.gen_ppl_metric.reset()
        if self.cfg.eval.disable_ema:
            self.logger.info("Disabling EMA for sampling.")
            self.model_instance.ema = None

    def _reset_peak_stats(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
            except RuntimeError:
                torch.cuda.reset_peak_memory_stats()

    def _capture_peak_vram(self) -> int:
        if torch.cuda.is_available():
            try:
                return torch.cuda.max_memory_allocated(torch.cuda.current_device())
            except RuntimeError:
                return torch.cuda.max_memory_allocated()
        return 0

    def _auto_batch_prompts(self, prompt_cap: Optional[int]) -> Dict[str, Any]:
        settings = self.generation_settings
        if not settings.auto_batch or not torch.cuda.is_available():
            return {"applied": False}
        target = settings.auto_batch_target_pct or 0.0
        if target <= 0.0:
            return {"applied": False}

        warmup_cap = min(len(self.prompts), max(settings.auto_batch_warmup_prompts, settings.batch_size))
        if warmup_cap <= 0 or self.tokenizer is None or self.model_instance is None:
            return {"applied": False}

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_mem = getattr(props, "total_memory", 0)
        trial_records = list(self.prompts[:warmup_cap])
        max_batch = settings.max_auto_batch_size or warmup_cap
        trial = max(1, min(settings.batch_size, warmup_cap))
        best_batch = trial
        peak_bytes = 0

        while trial <= max_batch:
            batch_sample = trial_records[:trial]
            settings.batch_size = len(batch_sample)
            _apply_batch_size(self.cfg, settings.batch_size)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_index)
            try:
                self._execute_conditioned_batch(batch_sample, prompt_cap, record_results=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    self.logger.warning("Auto-batch trial batch=%d hit OOM; reducing.", trial)
                    if trial == 1:
                        best_batch = 1
                        break
                    trial = max(1, trial // 2)
                    best_batch = trial
                    break
                raise

            torch.cuda.synchronize(device_index)
            peak_bytes = torch.cuda.max_memory_allocated(device_index)
            usage = peak_bytes / total_mem if total_mem else 0.0
            self.logger.info("Auto-batch trial batch=%d consumed %.1f%% VRAM.", trial, usage * 100)
            best_batch = trial
            if usage >= target:
                break
            next_trial = min(trial * 2, max_batch, warmup_cap)
            if next_trial == trial:
                break
            trial = next_trial

        settings.batch_size = best_batch
        _apply_batch_size(self.cfg, best_batch)
        self.logger.info(
            "Auto-batch settled on batch=%d (peak %.1f%% VRAM across %d warmup prompts).",
            best_batch,
            (peak_bytes / total_mem * 100) if total_mem else 0.0,
            warmup_cap,
        )
        return {
            "applied": True,
            "batch_size": best_batch,
            "peak_bytes": peak_bytes,
            "target_pct": target,
            "warmup_samples": warmup_cap,
        }

    def _execute_conditioned_batch(
        self,
        batch: Sequence[PromptRecord],
        prompt_cap: Optional[int],
        record_results: bool,
    ) -> List[GenerationResult]:
        if not batch or self.tokenizer is None or self.model_instance is None:
            return []

        init_tokens, prompt_lengths, truncated_flags = _build_initial_state(
            tokenizer=self.tokenizer,
            prompts=batch,
            max_length=self.cfg.model.length,
            max_prompt_tokens=prompt_cap,
            add_bos=self.generation_settings.add_bos,
            add_eos=self.generation_settings.add_eos,
            mask_token_id=self.model_instance.mask_index,
        )
        init_tokens = init_tokens.to(self.model_instance.device)
        sample_ids = self.model_instance.restore_model_and_sample(
            num_steps=self.generation_settings.sampling_steps,
            initial_state=init_tokens,
        ).cpu()

        if not record_results:
            return []

        mask_id = self.model_instance.mask_index
        stop_tokens = self.stop_tokens
        if stop_tokens is None and hasattr(self.tokenizer, "convert_tokens_to_ids"):
            stop_tokens = _resolve_stop_tokens(self.tokenizer)
        stop_ids: Set[int] = stop_tokens.stop_ids if stop_tokens else set()
        outputs: List[GenerationResult] = []
        for row, record in enumerate(batch):
            tokens = sample_ids[row].tolist()
            prompt_len = prompt_lengths[row]
            prompt_mask = [1] * int(prompt_len)
            if len(tokens) > prompt_len:
                prompt_mask.extend([0] * (len(tokens) - int(prompt_len)))
            completion_tokens, _, _ = _strip_completion_tokens(
                tokens=tokens,
                prompt_length=prompt_len,
                stop_ids=stop_ids,
                mask_id=mask_id,
                stop_sequences=stop_tokens.stop_sequences if stop_tokens else (),
            )

            if self.tokenizer is not None:
                completion_tokens = _truncate_at_turn_boundary(completion_tokens, self.tokenizer)

            raw_completion_text = self.tokenizer.decode(
                completion_tokens,
                skip_special_tokens=False,
            )
            completion_text = self.tokenizer.decode(
                completion_tokens,
                skip_special_tokens=True,
            ).strip()
            if stop_tokens is not None:
                _assert_no_extra_turn_tokens(
                    completion_tokens=completion_tokens,
                    decoded_completion=raw_completion_text,
                    prompt_length=prompt_len,
                    tokens=tokens,
                    stop_tokens=stop_tokens,
                    logger=self.logger,
                )
            metadata = _format_metadata(
                base=record.metadata,
                shard_metadata=self.shard_metadata,
                cfg=self.cfg,
                generation=self.generation_settings,
            )
            if truncated_flags[row]:
                metadata.setdefault("prompt_truncated", True)
            outputs.append(
                GenerationResult(
                    prompt_id=record.prompt_id,
                    prompt=record.prompt,
                    completion=completion_text,
                    full_text=completion_text,
                    token_ids=tokens,
                    prompt_length=prompt_len,
                    prompt_mask=prompt_mask,
                    metadata=metadata,
                )
            )
        return outputs

    def _generate_conditioned(self, prompt_cap: Optional[int]) -> List[GenerationResult]:
        if not self.prompts:
            return []
        results: List[GenerationResult] = []
        total = len(self.prompts)
        batch_size = self.generation_settings.batch_size
        self.logger.info("Beginning conditioned generation for %d prompts (batch=%d).", total, batch_size)
        checkpoints = {int(total * frac) for frac in [0.1, 0.25, 0.5, 0.75]}
        checkpoints.add(total)
        processed = 0
        for batch in _chunk(self.prompts, batch_size):
            processed += len(batch)
            results.extend(self._execute_conditioned_batch(batch, prompt_cap, record_results=True))
            if total <= 10 or processed in checkpoints:
                pct = (processed / total) * 100 if total else 100
                self.logger.info(
                    "Processed %d/%d prompts (%.1f%%) with batch=%d.",
                    processed,
                    total,
                    pct,
                    batch_size,
                )
        return results

    def _generate_unconditional(self) -> List[GenerationResult]:
        if self.generation_settings.unconditional_samples <= 0 or self.tokenizer is None or self.model_instance is None:
            return []
        remaining = self.generation_settings.unconditional_samples
        mask_id = self.model_instance.mask_index
        stop_tokens = self.stop_tokens
        if stop_tokens is None and hasattr(self.tokenizer, "convert_tokens_to_ids"):
            stop_tokens = _resolve_stop_tokens(self.tokenizer)
        stop_ids: Set[int] = stop_tokens.stop_ids if stop_tokens else set()
        results: List[GenerationResult] = []
        unconditional_index = 0
        total = remaining
        self.logger.info("Beginning unconditional generation for %d samples (batch=%d).", total, self.generation_settings.batch_size)
        checkpoints = {int(total * frac) for frac in [0.1, 0.25, 0.5, 0.75] if total}
        checkpoints.add(total)
        completed = 0
        while remaining > 0:
            batch_size = min(self.generation_settings.batch_size, remaining)
            sample_ids = self.model_instance.restore_model_and_sample(
                num_steps=self.generation_settings.sampling_steps,
                initial_state=None,
            ).cpu()
            tokens_batch = sample_ids[:batch_size]
            for tokens_tensor in tokens_batch:
                tokens = tokens_tensor.tolist()
                completion_tokens, _, _ = _strip_completion_tokens(
                    tokens=tokens,
                    prompt_length=0,
                    stop_ids=stop_ids,
                    mask_id=mask_id,
                    stop_sequences=stop_tokens.stop_sequences if stop_tokens else (),
                )

                if self.tokenizer is not None:
                    completion_tokens = _truncate_at_turn_boundary(completion_tokens, self.tokenizer)

                raw_completion_text = self.tokenizer.decode(
                    completion_tokens,
                    skip_special_tokens=False,
                )
                completion_text = self.tokenizer.decode(
                    completion_tokens,
                    skip_special_tokens=True,
                ).strip()
                if stop_tokens is not None:
                    _assert_no_extra_turn_tokens(
                        completion_tokens=completion_tokens,
                        decoded_completion=raw_completion_text,
                        prompt_length=0,
                        tokens=tokens,
                        stop_tokens=stop_tokens,
                        logger=self.logger,
                    )
                metadata = _format_metadata(
                    base={"prompt_type": "unconditional"},
                    shard_metadata=self.shard_metadata,
                    cfg=self.cfg,
                    generation=self.generation_settings,
                )
                results.append(
                    GenerationResult(
                        prompt_id=f"uncond:{unconditional_index}",
                        prompt="",
                        completion=completion_text,
                        full_text=completion_text,
                        token_ids=tokens,
                        prompt_length=0,
                        prompt_mask=[0] * len(tokens),
                        metadata=metadata,
                    )
                )
                unconditional_index += 1
            remaining -= batch_size
            completed += batch_size
            if total <= 10 or completed in checkpoints:
                pct = (completed / total) * 100 if total else 100
                self.logger.info(
                    "Generated %d/%d unconditional samples (%.1f%%) with batch=%d.",
                    completed,
                    total,
                    pct,
                    self.generation_settings.batch_size,
                )
        return results

def run_generation(
    prompts: Optional[Sequence[PromptRecord]],
    model: ModelSettings,
    generation: GenerationSettings,
    safety: SafetySettings,
    shard_metadata: Dict[str, Any],
) -> GenerationRun:
    engine = GenerationEngine(
        prompts=prompts,
        model=model,
        generation=generation,
        safety=safety,
        shard_metadata=shard_metadata,
    )
    with torch.inference_mode():
        return engine.run()


def run_llada_stub(*args, **kwargs) -> None:
    """Placeholder for future LLaDA integration."""
    raise NotImplementedError("LLaDA sampling is not yet integrated with the unsafe artifact pipeline.")
