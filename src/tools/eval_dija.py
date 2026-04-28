import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import importlib
import sys

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from sampling.sample_text import SafetySettings, resolve_eta_config, _prepare_unsafe_artifacts
from sampling.safe_hooks import build_llada_repellency_hook

logger = logging.getLogger(__name__)


def _import_third_party(*module_names: str):
    last_exc = None
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise ModuleNotFoundError("No module names provided.")


def _resolve_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value in (None, "", "null"):
        return None
    path_str = str(path_value)
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(to_absolute_path(path_str))


def _resolve_optional_str(value: Optional[str]) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    return str(value)


def _configure_safety(cfg: DictConfig) -> SafetySettings:
    eta_value, eta_from_scale = resolve_eta_config(cfg.safety)
    unsafe_artifacts = _resolve_path(cfg.safety.unsafe_artifacts)
    unsafe_artifact_root = _resolve_path(cfg.safety.unsafe_artifact_root)
    tokenizer_name = cfg.model.get("tokenizer_name") if hasattr(cfg, "model") else None
    if tokenizer_name in (None, "", "null"):
        tokenizer_name = cfg.model.checkpoint if hasattr(cfg, "model") else None
    return SafetySettings(
        enabled=cfg.safety.enabled,
        eta=eta_value,
        eta_from_scale=eta_from_scale,
        weight_mode=cfg.safety.get("weight_mode", "eta_beta_hat"),
        beta_hat_mode=cfg.safety.get("beta_hat_mode", "mc_mean"),
        beta_hat_clip_min=cfg.safety.get("beta_hat_clip_min"),
        beta_hat_clip_max=cfg.safety.get("beta_hat_clip_max"),
        schedule_mode=cfg.safety.get("schedule_mode", "hard_window"),
        scale=cfg.safety.get("scale"),
        unsafe_artifacts=unsafe_artifacts,
        unsafe_artifact_root=unsafe_artifact_root,
        unsafe_artifact_name=_resolve_optional_str(cfg.safety.unsafe_artifact_name),
        unsafe_prototypes=(
            Path(cfg.safety.unsafe_prototypes).expanduser()
            if cfg.safety.unsafe_prototypes
            else None
        ),
        critical_steps=cfg.safety.get("critical_steps"),
        t_start=cfg.safety.get("t_start"),
        t_end=cfg.safety.get("t_end"),
        use_semantic_gating=cfg.safety.get("use_semantic_gating", False),
        semantic_weight=float(cfg.safety.get("semantic_weight", 0.0)),
        semantic_temp=float(cfg.safety.get("semantic_temp", 1.0)),
        semantic_sigma=cfg.safety.get("semantic_sigma"),
        cache_semantic_ref=cfg.safety.get("cache_semantic_ref", False),
        semantic_ref_path=(
            Path(cfg.safety.semantic_ref_path).expanduser()
            if cfg.safety.get("semantic_ref_path")
            else None
        ),
        semantic_checkpoint=(
            Path(cfg.safety.semantic_checkpoint).expanduser()
            if cfg.safety.get("semantic_checkpoint")
            else None
        ),
        semantic_embed_attr=cfg.safety.get("semantic_embed_attr"),
        auto_build_unsafe_artifacts=bool(cfg.safety.get("auto_build_unsafe_artifacts", False)),
        tokenizer_name_or_path=str(tokenizer_name) if tokenizer_name else None,
    )


def _pick_behavior_text(item: dict) -> str:
    for key in (
        "Behavior",
        "behavior",
        "refined prompt",
        "refined_prompt",
        "refined_prompt_text",
        "vanilla prompt",
        "vanilla_prompt",
        "prompt",
        "goal",
        "target",
        "refined_goal",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _pick_refined_behavior(item: dict) -> str:
    for key in ("Refined_behavior", "refined_behavior", "refined prompt", "refined_prompt"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _normalize_attack_prompt(path: Path, output_dir: Path) -> Tuple[Path, bool]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read attack_prompt JSON at {path}: {exc}") from exc

    if not isinstance(data, list):
        raise SystemExit("DIJA attack_prompt must be a JSON list of records.")

    changed = False
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if "BehaviorID" not in item or not str(item.get("BehaviorID", "")).strip():
            item["BehaviorID"] = f"item_{idx}"
            changed = True
        if "Behavior" not in item or not str(item.get("Behavior", "")).strip():
            behavior = _pick_behavior_text(item)
            if behavior:
                item["Behavior"] = behavior
                changed = True
        if "Refined_behavior" not in item or not str(item.get("Refined_behavior", "")).strip():
            refined = _pick_refined_behavior(item)
            if refined:
                item["Refined_behavior"] = refined
                changed = True

    if not changed:
        return path, False

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "attack_prompt_normalized.json"
    normalized_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.warning("Normalized DIJA attack_prompt; writing %s", normalized_path)
    return normalized_path, True


def _iter_generate_modules():
    repo_root = Path(__file__).resolve().parents[2]
    dija_root = repo_root / "src" / "third_party" / "DIJA" / "run_harmbench"
    if dija_root.exists() and str(dija_root) not in sys.path:
        sys.path.insert(0, str(dija_root))
    module_names = [
        "utility.generate_function",
        "third_party.DIJA.run_harmbench.utility.generate_function",
        "src.third_party.DIJA.run_harmbench.utility.generate_function",
    ]
    seen = set()
    for name in module_names:
        if name in seen:
            continue
        seen.add(name)
        try:
            yield importlib.import_module(name)
        except ModuleNotFoundError:
            continue


def _install_llada_hook(safety: SafetySettings, generate_module=None) -> None:
    modules = list(_iter_generate_modules())
    if generate_module is not None and generate_module not in modules:
        modules.append(generate_module)
    if not modules:
        raise ModuleNotFoundError("Could not import DIJA generate_function module.")

    if not safety.enabled:
        for module in modules:
            module.set_logits_hook(
                logits_hook=None,
                logits_hook_ctx=None,
                t_start=None,
                t_end=None,
                hook_factory=None,
            )
        return

    try:
        unsafe_path = _prepare_unsafe_artifacts(safety)
    except Exception as exc:
        logger.warning("Failed to resolve unsafe artifacts for DIJA: %s", exc)
        print(f"[dija] unsafe artifact resolution failed: {exc}", flush=True)
        safety.enabled = False
        for module in modules:
            module.set_logits_hook(
                logits_hook=None,
                logits_hook_ctx=None,
                t_start=None,
                t_end=None,
                hook_factory=None,
            )
        return

    if unsafe_path is None:
        logger.warning("Unsafe artifacts not found; disabling DIJA safety hook.")
        print("[dija] unsafe artifacts not found; disabling safety hook.", flush=True)
        safety.enabled = False
        for module in modules:
            module.set_logits_hook(
                logits_hook=None,
                logits_hook_ctx=None,
                t_start=None,
                t_end=None,
                hook_factory=None,
            )
        return
    logger.info("Resolved unsafe artifacts to %s", unsafe_path)
    print(f"[dija] unsafe artifacts resolved: {unsafe_path}", flush=True)

    def _factory(tokenizer, device):
        return build_llada_repellency_hook(tokenizer, safety, device)

    for module in modules:
        module.set_logits_hook(
            hook_factory=_factory,
            logits_hook_ctx={},
            t_start=safety.t_start,
            t_end=safety.t_end,
        )
    logger.info("[INFO] Repellency is active (hook installed).")


def _build_args_from_cfg(cfg: DictConfig) -> list[str]:
    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_prompt = cfg.jailbreak.attack_prompt or cfg.data.dataset_json
    if not attack_prompt:
        raise SystemExit("Provide jailbreak.attack_prompt or data.dataset_json for DIJA evaluation.")
    attack_prompt = _resolve_path(str(attack_prompt)) or Path(str(attack_prompt))
    attack_prompt, _ = _normalize_attack_prompt(attack_prompt, output_dir)
    output_name = cfg.jailbreak.output_name or "dija_outputs.json"
    args = [
        "--model_path",
        str(cfg.model.checkpoint),
        "--attack_prompt",
        str(attack_prompt),
        "--output_json",
        str(output_dir / output_name),
        "--steps",
        str(cfg.jailbreak.steps),
        "--gen_length",
        str(cfg.jailbreak.gen_length),
        "--mask_id",
        str(cfg.jailbreak.mask_id),
        "--mask_counts",
        str(cfg.jailbreak.mask_counts),
        "--attack_method",
        str(cfg.jailbreak.attack_method),
    ]
    if cfg.jailbreak.defense_method not in (None, "", "null"):
        args.extend(["--defense_method", str(cfg.jailbreak.defense_method)])
    return args


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting LLaDA Evaluation (DIJA wrapper)...")
    logger.info("Config snapshot:\n%s", OmegaConf.to_yaml(cfg, resolve=False))
    args = _build_args_from_cfg(cfg)
    import sys
    sys.argv = [sys.argv[0]] + args
    target_script = _import_third_party(
        "third_party.DIJA.run_harmbench.models.harmbench_llada",
        "src.third_party.DIJA.run_harmbench.models.harmbench_llada",
    )
    safety = _configure_safety(cfg)
    generate_module = sys.modules.get(getattr(target_script.generate_llada, "__module__", ""))
    _install_llada_hook(safety, generate_module=generate_module)
    target_script.main()


if __name__ == "__main__":
    main()
