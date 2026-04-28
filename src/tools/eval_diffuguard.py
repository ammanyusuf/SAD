import logging
import os
from pathlib import Path
from typing import Optional
import importlib
import sys
import json

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from sampling.sample_text import SafetySettings, resolve_eta_config, _prepare_unsafe_artifacts
from sampling.safe_hooks import build_dream_repellency_hook, build_llada_repellency_hook

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
        unsafe_artifact_name=cfg.safety.unsafe_artifact_name,
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


def _iter_generate_modules():
    repo_root = Path(__file__).resolve().parents[2]
    diffuguard_root = repo_root / "src" / "third_party" / "DiffuGuard"
    if diffuguard_root.exists() and str(diffuguard_root) not in sys.path:
        sys.path.insert(0, str(diffuguard_root))
    module_names = [
        "utility.generate_function_llada",
        "third_party.DiffuGuard.utility.generate_function_llada",
        "src.third_party.DiffuGuard.utility.generate_function_llada",
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


def _iter_dream_generate_modules():
    repo_root = Path(__file__).resolve().parents[2]
    diffuguard_root = repo_root / "src" / "third_party" / "DiffuGuard"
    if diffuguard_root.exists() and str(diffuguard_root) not in sys.path:
        sys.path.insert(0, str(diffuguard_root))
    module_names = [
        "utility.generate_function_dream",
        "third_party.DiffuGuard.utility.generate_function_dream",
        "src.third_party.DiffuGuard.utility.generate_function_dream",
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
        raise ModuleNotFoundError("Could not import DiffuGuard generate_function_llada module.")

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
        logger.warning("Failed to resolve unsafe artifacts for DiffuGuard: %s", exc)
        print(f"[diffuguard] unsafe artifact resolution failed: {exc}", flush=True)
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
        logger.warning("Unsafe artifacts not found; disabling DiffuGuard safety hook.")
        print("[diffuguard] unsafe artifacts not found; disabling safety hook.", flush=True)
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
    print(f"[diffuguard] unsafe artifacts resolved: {unsafe_path}", flush=True)

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


def _install_dream_hook(safety: SafetySettings, generate_module=None) -> None:
    modules = list(_iter_dream_generate_modules())
    if generate_module is not None and generate_module not in modules:
        modules.append(generate_module)
    if not modules:
        raise ModuleNotFoundError("Could not import DiffuGuard generate_function_dream module.")

    if not safety.enabled:
        for module in modules:
            module.set_logits_hook(
                logits_hook=None,
                logits_hook_ctx=None,
                hook_factory=None,
            )
        return

    try:
        unsafe_path = _prepare_unsafe_artifacts(safety)
    except Exception as exc:
        logger.warning("Failed to resolve unsafe artifacts for DiffuGuard (Dream): %s", exc)
        print(f"[diffuguard] unsafe artifact resolution failed: {exc}", flush=True)
        safety.enabled = False
        for module in modules:
            module.set_logits_hook(
                logits_hook=None,
                logits_hook_ctx=None,
                hook_factory=None,
            )
        return

    if unsafe_path is None:
        logger.warning("Unsafe artifacts not found; disabling DiffuGuard safety hook (Dream).")
        print("[diffuguard] unsafe artifacts not found; disabling safety hook.", flush=True)
        safety.enabled = False
        for module in modules:
            module.set_logits_hook(
                logits_hook=None,
                logits_hook_ctx=None,
                hook_factory=None,
            )
        return
    logger.info("Resolved unsafe artifacts to %s", unsafe_path)
    print(f"[diffuguard] unsafe artifacts resolved: {unsafe_path}", flush=True)

    def _factory(tokenizer, device, **ctx):
        return build_dream_repellency_hook(tokenizer, safety, device, **ctx)

    for module in modules:
        module.set_logits_hook(
            hook_factory=_factory,
            logits_hook_ctx={},
        )
    logger.info("[INFO] Repellency is active (Dream hook installed).")


def _build_args_from_cfg(cfg: DictConfig, model_family: Optional[str] = None) -> list[str]:
    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_prompt = cfg.jailbreak.attack_prompt or cfg.data.dataset_json
    if not attack_prompt:
        raise SystemExit("Provide jailbreak.attack_prompt or data.dataset_json for DiffuGuard evaluation.")
    attack_prompt = _resolve_path(str(attack_prompt)) or Path(str(attack_prompt))
    block_length = cfg.jailbreak.block_length
    if block_length in (None, "", "null"):
        block_length = cfg.jailbreak.gen_length
    output_name = cfg.jailbreak.output_name or "diffuguard_outputs.json"
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
    if (model_family or "").lower() != "dream":
        args.extend(["--block_length", str(block_length)])
    prompt_limit_env = os.getenv("PROMPT_LIMIT")
    prompt_limit_cfg = None
    if hasattr(cfg, "data") and cfg.data is not None:
        prompt_limit_cfg = cfg.data.get("limit")
    prompt_limit_jb = getattr(cfg.jailbreak, "max_prompts", None)
    for limit in (prompt_limit_env, prompt_limit_cfg, prompt_limit_jb):
        if limit not in (None, "", "null"):
            args.extend(["--max_prompts", str(limit)])
            break
    defense_method = cfg.jailbreak.defense_method
    if defense_method not in (None, "", "null"):
        args.extend(["--defense_method", str(defense_method)])
    if cfg.jailbreak.temperature is not None:
        args.extend(["--temperature", str(cfg.jailbreak.temperature)])
    if cfg.jailbreak.cfg_scale is not None and (model_family or "").lower() != "dream":
        args.extend(["--cfg_scale", str(cfg.jailbreak.cfg_scale)])
    if cfg.jailbreak.remasking:
        args.extend(["--remasking", str(cfg.jailbreak.remasking)])
    if cfg.jailbreak.random_rate is not None:
        args.extend(["--random_rate", str(cfg.jailbreak.random_rate)])
    if cfg.jailbreak.injection_step not in (None, "", "null"):
        args.extend(["--injection_step", str(cfg.jailbreak.injection_step)])
    if cfg.jailbreak.alpha0 is not None:
        args.extend(["--alpha0", str(cfg.jailbreak.alpha0)])
    sp_mode = cfg.jailbreak.sp_mode
    if (
        defense_method
        and str(defense_method).lower() == "diffuguard"
        and str(sp_mode).lower() == "off"
    ):
        # Paper-style DiffuGuard uses hidden-state audit + repair; promote sp_mode.
        sp_mode = "hidden"
        logger.info("DiffuGuard defense requested with sp_mode=off; using sp_mode=hidden.")
    if sp_mode:
        args.extend(["--sp_mode", str(sp_mode)])
    if cfg.jailbreak.sp_threshold is not None:
        args.extend(["--sp_threshold", str(cfg.jailbreak.sp_threshold)])
    if cfg.jailbreak.refinement_steps is not None:
        args.extend(["--refinement_steps", str(cfg.jailbreak.refinement_steps)])
    if cfg.jailbreak.remask_ratio is not None:
        args.extend(["--remask_ratio", str(cfg.jailbreak.remask_ratio)])
    if cfg.jailbreak.suppression_value is not None:
        args.extend(["--suppression_value", str(cfg.jailbreak.suppression_value)])
    if cfg.jailbreak.fill_all_masks:
        args.append("--fill_all_masks")
    if cfg.jailbreak.debug_print:
        args.append("--debug_print")
    if cfg.jailbreak.correct_only_first_block is False:
        args.append("--no_correct_only_first_block")
    if cfg.jailbreak.auto_pick_gpu is False:
        args.append("--no_auto_pick_gpu")
    return args


def _convert_diffuguard_output_to_generations(
    output_json: Path,
    generations_path: Path,
) -> None:
    if not output_json.exists():
        logger.warning("DiffuGuard output JSON not found at %s; skipping generations.jsonl export.", output_json)
        return
    raw = json.loads(output_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a list in {output_json}, found {type(raw).__name__}.")
    generations_path.parent.mkdir(parents=True, exist_ok=True)
    with generations_path.open("w", encoding="utf-8") as fp:
        for idx, rec in enumerate(raw):
            if not isinstance(rec, dict):
                continue
            prompt = rec.get("final_prompt")
            if not isinstance(prompt, str) or not prompt:
                used_prompt_type = rec.get("used_prompt_type")
                if used_prompt_type == "refined":
                    prompt = rec.get("refined prompt") or rec.get("vanilla prompt") or ""
                else:
                    prompt = rec.get("vanilla prompt") or rec.get("refined prompt") or ""
            prompt_id = rec.get("prompt_id") or rec.get("id") or idx
            metadata = {}
            for key in (
                "category",
                "source",
                "goal",
                "refined_goal",
                "Behavior",
                "Refined_behavior",
                "target",
                "used_prompt_type",
            ):
                if key in rec:
                    metadata[key] = rec.get(key)
            out = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "completion": rec.get("response", ""),
                "metadata": metadata,
            }
            if "runtime_prompt_tokenized" in rec:
                out["runtime_prompt_tokenized"] = rec.get("runtime_prompt_tokenized")
            if "runtime_prompt_token_ids" in rec:
                out["runtime_prompt_token_ids"] = rec.get("runtime_prompt_token_ids")
            if "decode_start_token_index" in rec:
                out["decode_start_token_index"] = rec.get("decode_start_token_index")
            if "decode_start_reason" in rec:
                out["decode_start_reason"] = rec.get("decode_start_reason")
            for key in (
                "timing_total_sec",
                "timing_defense_sec",
                "timing_prompt_build_sec",
                "timing_detection_sec",
                "timing_generation_sec",
            ):
                if key in rec:
                    out[key] = rec.get(key)
            if "hardware" in rec:
                out["hardware"] = rec.get("hardware")
            fp.write(json.dumps(out) + "\n")
    logger.info("Wrote generations.jsonl to %s", generations_path)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting DiffuGuard evaluation...")
    seed = int(cfg.gen.get("seed", 0))
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info("RNG seeded with gen.seed=%d", seed)
    logger.info("Config snapshot:\n%s", OmegaConf.to_yaml(cfg, resolve=False))
    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config_merged.yaml", resolve=False)
    model_family = str(getattr(cfg.model, "family", "llada")).lower()
    safety = _configure_safety(cfg)
    args = _build_args_from_cfg(cfg, model_family=model_family)
    import sys
    sys.argv = [sys.argv[0]] + args
    if model_family == "dream":
        target_script = _import_third_party(
            "third_party.DiffuGuard.models.jailbreakbench_dream",
            "src.third_party.DiffuGuard.models.jailbreakbench_dream",
        )
        generate_module = sys.modules.get(getattr(target_script.generate_dream_hidden, "__module__", ""))
        _install_dream_hook(safety, generate_module=generate_module)
    else:
        target_script = _import_third_party(
            "third_party.DiffuGuard.models.jailbreakbench_llada",
            "src.third_party.DiffuGuard.models.jailbreakbench_llada",
        )
        generate_module = sys.modules.get(getattr(target_script.generate_llada, "__module__", ""))
        _install_llada_hook(safety, generate_module=generate_module)
    target_script.main()
    output_name = cfg.jailbreak.output_name or "diffuguard_outputs.json"
    output_json = output_dir / output_name
    generations_path = output_dir / "generations.jsonl"
    _convert_diffuguard_output_to_generations(output_json, generations_path)


if __name__ == "__main__":
    main()
