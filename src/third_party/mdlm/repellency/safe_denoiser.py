"""Discrete repellency method using a mask-based kernel, for use with MDLM."""

import logging
import math
import os
import csv
import weakref
from pathlib import Path
from typing import Optional
import re

import torch
from diagnostics.dist_logging import DistributionLogger
from utils.rbf_utils import median_heuristic_sigma, normalize_embeddings, rbf_kernel_matrix

from unsafe_prep.prototypes import load_unsafe_prototypes, UnsafePrototypes
from unsafe_prep.semantic_utils import masked_mean_pool
from .alignment import AlignmentResult, build_alignment_strategy
from .repellency_methods_fast import RepellencyMethod, register_conditioning_method

logger = logging.getLogger(__name__)


@register_conditioning_method(name='mask_kernel_discrete')
class MaskKernelRepellency(RepellencyMethod):
    """
    ref_data: LongTensor [N, L] of tokenized 'unsafe' reference continuations
    forward_fn: unused here (kept for parity with base)
    embed_fn: encoder used for semantic gating (ignored if use_semantic_gating=False)
    pad_index: token id treated as wildcard in refs (ignored for matching/counts)
    """
    _csv_reset_done: bool = False

    def __init__(self, ref_data, embed_fn, forward_fn,
                 num_timesteps, max_idx, beta_min, beta_max,
                 vocab_size, mask_index, pad_index=None,
                 eos_id=None,
                 unsafe_prototypes_path: Optional[str] = None,
                 use_semantic_gating: bool = False,
                 semantic_weight: float = 0.0,
                 alignment_strategy: Optional[str] = None,
                 semantic_sigma: Optional[float] = None,
                 semantic_temp: float = 1.0,
                 **kwargs):
        ref_data = self._sanitize_ref_data(ref_data, vocab_size, mask_index, pad_index)
        super().__init__(ref_data, embed_fn, forward_fn,
                         num_timesteps, max_idx, beta_min, beta_max, n_embed=0, **kwargs)
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.pad_index = pad_index
        self.eos_id = eos_id
        self.use_semantic_gating = use_semantic_gating
        self.semantic_weight = float(semantic_weight)
        semantic_sigma_raw = kwargs.get("semantic_sigma", semantic_sigma)
        self.semantic_sigma = float(semantic_sigma_raw) if semantic_sigma_raw is not None else None
        semantic_temp_raw = kwargs.get("semantic_temp", semantic_temp)
        self.semantic_temp = float(max(semantic_temp_raw, 1e-4))
        self.continuation_length = self._infer_continuation_length()
        self.alignment_strategy_name = alignment_strategy
        self.alignment_strategy = build_alignment_strategy(
            alignment_strategy,
            mask_index,
            pad_index,
            self.continuation_length,
            vocab_size,
        )
        self._warned_no_alignment = False
        if self.semantic_weight != 0.0 and semantic_temp_raw == 1.0:
            logger.warning(
                "semantic_weight is deprecated; use semantic_temp instead for semantic gating strength. "
                "semantic_weight is still applied for backward compatibility."
            )

        self.ignore_ids = self._compute_ignore_ids(kwargs.get('ignore_ids', []))
        if self.pad_index is not None and self.eos_id is not None and self.pad_index == self.eos_id:
            logger.warning("pad_index equals eos_id; treating both as ignored for repellency histograms.")
        if torch.is_tensor(self.ref_data):
            max_ref = int(self.ref_data.max().item()) if self.ref_data.numel() else -1
            if max_ref >= self.vocab_size:
                raise ValueError(f"Unsafe references contain id >= vocab_size ({max_ref} >= {self.vocab_size}) after sanitization.")

        self._unsafe_prototypes: Optional[UnsafePrototypes] = None
        self.proto_centroids = None          # [K, L]
        self.proto_cluster_sizes = None      # [K]
        self.proto_token_histograms = None   # [K, V]
        self.proto_cluster_embeddings = None # [K, D] or None
        self.semantic_ref_embeddings: Optional[torch.FloatTensor] = None  # [N, D]
        self.cache_semantic_ref = kwargs.get("cache_semantic_ref", False)
        self.semantic_ref_path = kwargs.get("semantic_ref_path", None)
        self.csv_log_path = kwargs.get("csv_log_path", os.getenv("SAFE_REPELLENCY_CSV_LOG"))
        unsafe_artifact_name = kwargs.get("unsafe_artifacts_name")
        run_tag = self._build_run_tag(unsafe_artifact_name)
        self._apply_run_tag_paths(run_tag)
        if self.csv_log_path and not MaskKernelRepellency._csv_reset_done:
            try:
                csv_path = Path(self.csv_log_path)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_path.write_text("", encoding="utf-8")
                MaskKernelRepellency._csv_reset_done = True
            except OSError as exc:
                logger.warning("Failed to reset repellency CSV log: %s", exc)
        self._metrics_buffer = []
        buffer_size_raw = os.getenv("SAFE_REPELLENCY_CSV_BUFFER_SIZE")
        buffer_size = 64
        if buffer_size_raw is not None:
            try:
                buffer_size = max(1, int(buffer_size_raw))
            except ValueError:
                pass
        self._csv_buffer_size = buffer_size
        self._csv_flush_finalizer = weakref.finalize(self, self._flush_metrics_buffer)
        self._distribution_logger = DistributionLogger.from_env(
            run_id=os.getenv("SAFE_DIST_LOG_RUN_ID"),
            t_start=self.t_start if self.t_start is not None else 0,
            t_end=self.t_end if self.t_end is not None else (self.num_timesteps or 0),
        )
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None and self._distribution_logger is not None:
            self._distribution_logger.set_decoder(
                lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
            )

        if unsafe_prototypes_path is not None:
            proto = load_unsafe_prototypes(Path(unsafe_prototypes_path))
            self._unsafe_prototypes = proto
            self.proto_centroids = proto.centroids.to('cuda')
            self.proto_cluster_sizes = proto.cluster_sizes.to('cuda')
            self.proto_token_histograms = proto.token_histograms.to(torch.float32).to('cuda')
            if (
                self.proto_token_histograms is not None
                and self.proto_token_histograms.size(-1) > self.vocab_size
            ):
                self.proto_token_histograms = self.proto_token_histograms[:, : self.vocab_size]
            if self.proto_token_histograms is not None and self.ignore_ids:
                ignore_tensor = torch.tensor(sorted(self.ignore_ids),
                                             device=self.proto_token_histograms.device,
                                             dtype=torch.long)
                if ignore_tensor.numel() > 0:
                    self.proto_token_histograms[:, ignore_tensor] = 0.0
                    denom = self.proto_token_histograms.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    self.proto_token_histograms = self.proto_token_histograms / denom
            if proto.cluster_embeddings is not None:
                self.proto_cluster_embeddings = (
                    proto.cluster_embeddings.to(torch.float32).to('cuda')
                )
        if isinstance(self.proj_refs, torch.Tensor):
            self.U_T = self.proj_refs.t().contiguous()
        else:
            self.U_T = None
        if self.cache_semantic_ref and self.semantic_ref_embeddings is None:
            self.semantic_ref_embeddings = self.import_semantic_ref(self.semantic_ref_path)
        if os.getenv("SAFE_REPELLENCY_DEBUG") and torch.is_tensor(self.ref_data):
            ref_tensor = self.ref_data
            pad_count = int((ref_tensor == self.pad_index).sum().item()) if self.pad_index is not None else 0
            eos_count = int((ref_tensor == self.eos_id).sum().item()) if self.eos_id is not None else 0
            logger.info(
                "Repellency refs stats: pad_index=%s eos_id=%s mask_index=%s vocab_size=%d ignore_ids=%s pad_count=%d eos_count=%d",
                str(self.pad_index),
                str(self.eos_id),
                str(self.mask_index),
                int(self.vocab_size),
                sorted(self.ignore_ids),
                pad_count,
                eos_count,
            )

    def _build_run_tag(self, unsafe_artifact_name: Optional[str]) -> str:
        eta_val = float(self.eta) if self.eta is not None else 0.0
        t_start = self.t_start if self.t_start is not None else 0
        t_end = self.t_end if self.t_end is not None else "end"
        unsafe_label = unsafe_artifact_name or os.getenv("SAFE_UNSAFE_LABEL", "unknown")
        prompt_variant = os.getenv("SAFE_PROMPT_VARIANT", "unknown")
        raw = f"eta{eta_val:.3g}_t{t_start}_t{t_end}_{prompt_variant}_{unsafe_label}"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)

    def _apply_run_tag_paths(self, run_tag: str) -> None:
        if not run_tag:
            return
        dist_base = Path(os.getenv("SAFE_DIST_LOG_DIR", "diagnostics/dist_logs")).expanduser()
        tagged_dist_dir = dist_base / run_tag
        os.environ["SAFE_DIST_LOG_DIR"] = str(tagged_dist_dir)
        if self.csv_log_path:
            csv_path = Path(self.csv_log_path).expanduser()
            if csv_path.suffix == ".csv":
                filename = csv_path.name
            else:
                filename = "repellency_stats.csv"
            tagged_dist_dir.mkdir(parents=True, exist_ok=True)
            self.csv_log_path = str(tagged_dist_dir / filename)
        if os.getenv("SAFE_DIST_LOG_PATH"):
            dist_path = Path(os.getenv("SAFE_DIST_LOG_PATH", "")).expanduser()
            if dist_path.is_absolute():
                os.environ["SAFE_DIST_LOG_PATH"] = str(tagged_dist_dir / dist_path.name)
            else:
                os.environ["SAFE_DIST_LOG_PATH"] = str(tagged_dist_dir / dist_path)
    def set_proj_ref(self):
        return self.ref_data.to('cuda')

    @staticmethod
    def _sanitize_ref_data(ref_data, vocab_size: int, mask_index: int, pad_index: Optional[int]):
        if not torch.is_tensor(ref_data):
            return ref_data
        ref_data = ref_data.to(torch.long)
        vocab_mask = ref_data >= vocab_size
        if vocab_mask.any():
            replace_candidates = [idx for idx in (mask_index, pad_index, 0) if idx is not None and idx < vocab_size]
            replace_id = replace_candidates[0] if replace_candidates else 0
            ref_data = ref_data.clone()
            ref_data[vocab_mask] = replace_id
            logger.warning(
                "Clamped %d unsafe reference tokens outside vocab_size=%d (shape preserved).",
                int(vocab_mask.sum().item()),
                vocab_size,
            )
        return ref_data

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return str(val).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_choice(name: str, default: str, allowed: set[str]) -> str:
        val = os.getenv(name, default)
        val = str(val).lower() if val is not None else default
        if val not in allowed:
            return default
        return val

    def _debug_log_kernel(
        self,
        logqt: torch.Tensor,
        score: torch.Tensor,
        kernel_mode: str,
        semantic_present: bool,
    ):
        if not self._env_flag("SAFE_REPELLENCY_DEBUG"):
            return
        logger.info(
            "logqt stats (mode=%s): min %.4f max %.4f mean %.4f; row0[:5]=%s",
            kernel_mode,
            float(logqt.min().item()),
            float(logqt.max().item()),
            float(logqt.mean().item()),
            logqt[0, :5].detach().cpu().tolist() if logqt.numel() else [],
        )
        logger.info(
            "score stats: min %.4f max %.4f mean %.4f; row0[:5]=%s",
            float(score.min().item()),
            float(score.max().item()),
            float(score.mean().item()),
            score[0, :5].detach().cpu().tolist() if score.numel() else [],
        )
        logger.info(
            "semantic_sigma=%s, semantic_temp=%.4f, semantic_weight=%.4f, semantic_present=%s",
            str(self.semantic_sigma),
            float(self.semantic_temp),
            float(self.semantic_weight),
            str(semantic_present),
        )

    def _debug_log_weights(self, w: torch.Tensor, rbf_logits: Optional[torch.Tensor]):
        if not self._env_flag("SAFE_REPELLENCY_DEBUG"):
            return
        logger.info(
            "weights stats: min %.4f max %.4f mean %.4f var %.4f; row0[:5]=%s",
            float(w.min().item()),
            float(w.max().item()),
            float(w.mean().item()),
            float(w.var(dim=-1).mean().item()),
            w[0, :5].detach().cpu().tolist() if w.numel() else [],
        )
        if rbf_logits is not None:
            logger.info(
                "rbf_logits stats: min %.4f, max %.4f, mean %.4f",
                rbf_logits.min().item(),
                rbf_logits.max().item(),
                rbf_logits.mean().item(),
            )

    def _debug_log_beta_metrics(self, metrics: dict):
        if not self._env_flag("SAFE_REPELLENCY_DEBUG"):
            return
        logger.info(
            "beta_hat_raw stats: mean=%.4e p50=%.4e p95=%.4e max=%.4e; beta_hat_len stats: mean=%.4e p50=%.4e p95=%.4e max=%.4e",
            metrics.get("beta_hat_raw_mean", float("nan")),
            metrics.get("beta_hat_raw_p50", float("nan")),
            metrics.get("beta_hat_raw_p95", float("nan")),
            metrics.get("beta_hat_raw_max", float("nan")),
            metrics.get("beta_hat_len_mean", float("nan")),
            metrics.get("beta_hat_len_p50", float("nan")),
            metrics.get("beta_hat_len_p95", float("nan")),
            metrics.get("beta_hat_len_max", float("nan")),
        )
        logger.info(
            "log_beta_raw_raw mean=%.4f max=%.4f log_beta_raw_len mean=%.4f max=%.4f log_beta_rel_raw mean=%.4f max=%.4f log_beta_rel_len mean=%.4f max=%.4f",
            metrics.get("log_beta_raw_raw_mean", float("nan")),
            metrics.get("log_beta_raw_raw_max", float("nan")),
            metrics.get("log_beta_raw_len_mean", float("nan")),
            metrics.get("log_beta_raw_len_max", float("nan")),
            metrics.get("log_beta_rel_raw_mean", float("nan")),
            metrics.get("log_beta_rel_raw_max", float("nan")),
            metrics.get("log_beta_rel_len_mean", float("nan")),
            metrics.get("log_beta_rel_len_max", float("nan")),
        )

    def _debug_log_validation(
        self,
        max_abs_diff: float,
        mean_abs_diff: float,
        disagreement: float,
        p_unsafe: torch.Tensor,
        p_unsafe_slow: torch.Tensor,
        L: int,
        V: int,
    ):
        if not self._env_flag("SAFE_REPELLENCY_DEBUG"):
            return
        logger.info(
            "Unsafe posterior validation: max_abs_diff=%.4e mean_abs_diff=%.4e argmax_disagree=%.4f",
            max_abs_diff,
            mean_abs_diff,
            disagreement,
        )
        if max_abs_diff > 1e-4:
            b0 = 0
            l_slice = slice(0, min(L, 5))
            topk = min(10, V)
            fast_slice = p_unsafe[b0, l_slice]
            slow_slice = p_unsafe_slow[b0, l_slice]
            topk_fast = fast_slice.topk(topk, dim=-1)
            topk_slow = slow_slice.topk(topk, dim=-1)
            logger.info("p_unsafe_fast topk tokens (b=0): %s", topk_fast.indices.detach().cpu().tolist())
            logger.info("p_unsafe_slow topk tokens (b=0): %s", topk_slow.indices.detach().cpu().tolist())
            logger.info("p_unsafe_fast topk probs (b=0): %s", topk_fast.values.detach().cpu().tolist())
            logger.info("p_unsafe_slow topk probs (b=0): %s", topk_slow.values.detach().cpu().tolist())

    def _debug_log_kernel_compare(self, top_overlap: float, rho_relaxed: Optional[float], rho_strict: Optional[float]):
        if not self._env_flag("SAFE_REPELLENCY_DEBUG"):
            return
        logger.info(
            "Kernel comparison: top1_overlap=%.4f rho_relaxed=%s rho_strict=%s",
            top_overlap,
            "nan" if rho_relaxed is None else f"{rho_relaxed:.4f}",
            "nan" if rho_strict is None else f"{rho_strict:.4f}",
        )

    def _debug_log_posterior_health(
        self,
        metrics: dict,
        w: torch.Tensor,
        rho: torch.Tensor,
        p_unsafe: torch.Tensor,
        mask_pos: torch.Tensor,
    ):
        if not self._env_flag("SAFE_REPELLENCY_DEBUG"):
            return
        def _entropy(p_dist: torch.Tensor) -> torch.Tensor:
            return -(p_dist.clamp_min(1e-12) * p_dist.clamp_min(1e-12).log()).sum(dim=-1)

        ent_masked = _entropy(p_unsafe).masked_select(mask_pos)
        if ent_masked.numel():
            metrics["p_unsafe_entropy_mean"] = float(ent_masked.mean().item())
            metrics["p_unsafe_entropy_p95"] = float(torch.quantile(ent_masked, 0.95))
        metrics["w_variance_mean"] = float(w.var(dim=-1).mean().item())
        metrics["rho_mean"] = float(rho.mean().item())
        clamp_min = -80.0
        clamp_max = 0.0
        for name in ("log_beta_raw_raw", "log_beta_raw_len"):
            tensor = metrics.get(name, None)
            if tensor is None:
                continue
            hits_min = (tensor <= clamp_min).float().mean().item()
            hits_max = (tensor >= clamp_max).float().mean().item()
            metrics[f"{name}_clamp_min_frac"] = hits_min
            metrics[f"{name}_clamp_max_frac"] = hits_max
            logger.info(
                "%s clamp stats: min=%.4f max=%.4f hits_min=%.4f hits_max=%.4f",
                name,
                float(tensor.min().item()),
                float(tensor.max().item()),
                hits_min,
                hits_max,
            )

    def _debug_log_guidance_pre(
        self,
        p: torch.Tensor,
        p_unsafe: torch.Tensor,
        strength_stats: dict,
    ):
        if not self._env_flag("SAFE_REPELLENCY_DEBUG"):
            return
        delta_pu = (p - p_unsafe).abs()
        mean_delta_pu = delta_pu.mean().item()
        max_delta_pu = delta_pu.max().item()
        arg_p = p.argmax(dim=-1)
        arg_pu = p_unsafe.argmax(dim=-1)
        frac_diff = (arg_p != arg_pu).float().mean().item()
        logger.info(
            "[SAFE conditioning_1] eta=%.3f, mean|p-p_unsafe|=%.3e, "
            "max|p-p_unsafe|=%.3e, frac_argmax_diff=%.4f, "
            "beta_hat_mean=%.4e, beta_hat_max=%.4e, log_beta_raw_mean=%.4f, log_beta_raw_max=%.4f, "
            "log_beta_rel_mean=%.4f, log_beta_rel_max=%.4f, "
            "g_t=%.3f, strength_mean=%.4e",
            float(self.eta),
            mean_delta_pu,
            max_delta_pu,
            frac_diff,
            strength_stats.get("beta_hat_mean", float("nan")),
            strength_stats.get("beta_hat_max", float("nan")),
            strength_stats.get("log_beta_raw_mean", float("nan")),
            strength_stats.get("log_beta_raw_max", float("nan")),
            strength_stats.get("log_beta_rel_mean", float("nan")) if strength_stats.get("log_beta_rel_mean", None) is not None else float("nan"),
            strength_stats.get("log_beta_rel_max", float("nan")) if strength_stats.get("log_beta_rel_max", None) is not None else float("nan"),
            strength_stats.get("g_t_mean", float("nan")),
            strength_stats.get("strength_mean", float("nan")),
        )

    def _debug_log_guidance_effects(
        self,
        p: torch.Tensor,
        p_safe: torch.Tensor,
        p_safe_logit: torch.Tensor,
        p_safe_prob: torch.Tensor,
        p_unsafe: torch.Tensor,
        alignment_x_t: torch.Tensor,
        strength_stats: dict,
        weights: Optional[torch.Tensor],
        strength: torch.Tensor,
        neg_item: float,
        compute_both: bool,
        t_idx: Optional[int] = None,
        prompt_id: Optional[str] = None,
        prompt_width: Optional[int] = None,
        prompt_variant: Optional[str] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        prompt_token_ids: Optional[torch.Tensor] = None,
    ):
        log_debug = self._env_flag("SAFE_REPELLENCY_DEBUG")
        dist_log_enabled = (
            self._distribution_logger is not None
            and getattr(self._distribution_logger, "_enabled", False)
        )
        if not log_debug and not dist_log_enabled:
            return {}
        mask_positions = alignment_x_t == self.mask_index
        if self.pad_index is not None:
            mask_positions = mask_positions | (alignment_x_t == self.pad_index)
        p_argmax = p.argmax(dim=-1)
        p_safe_argmax = p_safe.argmax(dim=-1)
        p_safe_argmax_logit = p_safe_logit.argmax(dim=-1)
        p_safe_argmax_prob = p_safe_prob.argmax(dim=-1)
        p_unsafe_argmax = p_unsafe.argmax(dim=-1)

        mask_frac = mask_positions.float().mean().item() if mask_positions.numel() else float("nan")

        def _mean_masked(tensor: torch.Tensor):
            if mask_positions.any():
                return tensor.masked_select(mask_positions).mean().item()
            return float("nan")

        def _kl_div(new_p: torch.Tensor):
            log_new = new_p.clamp_min(1e-30).log()
            log_old = p.clamp_min(1e-30).log()
            return (new_p * (log_new - log_old)).sum(dim=-1)

        eps = 1e-30
        p_safe_clamped = p_safe.clamp_min(eps)
        p_data_clamped = p.clamp_min(eps)
        p_unsafe_clamped = p_unsafe.clamp_min(eps)
        log_p_safe = torch.log(p_safe_clamped)
        log_p_data = torch.log(p_data_clamped)
        log_p_unsafe = torch.log(p_unsafe_clamped)
        tv_safe_data = 0.5 * (p_safe_clamped - p_data_clamped).abs().sum(dim=-1)
        tv_safe_unsafe = 0.5 * (p_safe_clamped - p_unsafe_clamped).abs().sum(dim=-1)
        tv_data_unsafe = 0.5 * (p_data_clamped - p_unsafe_clamped).abs().sum(dim=-1)
        kl_safe_data = torch.sum(p_safe_clamped * (log_p_safe - log_p_data), dim=-1)
        kl_safe_unsafe = torch.sum(p_safe_clamped * (log_p_safe - log_p_unsafe), dim=-1)
        kl_data_unsafe = torch.sum(p_data_clamped * (log_p_data - log_p_unsafe), dim=-1)
        mixture = 0.5 * (p_safe_clamped + p_data_clamped)
        mixture_log = torch.log(mixture.clamp_min(eps))
        js_safe_data = 0.5 * (
            torch.sum(p_safe_clamped * (log_p_safe - mixture_log))
            + torch.sum(p_data_clamped * (log_p_data - mixture_log))
        )
        overlap_data_unsafe = _mean_masked((p_argmax == p_unsafe_argmax).float())
        per_pos_delta_logit = (p_safe_logit - p).abs().sum(dim=-1)
        per_pos_delta_prob = (p_safe_prob - p).abs().sum(dim=-1)

        kl_logit = _kl_div(p_safe_logit)
        kl_prob = _kl_div(p_safe_prob)

        unsafe_idx = p_unsafe_argmax.unsqueeze(-1)
        unsafe_prob_before = p.gather(-1, unsafe_idx).squeeze(-1)
        unsafe_prob_after_logit = p_safe_logit.gather(-1, unsafe_idx).squeeze(-1)
        unsafe_prob_after_prob = p_safe_prob.gather(-1, unsafe_idx).squeeze(-1)
        unsafe_prob_rel_shift_logit = _mean_masked(
            (unsafe_prob_after_logit - unsafe_prob_before) / unsafe_prob_before.clamp_min(1e-12)
        )
        unsafe_prob_rel_shift_prob = _mean_masked(
            (unsafe_prob_after_prob - unsafe_prob_before) / unsafe_prob_before.clamp_min(1e-12)
        )

        top2_vals = p.topk(2, dim=-1).values
        top2_margin = top2_vals[..., 0] - top2_vals[..., 1]
        top2_vals_logit = p_safe_logit.topk(2, dim=-1).values
        top2_margin_logit = top2_vals_logit[..., 0] - top2_vals_logit[..., 1]
        top2_vals_prob = p_safe_prob.topk(2, dim=-1).values
        top2_margin_prob = top2_vals_prob[..., 0] - top2_vals_prob[..., 1]

        changed_masked = _mean_masked((p_argmax != p_safe_argmax).float())
        changed_logit = _mean_masked((p_argmax != p_safe_argmax_logit).float())
        changed_prob = _mean_masked((p_argmax != p_safe_argmax_prob).float())
        argmax_diff_masked = _mean_masked((p_argmax != p_unsafe_argmax).float())
        mean_delta_logit = _mean_masked(per_pos_delta_logit)
        mean_delta_prob = _mean_masked(per_pos_delta_prob)
        kl_logit_mean = _mean_masked(kl_logit)
        kl_prob_mean = _mean_masked(kl_prob)
        unsafe_shift_logit = _mean_masked(unsafe_prob_after_logit - unsafe_prob_before)
        unsafe_shift_prob = _mean_masked(unsafe_prob_after_prob - unsafe_prob_before)
        top2_margin_masked = _mean_masked(top2_margin)
        top2_margin_logit_masked = _mean_masked(top2_margin_logit)
        top2_margin_prob_masked = _mean_masked(top2_margin_prob)

        top_k = min(50, p.size(-1))
        top_idx = p.topk(top_k, dim=-1).indices
        p_top = p.gather(-1, top_idx)
        p_unsafe_top = p_unsafe.gather(-1, top_idx)
        p_safe_top = p_safe.gather(-1, top_idx)
        topk_l1 = (p_top - p_safe_top).abs()
        topk_pu_l1 = (p_top - p_unsafe_top).abs()
        mask_expanded = mask_positions.unsqueeze(-1).expand_as(topk_l1)
        topk_l1_masked = topk_l1.masked_select(mask_expanded).mean().item() if mask_expanded.numel() else float("nan")
        delta_pu_topk_masked = topk_pu_l1.masked_select(mask_expanded).mean().item() if mask_expanded.numel() else float("nan")
        topk_mass = topk_l1.sum(dim=-1)
        topk_mass_shift = topk_mass.masked_select(mask_positions).mean().item() if mask_positions.any() else float("nan")

        if log_debug:
            logger.info(
                "Repellency applied: mask_frac=%.4f, argmax_diff_masked=%.4f, top2_margin_masked=%.4e, "
                "mean|p-p_unsafe|_top50_masked=%.4e, mean|Δp|_masked=%.4e, top50_L1_masked=%.4e, "
                "top50_mass_shift_masked=%.4e, unsafe_shift_logit=%.4e, unsafe_shift_prob=%.4e, kl_masked=%.4e, changed_frac_masked=%.4f, "
                "changed_logit=%.4f, changed_prob=%.4f, kl_logit=%.4e, kl_prob=%.4e, unsafe_shift_logit=%.4e, unsafe_shift_prob=%.4e, "
                "top2_margin_logit=%.4e, top2_margin_prob=%.4e, "
                "beta_hat_mean=%.4e, g_t=%.3f, strength_mean=%.4e, mean_rho=%.4f",
                mask_frac,
                argmax_diff_masked,
                top2_margin_masked,
                delta_pu_topk_masked,
                mean_delta_logit,
                topk_l1_masked,
                topk_mass_shift,
                unsafe_shift_logit,
                unsafe_shift_prob,
                _mean_masked((p_safe * (p_safe.clamp_min(1e-30).log() - p.clamp_min(1e-30).log())).sum(dim=-1)),
                changed_masked,
                changed_logit,
                changed_prob,
                kl_logit_mean,
                kl_prob_mean,
                unsafe_shift_logit,
                unsafe_shift_prob,
                top2_margin_logit_masked,
                top2_margin_prob_masked,
                strength_stats.get("beta_hat_mean", float("nan")),
                strength_stats.get("g_t_mean", float("nan")),
                strength_stats.get("strength_mean", float("nan")),
                float(neg_item),
            )
            if compute_both:
                logger.info(
                    "Repellency debug (prob vs logit): unsafe_prob_rel_shift_logit=%.4e unsafe_prob_rel_shift_prob=%.4e delta_mean_logit=%.4e delta_mean_prob=%.4e",
                    unsafe_prob_rel_shift_logit,
                    unsafe_prob_rel_shift_prob,
                    mean_delta_logit,
                    mean_delta_prob,
                )
        strength_zero_frac = float((strength == 0).float().mean().item())
        metrics = {
            "strength_zero_frac": strength_zero_frac,
            "mask_frac": mask_frac,
            "changed_frac_masked": changed_masked,
            "changed_frac_logit": changed_logit,
            "changed_frac_prob": changed_prob,
            "kl_logit_mean": kl_logit_mean,
            "kl_prob_mean": kl_prob_mean,
            "unsafe_shift_logit": unsafe_shift_logit,
            "unsafe_shift_prob": unsafe_shift_prob,
            "unsafe_prob_rel_shift_logit": unsafe_prob_rel_shift_logit,
            "unsafe_prob_rel_shift_prob": unsafe_prob_rel_shift_prob,
            "top2_margin": top2_margin_masked,
            "top2_margin_logit": top2_margin_logit_masked,
            "top2_margin_prob": top2_margin_prob_masked,
        }
        if log_debug:
            logger.info("Strength zero fraction=%.4f", strength_zero_frac)
        mask_count = int(mask_positions.float().sum().item()) if mask_positions.numel() else 0
        strength_mean_value = float(strength_stats.get("strength_mean", 0.0))
        effective_strength_value = float(strength_mean_value * mask_frac) if not math.isnan(mask_frac) else float("nan")
        stats = {
            "tv_safe_data_mean": _mean_masked(tv_safe_data),
            "tv_safe_unsafe_mean": _mean_masked(tv_safe_unsafe),
            "tv_data_unsafe_mean": _mean_masked(tv_data_unsafe),
            "kl_safe_data_mean": _mean_masked(kl_safe_data),
            "kl_safe_unsafe_mean": _mean_masked(kl_safe_unsafe),
            "kl_data_unsafe_mean": _mean_masked(kl_data_unsafe),
            "js_safe_data_mean": _mean_masked(js_safe_data),
            "top1_change_rate": changed_masked,
            "top1_overlap_data_unsafe": overlap_data_unsafe,
            "effective_strength": effective_strength_value,
        }
        metrics = {
            "strength_zero_frac": strength_zero_frac,
            "mask_frac": mask_frac,
            "changed_frac_masked": changed_masked,
            "changed_frac_logit": changed_logit,
            "changed_frac_prob": changed_prob,
            "kl_logit_mean": kl_logit_mean,
            "kl_prob_mean": kl_prob_mean,
            "unsafe_shift_logit": unsafe_shift_logit,
            "unsafe_shift_prob": unsafe_shift_prob,
            "unsafe_prob_rel_shift_logit": unsafe_prob_rel_shift_logit,
            "unsafe_prob_rel_shift_prob": unsafe_prob_rel_shift_prob,
            "top2_margin": top2_margin_masked,
            "top2_margin_logit": top2_margin_logit_masked,
            "top2_margin_prob": top2_margin_prob_masked,
            **stats,
        }
        if weights is not None:
            emp_weights = weights.clamp_min(1e-30)
            ess = (1.0 / emp_weights.pow(2).sum(dim=-1).clamp_min(1e-12)).mean().item()
            entropy = -(emp_weights * emp_weights.clamp_min(1e-30).log()).sum(dim=-1).mean().item()
            metrics["ess_weights"] = float(ess)
            metrics["max_weight"] = float(emp_weights.max().item())
            metrics["entropy_weights"] = float(entropy)
        if self._distribution_logger is not None:
            step_header = {
                "g_t": strength_stats.get("g_t_mean"),
                "strength": strength_stats.get("strength_mean"),
                "beta_hat_mean": strength_stats.get("beta_hat_mean"),
                "beta_hat_raw_mean": strength_stats.get("beta_hat_raw_mean"),
                "beta_hat_len_mean": strength_stats.get("beta_hat_len_mean"),
                "kl_logit_mean": strength_stats.get("kl_logit_mean"),
                "tv_safe_data_mean": stats.get("tv_safe_data_mean"),
                "top1_change_rate": stats.get("top1_change_rate"),
                "unsafe_shift_logit": strength_stats.get("unsafe_shift_logit"),
                "unsafe_shift_prob": strength_stats.get("unsafe_shift_prob"),
                "prompt_variant": prompt_variant,
                "notes": {
                    "eta": float(self.eta),
                    "t_start": self.t_start,
                    "t_end": self.t_end,
                },
            }
            self._distribution_logger.maybe_log(
                step=t_idx if t_idx is not None else -1,
                token_ids=alignment_x_t,
                mask=mask_positions,
                logits_data=log_p_data,
                logits_unsafe=log_p_unsafe,
                logits_safe=log_p_safe,
                prompt_id=prompt_id,
                effective_strength=effective_strength_value,
                mask_frac=mask_frac,
                num_masked=mask_count,
                seq_len=alignment_x_t.size(1),
                extra={"prompt_variant": prompt_variant},
                step_header=step_header,
                mask_token_id=self.mask_index,
                prompt_len=prompt_width,
                prompt_mask=prompt_mask,
                prompt_token_ids=prompt_token_ids,
            )
        return metrics

    def _compute_ignore_ids(self, extra_ignore) -> set[int]:
        ignore_ids: set[int] = set(extra_ignore or [])
        for maybe_id in (self.pad_index, self.eos_id):
            if maybe_id is not None:
                ignore_ids.add(int(maybe_id))
        ignore_ids = {idx for idx in ignore_ids if 0 <= idx < self.vocab_size}
        return ignore_ids

    def _infer_continuation_length(self) -> int:
        if torch.is_tensor(self.ref_data):
            return self.ref_data.size(1)
        if torch.is_tensor(self.proj_refs):
            return self.proj_refs.size(1)
        raise RuntimeError("Unsafe references are required for continuation alignment.")

    def export_semantic_ref(self, semantic_ref: torch.Tensor, path: Optional[str]):
        if path is None:
            return
        dir_path = os.path.split(path)[0]
        os.makedirs(dir_path, exist_ok=True)
        torch.save(semantic_ref.cpu(), path)

    def import_semantic_ref(self, path: Optional[str]):
        if path is None or (path is not None and not os.path.exists(path)):
            return None
        return torch.load(path, map_location='cpu')

    @staticmethod
    def _move_from_sigma(sigma):
        # MDLM uses move = 1 - exp(-sigma)
        return 1.0 - torch.exp(-sigma)

    def _beta_hat_from_logq(
        self,
        log_q: torch.Tensor,
        eff_length: Optional[torch.Tensor] = None,
        prior_log_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute beta_hat as the average likelihood across unsafe refs/prototypes.

        log_q: [B, N]
        eff_length: optional [B, N] effective token counts to length-normalize log_q
        prior_log_weights: optional [B, N] to reweight refs (e.g., cluster priors)
        """
        if log_q is None:
            raise RuntimeError("log_q must be provided to compute beta_hat.")
        log_q_float = log_q.to(torch.float64)
        if eff_length is not None:
            eff_length = eff_length.to(log_q_float.device, dtype=log_q_float.dtype)
            score = log_q_float / eff_length.clamp_min(1.0)
        else:
            score = log_q_float
        if prior_log_weights is not None:
            score = score + prior_log_weights.to(score.device, dtype=score.dtype)

        logN = math.log(max(score.size(-1), 1))
        if prior_log_weights is not None:
            denom_tensor = torch.logsumexp(prior_log_weights.to(score.device, dtype=score.dtype), dim=-1, keepdim=True)
        else:
            denom_tensor = torch.tensor(logN, device=score.device, dtype=score.dtype)

        log_beta_raw = torch.logsumexp(score, dim=-1, keepdim=True) - denom_tensor
        log_beta_raw_for_exp = log_beta_raw.clamp(min=-80.0, max=0.0)
        beta_hat = torch.exp(log_beta_raw_for_exp).to(torch.float32)

        max_score = score.max(dim=-1, keepdim=True).values
        log_beta_rel = torch.logsumexp(score - max_score, dim=-1, keepdim=True) - denom_tensor

        beta_hat = beta_hat.view(-1, 1, 1)

        if os.getenv("SAFE_REPELLENCY_DEBUG"):
            logger.info(
                "beta_hat score stats: min %.4f max %.4f mean %.4f; row0[:5]=%s",
                float(score.min().item()),
                float(score.max().item()),
                float(score.mean().item()),
                score[0, :5].detach().cpu().tolist() if score.numel() else [],
            )
            logger.info(
                "beta_hat log_beta_raw stats: min %.4f max %.4f mean %.4f",
                float(log_beta_raw.min().item()),
                float(log_beta_raw.max().item()),
                float(log_beta_raw.mean().item()),
            )
            logger.info(
                "beta_hat stats: mean %.4e p95 %.4e max %.4e",
                float(beta_hat.view(-1).mean().item()),
                float(torch.quantile(beta_hat.view(-1), 0.95)),
                float(beta_hat.max().item()),
            )
            if not getattr(self, "_logged_beta_no_semantic", False):
                logger.info("beta_hat excludes semantic gating; only w incorporates semantic logits.")
                self._logged_beta_no_semantic = True
        return beta_hat, log_beta_raw.to(torch.float32), log_beta_rel.to(torch.float32)

    def _schedule_weight(self, t_idx: Optional[int], batch_size: int, device) -> torch.Tensor:
        if not self.should_apply_at(t_idx):
            return torch.zeros(batch_size, 1, 1, device=device)
        mode = (self.schedule_mode or "hard_window").lower()
        if mode == "hard_window":
            return torch.ones(batch_size, 1, 1, device=device)
        if mode == "cosine_ramp":
            start = int(self.t_start or 0)
            end = int(self.t_end if self.t_end is not None else start)
            if t_idx is None:
                return torch.ones(batch_size, 1, 1, device=device)
            t_val = int(t_idx)
            if t_val < start or t_val > end:
                return torch.zeros(batch_size, 1, 1, device=device)
            span = max(end - start, 1)
            progress = float(t_val - start) / span
            weight_scalar = 0.5 * (1.0 - math.cos(2.0 * math.pi * progress))
            return torch.full((batch_size, 1, 1), weight_scalar, device=device)
        logger.warning("Unknown schedule_mode=%s, defaulting to hard_window.", self.schedule_mode)
        return torch.ones(batch_size, 1, 1, device=device)

    def compute_guidance_strength(
        self,
        beta_hat: Optional[torch.Tensor],
        t_idx: Optional[int],
        batch_size: int,
        device,
        log_beta_raw: Optional[torch.Tensor] = None,
        log_beta_rel: Optional[torch.Tensor] = None,
        beta_hat_raw: Optional[torch.Tensor] = None,
        beta_hat_len: Optional[torch.Tensor] = None,
        beta_hat_mode: Optional[str] = None,
        log_beta_raw_raw: Optional[torch.Tensor] = None,
        log_beta_rel_raw: Optional[torch.Tensor] = None,
        log_beta_raw_len: Optional[torch.Tensor] = None,
        log_beta_rel_len: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        beta_mode_env = self._env_choice(
            "SAFE_BETA_MODE",
            "len",
            {"raw", "len", "both"},
        )
        beta_mode = (beta_hat_mode or beta_mode_env or "len").lower()
        beta_candidates: dict[str, Optional[torch.Tensor]] = {
            "raw": beta_hat_raw,
            "len": beta_hat_len,
        }
        beta_hat_use: Optional[torch.Tensor] = None
        beta_mode_applied = None
        beta_mode_alt = None
        if beta_mode == "both":
            primary = beta_hat_len if beta_hat_len is not None else beta_hat_raw
            secondary = beta_hat_raw if primary is beta_hat_len else beta_hat_len
            if primary is not None:
                beta_hat_use = primary
                beta_mode_applied = "len" if primary is beta_hat_len else "raw"
            if secondary is not None:
                beta_mode_alt = "raw" if secondary is beta_hat_raw else "len"
        else:
            beta_hat_use = beta_candidates.get(beta_mode, None)
            beta_mode_applied = beta_mode if beta_hat_use is not None else None

        if beta_hat_use is None:
            beta_hat_use = beta_hat_len if beta_hat_len is not None else beta_hat_raw
            if beta_hat_use is not None and beta_mode_applied is None:
                beta_mode_applied = "len" if beta_hat_use is beta_hat_len else "raw"
        if beta_hat_use is None:
            beta_hat_use = beta_hat if beta_hat is not None else torch.ones(batch_size, 1, 1, device=device)
            if beta_mode_applied is None:
                beta_mode_applied = "fallback"

        def _summarize_beta(name: str, tensor: Optional[torch.Tensor], stats: dict):
            if tensor is None:
                return
            flat = tensor.view(-1)
            stats[f"{name}_mean"] = flat.mean().item()
            stats[f"{name}_p50"] = float(torch.quantile(flat, 0.50))
            stats[f"{name}_p95"] = float(torch.quantile(flat, 0.95))
            stats[f"{name}_max"] = flat.max().item()

        beta_hat_stats: dict[str, float] = {}
        _summarize_beta("beta_hat_raw", beta_hat_raw, beta_hat_stats)
        _summarize_beta("beta_hat_len", beta_hat_len, beta_hat_stats)
        if not beta_hat_stats:
            _summarize_beta("beta_hat", beta_hat_use, beta_hat_stats)
        else:
            _summarize_beta("beta_hat", beta_hat_use, beta_hat_stats)
        if log_beta_raw is not None:
            beta_hat_stats["log_beta_raw_mean"] = log_beta_raw.mean().item()
            beta_hat_stats["log_beta_raw_max"] = log_beta_raw.max().item()
        if log_beta_rel is not None:
            beta_hat_stats["log_beta_rel_mean"] = log_beta_rel.mean().item()
            beta_hat_stats["log_beta_rel_max"] = log_beta_rel.max().item()
        if log_beta_raw_raw is not None:
            beta_hat_stats["log_beta_raw_raw_mean"] = log_beta_raw_raw.mean().item()
            beta_hat_stats["log_beta_raw_raw_max"] = log_beta_raw_raw.max().item()
        if log_beta_rel_raw is not None:
            beta_hat_stats["log_beta_rel_raw_mean"] = log_beta_rel_raw.mean().item()
            beta_hat_stats["log_beta_rel_raw_max"] = log_beta_rel_raw.max().item()
        if log_beta_raw_len is not None:
            beta_hat_stats["log_beta_raw_len_mean"] = log_beta_raw_len.mean().item()
            beta_hat_stats["log_beta_raw_len_max"] = log_beta_raw_len.max().item()
        if log_beta_rel_len is not None:
            beta_hat_stats["log_beta_rel_len_mean"] = log_beta_rel_len.mean().item()
            beta_hat_stats["log_beta_rel_len_max"] = log_beta_rel_len.max().item()
        
        g_t = self._schedule_weight(t_idx, batch_size, device)
        
        eta_value = float(self.eta)
        
        # Set eta to scale for legacy weight_mode compatibility
        if self.weight_mode and self.weight_mode != "eta_beta_hat":
            if not getattr(self, "_warned_weight_mode", False):
                logger.warning(
                    "Using legacy weight_mode=%s; falling back to scale=%.4f",
                    self.weight_mode,
                    float(self.scale),
                )
                self._warned_weight_mode = True
            eta_value = float(self.scale)
        
        # g_t determines the timing schedule of repellency application
        strength = eta_value * beta_hat_use * g_t
        strength_alt = None
        if beta_mode == "both" and beta_hat_raw is not None and beta_hat_len is not None:
            other = beta_hat_len if beta_hat_use is beta_hat_raw else beta_hat_raw
            strength_alt = eta_value * other * g_t

        stats = {
            **beta_hat_stats,
            "g_t_mean": g_t.view(-1).mean().item() if g_t.numel() else 0.0,
            "strength_mean": strength.view(-1).mean().item() if strength.numel() else 0.0,
        }
        if strength_alt is not None and strength_alt.numel():
            stats["strength_alt_mean"] = strength_alt.view(-1).mean().item()
        if beta_mode_applied is not None:
            stats["beta_hat_mode_active"] = beta_mode_applied
        if beta_mode_alt is not None:
            stats["beta_hat_mode_alt"] = beta_mode_alt
        if beta_hat_use is not None:
            flat_use = beta_hat_use.view(-1)
            stats["beta_hat_use_mean"] = flat_use.mean().item()
            stats["beta_hat_use_p95"] = float(torch.quantile(flat_use, 0.95))
            stats["beta_hat_use_max"] = flat_use.max().item()
        if os.getenv("SAFE_REPELLENCY_DEBUG"):
            logger.info(
                "[Repellency] t=%s mode=%s eta=%.4f g(t)=%.4f beta_hat[%s] (mean/p95/max)=(%.4e/%.4e/%.4e) strength_mean=%.4e",
                str(t_idx),
                self.schedule_mode,
                float(self.eta),
                stats["g_t_mean"],
                stats.get("beta_hat_mode_active", "na"),
                stats.get("beta_hat_use_mean", beta_hat_stats.get("beta_hat_mean", 0.0)),
                stats.get("beta_hat_use_p95", beta_hat_stats.get("beta_hat_p95", 0.0)),
                stats.get("beta_hat_use_max", beta_hat_stats.get("beta_hat_max", 0.0)),
                stats["strength_mean"],
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[Repellency] t=%s mode=%s eta=%.4f g(t)=%.4f beta_hat[%s] (mean/p95/max)=(%.4e/%.4e/%.4e) strength_mean=%.4e",
                str(t_idx),
                self.schedule_mode,
                float(self.eta),
                stats["g_t_mean"],
                stats.get("beta_hat_mode_active", "na"),
                stats.get("beta_hat_use_mean", beta_hat_stats.get("beta_hat_mean", 0.0)),
                stats.get("beta_hat_use_p95", beta_hat_stats.get("beta_hat_p95", 0.0)),
                stats.get("beta_hat_use_max", beta_hat_stats.get("beta_hat_max", 0.0)),
                stats["strength_mean"],
            )
        return strength, stats

    # @torch.no_grad()
    # def _log_qt_mask_kernel(self, xt, move):
    #     """
    #     Compute the forward-process log-likelihood log q_t(xt | x_ref[n]) for all unsafe refs.

    #     xt: LongTensor [B, L], Current noisy sequences at time t (contains normal tokens and mask tokens).
    #     move: FloatTensor [B] in (0,1), MDLM masking probability at time t: move(t) = 1 - exp(-sigma(t)).
    #     PAD tokens in references are treated as wildcards: they neither enforce equality nor count
    #     toward the masked/kept tallies.
    #     Returns: log q_t(xt | x_ref[n]) across N refs => [B, N]
    #     """
    #     if self.proj_refs is None:
    #         raise RuntimeError("Reference data is not available; prototypes are enabled.")
    #     U = self.proj_refs                          # U = unsafe references, [N, L]
    #     B, L = xt.shape                             # L = seq length, B = batch size
    #     N = U.size(0)                               # number of refs
        
    #     xt_b = xt[:, None, :].expand(B, N, L)       # [B,N,L], broadcasting xt
    #     U_b  = U[None, :, :].expand(B, N, L)        # [B,N,L], broadcasting U
    #     is_mask = (xt_b == self.mask_index)         # [B,N,L], mask positions in xt
    #     if self.pad_index is None:
    #         is_pad = torch.zeros_like(U_b, dtype=torch.bool)
    #     else:
    #         is_pad = (U_b == self.pad_index)        # [B,N,L], PAD positions in refs

    #     must_match = (~is_mask) & (~is_pad)         # positions with real tokens in both xt and ref
    #     # validity: all non-mask, non-pad positions must match the ref
    #     valid = torch.where(
    #         must_match,
    #         (xt_b == U_b),
    #         torch.ones_like(xt_b, dtype=torch.bool)
    #     ).all(dim=-1)  # [B,N]

    #     M = (is_mask & ~is_pad).sum(dim=-1).to(torch.float32)   # [B,N], masked positions with content
    #     K = ((~is_mask) & ~is_pad).sum(dim=-1).to(torch.float32) # [B,N], known positions with content
    #     mv = move[:, None]                          # [B,1], move broadcasted
    #     kv = (1.0 - move)[:, None]                  # [B,1], keep broadcasted

    #     logqt = torch.full((B, N), -1e9, device=xt.device, dtype=torch.float32)
    #     eps = 1e-12
    #     if valid.any():
    #         logqt_valid = M * (mv + eps).log() + K * (kv + eps).log()
    #         logqt[valid] = logqt_valid[valid]
    #     return logqt  # [B,N]

    @torch.no_grad()
    def _log_qt_mask_kernel_strict(self, xt, move):
        """
        Valid pairs require xt tokens to exactly match the reference wherever
        the reference has a real token (non-PAD/EOS) and xt is not MASK.
        """
        if self.proj_refs is None:
            raise RuntimeError("Reference data is not available; prototypes are enabled.")

        B, L = xt.shape
        U = self.proj_refs[:, :L].to(xt.device)  # [N, L]
        N = U.size(0)

        xt_b = xt.unsqueeze(1).expand(B, N, L)    # [B,N,L]
        U_b = U.unsqueeze(0).expand(B, N, L)      # [B,N,L]

        is_mask_xt = (xt_b == self.mask_index)
        is_pad_ref = torch.zeros_like(U_b, dtype=torch.bool)
        if self.pad_index is not None:
            is_pad_ref |= (U_b == self.pad_index)
        if self.eos_id is not None:
            is_pad_ref |= (U_b == self.eos_id)
        eff = ~is_pad_ref                           # effective sites in refs

        is_pad_xt = torch.zeros_like(xt_b, dtype=torch.bool)
        if self.pad_index is not None:
            is_pad_xt |= (xt_b == self.pad_index)
        if self.eos_id is not None:
            is_pad_xt |= (xt_b == self.eos_id)

        visible = eff & (~is_mask_xt) & (~is_pad_xt)
        mismatches = visible & (xt_b != U_b)
        valid = ~mismatches
        valid_rows = valid.all(dim=-1)             # [B,N]

        M = (eff & (is_mask_xt | is_pad_xt)).sum(dim=-1).to(torch.float32)  # [B,N]
        K = (eff & (~is_mask_xt) & (~is_pad_xt)).sum(dim=-1).to(torch.float32)  # [B,N]

        mv = move.view(B, 1).clamp(1e-6, 1.0 - 1e-6)
        kv = (1.0 - mv).clamp_min(1e-6)

        logqt = torch.full((B, N), -1e9, device=xt.device, dtype=torch.float32)
        eps = 1e-12
        if valid_rows.any():
            logqt_valid = M * (mv + eps).log() + K * (kv + eps).log()
            logqt[valid_rows] = logqt_valid[valid_rows]

        L_eff = (M + K).clamp_min(1.0)
        return logqt, L_eff



    @torch.no_grad()
    def _log_qt_mask_kernel(self, xt, move):
        """
        Relaxed discrete kernel for text, based on MDLM-style keep/mask rates.

        xt   : LongTensor [B, L], current noisy sequences at time t
        move : FloatTensor [B] in (0,1), MDLM masking probability at time t

        We treat:
        - MASK in xt = masked site (prob ~ move)
        - Visible tokens in xt that match ref = "good keeps"
        - Visible tokens in xt that mismatch ref = "bad keeps" (small but non-zero prob)

        PAD tokens in references are wildcards (ignored).
        Returns:
        logqt : FloatTensor [B, N]
        """
        if self.proj_refs is None:
            raise RuntimeError("Reference data is not available; prototypes are enabled.")

        B, L = xt.shape
        U = self.proj_refs[:, :L].to(xt.device)          # [N, L]
        N = U.size(0)

        xt_b = xt.unsqueeze(1).expand(B, N, L)    # [B,N,L]
        U_b  = U.unsqueeze(0).expand(B, N, L)     # [B,N,L]

        is_mask = (xt_b == self.mask_index)       # [B,N,L] positions in xt that are MASK
        if self.pad_index is None and self.eos_id is None:
            is_pad = torch.zeros_like(U_b, dtype=torch.bool)
        else:
            pad_ref = (U_b == self.pad_index) if self.pad_index is not None else torch.zeros_like(U_b, dtype=torch.bool)
            eos_ref = (U_b == self.eos_id) if self.eos_id is not None else torch.zeros_like(U_b, dtype=torch.bool)
            is_pad = pad_ref | eos_ref             # [B,N,L] PAD/EOS in refs

        eff = ~is_pad                             # effective positions
        is_pad_xt = torch.zeros_like(xt_b, dtype=torch.bool)
        if self.pad_index is not None:
            is_pad_xt |= (xt_b == self.pad_index)
        if self.eos_id is not None:
            is_pad_xt |= (xt_b == self.eos_id)

        # for effective positions, split into:
        #   - masked/pad in xt (treated as non-evidence)
        #   - unmasked in xt: matches vs mismatches
        unmasked_eff = eff & (~is_mask) & (~is_pad_xt)

        matches    = unmasked_eff & (xt_b == U_b)   # visible tokens that match ref
        mismatches = unmasked_eff & (xt_b != U_b)   # visible tokens that mismatch ref

        M       = ((is_mask | is_pad_xt) & eff).float().sum(dim=-1)      # [B,N] masked or pad sites with content
        K_match = matches.float().sum(dim=-1)              # [B,N] matched visible tokens
        K_mis   = mismatches.float().sum(dim=-1)           # [B,N] mismatched visible tokens

        # move / keep probabilities
        mv = move.view(B, 1).clamp(1e-4, 1.0 - 1e-4)       # [B,1], masking prob
        kv = (1.0 - mv)                                    # [B,1], keep prob

        # mismatches are allowed but penalized: they get a tiny slice of the keep mass.
        V = self.vocab_size
        beta_floor = 1e-6
        beta_cap = 0.1
        beta = 1.0 / float(max(V - 1, 1))
        beta = min(max(beta_floor, beta), beta_cap)

        p_keep_good = (kv * (1.0 - beta)).clamp_min(1e-12) # [B,1]
        p_keep_bad  = (kv * beta).clamp_min(1e-12)         # [B,1]

        eps = 1e-12
        logqt = (
            M       * torch.log(mv + eps) +
            K_match * torch.log(p_keep_good + eps) +
            K_mis   * torch.log(p_keep_bad  + eps)
        )                                                  # [B,N]
        L_eff = (M + K_match + K_mis).clamp_min(1.0)        # [B,N]
        if self._env_flag("SAFE_REPELLENCY_DEBUG"):
            score_dbg = logqt / L_eff.clamp_min(1.0)
            logger.info(
                "mask_frac=%.4f score_std_over_refs=%.6f K_match_mean=%.3f K_mis_mean=%.3f M_mean=%.3f",
                (xt == self.mask_index).float().mean().item(),
                score_dbg.std(dim=-1).mean().item(),
                K_match.mean().item(),
                K_mis.mean().item(),
                M.mean().item(),
            )
        return logqt, L_eff

    @torch.no_grad()
    def _encode_semantic_state(self, x_t, x_0_hat=None, **kwargs):
        """
        Map the current noisy sequence into a pooled embedding z_t: [B, D].

        Builds a visible sequence by filling MASK sites with argmax(x_0_hat)
        when available. If x_0_hat is not provided, MASK sites fall back to PAD
        (when defined) or remain MASK. PAD/MASK tokens are excluded from pooling.
        """
        if self.embed_fn is None:
            raise RuntimeError("embed_fn is required for semantic encoding.")

        x_visible = x_t.clone()
        mask_pos = x_visible == self.mask_index

        if mask_pos.any():
            if x_0_hat is not None:
                fill_tokens = x_0_hat.argmax(dim=-1)
                x_visible = x_visible.masked_scatter(mask_pos, fill_tokens[mask_pos])
            else:
                fallback = self.pad_index if self.pad_index is not None else self.mask_index
                x_visible = x_visible.masked_fill(mask_pos, fallback)

        embeds = self.embed_fn(x_visible)
        if embeds.dim() == 2:
            return embeds  # already [B, D]
        if embeds.dim() != 3:
            raise ValueError(
                f"embed_fn must return [B, L, D] or [B, D], got {embeds.shape}"
            )

        z_t = masked_mean_pool(
            embeds,
            x_visible,
            pad_id=self.pad_index,
            mask_id=self.mask_index,
        )  # [B, D]
        return z_t

    @torch.no_grad()
    def _compute_semantic_ref_embeddings(self):
        """
        Encode all unsafe references once for semantic comparisons.
        Returns:
            semantic_ref_embeddings: FloatTensor [N, D]
        """
        if self.ref_data is None:
            raise RuntimeError("ref_data is required to compute semantic embeddings.")
        if not torch.is_tensor(self.ref_data):
            raise TypeError("ref_data must be a tensor to compute semantic embeddings.")

        ref_device = self.proj_refs.device if self.proj_refs is not None else self.ref_data.device
        ref_tokens = self.ref_data.to(ref_device)
        ref_embs = self._encode_semantic_state(ref_tokens)
        self.semantic_ref_embeddings = ref_embs.detach().to(torch.float32)
        if self.cache_semantic_ref and self.semantic_ref_path is not None:
            self.export_semantic_ref(self.semantic_ref_embeddings, self.semantic_ref_path)
        return self.semantic_ref_embeddings

    @torch.no_grad()
    def _semantic_scores_for_refs(self, x_t, x_0_hat=None, **kwargs):
        """
        Gather embeddings for semantic comparisons between current state and refs.

        Returns:
            (z_t, ref_embs): tuple of FloatTensor ([B, D], [N, D]) or None if semantic gating is off.
        """
        if not self.use_semantic_gating:
            if os.getenv("SAFE_REPELLENCY_DEBUG"):
                logger.info("Semantic gating disabled; skipping semantic embeddings.")
                logger.info(f'{self.use_semantic_gating}, {self.semantic_weight}, {self.semantic_temp}')
            return None

        if self.semantic_ref_embeddings is None:
            if self.cache_semantic_ref and self.semantic_ref_path is not None:
                self.semantic_ref_embeddings = self.import_semantic_ref(self.semantic_ref_path)
                if os.getenv("SAFE_REPELLENCY_DEBUG"):
                    if self.semantic_ref_embeddings is not None:
                        logger.info("Imported cached semantic ref embeddings from %s", self.semantic_ref_path)
                    else:
                        logger.info("No cached semantic ref embeddings found at %s; computing afresh.", self.semantic_ref_path)
            if self.semantic_ref_embeddings is None:
                self._compute_semantic_ref_embeddings()

        ref_embs = self.semantic_ref_embeddings.to(x_t.device)  # [N, D]
        z_t = self._encode_semantic_state(x_t, x_0_hat=x_0_hat, **kwargs)  # [B, D]

        if os.getenv("SAFE_REPELLENCY_DEBUG"):
            logger.info(
                "semantic embeddings acquired: z_t shape=%s, ref_embs shape=%s",
                tuple(z_t.shape),
                tuple(ref_embs.shape),
            )
        return z_t, ref_embs

    @staticmethod
    def _normalize_scores(scores: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """
        Per-example (row-wise) zero-mean, unit-std normalization for weighting terms.

        This keeps heterogeneous log-space factors (mask kernel vs. semantic)
        on comparable scales before the softmax.
        """
        center = scores - scores.mean(dim=-1, keepdim=True)
        scale = center.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return center / scale

    def _semantic_rbf_logits(
        self,
        z_t: torch.Tensor,
        ref_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log RBF kernel scores log k(z_t, z_n) for semantic gating.

        Returns:
            logits: FloatTensor [B, N], where
                logits[b, n] = - ||z_t[b] - ref_embs[n]||^2 / (2 * sigma^2),
            scaled by semantic_temp later.
        """
        z_t = z_t.float()
        ref_embs = ref_embs.float()
        z_t = normalize_embeddings(z_t)
        ref_embs = normalize_embeddings(ref_embs)

        if self.semantic_sigma is None:
            sigma = median_heuristic_sigma(ref_embs, normalize=False)
            self.semantic_sigma = float(sigma)

        B, D = z_t.shape
        N, D_ref = ref_embs.shape
        assert D == D_ref

        sigma = float(self.semantic_sigma)
        kernel = rbf_kernel_matrix(z_t, ref_embs, sigma)
        logits = kernel.clamp_min(1e-30).log()
        return logits


    @torch.no_grad()
    def _unsafe_posterior(self, xt, move, x_0_hat=None, **kwargs):
        """
        Build per-position categorical dist over tokens from unsafe refs.
        This is supposed to be an empirical posterior from the unsafe refs:

        Idea:
            - Weight each unsafe ref x_ref[n] by w_n \\propto q_t(xt | x_ref[n]).
            - For each position i, make a weighted histogram over vocab tokens v:
                p_unsafe[i, v] \\propto \\sum_n w_n * 1\\{ x_ref[n][i] == v \\}
            - Normalize over v (only want where xt is masked).
            - For unmasked positions, just put a delta on the observed xt token
                (i.e., we don't try to change visible/kept tokens).

        p_unsafe[b, i, v] = \\sum_n w_n * 1\\{U_n[i] == v\\}  (normalized over v)
        where w_n \\propto q_t(xt | U_n).
        
        Args:
          xt   : LongTensor [B, L]
          move : FloatTensor [B]  (move(t))
        
        Returns:
          p_unsafe : FloatTensor [B, L, V]  (categorical per position)
          metrics  : dict with rho_hat, beta_hat, and raw log_q stats
        """
        prompt_mask = kwargs.get("prompt_mask", None)
        if self.proj_refs is None or self.U_T is None:
            raise RuntimeError("Unsafe references not loaded.")
        B, L = xt.shape
        V = self.vocab_size
        debug = self._env_flag("SAFE_REPELLENCY_DEBUG")
        validate = self._env_flag("SAFE_REPELLENCY_VALIDATE") or self._env_flag("SAFE_REPELLENCY_DEBUG_VALIDATE")
        kernel_mode_env = self._env_choice("SAFE_KERNEL_MODE", "relaxed", {"relaxed", "strict", "both"})
        kernel_use = "strict" if kernel_mode_env == "strict" else "relaxed"
        compare_kernels = debug or kernel_mode_env == "both"
        need_paper_beta = kernel_use != "strict"

        def _compute_kernel(mode: str):
            if mode == "strict":
                return self._log_qt_mask_kernel_strict(xt, move)
            return self._log_qt_mask_kernel(xt, move)

        logqt_main, logqt_len_main = _compute_kernel(kernel_use)
        logqt_len_main = logqt_len_main.clamp_min(1.0)
        logqt_relaxed = logqt_len_relaxed = None
        logqt_strict = logqt_len_strict = None
        if kernel_use == "strict":
            logqt_strict, logqt_len_strict = logqt_main, logqt_len_main
        if compare_kernels:
            if kernel_use == "relaxed":
                logqt_relaxed, logqt_len_relaxed = logqt_main, logqt_len_main
                logqt_strict, logqt_len_strict = _compute_kernel("strict")
                logqt_len_strict = logqt_len_strict.clamp_min(1.0)
            elif kernel_use == "strict":
                logqt_strict, logqt_len_strict = logqt_main, logqt_len_main
                logqt_relaxed, logqt_len_relaxed = _compute_kernel("relaxed")
                logqt_len_relaxed = logqt_len_relaxed.clamp_min(1.0)
            else:  # kernel_use from env "both" defaults to relaxed for computation
                logqt_relaxed, logqt_len_relaxed = _compute_kernel("relaxed")
                logqt_relaxed = logqt_relaxed
                logqt_len_relaxed = logqt_len_relaxed.clamp_min(1.0)
                logqt_strict, logqt_len_strict = _compute_kernel("strict")
                logqt_len_strict = logqt_len_strict.clamp_min(1.0)
        elif need_paper_beta and logqt_strict is None and debug:
            # Only compute the strict kernel for logging/validation purposes.
            # In production (debug=False) this second forward pass is skipped.
            logqt_strict, logqt_len_strict = _compute_kernel("strict")
            logqt_len_strict = logqt_len_strict.clamp_min(1.0)

        beta_hat_raw, log_beta_raw_raw, log_beta_rel_raw = self._beta_hat_from_logq(
            logqt_main, eff_length=None
        )
        beta_hat_len, log_beta_raw_len, log_beta_rel_len = self._beta_hat_from_logq(
            logqt_main, eff_length=logqt_len_main
        )
        beta_hat_relaxed = beta_hat_strict = None
        beta_hat_relaxed_len = beta_hat_strict_len = None
        if logqt_relaxed is not None:
            beta_hat_relaxed, _, _ = self._beta_hat_from_logq(logqt_relaxed, eff_length=None)
            beta_hat_relaxed_len, _, _ = self._beta_hat_from_logq(logqt_relaxed, eff_length=logqt_len_relaxed)
        if logqt_strict is not None:
            beta_hat_strict, _, _ = self._beta_hat_from_logq(logqt_strict, eff_length=None)
            beta_hat_strict_len, _, _ = self._beta_hat_from_logq(logqt_strict, eff_length=logqt_len_strict)

        semantic = self._semantic_scores_for_refs(
            xt, x_0_hat=x_0_hat, **kwargs
        )
        if isinstance(semantic, tuple):
            z_t, ref_embs = semantic
        else:
            z_t, ref_embs = None, None

        def _compute_weights(log_q: torch.Tensor, log_q_len: torch.Tensor):
            score = log_q / log_q_len
            w_logits = score
            rbf_logits = None
            if self.use_semantic_gating and z_t is not None and ref_embs is not None:
                rbf_logits = self._semantic_rbf_logits(z_t, ref_embs)
                w_logits = w_logits + (self.semantic_weight * rbf_logits / self.semantic_temp)
            elif debug:
                logger.info("Skipping semantic gating in weight computation.")

            w_logits_stable = w_logits - w_logits.max(dim=-1, keepdim=True).values
            w = torch.softmax(w_logits_stable, dim=-1)
            return w, w_logits, rbf_logits, score

        w, w_logits, rbf_logits, score = _compute_weights(logqt_main, logqt_len_main)

        self._debug_log_kernel(logqt_main, score, kernel_use, z_t is not None and ref_embs is not None)
        self._debug_log_weights(w, rbf_logits)

        def _rho_from_weights(weight_tensor: torch.Tensor):
            N_local = weight_tensor.size(-1)
            w_max_local, _ = weight_tensor.max(dim=-1, keepdim=True)
            uniform_level_local = 1.0 / float(max(N_local, 1))
            rho_flat_local = ((w_max_local - uniform_level_local) / (1.0 - uniform_level_local)).clamp(0.0, 1.0)
            return rho_flat_local.view(-1, 1, 1)

        rho = _rho_from_weights(w)

        U = self.proj_refs[:, :L].to(xt.device)            # [N,L]
        assert U.dtype == torch.long, "Reference tokens must be torch.long"
        replace_id = (
            self.pad_index
            if self.pad_index is not None
            else (self.mask_index if self.mask_index is not None else 0)
        )
        bad_ref = (U < 0) | (U >= V)
        if bad_ref.any():
            U = U.clone()
            U[bad_ref] = replace_id
        idx = U.t().contiguous()                            # [L,N] long
        idx_L, idx_N = idx.shape
        assert idx_L == L and idx_N == w.size(-1), "Reference shape mismatch in unsafe posterior."

        p_unsafe = torch.zeros(B, L, V, device=xt.device, dtype=torch.float32)
        max_entries = 10_000_000
        total_entries = int(B * L * idx_N)
        if total_entries <= max_entries:
            idx_BLN = idx.unsqueeze(0).expand(B, L, idx_N)        # [B,L,N]
            src_BLN = w.unsqueeze(1).expand(B, L, idx_N)          # [B,L,N]
            if self.pad_index is not None:
                src_BLN = src_BLN.masked_fill(idx_BLN.eq(self.pad_index), 0.0)
            if self.eos_id is not None:
                src_BLN = src_BLN.masked_fill(idx_BLN.eq(self.eos_id), 0.0)
            p_unsafe.scatter_add_(dim=2, index=idx_BLN, src=src_BLN)
        else:
            chunk = 2048
            for s in range(0, idx_N, chunk):
                e = min(s + chunk, idx_N)
                idx_blk = idx[:, s:e].contiguous().unsqueeze(0).expand(B, L, e - s)
                src_blk = w[:, s:e].unsqueeze(1).expand(B, L, e - s)
                if self.pad_index is not None:
                    idx_pad_mask = idx_blk.eq(self.pad_index)
                    if idx_pad_mask.any():
                        src_blk = src_blk.masked_fill(idx_pad_mask, 0.0)
                if self.eos_id is not None:
                    idx_eos_mask = idx_blk.eq(self.eos_id)
                    if idx_eos_mask.any():
                        src_blk = src_blk.masked_fill(idx_eos_mask, 0.0)
                idx_blk = idx_blk.contiguous()
                p_unsafe.scatter_add_(dim=2, index=idx_blk, src=src_blk)

        if self.ignore_ids:
            ignore = torch.tensor(sorted(self.ignore_ids), device=xt.device, dtype=torch.long)
            if ignore.numel():
                p_unsafe.index_fill_(2, ignore, 0.0)

        denom = p_unsafe.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p_unsafe = p_unsafe / denom
        
        mask_pos = (xt == self.mask_index)
        if self.pad_index is not None:
            mask_pos = mask_pos | (xt == self.pad_index)
        unmasked = ~mask_pos
        if unmasked.any():
            p_unsafe[unmasked] = 0.0
            b_idx, l_idx = unmasked.nonzero(as_tuple=True)
            tok = xt[b_idx, l_idx].clamp(0, V - 1)
            p_unsafe[b_idx, l_idx, tok] = 1.0

        metrics = {
            "rho": rho,
            "beta_hat": beta_hat_len,
            "beta_hat_raw": beta_hat_raw,
            "beta_hat_len": beta_hat_len,
            "beta_hat_relaxed": beta_hat_relaxed,
            "beta_hat_relaxed_len": beta_hat_relaxed_len,
            "beta_hat_strict": beta_hat_strict,
            "beta_hat_strict_len": beta_hat_strict_len,
            "beta_kernel_mode_active": kernel_use,
            "log_beta_raw": log_beta_raw_len,
            "log_beta_rel": log_beta_rel_len,
            "log_beta_raw_raw": log_beta_raw_raw,
            "log_beta_rel_raw": log_beta_rel_raw,
            "log_beta_raw_len": log_beta_raw_len,
            "log_beta_rel_len": log_beta_rel_len,
            "log_q": logqt_main,
            "log_q_len": logqt_len_main,
            "weights": w,
        }

        def _summarize_beta_tensor(name: str, tensor: Optional[torch.Tensor]) -> dict:
            if tensor is None:
                return {}
            flat = tensor.view(-1)
            return {
                f"{name}_mean": float(flat.mean().item()),
                f"{name}_p50": float(torch.quantile(flat, 0.50)),
                f"{name}_p95": float(torch.quantile(flat, 0.95)),
                f"{name}_max": float(flat.max().item()),
            }

        metrics.update(_summarize_beta_tensor("beta_hat_raw", beta_hat_raw))
        metrics.update(_summarize_beta_tensor("beta_hat_len", beta_hat_len))
        metrics.update(_summarize_beta_tensor("beta_hat_relaxed", beta_hat_relaxed))
        metrics.update(_summarize_beta_tensor("beta_hat_relaxed_len", beta_hat_relaxed_len))
        metrics.update(_summarize_beta_tensor("beta_hat_strict", beta_hat_strict))
        metrics.update(_summarize_beta_tensor("beta_hat_strict_len", beta_hat_strict_len))
        if beta_hat_strict is not None:
            metrics["beta_hat_paper"] = beta_hat_strict_len
            metrics["beta_hat_paper_raw"] = beta_hat_strict
            metrics.update(_summarize_beta_tensor("beta_hat_paper", beta_hat_strict_len))
        if kernel_use != "strict":
            metrics["beta_kernel_mode_paper"] = "strict"
            if not getattr(self, "_warned_relaxed_kernel_beta", False):
                logger.warning(
                    "Paper-consistent beta_hat values come from the strict kernel; guidance is using %s kernel.",
                    kernel_use,
                )
                self._warned_relaxed_kernel_beta = True
        if log_beta_raw_raw is not None:
            metrics["log_beta_raw_raw_mean"] = float(log_beta_raw_raw.mean().item())
            metrics["log_beta_raw_raw_max"] = float(log_beta_raw_raw.max().item())
        if log_beta_rel_raw is not None:
            metrics["log_beta_rel_raw_mean"] = float(log_beta_rel_raw.mean().item())
            metrics["log_beta_rel_raw_max"] = float(log_beta_rel_raw.max().item())
        if log_beta_raw_len is not None:
            metrics["log_beta_raw_len_mean"] = float(log_beta_raw_len.mean().item())
            metrics["log_beta_raw_len_max"] = float(log_beta_raw_len.max().item())
        if log_beta_rel_len is not None:
            metrics["log_beta_rel_len_mean"] = float(log_beta_rel_len.mean().item())
            metrics["log_beta_rel_len_max"] = float(log_beta_rel_len.max().item())

        if metrics.get("beta_hat_len_mean", None) is not None:
            metrics["beta_hat_mean"] = metrics["beta_hat_len_mean"]
            metrics["beta_hat_p50"] = metrics.get("beta_hat_len_p50")
            metrics["beta_hat_p95"] = metrics.get("beta_hat_len_p95")
            metrics["beta_hat_max"] = metrics.get("beta_hat_len_max")

        self._debug_log_beta_metrics(metrics)

        if validate:
            p_unsafe_slow = self._unsafe_posterior_slow(
                xt,
                move,
                kernel_mode=kernel_use,
                w_logits=w_logits,
                semantic=semantic,
                logqt_override=logqt_main,
                logqt_len_override=logqt_len_main,
            )
            diff = (p_unsafe - p_unsafe_slow).abs()
            max_abs_diff = diff.max().item()
            mean_abs_diff = diff.mean().item()
            argmax_fast = p_unsafe.argmax(dim=-1)
            argmax_slow = p_unsafe_slow.argmax(dim=-1)
            masked_positions = mask_pos
            if masked_positions.any():
                disagreement = (
                    (argmax_fast != argmax_slow).float().masked_select(masked_positions).mean().item()
                )
            else:
                disagreement = 0.0
            metrics["validate_max_abs_diff"] = max_abs_diff
            metrics["validate_mean_abs_diff"] = mean_abs_diff
            metrics["validate_argmax_diff"] = disagreement
            if debug:
                self._debug_log_validation(
                    max_abs_diff,
                    mean_abs_diff,
                    disagreement,
                    p_unsafe,
                    p_unsafe_slow,
                    L,
                    V,
                )

        if compare_kernels and logqt_relaxed is not None and logqt_strict is not None:
            top_relaxed = logqt_relaxed.argmax(dim=1)
            top_strict = logqt_strict.argmax(dim=1)
            top_overlap = (top_relaxed == top_strict).float().mean().item()
            rho_relaxed = rho_strict = None
            try:
                w_relaxed, _, _, _ = _compute_weights(logqt_relaxed, logqt_len_relaxed)
                rho_relaxed = _rho_from_weights(w_relaxed).mean().item()
                w_strict, _, _, _ = _compute_weights(logqt_strict, logqt_len_strict)
                rho_strict = _rho_from_weights(w_strict).mean().item()
            except RuntimeError:
                rho_relaxed = rho_strict = None
            self._debug_log_kernel_compare(top_overlap, rho_relaxed, rho_strict)
            metrics["kernel_top1_overlap"] = top_overlap
            if beta_hat_relaxed is not None and beta_hat_strict is not None:
                metrics["beta_hat_relaxed_mean"] = float(beta_hat_relaxed.mean().item())
                metrics["beta_hat_strict_mean"] = float(beta_hat_strict.mean().item())
                if debug:
                    logger.info(
                        "Kernel beta compare: relaxed_mean=%.4e relaxed_len_mean=%.4e strict_mean=%.4e strict_len_mean=%.4e",
                        metrics.get("beta_hat_relaxed_mean", float("nan")),
                        metrics.get("beta_hat_relaxed_len_mean", float("nan")),
                        metrics.get("beta_hat_strict_mean", float("nan")),
                        metrics.get("beta_hat_strict_len_mean", float("nan")),
                    )

        self._debug_log_posterior_health(metrics, w, rho, p_unsafe, mask_pos)

        return p_unsafe, metrics

    @torch.no_grad()
    def _unsafe_posterior_slow(
        self,
        xt: torch.Tensor,
        move: torch.Tensor,
        kernel_mode: str = "relaxed",
        w_logits: Optional[torch.Tensor] = None,
        semantic=None,
        logqt_override: Optional[torch.Tensor] = None,
        logqt_len_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reference slow histogram builder for validation. Extremely expensive.
        """
        B, L = xt.shape
        V = self.vocab_size

        if logqt_override is None or logqt_len_override is None:
            if kernel_mode == "strict":
                logqt, logqt_len = self._log_qt_mask_kernel_strict(xt, move)
            else:
                logqt, logqt_len = self._log_qt_mask_kernel(xt, move)
            logqt_len = logqt_len.clamp_min(1.0)
        else:
            logqt, logqt_len = logqt_override, logqt_len_override

        if isinstance(semantic, tuple):
            z_t, ref_embs = semantic
        else:
            z_t, ref_embs = None, None

        if w_logits is None:
            score = logqt / logqt_len
            w_logits = score
            if self.use_semantic_gating and z_t is not None and ref_embs is not None:
                rbf_logits = self._semantic_rbf_logits(z_t, ref_embs)
                w_logits = w_logits + (self.semantic_weight * rbf_logits / self.semantic_temp)
        w_logits_stable = w_logits - w_logits.max(dim=-1, keepdim=True).values
        w = torch.softmax(w_logits_stable, dim=-1)

        U = self.proj_refs[:, :L].to(xt.device)  # [N, L]
        p_unsafe = torch.zeros(B, L, V, device=xt.device, dtype=torch.float32)
        ignore_ids = set(self.ignore_ids)
        if self.pad_index is not None:
            ignore_ids.add(int(self.pad_index))
        if self.eos_id is not None:
            ignore_ids.add(int(self.eos_id))

        for b in range(B):
            for n in range(U.size(0)):
                weight = float(w[b, n].item())
                if weight == 0.0:
                    continue
                for l in range(L):
                    tok = int(U[n, l].item())
                    if tok in ignore_ids or tok < 0 or tok >= V:
                        continue
                    p_unsafe[b, l, tok] += weight

        denom = p_unsafe.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p_unsafe = p_unsafe / denom

        mask_pos = (xt == self.mask_index)
        if self.pad_index is not None:
            mask_pos = mask_pos | (xt == self.pad_index)
        unmasked = ~mask_pos
        if unmasked.any():
            p_unsafe[unmasked] = 0.0
            b_idx, l_idx = unmasked.nonzero(as_tuple=True)
            tok = xt[b_idx, l_idx].clamp(0, V - 1)
            p_unsafe[b_idx, l_idx, tok] = 1.0

        return p_unsafe

    @torch.no_grad()
    def empirical_denoiser(self, x_t, sigma=None, move=None, x_0_hat=None, **kwargs):
        """
        x_t: LongTensor [B, L] (noisy sequence at time t)
        sigma: FloatTensor [B] or scalar (if you have sigma(t))
        move:  FloatTensor [B] or scalar (if you directly pass move(t))
        x_0_hat: optional FloatTensor [B, L, V] model posterior (for semantic encoding)
        returns:
           negative_score: Float [B, L, V]  (unsafe posterior to repel)
           metrics: dict with rho, beta_hat, and summary scalars for logging
        """
        assert (sigma is not None) or (move is not None), "Pass sigma or move"
        if move is None:
            if not torch.is_tensor(sigma): sigma = torch.tensor([float(sigma)], device=x_t.device)
            move = self._move_from_sigma(sigma).view(-1)   # [B]
        else:
            move = move.view(-1).to(x_t.device)

        if self.proto_centroids is not None:
            # p_unsafe, metrics = self._unsafe_posterior_clustered(
            #     x_t, move, x_0_hat=x_0_hat, **kwargs
            # )
            logger.warning("Clustered unsafe posterior not implemented yet; falling back to standard unsafe posterior.")
            p_unsafe, metrics = self._unsafe_posterior(
                x_t, move, x_0_hat=x_0_hat, **kwargs
            )
        else:
            p_unsafe, metrics = self._unsafe_posterior(
                x_t, move, x_0_hat=x_0_hat, **kwargs
            )  # [B,L,V], metrics
        rho_tensor = metrics.get("rho", None)
        negative_score_item = 0.0
        if rho_tensor is not None:
            negative_score_item = float(rho_tensor.mean().item())
        metrics["negative_score_item"] = negative_score_item
        return p_unsafe, metrics

    def _log_metrics_to_csv(self, t_idx: int, emp_metrics: dict, strength_metrics: dict):
        if not self.csv_log_path:
            return
        try:
            data = {"step": t_idx if t_idx is not None else -1}

            def _add(d):
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        data[k] = v
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        data[k] = v.item()

            _add(strength_metrics)
            _add(emp_metrics)

            log_q = emp_metrics.get("log_q")
            log_q_len = emp_metrics.get("log_q_len")
            if log_q is not None:
                data["log_q_t_mean"] = log_q.mean().item()
                data["q_t_mean"] = log_q.exp().mean().item()
                if log_q_len is not None:
                    per_token = (log_q / log_q_len).to(torch.float32)
                    per_token_mean = float(per_token.mean().item())
                    per_token_min = float(per_token.min().item())
                    per_token_max = float(per_token.max().item())
                    data["log_q_per_token_mean"] = per_token_mean
                    data["log_q_per_token_min"] = per_token_min
                    data["log_q_per_token_max"] = per_token_max
                    if self._env_flag("SAFE_REPELLENCY_DEBUG"):
                        logger.info(
                            "per-token log_q_t stats: mean %.4f min %.4f max %.4f (target range ~[-12,0], typical around -6~-8).",
                            per_token_mean,
                            per_token_min,
                            per_token_max,
                        )

            self._metrics_buffer.append(data)
            if len(self._metrics_buffer) >= self._csv_buffer_size:
                self._flush_metrics_buffer()
        except Exception as e:
            logger.warning(f"Failed to log repellency metrics to CSV: {e}")

    def _flush_metrics_buffer(self) -> None:
        if not self.csv_log_path:
            return
        if not self._metrics_buffer:
            return
        to_flush = self._metrics_buffer
        self._metrics_buffer = []
        path = self.csv_log_path
        try:
            file_exists = os.path.exists(path)
            existing_fields: list[str] = []
            if file_exists:
                with open(path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    try:
                        existing_fields = next(reader)
                    except StopIteration:
                        existing_fields = []
                if not existing_fields:
                    file_exists = False
            buffer_fields = {k for row in to_flush for k in row.keys()}
            if not file_exists:
                fieldnames = sorted(buffer_fields)
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in to_flush:
                        writer.writerow({fn: row.get(fn, "") for fn in fieldnames})
                return
            new_fields = [k for k in sorted(buffer_fields) if k not in existing_fields]
            if new_fields:
                fieldnames = existing_fields + new_fields
                existing_rows: list[dict[str, str]] = []
                with open(path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in existing_rows:
                        writer.writerow({fn: row.get(fn, "") for fn in fieldnames})
                    for row in to_flush:
                        writer.writerow({fn: row.get(fn, "") for fn in fieldnames})
                return
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=existing_fields, extrasaction='ignore')
                for row in to_flush:
                    writer.writerow(row)
        except Exception as exc:
            logger.warning("Failed to flush repellency CSV buffer: %s", exc)
            self._metrics_buffer = to_flush + self._metrics_buffer

    def conditioning_1(self, x_0_hat, **kwargs):
        """
        x_0_hat: FloatTensor [B, L, V] (model posterior p_theta(x0 | xt, t), already softmaxed)
        kwargs must contain:
        - 'x_t'  : LongTensor [B, L] (current noisy tokens)
        - 'move' : FloatTensor [B]    (move(t))  OR 'sigma' to derive move

        We apply:  p_safe = p + s * (p - p_unsafe); then renormalize across V.
        """
        x_t_full = kwargs['x_t']
        prompt_mask = kwargs.get("prompt_mask", None)
        x_0_hat = x_0_hat[..., : self.vocab_size]
        x_0_hat = x_0_hat / x_0_hat.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if self._env_flag("SAFE_REPELLENCY_DEBUG") and not getattr(self, "_logged_mode_defaults", False):
            kernel_mode_active = self._env_choice("SAFE_KERNEL_MODE", "relaxed", {"relaxed", "strict", "both"})
            beta_mode_active = self._env_choice("SAFE_BETA_MODE", "len", {"raw", "len", "both"})
            guidance_mode_active = self._env_choice("SAFE_GUIDANCE_MODE", "logit", {"logit", "prob", "both"})

            def _mode_source(name: str, default: str) -> str:
                return "env" if os.getenv(name) is not None else f"default:{default}"

            logger.info(
                "[Repellency] resolved modes kernel=%s (%s) beta=%s (%s) guidance=%s (%s)",
                kernel_mode_active,
                _mode_source("SAFE_KERNEL_MODE", "relaxed"),
                beta_mode_active,
                _mode_source("SAFE_BETA_MODE", "len"),
                guidance_mode_active,
                _mode_source("SAFE_GUIDANCE_MODE", "logit"),
            )
            self._logged_mode_defaults = True

        if prompt_mask is not None:
            prompt_width = kwargs.get("prompt_width", None)
            if prompt_width is None:
                prompt_width = int(prompt_mask.any(dim=0).sum().item())
                logger.warning("prompt_width not provided, inferred %d from prompt_mask", prompt_width)
            B, L_total = x_t_full.shape
            prompt_width = int(prompt_width)
            if prompt_width < 0 or prompt_width > L_total:
                raise ValueError(f"Invalid prompt_width={prompt_width} for sequence length {L_total}")
            max_avail = max(L_total - prompt_width, 0)
            target_len = self.continuation_length if self.continuation_length is not None else max_avail
            L_cont = int(min(int(target_len), max_avail)) if max_avail > 0 else 0

            if L_cont == 0:
                logger.debug("No continuation tokens to guide (L_cont=0), returning x_0_hat unchanged")
                return {"x_0_hat": x_0_hat, "mean_x_0_hat": 0.0}

            x_cont = x_t_full[:, prompt_width:prompt_width + L_cont].contiguous()
            cont_mask = torch.zeros((B, L_total), device=x_t_full.device, dtype=torch.bool)
            if L_cont > 0:
                cont_mask[:, prompt_width:prompt_width + L_cont] = True
            p_cont = self.alignment_strategy._create_continuation_probs(  # type: ignore[attr-defined]
                x_0_hat,
                cont_mask,
                L_cont,
            )
            alignment = AlignmentResult(
                x_t=x_cont,
                x_0_hat=p_cont,
                cont_mask=cont_mask,
                should_apply=True,
            )
            if (
                self._env_flag("SAFE_REPELLENCY_DEBUG")
                or self._env_flag("SAFE_REPELLENCY_VALIDATE")
                or self._env_flag("SAFE_REPELLENCY_DEBUG_VALIDATE")
            ):
                assert alignment.x_t.shape[1] == p_cont.shape[1], (
                    "Alignment mismatch: x_t length does not match p_cont length."
                )
                assert alignment.x_0_hat.shape[:2] == alignment.x_t.shape, (
                    "Alignment mismatch: x_0_hat batch/length does not match x_t."
                )
                assert int(alignment.cont_mask[:, :prompt_width].sum().item()) == 0, (
                    "Alignment mask includes prompt tokens."
                )
        else:
            alignment = self.alignment_strategy.align(x_t_full, x_0_hat)
            if alignment.cont_mask is None or not alignment.should_apply:
                if not self._warned_no_alignment:
                    logger.warning("Alignment strategy not implemented; proceeding without alignment.")
                    self._warned_no_alignment = True
                return {"x_0_hat": x_0_hat, "mean_x_0_hat": 0.0}
        if not alignment.should_apply:
            return {"x_0_hat": x_0_hat, "mean_x_0_hat": 0.0}

        align_kwargs = dict(kwargs)
        align_kwargs['x_t'] = alignment.x_t

        p_unsafe, metrics = self.empirical_denoiser(
            x_t=alignment.x_t,
            sigma=align_kwargs.get('sigma', None),
            move=align_kwargs.get('move', None),
            x_0_hat=alignment.x_0_hat,
            **{k: v for k, v in align_kwargs.items() if k not in {"x_t", "sigma", "move"}}
        )
        beta_hat_len = metrics.get("beta_hat_len", None)
        beta_hat_raw = metrics.get("beta_hat_raw", None)
        strength, strength_stats = self.compute_guidance_strength(
            beta_hat=beta_hat_len,
            t_idx=kwargs.get("t_idx", None),
            batch_size=alignment.x_t.size(0),
            device=alignment.x_t.device,
            beta_hat_raw=beta_hat_raw,
            beta_hat_len=beta_hat_len,
            beta_hat_mode=self._env_choice("SAFE_BETA_MODE", "len", {"raw", "len", "both"}),
            log_beta_raw=metrics.get("log_beta_raw_len"),
            log_beta_rel=metrics.get("log_beta_rel_len"),
            log_beta_raw_raw=metrics.get("log_beta_raw_raw"),
            log_beta_rel_raw=metrics.get("log_beta_rel_raw"),
            log_beta_raw_len=metrics.get("log_beta_raw_len"),
            log_beta_rel_len=metrics.get("log_beta_rel_len"),
        )
        p = alignment.x_0_hat
        neg_item = metrics.get("negative_score_item", 0.0)
        guidance_mode = self._env_choice("SAFE_GUIDANCE_MODE", "logit", {"logit", "prob", "both"})
        compute_both = guidance_mode == "both" or self._env_flag("SAFE_REPELLENCY_DEBUG")

        self._debug_log_guidance_pre(p, p_unsafe, strength_stats)

        V = p.size(-1)
        if torch.all(strength == 0):
            p_safe_logit = p
            p_safe_prob = p
        else:
            eps = 1e-4
            p_unsafe_smooth = (1.0 - eps) * p_unsafe + (eps / float(max(V, 1)))
            p_unsafe_smooth = p_unsafe_smooth / p_unsafe_smooth.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            log_p = torch.log(p.clamp_min(1e-30))
            log_pu = torch.log(p_unsafe_smooth.clamp_min(1e-30))
            delta = strength * (log_p - log_pu)
            log_p_safe = log_p + delta
            # Default (logit) mode: repel in log-prob space, then renormalize.
            # This avoids zeroing out low-probability tokens that prob-space subtraction can produce.
            p_safe_logit = torch.softmax(log_p_safe, dim=-1)

            # Prob-space variant (used when SAFE_GUIDANCE_MODE=prob): additive repel.
            p_safe_prob = p + strength * (p - p_unsafe)
            p_safe_prob = p_safe_prob.clamp_min(1e-12)
            p_safe_prob = p_safe_prob / p_safe_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)


        if guidance_mode == "prob":
            p_safe = p_safe_prob
            if self._distribution_logger is not None:
                _ = self._debug_log_guidance_effects(
                    p,
                    p_safe,
                    p_safe_logit,
                    p_safe_prob,
                    p_unsafe,
                    alignment.x_t,
                    strength_stats,
                    metrics.get("weights"),
                    strength,
                    neg_item,
                    compute_both,
                    kwargs.get("t_idx"),
                    kwargs.get("prompt_id"),
                    kwargs.get("prompt_width"),
                    kwargs.get("prompt_variant"),
                    kwargs.get("prompt_mask"),
                    x_t_full if prompt_mask is not None else None,
                )
        else:
            p_safe = p_safe_logit

            debug_strength_metrics = self._debug_log_guidance_effects(
                p, 
                p_safe, 
                p_safe_logit, 
                p_safe_prob,
                p_unsafe, 
                alignment.x_t, 
                strength_stats,
                metrics.get("weights"),
                strength, 
                neg_item, 
                compute_both, 
                kwargs.get("t_idx"),
                kwargs.get("prompt_id"),
                kwargs.get("prompt_width"),
                kwargs.get("prompt_variant"),
                kwargs.get("prompt_mask"),
                x_t_full if prompt_mask is not None else None,
            )
            if debug_strength_metrics:
                    strength_stats.update(debug_strength_metrics)

            if self._env_flag("SAFE_REPELLENCY_DEBUG"):
                self._log_metrics_to_csv(kwargs.get("t_idx"), metrics, strength_stats)
            
            if alignment.cont_mask is not None:
                if p_safe.dtype != x_0_hat.dtype:
                    p_safe = p_safe.to(x_0_hat.dtype)
            p_safe_full = self.alignment_strategy.scatter(p_safe, alignment.cont_mask, x_0_hat)
            return {
                "x_0_hat": p_safe_full,
                "mean_x_0_hat": neg_item,
                "beta_hat_mean": strength_stats.get("beta_hat_mean"),
                "beta_hat_p50": strength_stats.get("beta_hat_p50"),
                "beta_hat_p95": strength_stats.get("beta_hat_p95"),
                "beta_hat_max": strength_stats.get("beta_hat_max"),
                "beta_hat_raw_mean": strength_stats.get("beta_hat_raw_mean"),
                "beta_hat_raw_p50": strength_stats.get("beta_hat_raw_p50"),
                "beta_hat_raw_p95": strength_stats.get("beta_hat_raw_p95"),
                "beta_hat_raw_max": strength_stats.get("beta_hat_raw_max"),
                "beta_hat_len_mean": strength_stats.get("beta_hat_len_mean"),
                "beta_hat_len_p50": strength_stats.get("beta_hat_len_p50"),
                "beta_hat_len_p95": strength_stats.get("beta_hat_len_p95"),
                "beta_hat_len_max": strength_stats.get("beta_hat_len_max"),
                "guidance_strength_mean": strength_stats.get("strength_mean"),
                "schedule_weight_mean": strength_stats.get("g_t_mean"),
                "log_beta_raw_mean": strength_stats.get("log_beta_raw_mean"),
                "log_beta_raw_max": strength_stats.get("log_beta_raw_max"),
                "log_beta_rel_mean": strength_stats.get("log_beta_rel_mean"),
                "log_beta_rel_max": strength_stats.get("log_beta_rel_max"),
                "log_beta_raw_raw_mean": strength_stats.get("log_beta_raw_raw_mean"),
                "log_beta_raw_raw_max": strength_stats.get("log_beta_raw_raw_max"),
                "log_beta_rel_raw_mean": strength_stats.get("log_beta_rel_raw_mean"),
                "log_beta_rel_raw_max": strength_stats.get("log_beta_rel_raw_max"),
                "log_beta_raw_len_mean": strength_stats.get("log_beta_raw_len_mean"),
                "log_beta_raw_len_max": strength_stats.get("log_beta_raw_len_max"),
                "log_beta_rel_len_mean": strength_stats.get("log_beta_rel_len_mean"),
                "log_beta_rel_len_max": strength_stats.get("log_beta_rel_len_max"),
                "strength_zero_frac": strength_stats.get("strength_zero_frac"),
                "mask_frac": strength_stats.get("mask_frac"),
                "changed_frac_masked": strength_stats.get("changed_frac_masked"),
                "changed_frac_logit": strength_stats.get("changed_frac_logit"),
                "changed_frac_prob": strength_stats.get("changed_frac_prob"),
                "kl_logit_mean": strength_stats.get("kl_logit_mean"),
                "kl_prob_mean": strength_stats.get("kl_prob_mean"),
                "unsafe_shift_logit": strength_stats.get("unsafe_shift_logit"),
                "unsafe_shift_prob": strength_stats.get("unsafe_shift_prob"),
                "unsafe_prob_rel_shift_logit": strength_stats.get("unsafe_prob_rel_shift_logit"),
                "unsafe_prob_rel_shift_prob": strength_stats.get("unsafe_prob_rel_shift_prob"),
                "top2_margin": strength_stats.get("top2_margin"),
                "top2_margin_logit": strength_stats.get("top2_margin_logit"),
                "top2_margin_prob": strength_stats.get("top2_margin_prob"),
                "tv_safe_data_mean": strength_stats.get("tv_safe_data_mean"),
                "tv_safe_unsafe_mean": strength_stats.get("tv_safe_unsafe_mean"),
                "tv_data_unsafe_mean": strength_stats.get("tv_data_unsafe_mean"),
                "kl_safe_data_mean": strength_stats.get("kl_safe_data_mean"),
                "kl_safe_unsafe_mean": strength_stats.get("kl_safe_unsafe_mean"),
                "kl_data_unsafe_mean": strength_stats.get("kl_data_unsafe_mean"),
                "js_safe_data_mean": strength_stats.get("js_safe_data_mean"),
                "top1_change_rate": strength_stats.get("top1_change_rate"),
                "top1_overlap_data_unsafe": strength_stats.get("top1_overlap_data_unsafe"),
                "ess_weights": strength_stats.get("ess_weights"),
                "max_weight": strength_stats.get("max_weight"),
                "entropy_weights": strength_stats.get("entropy_weights"),
                "effective_strength": strength_stats.get("effective_strength"),
                "beta_hat_mode_active": strength_stats.get("beta_hat_mode_active"),
                "beta_hat_mode_alt": strength_stats.get("beta_hat_mode_alt"),
                "guidance_mode": guidance_mode,
            }
        return {
            "x_0_hat": p_safe,
            "mean_x_0_hat": neg_item,
            "beta_hat_mean": strength_stats.get("beta_hat_mean"),
            "beta_hat_p50": strength_stats.get("beta_hat_p50"),
            "beta_hat_p95": strength_stats.get("beta_hat_p95"),
            "beta_hat_max": strength_stats.get("beta_hat_max"),
            "beta_hat_raw_mean": strength_stats.get("beta_hat_raw_mean"),
            "beta_hat_raw_p50": strength_stats.get("beta_hat_raw_p50"),
            "beta_hat_raw_p95": strength_stats.get("beta_hat_raw_p95"),
            "beta_hat_raw_max": strength_stats.get("beta_hat_raw_max"),
            "beta_hat_len_mean": strength_stats.get("beta_hat_len_mean"),
            "beta_hat_len_p50": strength_stats.get("beta_hat_len_p50"),
            "beta_hat_len_p95": strength_stats.get("beta_hat_len_p95"),
            "beta_hat_len_max": strength_stats.get("beta_hat_len_max"),
            "guidance_strength_mean": strength_stats.get("strength_mean"),
            "schedule_weight_mean": strength_stats.get("g_t_mean"),
            "log_beta_raw_mean": strength_stats.get("log_beta_raw_mean"),
            "log_beta_raw_max": strength_stats.get("log_beta_raw_max"),
            "log_beta_rel_mean": strength_stats.get("log_beta_rel_mean"),
            "log_beta_rel_max": strength_stats.get("log_beta_rel_max"),
            "log_beta_raw_raw_mean": strength_stats.get("log_beta_raw_raw_mean"),
            "log_beta_raw_raw_max": strength_stats.get("log_beta_raw_raw_max"),
            "log_beta_rel_raw_mean": strength_stats.get("log_beta_rel_raw_mean"),
            "log_beta_rel_raw_max": strength_stats.get("log_beta_rel_raw_max"),
            "log_beta_raw_len_mean": strength_stats.get("log_beta_raw_len_mean"),
            "log_beta_raw_len_max": strength_stats.get("log_beta_raw_len_max"),
            "log_beta_rel_len_mean": strength_stats.get("log_beta_rel_len_mean"),
            "log_beta_rel_len_max": strength_stats.get("log_beta_rel_len_max"),
            "strength_zero_frac": strength_stats.get("strength_zero_frac"),
            "mask_frac": strength_stats.get("mask_frac"),
            "changed_frac_masked": strength_stats.get("changed_frac_masked"),
            "changed_frac_logit": strength_stats.get("changed_frac_logit"),
            "changed_frac_prob": strength_stats.get("changed_frac_prob"),
            "kl_logit_mean": strength_stats.get("kl_logit_mean"),
            "kl_prob_mean": strength_stats.get("kl_prob_mean"),
            "unsafe_shift_logit": strength_stats.get("unsafe_shift_logit"),
            "unsafe_shift_prob": strength_stats.get("unsafe_shift_prob"),
            "unsafe_prob_rel_shift_logit": strength_stats.get("unsafe_prob_rel_shift_logit"),
            "unsafe_prob_rel_shift_prob": strength_stats.get("unsafe_prob_rel_shift_prob"),
            "top2_margin": strength_stats.get("top2_margin"),
            "top2_margin_logit": strength_stats.get("top2_margin_logit"),
            "top2_margin_prob": strength_stats.get("top2_margin_prob"),
            "beta_hat_mode_active": strength_stats.get("beta_hat_mode_active"),
            "beta_hat_mode_alt": strength_stats.get("beta_hat_mode_alt"),
            "guidance_mode": guidance_mode,
        }
        # return {"x_0_hat": p_unsafe, "mean_x_0_hat": neg_item}
