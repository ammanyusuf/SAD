'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import logging
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
import gc
import json
from datasets import Dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import sys
import os
import resource

from transformers import AutoTokenizer, AutoModel
from generate import generate

try:
    from utils.safety_env import safety_settings_from_env
    from sampling.safe_hooks import build_llada_repellency_hook
except Exception:  # pragma: no cover - optional safety hook support
    safety_settings_from_env = None
    build_llada_repellency_hook = None

_log_level = os.getenv("LLADA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="[%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
eval_logger = logging.getLogger(__name__)

print(f"[DEBUG] eval_llada path: {__file__}", flush=True)
print(f"[DEBUG] eval_llada argv: {sys.argv}", flush=True)
try:
    import lm_eval as _lm_eval

    print(f"[DEBUG] lm_eval module: {_lm_eval.__file__}", flush=True)
    print(f"[DEBUG] lm_eval version: {getattr(_lm_eval, '__version__', 'unknown')}", flush=True)
except Exception as exc:  # pragma: no cover - debug only
    print(f"[DEBUG] lm_eval import failed: {exc}", flush=True)


def _read_proc_status_memory():
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return {}
    values = {}
    try:
        with open(status_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "VmPeak:")):
                    parts = line.split()
                    if len(parts) >= 2:
                        values[parts[0].rstrip(":")] = " ".join(parts[1:])
    except Exception:
        return {}
    return values


def log_memory(tag: str) -> None:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    proc_vals = _read_proc_status_memory()
    cuda_stats = ""
    try:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() // (1024 * 1024)
            reserved = torch.cuda.memory_reserved() // (1024 * 1024)
            cuda_stats = f" | cuda_alloc_mb={alloc} cuda_reserved_mb={reserved}"
    except Exception:
        cuda_stats = ""
    if proc_vals:
        print(
            f"[MEM] {tag} | ru_maxrss_kb={rss_kb} | "
            f"VmRSS={proc_vals.get('VmRSS')} VmHWM={proc_vals.get('VmHWM')} "
            f"VmSize={proc_vals.get('VmSize')} VmPeak={proc_vals.get('VmPeak')}{cuda_stats}",
            flush=True,
        )
    else:
        print(f"[MEM] {tag} | ru_maxrss_kb={rss_kb}{cuda_stats}", flush=True)

def _checkpoint_path_for_rank(base_path: str, rank: int | None) -> str:
    if rank is None:
        return base_path
    suffix = f".rank{rank}"
    if base_path.endswith(".jsonl"):
        return base_path[:-6] + suffix + ".jsonl"
    return base_path + suffix


def _load_gen_checkpoint(path: str) -> dict[int, str]:
    cache: dict[int, str] = {}
    if not path or not os.path.exists(path):
        return cache
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = obj.get("idx")
                output = obj.get("output")
                if isinstance(idx, int) and isinstance(output, str):
                    cache[idx] = output
    except Exception as exc:
        eval_logger.warning("Failed to load checkpoint %s: %s", path, exc)
    return cache


def _append_gen_checkpoint(path: str, idx: int, output: str) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"idx": idx, "output": output}) + "\n")
            fh.flush()
    except Exception as exc:
        eval_logger.warning("Failed to write checkpoint %s: %s", path, exc)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        log_memory("init:start")
        self.model_path = model_path
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()
        log_memory("init:after_model_load")

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log_memory("init:after_tokenizer")

        self.mc_num = int(mc_num)
        self.batch_size = self._resolve_batch_size(batch_size)
        assert self.mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking    
        self.safety_settings = None
        self.logits_hook = None
        if safety_settings_from_env is not None and build_llada_repellency_hook is not None:
            try:
                safety = safety_settings_from_env()
                if safety.enabled:
                    self.safety_settings = safety
                    self.logits_hook = build_llada_repellency_hook(
                        self.tokenizer, safety, self.device
                    )
                    eval_logger.info("LLaDA safety hook enabled for capability eval.")
            except Exception as exc:
                eval_logger.warning("Failed to enable LLaDA safety hook: %s", exc)

    def _resolve_batch_size(self, batch_size):
        if isinstance(batch_size, str):
            normalized = batch_size.strip().lower()
            if normalized in {"auto", "probe", "oom"}:
                return self._probe_batch_size()
            if normalized.isdigit():
                return int(normalized)
            raise ValueError(f"Invalid batch_size value: {batch_size}")
        return int(batch_size)

    def _probe_batch_size(self) -> int:
        if self.device.type != "cuda":
            eval_logger.info("Auto batch_size disabled: non-CUDA device.")
            return 1
        candidates_raw = os.getenv("LLADA_BATCHSIZE_CANDIDATES", "64,32,16,8,4,2,1")
        try:
            candidates = [int(x) for x in candidates_raw.split(",") if x.strip()]
        except ValueError:
            candidates = [64, 32, 16, 8, 4, 2, 1]
        max_bs_env = os.getenv("LLADA_BATCHSIZE_PROBE_MAX", "").strip()
        max_bs = int(max_bs_env) if max_bs_env.isdigit() else None
        probe_tokens_env = os.getenv("LLADA_BATCHSIZE_PROBE_TOKENS", "2048").strip()
        probe_tokens = int(probe_tokens_env) if probe_tokens_env.isdigit() else 2048
        seq_len = min(self.max_length, probe_tokens) if self.max_length else probe_tokens

        eval_logger.info(
            "Auto batch_size probe starting: candidates=%s seq_len=%s mc_num=%s",
            candidates,
            seq_len,
            self.mc_num,
        )
        for bs in candidates:
            if max_bs is not None and bs > max_bs:
                continue
            if bs <= 0 or (self.mc_num % bs) != 0:
                continue
            try:
                self._probe_forward(bs, seq_len)
                eval_logger.info("Auto batch_size selected: %s", bs)
                return bs
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                    eval_logger.warning("OOM at batch_size=%s; trying smaller.", bs)
                    self._clear_cuda()
                    continue
                raise
        eval_logger.warning("Auto batch_size fallback to 1.")
        return 1

    def _probe_forward(self, batch_size: int, seq_len: int) -> None:
        input_ids = torch.full(
            (batch_size, seq_len),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            _ = self.model(input_ids).logits
        del input_ids, _
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _clear_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def tokenizer_name(self):
        name = getattr(self.tokenizer, "name_or_path", "")
        return name or self.model_path or ""
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for step_idx in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
            del loss, logits, mask_indices, perturbed_seq, p_mask

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        tokenized = []
        for entry in ds:
            prefix, target = self._encode_pair(entry["prefix"], entry["target"])
            tokenized.append(
                {
                    "prefix_text": entry["prefix"],
                    "target_text": entry["target"],
                    "prefix": torch.tensor(prefix, dtype=torch.long),
                    "target": torch.tensor(target, dtype=torch.long),
                }
            )
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in tokenized]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for idx, elem in enumerate(tqdm(tokenized, desc="Computing likelihood...")):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        ckpt_base = os.getenv("LLADA_GEN_CHECKPOINT_PATH", "").strip()
        ckpt_path = ""
        if ckpt_base:
            ckpt_path = _checkpoint_path_for_rank(ckpt_base, getattr(self, "_rank", None))
        cache = _load_gen_checkpoint(ckpt_path) if ckpt_path else {}
        if ckpt_path:
            if os.path.exists(ckpt_path):
                print(
                    f"[CHECKPOINT] enabled: {ckpt_path} (loaded {len(cache)} cached outputs)",
                    flush=True,
                )
            else:
                print(
                    f"[CHECKPOINT] enabled: {ckpt_path} (file not found; starting fresh)",
                    flush=True,
                )

        tokenized = []
        for req in requests:
            question = req.args[0]
            tokenized.append(
                {
                    "question": torch.tensor(
                        self.tokenizer(question)["input_ids"], dtype=torch.long
                    ),
                    "question_text": question,
                    "until": req.args[1]["until"],
                }
            )

        out = [None] * len(tokenized)
        saved_since_log = 0
        for idx, elem in enumerate(tqdm(tokenized, desc="Generating...")):
            cached = cache.get(idx)
            if cached is not None:
                out[idx] = cached
                continue
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
 
            generated_answer = generate(self.model, prompt, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, cfg_scale=self.cfg, remasking=self.remasking, mask_id=self.mask_id,
                                        logits_hook=self.logits_hook, logits_hook_ctx={"prompt_variant": "capability"})
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out[idx] = generated_answer
            if ckpt_path:
                _append_gen_checkpoint(ckpt_path, idx, generated_answer)
                saved_since_log += 1
                if saved_since_log % 5 == 0:
                    print(
                        f"[CHECKPOINT] saved {saved_since_log} new outputs (last idx={idx}).",
                        flush=True,
                    )

            if getattr(self, "accelerator", None) is not None:
                self.accelerator.wait_for_everyone()

        return [x if x is not None else "" for x in out]


def _patch_cache_requests_arg(parser):
    """Work around lm_eval 0.4.10 cache_requests argparse typing bug."""
    patched = 0
    for action in getattr(parser, "_actions", []):
        option_strings = getattr(action, "option_strings", [])
        dest = getattr(action, "dest", None)
        if "--cache_requests" in option_strings or dest == "cache_requests":
            # Ensure argparse keeps the raw string value.
            if getattr(action, "type", None) is not None and action.type is not str:
                action.type = str
            # Some versions incorrectly combine type conversion with choices.
            action.choices = None
            patched += 1
        # Recurse into subparsers (lm-eval "run" subcommand).
        if action.__class__.__name__ == "_SubParsersAction":
            for subparser in action.choices.values():
                patched += _patch_cache_requests_arg(subparser)
    return patched


def _cli_evaluate_compat():
    try:
        import sys as _sys
        from lm_eval import __main__ as lm_eval_main

        original_argv = list(_sys.argv)
        stripped_argv = []
        cache_requests_value = None
        idx = 0
        while idx < len(original_argv):
            arg = original_argv[idx]
            if arg == "--cache_requests" and idx + 1 < len(original_argv):
                cache_requests_value = original_argv[idx + 1]
                idx += 2
                continue
            if arg.startswith("--cache_requests="):
                cache_requests_value = arg.split("=", 1)[1]
                idx += 1
                continue
            stripped_argv.append(arg)
            idx += 1
        if cache_requests_value is not None:
            _sys.argv = stripped_argv

        if hasattr(lm_eval_main, "setup_parser") and hasattr(
            lm_eval_main, "parse_eval_args"
        ):
            parser = lm_eval_main.setup_parser()
            _patch_cache_requests_arg(parser)
            args = lm_eval_main.parse_eval_args(parser)
            if cache_requests_value is not None:
                args.cache_requests = cache_requests_value
            lm_eval_main.cli_evaluate(args)
            _sys.argv = original_argv
            return
    except Exception:
        # Fall back to the standard CLI behavior if anything goes wrong.
        pass
    from lm_eval import __main__ as lm_eval_main
    lm_eval_main.cli_evaluate()


if __name__ == "__main__":
    set_seed(1234)
    _cli_evaluate_compat()
    
