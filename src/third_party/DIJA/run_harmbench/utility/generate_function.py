from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import pdb
import random 

_DEFAULT_LOGITS_HOOK = None
_DEFAULT_LOGITS_HOOK_CTX: Optional[dict] = None
_DEFAULT_T_START: Optional[int] = None
_DEFAULT_T_END: Optional[int] = None
_DEFAULT_HOOK_FACTORY = None


def set_logits_hook(
    logits_hook=None,
    *,
    logits_hook_ctx: Optional[dict] = None,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
    hook_factory=None,
) -> None:
    global _DEFAULT_LOGITS_HOOK, _DEFAULT_LOGITS_HOOK_CTX, _DEFAULT_T_START, _DEFAULT_T_END, _DEFAULT_HOOK_FACTORY
    _DEFAULT_LOGITS_HOOK = logits_hook
    _DEFAULT_LOGITS_HOOK_CTX = logits_hook_ctx
    _DEFAULT_T_START = t_start
    _DEFAULT_T_END = t_end
    _DEFAULT_HOOK_FACTORY = hook_factory


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)

  


def generate(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x = input_ids
        # feature_cache = dLLMCache()
        # feature_cache.reset_cache(0) 
        num_transfer_tokens = 1
        while (x == mask_id).any():
            mask_index = x == mask_id
            # pdb.set_trace()
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(
                mask_index, x0, x
            )
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens
                    ).indices
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            # pdb.set_trace()
        return x
    

def generate_llada(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    *,
    logits_hook=None,
    logits_hook_ctx: Optional[dict] = None,
    t_start: Optional[int] = None,
    t_end: Optional[int] = None,
    tokenizer=None,
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x = input_ids
        prompt_index = x != mask_id
        if (input_ids == mask_id).any():
            pos = torch.arange(prompt_length, device=input_ids.device)[None, :].expand_as(input_ids)
            masked_pos = torch.where(input_ids == mask_id, pos, torch.full_like(pos, prompt_length))
            prompt_width = int(masked_pos.min(dim=1).values.min().item())
        else:
            prompt_width = prompt_length

        if logits_hook is None and _DEFAULT_HOOK_FACTORY is not None and tokenizer is not None:
            logits_hook = _DEFAULT_HOOK_FACTORY(tokenizer, model.device)
        if logits_hook is None:
            logits_hook = _DEFAULT_LOGITS_HOOK
        if logits_hook_ctx is None:
            logits_hook_ctx = dict(_DEFAULT_LOGITS_HOOK_CTX or {})
        else:
            logits_hook_ctx = dict(logits_hook_ctx)
        if t_start is None:
            t_start = _DEFAULT_T_START
        if t_end is None:
            t_end = _DEFAULT_T_END

        total_steps = max(int(steps), 1)
        global_step = 0

        # feature_cache = dLLMCache()
        # feature_cache.reset_cache(0) 
        num_transfer_tokens = 1
        while (x == mask_id).any():
            mask_index = x == mask_id
            # pdb.set_trace()
            logits = model(x, attention_mask=attention_mask).logits
            if logits_hook is not None:
                t_current = max((total_steps - 1) - global_step, 0)
                if t_start is None or t_end is None or (t_start <= t_current <= t_end):
                    extra = dict(logits_hook_ctx)
                    extra["prompt_width"] = prompt_width
                    extra.setdefault("total_steps", total_steps)
                    extra.setdefault("global_step", global_step)
                    extra.setdefault("vocab_size", logits.shape[-1])
                    logits = logits_hook(
                        logits,
                        x=x,
                        t=t_current,
                        mask_index=mask_index,
                        prompt_index=prompt_index,
                        attention_mask=attention_mask,
                        extra=extra,
                    )
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(
                mask_index, x0, x
            )
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens
                    ).indices
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            global_step += 1
            # pdb.set_trace()
        return x
    


def generate_mmada(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x = input_ids
        # feature_cache = dLLMCache()
        # feature_cache.reset_cache(0) 
        num_transfer_tokens = 1
        while (x == mask_id).any():
            mask_index = x == mask_id
            # pdb.set_trace()
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0 = torch.where(
                mask_index, x0, x
            )
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                if (x[j] == mask_id).any():
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens
                    ).indices
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            # pdb.set_trace()
        return x
