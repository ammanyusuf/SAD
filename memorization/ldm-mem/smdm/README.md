# Memorization in Diffusion Language Models (Reproduction Code)

This repository contains the reproduction code for **“Characterizing Memorization in Diffusion Language Models: Generalized Extraction and Sampling Effects”**.

## Overview

We study memorization in **diffusion language models (DLMs)** under a generalized discoverable extraction framework, and empirically analyze how **sampling resolution** affects verbatim extraction, plus an aligned **PII leakage** comparison between DLMs and autoregressive models (ARMs). 

The codebase is built on top of **Nie et al. (SMDM)**:

- **Pretraining setup and data filtering** fully follow Nie et al.
- Our main extension is to **support training-data auditing** during (pre)training by logging the exact training bins (and optionally block ids) that were consumed.

Upstream pretrain reference (Nie et al. / SMDM):

```text
https://github.com/ML-GSAI/SMDM/tree/main/pretrain
```

## Repository Structure (high level)

- `smdm/`: forked/embedded SMDM codebase + our evaluation utilities
- `smdm/pretrain/`: pretraining / finetuning scripts (audit-enabled)
- `data_process/enron/enron_process.py`: Enron preprocessing (packed dataset) aligned with pretraining tokenizer
- `*.sh` in the repo root: commands used to reproduce **Section 6 (Analysis and Results)**

## Environment

A minimal setup is to use the provided conda environment:

```
conda env create -f smdm/environment.yaml
conda activate smdm
```

(If you already use the official SMDM/TinyLlama environment, this repo is designed to be compatible.)

## Pretraining (Nie et al. recipe + audit extension)

### What is identical to Nie et al.

- Dataset preparation / filtering and the overall pretraining recipe follow Nie et al. (SMDM).
- Please use the upstream SMDM pretrain documentation for the full pipeline.

### What we add for memorization: training-data audit

We extend the training scripts to optionally write audit logs (JSONL) that record which training bins were used in each optimizer-step window.

Relevant scripts:

- `smdm/pretrain/train_mdm.py` (audit-enabled)
- `smdm/pretrain/finetune_mdm.py` (audit-enabled)

Audit flags (available in both):

- `--audit_enable`: enable audit logging
- `--audit_every`: flush once every N optimizer steps (windowed aggregation)
- `--audit_level {bin,block}`: record either only bin names, or (bin, block_id)
- `--audit_fsync`: stronger persistence (slower)

Audit outputs are written under the run directory, e.g.:

- `workdir/.../audit/audit_rankXXXX.jsonl`

(You can post-process audit logs with `smdm/sampling_from_audit_bin.py` or `smdm/process_fre.py` depending on your analysis needs.)



## Enron Finetuning (aligned with pretraining data)

We finetune on the Enron email dataset to align evaluation across model families (including LLaDA-8B).

### 1) Preprocess Enron (Maildir → packed dataset)

Use **the same tokenizer as pretraining** to ensure strict alignment:

```
python data_process/enron/enron_process.py \
  --source_path /path/to/enron/maildir \
  --tokenizer_path /path/to/TinyLlama/checkpoints \
  --destination_path /path/to/output/finetuning_enron_packed \
  --prefix train_enron \
  --percentage 1.0 \
  --num_processes 32
```

This produces packed `*.bin` shards compatible with the SMDM dataloader.



### 2) Finetune

Run the corresponding finetuning scripts under `smdm/pretrain/` (e.g., `finetune_mdm.py` for diffusion; `finetune_ar.py` for autoregressive baselines). If you want audit logs during finetuning, add `--audit_enable ...` as above.



> ## Reproducing “Analysis and Results” (Paper Section 6)
>
> The root `*.sh` scripts are organized to match each subsection in **Section 6**.
>
> > **Note:** You can generate the **100-token evaluation windows** by sampling from the packed `.bin` shards using `smdm/sampling_from_audit_bin.py`, which draws windows from the **audit-logged bins**, decodes them back to text, and exports them as **JSONL** (optionally applying an English-only filter by default).  
> > **Note:** You can generate the **Enron PII prompt JSONL** using `data_process/enron/pii.py`, which constructs prefix-conditioned prompts and corresponding PII targets (e.g., email/phone) in the format expected by the PII evaluation scripts.

### 6.1 Rationality of generalized discoverable extraction in DLM

Goal: compare **theoretical** vs **empirical** memorization probability on sampled SlimPajama windows. 

Run:

- **Sample mask-pattern trajectories (for theoretical estimation / screening):**

  ```
  bash new_diff_saptimes.sh
  ```

- **Empirical estimation via large-scale random-decoding generations (100,000 trials):**

  ```
  bash new_diff_generation.sh
  ```

### 6.2 Impact of Sampling Resolution on Verbatim Memorization

Goal: show that increasing sampling resolution (more steps / per-token) increases exact recovery probability.

Run:

- **Sampling target and Multi-resolution generation on the selected trajectories (steps {1,2,5,10,Max}):**

  ```
  bash sampling_mask-pattern_trj.sh
  bash empricial_generation_mask-pattern_trj.sh
  ```

### 6.3 Empirical Analysis of PII Leakage via Aligned AR–MDM Measurement

Goal: aligned prefix-conditioned PII completion, reporting (n, p)-discoverable leakage under the same protocol. 

Run:

- **Diffusion (local checkpoints) PII evaluation:**

  ```
  bash new_pii_diff.sh
  ```

- **Autoregressive baseline PII evaluation:**

  ```
  bash new_pii_ar.sh
  ```

- **LLaDA-8B (HF) PII evaluation:**

  ```
  bash new_pii_diff_hf.sh
  ```

Input JSONL format (required by `smdm/n_eval_pii_*.py`):

- Must contain `context_text` (the prefix) and at least one suffix field among `{email, name, phone_number}`.

“**One**” vs “**Max**” steps:

- “One” corresponds to single-step recovery (set `--recover_steps 1` and do **not** use per-token override).
- “Max” corresponds to per-token recovery (enable `--recover_each_token`), matching the paper’s “per-token” setting

### 6.4 Validating Memorization Beyond Generalization (Train vs disjoint same-domain test)

Goal: verify the extraction metric reflects **memorization** (Enron train) rather than generalization by comparing against **TREC 2007 Spam** (same domain, disjoint)

Run:

- **Diffusion: reconstruction likelihood / extraction metric under fixed mask rate and 512 queries**

  ```
  bash different_rate.sh
  ```

  Run it twice by swapping `--samples_path`:

  - once for **Enron** windows (train)
  - once for **TREC 2007 Spam** windows (test)

- **Autoregressive counterpart (optional, if you reproduce AR-side curves):**

  ```
  bash different_ar_rate.sh
  ```

