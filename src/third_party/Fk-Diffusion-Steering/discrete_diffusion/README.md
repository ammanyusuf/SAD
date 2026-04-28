# FK Steering x Discrete Diffusion 

This repository contains our implementation for sampling from a discrete text diffusion model ([MDLM](https://github.com/kuleshov-group/mdlm)) with FK Steering.

## Installation

In your (new?) python environment, you can install the required dependencies by running:

```bash
chmod +x setup.sh
./setup.sh
```

Python 3.9+ is recommended.

## Structure

### Sampling and FK Steering Code

- [**generate_with_fk.py**](generate_with_fk.py): Main script for generating text samples using FK Steering and a trained MDLM-style diffusion model.

- [**fk_diffusion.py**](fk_diffusion.py): Implements FK Steering on a discrete diffusion model, integrating reward functions for controlled text generation.

- [**fkd_class.py**](fkd_class.py): Contains the FKD (Feynman-Kac Diffusion) steering mechanism for resampling during the diffusion process. This class is used in ``fk_diffusion.py``.

- [**reward_functions.py**](reward_functions.py): Defines various reward functions, including toxicity, GPT-2 perplexity, CoLA acceptability, and InfiniGram perplexity.

### Configuration
The `configs/` folder contains configuration files for FK Steering and sampling parameters.

- [**fk_steering_config.yaml**](configs/fk_steering_config.yaml): Configuration file defining FK Steering parameters, including:
  - `potential_type`: FK Steering potential type (e.g., `diff`, `max`, `bon`. or `rt`).
  - `k_particles`: Number of particles used in resampling.
  - `lmbda`: Lambda hyperparameter for reward scaling.
  - `reward_fn`: The reward function to use (e.g., `toxicity`, `cola`, `gpt2_perp`).
  - `reward_trim_length`: Number of tokens considered for reward computation.
  - `resample_frequency`: Defines resampling frequency (-1 disables resampling).
  - `num_x0_samples`: Number of x0 samples to use for reward computation.


### Experiment Scripts

The following scripts can be used for generating samples:

- [**run_cola_reward.sh**](run_cola_reward.sh): CoLA grammatical acceptability steering

- [**run_gpt2_reward.sh**](run_gpt2_reward.sh): GPT-2 perplexity steering

- [**run_infinigram_reward.sh**](run_infinigram_reward.sh): InfiniGram perplexity steering (a bit slow!)

- [**run_toxicity_reward.sh**](run_toxicity_reward.sh): Toxicity steering. **WARNING: Results in potentially harmful and offensive outputs.**

### Evaluation

The `evaluation/` folder contains scripts for evaluating generated text samples.

Once you have generated samples, you can evaluate them using [**compute_metrics.sh**](./evaluation/compute_metrics.sh). 
Resulting outputs will be saved in `./outputs/` folder in the `fk_steering` child directory.

- [**compute_metrics.sh**](evaluation/compute_metrics.sh): Shell script for automating evaluation on multiple generated samples.

- [**mdlm_to_eval_format.py**](evaluation/mdlm_to_eval_format.py): Converts MDLM-generated samples into an evaluation-ready format.

- [**evaluate.py**](evaluation/evaluate.py): Computes various evaluation metrics on generated text, including perplexity, CoLA acceptability, distinctiveness, and toxicity.

- [**aggregate_over_seeds_mdlm.py**](evaluation/aggregate_over_seeds_mdlm.py): After running [compute_metrics.sh](evaluation/compute_metrics.sh), you can use this script to aggegrate over seeds.

- [**pplm_discrim_prompts_orig.jsonl**](evaluation/pplm_discrim_prompts_orig.jsonl): Contains prompts used for evaluating text generation (from [SSD-LM](https://github.com/xhan77/ssd-lm)).
