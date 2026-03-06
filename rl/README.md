# RL

This directory contains two RL tracks:
- `prereqs/`: small REINFORCE intuition builders (coin toss, LLM toy policy, reward simulationxw).
- `reasoning/`: math-reasoning pipeline using Qwen2.5-Math with SFT + GRPO-style RLVR + custom grading.

`rl/alignment/` is currently empty and intentionally ignored here.

## Directory Map

- `prereqs/`
  - `policy.py`: policy basics on logits/probs/logprobs/entropy; includes GPT-2 token sampling demo.
  - `rewards.py`: trajectory-level reward simulation for coin flips and toy LLM episodes.
  - `reinforce_coin.py`: REINFORCE on a Bernoulli coin policy.
  - `reinforce_llm.py`: toy REINFORCE loop directly over a causal LM on prompt `"2 + 2 ="`.
- `reasoning/`
  - `prompts/r1_zero.prompt`: prompt format requiring `<think>...</think><answer>...</answer>`.
  - `grader/drgrpo_grader.py`: math grader + reward functions (`r1_zero_reward_fn`, `question_only_reward_fn`).
  - `sft/`: SFT data generation, split/filter helpers, training, and vLLM validation helpers.
  - `grpo/`: GRPO/RLVR training loop, losses/utilities, and vLLM weight-sync helpers.
  - `docker/`: Dockerfile + dependency list for reproducible environment.
  - `runpod_requirements.txt`, `runpod_pips.sh`: environment setup helpers.
  - `zeroshot.py`, `vllm_test.py`: baseline/evaluation scripts.

## Dependencies

From `reasoning/runpod_requirements.txt`:

- `datasets`
- `transformers>=4.48.2,<5`
- `vllm==0.7.2`
- `latex2sympy2_extended`
- `math_verify`
- `pylatexenc`
- `hf_transfer`
- `scikit-learn`
- `wandb`

Install quickly:

```bash
cd rl/reasoning
pip install -r runpod_requirements.txt
```

## Hardware Assumptions

`reasoning/sft/run.py` and `reasoning/grpo/run.py` are written for multi-GPU setups:
- policy HF model on `cuda:0`
- vLLM model on `cuda:1` for generation/validation

`grpo/run.py` explicitly asserts CUDA.

## End-to-End Reasoning Workflow

Run commands from `rl/reasoning/` unless noted.

### 1) Build SFT dataset

From `rl/reasoning/sft`:

```bash
cd sft
python dataset_generator.py
python train_val_generator.py
```

What it does:
- streams math data from `a-m-team/AM-DeepSeek-R1-Distilled-1.4M`
- filters by source + non-MCQ + `<think>/<answer>` structure + think-token cap
- rewrites prompts with `../prompts/r1_zero.prompt`
- writes JSONL examples
- splits into train/validation JSONL files

### 2) (Optional) Filter training JSONL by answer-correct subset

```bash
python model_vllm_test.py
python filter_jsonl.py
```

- `model_vllm_test.py` can produce `data/correct_answers_indices.pkl` (toggle `IS_FILTERING`).
- `filter_jsonl.py` keeps only those indexed rows.

### 3) Train SFT model

```bash
python run.py --lr 1e-4 --batchsize 4 --tfile data/sft_train.jsonl --vfile data/sft_valdn.jsonl
```

Behavior:
- base model: `Qwen/Qwen2.5-Math-1.5B`
- gradient accumulation (`grad_acc_steps=16`)
- per-epoch validation through vLLM + grader reward metrics
- saves model to `sft_model_bs{effective_batch}_lr{lr}`
- wandb enabled by default (`IS_WANDB=True`)

### 4) Run GRPO/RLVR

From `rl/reasoning/grpo`:

```bash
cd grpo
python run.py
```

Core loop:
- sample training prompts
- generate grouped rollouts with vLLM (`n=group_size`)
- compute rewards using `grader.drgrpo_grader.r1_zero_reward_fn`
- build advantages with group normalization
- optimize policy using one of:
  - `no_baseline`
  - `reinforce_with_baseline`
  - `grpo_clip` (default)
- sync updated HF policy weights back into vLLM for next rollout/eval
- logs train/val reward metrics + optimization stats to wandb

### 5) Zero-shot baseline / quick generation checks

From `rl/reasoning`:

```bash
python zeroshot.py
python vllm_test.py
```

## Script Notes

- `sft/vllm_helper.py` and `grpo/vllm_helper.py` are identical helper modules for:
  - initializing vLLM with monkeypatches for this setup
  - loading HF policy weights into a live vLLM instance
- `sft/utils.py` and `grpo/utils.py` hold most token/logprob/loss math utilities.
- `grpo/wandb_utils.py` is a small logging wrapper.

## Prereq RL Demos

From `rl/prereqs`:

```bash
python policy.py
python rewards.py
python reinforce_coin.py
python reinforce_llm.py
```

These are educational scripts; they are not wired into the `reasoning/` training pipeline.

## Python File Inventory

- `prereqs/policy.py`
- `prereqs/reinforce_coin.py`
- `prereqs/reinforce_llm.py`
- `prereqs/rewards.py`
- `reasoning/grader/drgrpo_grader.py`
- `reasoning/grpo/run.py`
- `reasoning/grpo/utils.py`
- `reasoning/grpo/vllm_helper.py`
- `reasoning/grpo/wandb_utils.py`
- `reasoning/sft/dataset_generator.py`
- `reasoning/sft/filter_jsonl.py`
- `reasoning/sft/model_vllm_test.py`
- `reasoning/sft/run.py`
- `reasoning/sft/train_val_generator.py`
- `reasoning/sft/utils.py`
- `reasoning/sft/vllm_helper.py`
- `reasoning/vllm_test.py`
- `reasoning/zeroshot.py`

