# LLM From Scratch

This repository is a code-first workspace for building language model systems from core components. It includes tokenization utilities, transformer training experiments, and reinforcement learning workflows for reasoning tasks.

## Repository Layout

- [`tokenizer/`](tokenizer/): custom byte-pair encoding (BPE) training, GPT-2 tokenization, chunking utilities, and encode/decode helpers.
- [`transformer/`](transformer/): transformer model implementations, training scripts, inference utilities, and distributed training experiments.
- [`rl/`](rl/): supervised fine-tuning and reinforcement learning pipelines for reasoning-focused experiments.

## How The Pieces Fit Together

1. [`tokenizer/`](tokenizer/) prepares tokenized corpora and tokenizer artifacts.
2. [`transformer/`](transformer/) consumes tokenized datasets for model training and evaluation.
3. [`rl/`](rl/) builds on trained models for post-training and reasoning experiments.

## Tokenizer Workflows

The tokenizer directory currently supports two main paths:

- A custom BPE trainer in [`tokenizer/tokenizer.py`](tokenizer/tokenizer.py) with optional parallel pretokenization, optional in-place pair-count updates, and pickle export for learned vocabulary and merges.
- A GPT-2 based tokenizer in [`tokenizer/off_the_shelf_tokenizer.py`](tokenizer/off_the_shelf_tokenizer.py) that safely tokenizes large UTF-8 files in binary chunks and writes token IDs to `.npy`.

Example commands:

```bash
cd tokenizer

python off_the_shelf_tokenizer.py \
  --input path/to/corpus.txt \
  --out-npy tiny_valid.npy \
  --special "<|endoftext|>" "<|assistant|>" \
  --print-preview

python tokenizer.py \
  --tfile path/to/corpus.txt \
  --vocabsize 8000 \
  --inplace \
  --parallel \
  --store

python tokenizer.py \
  --tfile path/to/corpus.txt \
  --vocabsize 2048 \
  --compare
```

Large corpora and generated `.npy` files are typically kept local and are not tracked in Git.

Outputs produced by the tokenizer workflow include:

- `.npy` token ID arrays for downstream training
- `vocab.pkl` and `merges.pkl` when `tokenizer.py` is run with `--store`

Additional details are documented in [`tokenizer/README.md`](tokenizer/README.md).

## Transformer Workflows

The transformer directory contains the model implementation, training loop, inference scripts, and profiling utilities built on top of tokenized `.npy` datasets.

Example commands:

```bash
cd transformer

python train.py \
  --tfile temp/temp_train.npy \
  --vfile temp/temp_valn.npy

python test.py --sampling greedy

#To profile the run
cd nvtx_runs
python run.py --tfile ../temp/temp.npy --steps 20
```

Typical transformer workflow:

- train a language model with [`transformer/train.py`](transformer/train.py) using tokenized `.npy` arrays
- generate text or inspect checkpoints with [`transformer/test.py`](transformer/test.py)
- profile execution with [`transformer/nvtx_runs/README.md`](transformer/nvtx_runs/README.md) and [`transformer/timing_comparisons/README.md`](transformer/timing_comparisons/README.md)
- explore multi-GPU experiments in [`transformer/distributed_training/README.md`](transformer/distributed_training/README.md)

## Reasoning Workflows

The reasoning pipeline under [`rl/reasoning/`](rl/reasoning/) builds on a base model with supervised fine-tuning in `sft/` and reinforcement-learning-style optimization in `grpo/`.

### SFT

The `sft/` workflow prepares math reasoning data, creates train/validation splits, and fine-tunes the policy model.

Example commands:

```bash
cd rl/reasoning/sft

python dataset_generator.py
python train_val_generator.py
python run.py --lr 1e-4 --batchsize 4 --tfile data/sft_train.jsonl --vfile data/sft_valdn.jsonl
```

Typical `sft/` flow:

- build `sft.jsonl` from the source dataset
- split it into train and validation JSONL files
- optionally filter examples with `model_vllm_test.py` and `filter_jsonl.py`
- train the SFT model and validate it through vLLM-based evaluation

### GRPO

The `grpo/` workflow takes the reasoning setup further with grouped rollouts, reward computation, and policy optimization.

Example commands:

```bash
cd rl/reasoning/grpo

python run.py
```

Typical `grpo/` flow:

- load prompts and grouped rollouts
- score generations with the math grader in `grader/drgrpo_grader.py`
- compute normalized advantages
- optimize the policy with `grpo_clip`, `reinforce_with_baseline`, or `no_baseline`
- sync updated policy weights back into vLLM for the next evaluation cycle

Additional details are documented in [`rl/README.md`](rl/README.md).

## Other Documentation

- [`tokenizer/README.md`](tokenizer/README.md)
- [`transformer/README.md`](transformer/README.md)
- [`transformer/nvtx_runs/README.md`](transformer/nvtx_runs/README.md)
- [`transformer/timing_comparisons/README.md`](transformer/timing_comparisons/README.md)
- [`transformer/distributed_training/README.md`](transformer/distributed_training/README.md)
- [`rl/README.md`](rl/README.md)

## Environment Notes

- Python 3.10 or newer is recommended.
- Most scripts are organized as local modules rather than an installed package, so run commands from the relevant subdirectory.
- Tokenizer scripts rely on `regex`, `numpy`, and `tiktoken`.
- Some transformer and RL workflows assume CUDA-capable hardware and a multi-GPU environment.
