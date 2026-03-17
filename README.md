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

- A custom BPE trainer in [`tokenizer/tokenizer.py`](tokenizer/tokenizer.py) with optional parallel pretokenization and optional in-place pair-count updates, and produces `merges and vocab`.
- A GPT-2 based tokenizer in [`tokenizer/off_the_shelf_tokenizer.py`](tokenizer/off_the_shelf_tokenizer.py) that safely tokenizes large UTF-8 files in binary chunks and writes token IDs to `.npy`.

Example commands:

```bash
cd tokenizer

python off_the_shelf_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --out-npy tiny_valid.npy \
  --special "<|endoftext|>" "<|assistant|>" \
  --print-preview

python tokenizer.py \
  --tfile data/TinyStoriesV2-GPT4-valid.txt \
  --vocabsize 8000 \
  --inplace \
  --parallel \
  --store

python tokenizer.py \
  --tfile data/TinyStoriesV2-GPT4-valid.txt \
  --vocabsize 2048 \
  --compare
```

Outputs produced by the tokenizer workflow include:

- `.npy` token ID arrays for downstream training
- `vocab.pkl` and `merges.pkl` when `tokenizer.py` is run with `--store`

Additional details are documented in [`tokenizer/README.md`](tokenizer/README.md).

## Other Documentation

- [`tokenizer/README.md`](tokenizer/README.md)
- [`transformer/README.md`](transformer/README.md)
- [`transformer/distributed_training/README.md`](transformer/distributed_training/README.md)
- [`rl/README.md`](rl/README.md)

## Environment Notes

- Python 3.10 or newer is recommended.
- Most scripts are organized as local modules rather than an installed package, so run commands from the relevant subdirectory.
- Tokenizer scripts rely on `regex`, `numpy`, and `tiktoken`.
- Some transformer and RL workflows assume CUDA-capable hardware and a multi-GPU environment.
