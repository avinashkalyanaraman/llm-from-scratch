# LLM From Scratch

A practical, code-first repository for building language-model systems from core components:
- tokenization (`tokenizer/`)
- transformer modeling + training (`transformer/`)
- RL for reasoning (`rl/`)

This repo is organized as an experimentation workspace rather than a single packaged library.

## Repository Structure

### Core directories
- [`tokenizer/`](tokenizer): byte/BPE tokenizer experiments, custom encoder/decoder, and tiktoken-based tokenization scripts.
- [`transformer/`](transformer): from-scratch Transformer modules, training/inference scripts, profiling, and distributed training experiments.
- [`rl/`](rl): RL prerequisites + reasoning-focused SFT/GRPO workflows with vLLM and custom grading.



## How The Pieces Connect

1. `tokenizer/` prepares tokenized data artifacts (`.npy` token id arrays).
2. `transformer/` trains and evaluates Transformer language models on tokenized corpora.
3. `rl/reasoning/` runs post-training pipelines:
   - supervised fine-tuning (`sft/`)
   - RLVR/GRPO-style updates (`grpo/`)
   - grading and reward computation (`grader/`)

## Quick Start

### 1) Tokenization

Examples from [`tokenizer/`](tokenizer):

```bash
cd tokenizer
python off_the_shelf_tokenizer.py --input data/TinyStoriesV2-GPT4-valid.txt --out-npy tiny_valid.npy
python tokenizer.py
```

### 2) Transformer training/inference

Examples from [`transformer/`](transformer):

```bash
cd transformer
python train.py --tfile temp/temp.npy --vfile temp/temp.npy
python test.py --sampling greedy
```

Distributed experiments:

```bash
cd transformer/distributed_training
# see subdirectory README for per-experiment commands
```

### 3) RL reasoning pipeline

Examples from [`rl/reasoning/`](rl/reasoning):

```bash
cd rl/reasoning/sft
python dataset_generator.py
python train_val_generator.py
python run.py

cd ../grpo
python run.py
```

## Detailed READMEs

- [`tokenizer/`](tokenizer) (key scripts: `tokenizer.py`, `off_the_shelf_tokenizer.py`, `encoder_decoder.py`)
- [`transformer/README.md`](transformer/README.md)
- [`transformer/distributed_training/README.md`](transformer/distributed_training/README.md)
- [`rl/README.md`](rl/README.md)

## Environment Notes

- Python 3.10+ recommended.
- Most scripts are standalone and rely on local imports, so run them from the directory shown in examples.
- `rl/reasoning` workflows use `vllm` and are written for CUDA multi-GPU setups (`cuda:0` + `cuda:1`).
- `transformer/distributed_training` scripts use `torch.multiprocessing.spawn` and fixed localhost process-group settings.

