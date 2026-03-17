# Transformer

A from-scratch PyTorch Transformer stack with:
- custom layers (`Linear`, `RMSNorm`, `ROPE`, MHA, SwiGLU FFN)
- custom optimizers (`SGD`, `AdamW`) + LR schedule + grad clipping
- training/inference scripts
- profiling/timing experiments
- distributed training experiments (manual DDP variants and optimizer sharding)

## Directory Overview

### Core model and training
- `transformer_pipeline.py`: full language model (`Embedding -> N x TransformerBlock -> RMSNorm -> Linear logits`).
- `transformer_block.py`: pre-norm block with residuals (`RMSNorm -> MHA`, then `RMSNorm -> FFN`).
- `embedding.py`: learned token embedding table (truncated normal init).
- `multiHeadAttention.py`: causal self-attention with optional ROPE, plus a slower reference path.
- `rope.py`: rotary positional embedding (`ROPE2` is the main implementation used by MHA).
- `ffn.py`: SwiGLU FFN (`W2(SiLU(W1x) * W3x)`).
- `linear.py`: custom linear projection layer (`x @ W`) with truncated-normal init.
- `rmsnorm.py`: custom RMSNorm with upcast-to-fp32 for stable norm calculation.
- `loss.py`: custom cross-entropy/perplexity helpers over logits.
- `optimizer.py`: custom `SGD`, custom `AdamW`, cosine LR schedule, gradient clipping.
- `train.py`: end-to-end training loop (checkpointing, validation, optional timing, wandb logging).
- `test.py`: checkpointed text generation (greedy or nucleus/top-p).
- `checkpoint.py`: save/load tuple `(model_state, optimizer_state, iteration)`.
- `filehandler.py`: mmap-backed token dataset with block cache for low-overhead sequential access.
- `utils.py`: stable softmax + scaled dot-product attention + basic unit tests.

### Profiling and benchmarking
- [`nvtx_runs/README.md`](nvtx_runs/README.md): guide to the CUDA NVTX profiling scripts and tracked profiling notes.
- [`timing_comparisons/README.md`](timing_comparisons/README.md): guide to the timing benchmarks and tracked DDP timing outputs.
- `nvtx_runs/run.py`: CUDA-only training trace with NVTX ranges (supports optional stream overlap path).
- `nvtx_runs/stream_comparison.py`: compares single-stream vs explicit copy-stream overlap.
- `timing_comparisons/linear_comparison.py`: compares three linear forward implementations.
- `timing_comparisons/mha_comparison.py`: baseline-vs-this-impl MHA forward/backward timing harness.
- `compute_flops.py`: quick FLOP composition calculator for two model configurations.

### Distributed training
- `distributed_training/...`: collective communication basics and custom DDP experiments.
- Detailed guide: `distributed_training/README.md`.

### Misc / demos
- `module_list_learn.py`: demonstrates `list` vs `nn.ModuleList` parameter registration.
- `wandb_play.py`: minimal wandb logging demo script.
- `wandb_utils.py`: read/write wandb run ID helper.
- `__init__.py`: empty package marker.

## Requirements

- Python 3.10+
- `torch`
- `numpy`
- `tiktoken` (for `test.py` and `filehandler.py` helper path)
- `wandb` (training script imports it)
- `matplotlib` (distributed timing plot)

Install example:

```bash
pip install torch numpy tiktoken wandb matplotlib torchvision
```

## Expected Data Format

`train.py`/`nvtx` scripts expect `.npy` files containing a flat token-id array.

- `filehandler.myDataset(filepath, context_len)` creates `(x, y)` pairs:
  - `x = tokens[i : i+context_len]`
  - `y = tokens[i+1 : i+1+context_len]`
- Data is loaded with `np.load(..., mmap_mode='r+')` and cached in large chunks to reduce repeated disk access.

## Run From This Directory

Run commands from `transformer/` unless noted.

```bash
cd transformer
```

## Train

Basic training:

```bash
python train.py --tfile temp/temp.npy --vfile temp/temp.npy
```

Useful args:
- `--lr --beta1 --beta2`
- `--d_model --seqlen --heads --num_layers --vocabsize`
- `--batchsize --epochs`
- `--tokenlimit --val_tokenlimit`
- `--time` (records step timing/throughput)

Behavior baked into `train.py`:
- wandb logging is enabled via `ISWANDB = True`
- checkpoint every `5000` steps
- validation every `1000` steps
- wandb log every `50` steps
- auto-resume from `checkpoints/step9600.pt` if that file exists

## Generate Text

```bash
python test.py --sampling greedy
python test.py --sampling nucleus --p 0.9 --temperature 0.8
```

Notes:
- `test.py` currently hardcodes `checkpoint_file = 'checkpoints/step10000.pt'` and asserts it exists.
- prompt string is currently hardcoded in script (`"tell me a story set in istanbul!"`).

## Profiling / Timing Scripts

Detailed subdirectory guides:

- [`nvtx_runs/README.md`](nvtx_runs/README.md)
- [`timing_comparisons/README.md`](timing_comparisons/README.md)

### NVTX trace (CUDA only)

From `transformer/nvtx_runs`:

```bash
cd nvtx_runs
python run.py --tfile ../temp/temp.npy --steps 20
python run.py --tfile ../temp/temp.npy --steps 20 --stream
```

`run.py` marks H2D, forward, loss, LR update, backward, clipping, and optimizer-step ranges via `torch.cuda.nvtx`.

### Stream overlap comparison

From `transformer/nvtx_runs`:

```bash
python stream_comparison.py --tfile ../temp/temp.npy --steps 20
python stream_comparison.py --tfile ../temp/temp.npy --steps 20 --stream
```

### Linear micro-benchmark

From `transformer/timing_comparisons`:

```bash
cd timing_comparisons
python linear_comparison.py
```

### MHA/model timing comparison

From `transformer/timing_comparisons`:

```bash
python mha_comparison.py --type comparison --epochs 5 --warmups 1
python mha_comparison.py --type baseline --epochs 5 --warmups 1
```

`--type baseline` depends on external modules under `../../assignment2-systems`.

## Distributed Training

A full walkthrough and per-experiment commands are in:
- `transformer/distributed_training/README.md`

Highlights covered there:
- collectives basics (`all_reduce`, `broadcast`, `reduce_scatter`)
- naive manual DDP
- flattened gradient all-reduce
- overlap of communication and gradient computation (hooks)
- bucketed gradient overlap
- optimizer sharding + post-step param broadcast

## Standalone Utility/Demo Scripts

From `transformer/`:

```bash
python utils.py              # softmax/attention unit-style checks
python rope.py               # ROPE2 tests
python compute_flops.py      # FLOP breakdown prints
python module_list_learn.py  # ModuleList registration demo
python wandb_play.py         # wandb demo
```

## Notes and Caveats

- Imports are mostly local-module style (not package-relative), so working directory matters.
- Several scripts use custom implementations instead of `torch.nn` / `torch.optim` built-ins intentionally for learning.
- Some scripts assume specific checkpoint filenames and directory layout.
- `train.py` and `test.py` use `torch.compile` conditionally by device type.

## Python File Inventory

Top-level:
- `__init__.py`
- `checkpoint.py`
- `compute_flops.py`
- `embedding.py`
- `ffn.py`
- `filehandler.py`
- `linear.py`
- `loss.py`
- `module_list_learn.py`
- `multiHeadAttention.py`
- `optimizer.py`
- `rmsnorm.py`
- `rope.py`
- `test.py`
- `train.py`
- `transformer_block.py`
- `transformer_pipeline.py`
- `utils.py`
- `wandb_play.py`
- `wandb_utils.py`

NVTX:
- `nvtx_runs/README.md`
- `nvtx_runs/run.py`
- `nvtx_runs/stream_comparison.py`

Timing comparisons:
- `timing_comparisons/README.md`
- `timing_comparisons/linear_comparison.py`
- `timing_comparisons/mha_comparison.py`

Distributed training:
- `distributed_training/naive_ddp/run.py`
- `distributed_training/flattened_ddp/run.py`
- `distributed_training/overlap_commnxn_compn/run.py`
- `distributed_training/overlap_commnxn_compn/run2.py`
- `distributed_training/overlap_commnxn_compn/ddp_overlap_indiv_params.py`
- `distributed_training/bucketed_overlap_commnxn_compn/run.py`
- `distributed_training/bucketed_overlap_commnxn_compn/ddp_overlap_bucketed.py`
- `distributed_training/optimizer_sharding/run.py`
- `distributed_training/optimizer_sharding/sharded_optimizer.py`
- `distributed_training/msg_passing_basics/plot_utils.py`
- `distributed_training/msg_passing_basics/single_node_toy.py`
- `distributed_training/msg_passing_basics/single_node_tensor_bcast.py`
- `distributed_training/msg_passing_basics/single_node_reduce_scatter.py`
- `distributed_training/msg_passing_basics/single_node_time_sweep.py`
