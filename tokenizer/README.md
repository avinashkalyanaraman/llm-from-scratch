# Tokenizer

This directory contains the tokenization utilities used by the repository. It includes a custom BPE trainer, a GPT-2 based tokenizer for producing `.npy` datasets, and helper utilities for chunking files and validating encode/decode behavior.

## Contents

- [`tokenizer.py`](tokenizer.py): custom BPE trainer with GPT-2 style pretokenization, optional parallel pretokenization, optional in-place pair-count updates, and artifact export support.
- [`off_the_shelf_tokenizer.py`](off_the_shelf_tokenizer.py): GPT-2 tokenizer built on `tiktoken`, with chunked binary reads, UTF-8 safe boundaries, and special-token-aware streaming.
- [`encoder_decoder.py`](encoder_decoder.py): helper script for encoding text with learned merges and decoding token IDs back to text.
- [`chunker.py`](chunker.py): utility for finding chunk boundaries aligned to a special token, used by the parallel pretokenizer.
- [`settings.py`](settings.py): GPT-2 regex pretokenization pattern and configured special tokens.
- [`filehandler.py`](filehandler.py): small helpers for writing tokenizer artifacts.
- [`tokenizer_timing_numbers.txt`](tokenizer_timing_numbers.txt): sample runtime comparison for the custom tokenizer variants.

## Dependencies

Install the tokenizer dependencies from this directory or from the repository root:

```bash
pip install regex numpy tiktoken
```

Python 3.10 or newer is recommended.

## Running The Scripts

The tokenizer modules use local imports, so run them from [`tokenizer/`](./):

```bash
cd tokenizer
```

Large corpus files under `data/` and generated `.npy` outputs are intentionally kept out of version control. The commands below assume you provide your own local UTF-8 text file path.

## Workflow 1: Tokenize A Corpus With GPT-2

[`off_the_shelf_tokenizer.py`](off_the_shelf_tokenizer.py) is the fastest path for producing token ID arrays for downstream model training.

```bash
python off_the_shelf_tokenizer.py \
  --input path/to/corpus.txt \
  --out-npy tiny_valid.npy \
  --special "<|endoftext|>" "<|assistant|>" \
  --print-preview
```

Pass multiple special tokens as separate values after a single `--special` flag.

Key behavior:

- Reads the input file in binary chunks rather than loading the full file into memory at once.
- Preserves UTF-8 codepoint boundaries across chunk edges.
- Protects custom special tokens from being split across chunks.
- Saves token IDs as `int32` when `--out-npy` is provided.

Useful arguments:

- `--input`: input UTF-8 text file
- `--out-npy`: destination `.npy` file for token IDs
- `--special`: one or more allowed special tokens
- `--chunk-bytes`: chunk size in bytes, default `2000000`
- `--print-preview`: prints the first token IDs and a short decoded preview

## Workflow 2: Train A Custom BPE Tokenizer

[`tokenizer.py`](tokenizer.py) trains a byte-level BPE tokenizer from raw text using GPT-2 style pretokenization.

Example:

```bash
python tokenizer.py \
  --tfile path/to/corpus.txt \
  --vocabsize 8000 \
  --inplace \
  --parallel \
  --store
```

This script:

- initializes the vocabulary with the 256 byte values
- pretokenizes text with the GPT-2 regex defined in [`settings.py`](settings.py)
- learns merges until the target vocabulary size is reached
- appends configured special tokens to the final vocabulary
- optionally writes `vocab.pkl` and `merges.pkl`

Useful arguments:

- `--tfile`: input training text file
- `--vocabsize`: target vocabulary size
- `--inplace`: enables in-place pair-count updates during merge iterations
- `--parallel`: enables multiprocessing during pretokenization
- `--compare`: runs all four combinations of `inplace` and `parallel`, prints timings, and asserts identical outputs
- `--store`: writes `vocab.pkl` and `merges.pkl`

Implementation notes:

- The parallel pretokenizer splits work using chunk boundaries aligned to `<|endoftext|>`.
- The comparison mode is useful for validating correctness after optimization changes.
- Sample benchmark numbers for a vocabulary size of `2048` are recorded in [`tokenizer_timing_numbers.txt`](tokenizer_timing_numbers.txt).

## Workflow 3: Encode And Decode With Learned Artifacts

[`encoder_decoder.py`](encoder_decoder.py) applies saved merges to new text and verifies round-trip decoding.

```bash
python encoder_decoder.py "hello world"
```

Current expectations:

- the script loads `vocab_10000.pkl`
- the script loads `merges_10000.pkl`

If you generated artifacts with `tokenizer.py --store`, the files written are `vocab.pkl` and `merges.pkl`. Rename them to the expected filenames or update the script before running the encoder/decoder utility.

## Local Inputs And Outputs

Typical local inputs:

- one or more UTF-8 text corpora, often placed under `tokenizer/data/`
- optional special tokens such as `<|endoftext|>` and task-specific markers

Typical generated outputs:

- `.npy` token ID arrays from [`off_the_shelf_tokenizer.py`](off_the_shelf_tokenizer.py)
- `vocab.pkl` and `merges.pkl` from [`tokenizer.py`](tokenizer.py) when `--store` is enabled

These large datasets and generated artifacts are usually gitignored to avoid inflating repository size.
