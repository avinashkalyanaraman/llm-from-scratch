# Tokenizer

Tokenizer experiments and utilities used by this repo’s language-model pipeline.

This directory has two main tracks:
- custom BPE training/tokenization from raw text (`tokenizer.py`, `barebones_tokenizer.py`)
- off-the-shelf GPT-2 tokenization with robust chunked file encoding (`off_the_shelf_tokenizer.py`)

## What’s Here

- `tokenizer.py`: main custom BPE trainer with:
  - GPT-2 regex pretokenization pattern (from `settings.py`)
  - optional multiprocessing pretokenization split by special-token boundaries
  - optional in-place pair-count updates for faster merge iterations
- `barebones_tokenizer.py`: simpler reference BPE trainer (serial, no in-place pair updates).
- `encoder_decoder.py`: encode/decode helper using saved `vocab_*.pkl` and `merges_*.pkl`.
- `off_the_shelf_tokenizer.py`: tiktoken-based tokenizer with chunked byte-safe streaming and dynamic special-token handling.
- `chunker.py`: finds safe file chunk boundaries at a special token (default usage uses `<|endoftext|>`).
- `settings.py`: GPT-2 pretokenization regex and special token list.
- `filehandler.py`: helpers to serialize vocab/merges to pickle or text.

Data/artifacts currently present:
- `data/TinyStoriesV2-GPT4-train.txt`
- `data/TinyStoriesV2-GPT4-valid.txt`
- `tinystoriesV2_train_tokenized.npy`
- `tinystoriesV2_valid_tokenized.npy`

## Dependencies

- Python 3.10+
- `regex`
- `numpy`
- `tiktoken` (for `off_the_shelf_tokenizer.py`)

Install:

```bash
pip install regex numpy tiktoken
```

## Run From This Directory

These scripts rely on local-module imports, so run from `tokenizer/`:

```bash
cd tokenizer
```

## Typical Workflows

### 1) Tokenize with GPT-2 (off-the-shelf, robust for large files)

```bash
python off_the_shelf_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --out-npy tiny_valid.npy \
  --special "<|endoftext|>" \
  --print-preview
```

Notes:
- Reads file in binary chunks (`--chunk-bytes`, default `2_000_000`).
- Preserves UTF-8 boundaries across chunks.
- Avoids splitting special tokens across chunk boundaries.

### 2) Train custom BPE merges/vocab (full script)

```bash
python tokenizer.py
```

What `tokenizer.py` currently does in `__main__`:
- uses `data/TinyStoriesV2-GPT4-valid.txt`
- trains to vocab size `8000`
- runs four variants (in-place update on/off, parallel pretokenization on/off)
- asserts all variants produce matching vocab/merges

Important:
- Saving vocab/merges is currently commented out at the bottom of `tokenizer.py`.
- Uncomment `filehandler.serializedWrite(...)` lines there if you want persistent `.pkl` outputs.

### 3) Train custom BPE (minimal baseline)

```bash
python barebones_tokenizer.py
```

This writes a text vocab dump (`vocab_tiny_5000.txt`) via `filehandler.textWrite(...)`.

### 4) Encode/decode with saved merges

```bash
python encoder_decoder.py "hello world"
```

`encoder_decoder.py` expects:
- `vocab_10000.pkl`
- `merges_10000.pkl`

in the current directory. Generate or rename your saved artifacts accordingly.

## Script-Level Notes

- `settings.py` defaults:
  - `gpt2_pat`: GPT-2 regex pretokenization pattern
  - `special_tokens`: `["<|endoftext|>"]`
- `chunker.py` can be used directly to inspect chunk boundaries for parallel pretokenization.
- `tokenizer.py` and `encoder_decoder.py` share merge logic, so order of `merges` insertion is significant (Python 3.7+ ordered dict behavior).

## Relation to Other Parts of Repo

- Token IDs generated here are consumed by training scripts under `transformer/`.
- `.npy` token arrays are the expected format for `transformer/filehandler.py` datasets.
