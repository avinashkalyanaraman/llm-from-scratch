import argparse, os
import numpy as np
import tiktoken

# ---------- Build a new Encoding that includes your custom specials ----------
def build_gpt2_with_specials(extra_specials=None):
    """
    Returns (enc, specials_set, specials_bytes):
      - enc: a new tiktoken.Encoding named "gpt2+custom" that contains GPT-2 vocab + your specials
      - specials_set: set[str] of all special strings (for allowed_special)
      - specials_bytes: list[bytes] of specials (for overlap checks)
    """
    base = tiktoken.get_encoding("gpt2")  # frozen base
    extra_specials = extra_specials or []

    # Merge existing specials with new ones; assign new ids after base.n_vocab
    base_specials = dict(getattr(base, "_special_tokens", {}))  # {str: id}
    next_id = base.n_vocab
    add = {s: next_id + i for i, s in enumerate(extra_specials) if s not in base_specials}
    all_specials = {**base_specials, **add}

    # Construct a NEW encoding that reuses GPT-2's pattern & ranks but with extended specials
    enc = tiktoken.Encoding(
        name="gpt2+custom",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=all_specials,
    )

    specials_set = set(all_specials.keys())
    specials_bytes = [s.encode("utf-8") for s in specials_set]
    return enc, specials_set, specials_bytes

# ---------- Bytes helpers: dynamic special overlap + UTF-8 safe cutoff ----------
def suffix_special_overlap_len_bytes(buf: bytes, specials_b: list[bytes]) -> int:
    """
    Length of the longest suffix of `buf` that is also a prefix of ANY special (bytes).
    If it equals a special's full length, we still keep it for next round to avoid double-encode.
    """
    if not buf or not specials_b:
        return 0
    max_k = 0
    for s in specials_b:
        maxlen = min(len(s), len(buf))
        for k in range(maxlen, 0, -1):  # longest-first
            if s.startswith(buf[-k:]):
                if k > max_k:
                    max_k = k
                break
    return max_k

def utf8_safe_cutoff(b: bytes, upto: int) -> int:
    """
    Largest index <= upto that ends on a valid UTF-8 codepoint boundary.
    We back off up to 3 bytes (UTF-8 max 4-byte sequence).
    """
    if upto <= 0:
        return 0
    cut = upto
    try:
        b[:cut].decode("utf-8", errors="strict")
        return cut
    except UnicodeDecodeError:
        pass
    for back in range(1, min(4, cut) + 1):
        try:
            b[:cut - back].decode("utf-8", errors="strict")
            return cut - back
        except UnicodeDecodeError:
            continue
    return 0  # extremely unlikely on valid UTF-8

# ---------- Streaming encode (byte-accurate, chunked) ----------
def encode_file_bytes(path: str, enc, specials_set: set[str], specials_b: list[bytes],
                      chunk_bytes: int = 2_000_000) -> list[int]:
    """
    Read file in binary; avoid splitting specials and UTF-8 codepoints across boundaries.
    """
    ids_out: list[int] = []
    buf = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)

            if not chunk:
                if buf:
                    s = buf.decode("utf-8", errors="strict")
                    ids_out.extend(enc.encode(s, allowed_special=specials_set))
                break

            buf += chunk
            overlap = suffix_special_overlap_len_bytes(buf, specials_b)  # hold possible special prefix
            flush_upto = len(buf) - overlap

            if flush_upto > 0:
                cutoff = utf8_safe_cutoff(buf, flush_upto)
                if cutoff > 0:
                    s = buf[:cutoff].decode("utf-8", errors="strict")
                    ids_out.extend(enc.encode(s, allowed_special=specials_set))
                    buf = buf[cutoff:]  # keep tail (may include special prefix or partial UTF-8)
                # else: keep all bytes; wait for more to complete a codepoint
    return ids_out

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Byte-accurate GPT-2 tokenization with dynamic special-token protection."
    )
    parser.add_argument("--input", help="Input UTF-8 text file")
    parser.add_argument("--out-npy", help="Optional .npy path to save token ids")
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"],
                        help="Special tokens to include/allow (default: <|endoftext|>)")
    parser.add_argument("--print-preview", action="store_true",
                        help="Print a short preview of tokens")
    parser.add_argument("--chunk-bytes", type=int, default=2_000_000,
                        help="Read size per chunk in bytes")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"File not found: {args.input}")

    enc, specials_set, specials_b = build_gpt2_with_specials(args.special)
    ids = encode_file_bytes(args.input, enc, specials_set, specials_b, args.chunk_bytes)

    if args.out_npy:
        np.save(args.out_npy, np.array(ids, dtype=np.int32))
        print(f"Saved {len(ids)} token ids to {args.out_npy}")

    if args.print_preview:
        print("First 100 token ids:", ids[:100])
        print("Decoded preview:", enc.decode(ids[:100]))

if __name__ == "__main__":
    main()
