"""HF to Tiktoken Converter

Converts HuggingFace tokenizer JSON files to tiktoken-compatible format.
Handles byte-level BPE tokenizers (GPT-2 style).
"""

import json
from pathlib import Path


def _bytes_to_unicode() -> dict[int, str]:
    """Returns the GPT-2 byte-to-unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _unicode_to_bytes() -> dict[str, int]:
    """Returns the reverse mapping: unicode characters back to bytes."""
    return {v: k for k, v in _bytes_to_unicode().items()}


def convert_vocab_to_mergeable_ranks(hf_tokenizer_path: str) -> dict[bytes, int]:
    """Convert HF vocab to tiktoken mergeable_ranks format."""
    with open(hf_tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = data["model"]["vocab"]
    byte_decoder = _unicode_to_bytes()

    mergeable_ranks = {}

    for token, rank in vocab.items():
        try:
            token_bytes = bytes([byte_decoder[c] for c in token])
            mergeable_ranks[token_bytes] = rank
        except KeyError:
            continue

    return mergeable_ranks


def extract_pattern(hf_tokenizer_path: str) -> str:
    """Extract the pre-tokenization regex pattern from the tokenizer."""
    with open(hf_tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "pre_tokenizer" in data:
        pre_tok = data["pre_tokenizer"]
        if pre_tok.get("type") == "ByteLevel":
            return r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\r?\n|\s+(?!\S)|\s+"

    return r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\r?\n|\s+(?!\S)|\s+"


def extract_special_tokens(hf_tokenizer_path: str) -> dict[str, int]:
    """Extract special tokens mapping from the tokenizer."""
    with open(hf_tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    special_tokens = {}

    if "added_tokens" in data:
        for token in data["added_tokens"]:
            if "id" in token and "content" in token:
                special_tokens[token["content"]] = token["id"]

    return special_tokens


def convert_hf_to_tiktoken(hf_tokenizer_path: str, output_path: str) -> None:
    """Convert HF tokenizer to tiktoken format and save."""
    mergeable_ranks = convert_vocab_to_mergeable_ranks(hf_tokenizer_path)
    pat_str = extract_pattern(hf_tokenizer_path)
    special_tokens = extract_special_tokens(hf_tokenizer_path)

    data = {
        "mergeable_ranks": {},
        "pat_str": pat_str,
        "special_tokens": special_tokens,
    }

    for token_bytes, rank in mergeable_ranks.items():
        key = list(token_bytes)
        data["mergeable_ranks"][str(key)] = rank

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Converted tokenizer saved to: {output_path}")
