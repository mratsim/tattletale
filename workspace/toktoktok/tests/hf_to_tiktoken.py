"""HF to Tiktoken Converter

Converts HuggingFace tokenizer JSON files to tiktoken-compatible format.
Handles byte-level BPE tokenizers (GPT-2 style).
"""

import base64
import json
from pathlib import Path


# Kimi-K2.5 pattern from https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenization_kimi.py
KIMI_K25_PATTERN = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


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
    """Convert HF vocab to tiktoken mergeable_ranks format with bytes keys."""
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


def parse_tiktoken_file(tiktoken_path: str) -> dict[bytes, int]:
    """Parse a tiktoken file and return mergeable_ranks dict."""
    mergeable_ranks = {}
    with open(tiktoken_path, "r") as f:
        content = f.read()
    for line in content.strip().split("\n"):
        parts = line.split()
        if len(parts) == 2:
            try:
                token_bytes = base64.b64decode(parts[0])
                rank = int(parts[1])
                mergeable_ranks[token_bytes] = rank
            except:
                pass
    return mergeable_ranks


def extract_pattern(hf_tokenizer_path: str) -> str:
    """Extract the pre-tokenization regex pattern from the tokenizer."""
    with open(hf_tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "pre_tokenizer" in data:
        pre_tok = data["pre_tokenizer"]
        if pre_tok.get("type") == "ByteLevel":
            return r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\r?\n|\s+(?!\S)|\s+"
        elif pre_tok.get("type") == "Sequence":
            # Check for Split pretokenizer with pattern
            if "pretokenizers" in pre_tok:
                for step in pre_tok["pretokenizers"]:
                    if step.get("type") == "Split" and "pattern" in step:
                        return step["pattern"].get("Regex", "")

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


def convert_hf_to_tiktoken(
    hf_tokenizer_path: str, output_path: str | None = None
) -> str:
    """Convert HF tokenizer to tiktoken format.

    If output_path is provided, saves to file and returns the path.
    If output_path is None, returns the JSON string directly.
    """
    mergeable_ranks = convert_vocab_to_mergeable_ranks(hf_tokenizer_path)
    pat_str = extract_pattern(hf_tokenizer_path)
    special_tokens = extract_special_tokens(hf_tokenizer_path)

    data = {
        "mergeable_ranks": mergeable_ranks,
        "pat_str": pat_str,
        "special_tokens": special_tokens,
    }

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, separators=(",", ": "))
        print(f"[OK] Converted tokenizer saved to: {output_path}")
        return output_path
    else:
        return json.dumps(data, ensure_ascii=False, indent=2, separators=(",", ": "))
