#!/usr/bin/env python3
"""
Test HF to tiktoken conversion on the fly against HuggingFace tokenizers.

This test verifies that the conversion from HF format to tiktoken format
works correctly by comparing against the HuggingFace tokenizers library.

Usage:
    python test_vs_hf_tokenizers.py --tokenizer gpt2
    python test_vs_hf_tokenizers.py --tokenizer llama
"""

from pathlib import Path
from typing import List, Tuple
import json
import time
import argparse

from tokenizers import Tokenizer
from hf_to_tiktoken import convert_hf_to_tiktoken

TEST_DIR = Path(__file__).parent.resolve()


def load_hf_tokenizer(tokenizer_path: str):
    """Load HuggingFace tokenizer from JSON file."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(path))
    print(f"[OK] HuggingFace tokenizer loaded ({path.name})")
    return tokenizer


def get_tiktoken_format(hf_tokenizer_path: str):
    """Get tiktoken format by converting HF to tiktoken format in memory."""
    hf_path = Path(hf_tokenizer_path)
    if not hf_path.exists():
        raise FileNotFoundError(f"HF tokenizer file not found: {hf_tokenizer_path}")
    tiktoken_json = convert_hf_to_tiktoken(hf_tokenizer_path, None)
    print(f"[OK] HF converted to tiktoken format (in memory)")
    return json.loads(tiktoken_json)


def get_test_samples() -> List[Tuple[str, str]]:
    """Get test samples from fixture files."""
    samples = []
    fixtures_dir = TEST_DIR / "fixtures"

    fixture_files = [
        ("shakespeare", fixtures_dir / "pg100-shakespeare.txt", 10000),
        (
            "sanguozhi",
            fixtures_dir / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt",
            5000,
        ),
        ("verne", fixtures_dir / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt", 5000),
    ]

    for name, path, max_chars in fixture_files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(max_chars)
            samples.append((name, sample))

    synthetic_samples = [
        ("hello_world", "Hello, world! This is a test."),
        ("unicode", "你好世界 🌍 Unicode test"),
        ("numbers", "123 456.789 @#$%"),
        ("code", "def foo():\n    return 42"),
        ("repetitive", "the " * 50),
    ]

    samples.extend(synthetic_samples)
    return samples


def compare_encoding(
    text: str, hf_tokenizer, tt_format
) -> Tuple[List[int], float, bool]:
    """Compare encoding between HF tokenizer and our tiktoken format."""
    start = time.perf_counter()
    output = hf_tokenizer.encode(text)
    hf_tokens = output.ids
    hf_time = time.perf_counter() - start

    start = time.perf_counter()
    tt_tokens = encode_with_format(text, tt_format)
    tt_time = time.perf_counter() - start

    match = hf_tokens == tt_tokens
    return hf_tokens, hf_time, match


def encode_with_format(text: str, format_dict: dict) -> List[int]:
    """Encode text using a tiktoken format dict (simplified implementation)."""
    from hf_to_tiktoken import _bytes_to_unicode, _unicode_to_bytes

    byte_encoder = _unicode_to_bytes()
    mergeable_ranks = format_dict["mergeable_ranks"]

    def get_pairs(word):
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def byte_pair_encode(word, ranks):
        if len(word) == 1:
            return []

        pairs = get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda p: ranks.get(str(list(p)), float("inf")))
            if bigram not in ranks:
                break

            i = word.index(bigram[0])
            if bigram[1] == word[i + 1]:
                word = word[:i] + [bigram[1]] + word[i + 2 :]
            else:
                word = word[: i + 1] + word[i + 2 :]

            if len(word) > 1:
                pairs = get_pairs(word)

        result = []
        for token in word:
            if str(token) in mergeable_ranks:
                result.append(mergeable_ranks[str(token)])
        return result

    text_bytes = []
    for char in text:
        if char in byte_encoder:
            text_bytes.append(byte_encoder[char])
        else:
            text_bytes.append(ord(char))

    tokens = []
    i = 0
    while i < len(text_bytes):
        if i + 1 < len(text_bytes):
            pair = (text_bytes[i], text_bytes[i + 1])
            pair_str = str(list(pair))
            if pair_str in mergeable_ranks:
                tokens.append(mergeable_ranks[pair_str])
                i += 2
                continue
        tokens.append(mergeable_ranks.get(str([text_bytes[i]]), text_bytes[i]))
        i += 1

    return tokens


def run_encoding_tests(hf_tokenizer, tt_format) -> int:
    """Run encoding comparison tests."""
    print("\n" + "=" * 70)
    print("ENCODING TESTS")
    print("=" * 70)

    samples = get_test_samples()
    errors = 0

    for i, (name, text) in enumerate(samples):
        display_text = text[:40].replace("\n", " ")
        print(
            f"\n[{i + 1}/{len(samples)}] {name}: {display_text}...", end=" ", flush=True
        )

        hf_tokens, _, match = compare_encoding(text, hf_tokenizer, tt_format)

        if match:
            print("[OK] MATCH")
        else:
            hf_len = len(hf_tokens)
            print(f"[FAIL] MISMATCH (HF:{hf_len})")
            errors += 1

    return errors


def run_length_tests(hf_tokenizer, tt_format) -> int:
    """Test encoding at various text lengths."""
    print("\n" + "=" * 70)
    print("LENGTH-BASED TESTS")
    print("=" * 70)

    base_text = "The quick brown fox jumps over the lazy dog. "
    lengths = [10, 50, 100, 500, 1000]

    errors = 0

    for length in lengths:
        text = base_text * (length // len(base_text) + 1)
        text = text[:length]

        print(f"\nLength {length:4d}: ", end="", flush=True)

        hf_tokens, _, match = compare_encoding(text, hf_tokenizer, tt_format)

        if match:
            print("[OK] MATCH")
        else:
            hf_len = len(hf_tokens)
            print(f"[FAIL] MISMATCH (HF:{hf_len})")
            errors += 1

    return errors


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="HF to tiktoken Conversion Test vs HuggingFace"
    )
    parser.add_argument(
        "--tokenizer",
        choices=["gpt2", "llama"],
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )

    args = parser.parse_args()

    tokenizer_type = args.tokenizer.lower()
    tokenizer_path = TEST_DIR / "tokenizers" / f"{tokenizer_type}-tokenizer.json"

    print("HF to tiktoken Conversion Test vs HuggingFace")
    print(f"Tokenizer: {tokenizer_type.upper()}")
    print("=" * 70)

    hf_tokenizer = load_hf_tokenizer(str(tokenizer_path))
    tt_format = get_tiktoken_format(str(tokenizer_path))

    errors = 0
    errors += run_encoding_tests(hf_tokenizer, tt_format)
    errors += run_length_tests(hf_tokenizer, tt_format)

    if errors == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[FAIL] {errors} test(s) failed")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    exit(main())
