#!/usr/bin/env python3
"""
Test Nim BPETokenizer against tiktoken for both:
1. Tiktoken file format (base64 line format)
2. HF tokenizer JSON format (after conversion)
"""

import pytoktoktok
from pathlib import Path
from typing import List, Tuple
import tiktoken
from enum import IntEnum

TEST_DIR = Path(__file__).parent.resolve()

class RegexPattern(IntEnum):
    r50k = 1
    p50k = 2
    cl100k = 3
    o200k = 4

TIKTOKEN_FILES = {
    "r50k_base": "r50k_base.tiktoken",
    "p50k_base": "p50k_base.tiktoken",
    "cl100k_base": "cl100k_base.tiktoken",
    "o200k_base": "o200k_base.tiktoken",
}
REGEX_MAP = {
    "r50k_base": RegexPattern.r50k,
    "p50k_base": RegexPattern.p50k,
    "cl100k_base": RegexPattern.cl100k,
    "o200k_base": RegexPattern.o200k,
}

HF_FILES = {
    "gpt2": "gpt2-tokenizer.json",
    "llama3": "llama3-tokenizer.json",
}


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


def run_tiktoken_format_tests(
    nim_tokenizer_name: str, tiktoken_path: Path, tik_encoding, tokenizer_type: str
) -> int:
    """Run encoding comparison tests using tiktoken file format."""
    print("\n" + "=" * 70)
    print(f"TIKTOKEN FILE FORMAT TESTS ({nim_tokenizer_name})")
    print("=" * 70)

    pattern = REGEX_MAP[tokenizer_type]
    nim_tokenizer = pytoktoktok.load_tokenizer_tiktoken(str(tiktoken_path), pattern)
    print(f"[OK] Nim BPETokenizer loaded ({nim_tokenizer})")

    samples = get_test_samples()
    errors = 0

    for i, (name, text) in enumerate(samples):
        display_text = text[:40].replace("\n", " ")
        print(
            f"\n[{i + 1}/{len(samples)}] {name}: {display_text}...", end=" ", flush=True
        )

        hf_tokens = tik_encoding.encode_ordinary(text)
        nim_tokens = nim_tokenizer.encode_ordinary(text)

        match = hf_tokens == nim_tokens

        if match:
            print("[OK] MATCH")
        else:
            hf_len = len(hf_tokens)
            nim_len = len(nim_tokens)
            print(f"[FAIL] MISMATCH (HF:{hf_len}, Nim:{nim_len})")
            errors += 1

    return errors


def run_hf_conversion_tests(model_name: str, hf_path: Path, tik_encoding) -> int:
    """Run encoding comparison tests using HF tokenizer JSON format (after conversion)."""
    print("\n" + "=" * 70)
    print(f"HF TOKENIZER CONVERSION TESTS ({model_name})")
    print("=" * 70)

    nim_tokenizer = pytoktoktok.load_tokenizer_hf(str(hf_path))
    print(f"[OK] Nim BPETokenizer loaded ({nim_tokenizer})")

    samples = get_test_samples()
    errors = 0

    for i, (name, text) in enumerate(samples):
        display_text = text[:40].replace("\n", " ")
        print(
            f"\n[{i + 1}/{len(samples)}] {name}: {display_text}...", end=" ", flush=True
        )

        hf_tokens = tik_encoding.encode_ordinary(text)
        nim_tokens = nim_tokenizer.encode_ordinary(text)

        match = hf_tokens == nim_tokens

        if match:
            print("[OK] MATCH")
        else:
            hf_len = len(hf_tokens)
            nim_len = len(nim_tokens)
            print(f"[FAIL] MISMATCH (HF:{hf_len}, Nim:{nim_len})")
            errors += 1

    return errors


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Nim BPETokenizer vs tiktoken Test")
    parser.add_argument(
        "--format",
        choices=["tiktoken", "hf", "all"],
        default="all",
        help="Tokenizer format to test (default: all)",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"],
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )

    args = parser.parse_args()

    tokenizer_type = args.tokenizer.lower()
    errors = 0

    print("Nim BPETokenizer vs tiktoken Test")
    print("=" * 70)

    # Tiktoken file format tests
    if args.format in ("tiktoken", "all"):
        tiktoken_file = TIKTOKEN_FILES[tokenizer_type]
        tiktoken_path = TEST_DIR / "tokenizers" / tiktoken_file
        tik_encoding = tiktoken.get_encoding(tokenizer_type)
        print(f"[OK] tiktoken loaded ({tokenizer_type})")
        errors += run_tiktoken_format_tests(
            tokenizer_type, tiktoken_path, tik_encoding, tokenizer_type
        )

    # HF tokenizer conversion tests
    if args.format in ("hf", "all"):
        hf_file = HF_FILES[tokenizer_type]
        hf_path = TEST_DIR / "tokenizers" / hf_file
        model = "gpt2" if tokenizer_type == "gpt2" else tokenizer_type
        tik_encoding = tiktoken.encoding_for_model(model)
        print(f"[OK] tiktoken loaded ({model})")
        errors += run_hf_conversion_tests(tokenizer_type, hf_path, tik_encoding)

    if errors == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[FAIL] {errors} test(s) failed")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    exit(main())
