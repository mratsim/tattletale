#!/usr/bin/env python3
"""
Test Toktoktok against OpenAI's tiktoken tokenizer.

Compares tokenization output for GPT-2 and LLaMA tokenizers.
Uses HF to tiktoken conversion to ensure proper byte-level BPE format.

Usage:
    python test_vs_tiktoken.py --tokenizer gpt2 --quick
    python test_vs_tiktoken.py --tokenizer llama --quick
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import time
import argparse

import tiktoken
import pytoktoktok
from hf_to_tiktoken import (
    convert_hf_to_tiktoken,
    convert_vocab_to_mergeable_ranks,
    extract_pattern,
    extract_special_tokens,
)

TEST_DIR = Path(__file__).parent.resolve()
CONVERTED_DIR = TEST_DIR / "tokenizers" / "converted"


def load_tiktoken(model: str):
    """Load tiktoken tokenizer."""
    encoding = tiktoken.encoding_for_model(model)
    print(f"✓ tiktoken loaded ({model})")
    return encoding


def load_toktoktok_from_converted(converted_path: str):
    """Load toktoktok tokenizer from converted tiktoken format."""
    path = Path(converted_path)
    if not path.exists():
        raise FileNotFoundError(f"Converted tokenizer not found: {converted_path}")
    tokenizer = pytoktoktok.load_tokenizer(str(path))
    print(f"✓ Toktoktok loaded (converted format)")
    return tokenizer


def load_toktoktok(hf_tokenizer_path: str) -> str:
    """Convert HF tokenizer to tiktoken format and return path."""
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)

    hf_path = Path(hf_tokenizer_path)
    converted_path = CONVERTED_DIR / f"{hf_path.stem}_tiktoken.json"

    if not converted_path.exists():
        convert_hf_to_tiktoken(hf_tokenizer_path, str(converted_path))
    else:
        print(f"✓ Using cached converted tokenizer: {converted_path}")

    return str(converted_path)


def load_fixture_sample(fixture_path: Path, max_chars: int = 10000) -> str:
    """Load a sample from fixture file."""
    if not fixture_path.exists():
        return ""
    with open(fixture_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def get_test_samples(tokenizer_type: str, quick: bool) -> List[Tuple[str, str]]:
    """Get test samples from fixture files."""
    samples = []
    fixtures_dir = TEST_DIR / "fixtures"

    sample_sizes = {"shakespeare": 10000, "sanguozhi": 5000, "verne": 5000}
    if quick:
        sample_sizes = {"shakespeare": 1000, "sanguozhi": 500, "verne": 1000}

    fixture_files = [
        ("shakespeare", fixtures_dir / "pg100-shakespeare.txt"),
        (
            "sanguozhi",
            fixtures_dir / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt",
        ),
        ("verne", fixtures_dir / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt"),
    ]

    for name, path in fixture_files:
        if path.exists():
            sample = load_fixture_sample(path, sample_sizes.get(name, 5000))
            if sample:
                samples.append((name, sample))

    synthetic_samples = [
        ("hello_world", "Hello, world! This is a test."),
        ("unicode", "你好世界 🌍 Unicode test"),
        ("numbers", "123 456.789 @#$%"),
        ("code", "def foo():\n    return 42"),
        ("repetitive", "the " * 50),
    ]

    if quick:
        synthetic_samples = synthetic_samples[:2]

    samples.extend(synthetic_samples)
    return samples


def compare_encoding(
    text: str, tik_encoding, tt_tokenizer
) -> Tuple[Optional[List[int]], Optional[List[int]], float, float, bool]:
    """Compare encoding between tiktoken and toktoktok."""
    hf_tokens = None
    hf_time = 0.0
    tt_tokens = None
    tt_time = 0.0
    match = False

    try:
        start = time.perf_counter()
        hf_tokens = tik_encoding.encode_ordinary(text)
        hf_time = time.perf_counter() - start
    except Exception:
        pass

    if tt_tokenizer is not None:
        try:
            start = time.perf_counter()
            tt_tokens = tt_tokenizer.encode(text)
            tt_time = time.perf_counter() - start
        except Exception:
            pass

    if hf_tokens is not None and tt_tokens is not None:
        match = hf_tokens == tt_tokens

    return hf_tokens, tt_tokens, hf_time, tt_time, match


def run_encoding_tests(
    tokenizer_type: str, quick: bool, tik_encoding, tt_tokenizer
) -> int:
    """Run encoding comparison tests."""
    print("\n" + "=" * 70)
    print("ENCODING TESTS")
    print("=" * 70)

    samples = get_test_samples(tokenizer_type, quick)
    errors = 0

    for i, (name, text) in enumerate(samples):
        display_text = text[:40].replace("\n", " ")
        print(
            f"\n[{i + 1}/{len(samples)}] {name}: {display_text}...", end=" ", flush=True
        )

        hf_tokens, tt_tokens, _, _, match = compare_encoding(
            text, tik_encoding, tt_tokenizer
        )

        if match:
            print("✓ MATCH")
        elif hf_tokens and tt_tokens and len(hf_tokens) == len(tt_tokens):
            print("✗ TOKENS MISMATCH (same length)")
        else:
            hf_len = len(hf_tokens) if hf_tokens else "err"
            tt_len = len(tt_tokens) if tt_tokens else "err"
            print(f"✗ LENGTH MISMATCH (HF:{hf_len} vs TT:{tt_len})")
            errors += 1

    return errors


def run_length_tests(quick: bool, tik_encoding, tt_tokenizer) -> int:
    """Test encoding at various text lengths."""
    print("\n" + "=" * 70)
    print("LENGTH-BASED TESTS")
    print("=" * 70)

    base_text = "The quick brown fox jumps over the lazy dog. "
    lengths = [10, 50, 100, 500, 1000]

    if quick:
        lengths = [10, 50, 100]

    errors = 0

    for length in lengths:
        text = base_text * (length // len(base_text) + 1)
        text = text[:length]

        print(f"\nLength {length:4d}: ", end="", flush=True)

        hf_tokens, tt_tokens, _, _, match = compare_encoding(
            text, tik_encoding, tt_tokenizer
        )

        if match:
            print("✓ MATCH")
        elif hf_tokens and tt_tokens and len(hf_tokens) == len(tt_tokens):
            print(f"✗ tokens differ (HF:{len(hf_tokens)} vs TT:{len(tt_tokens)})")
            errors += 1
        else:
            hf_len = len(hf_tokens) if hf_tokens else "err"
            tt_len = len(tt_tokens) if tt_tokens else "err"
            print(f"✗ length diff (HF:{hf_len} vs TT:{tt_len})")
            errors += 1

    return errors


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Toktoktok vs tiktoken Correctness Test"
    )
    parser.add_argument(
        "--tokenizer",
        choices=["gpt2", "llama"],
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with fewer samples"
    )
    parser.add_argument(
        "--convert", action="store_true", help="Force reconversion of tokenizer"
    )

    args = parser.parse_args()

    tokenizer_type = args.tokenizer.lower()
    model = "gpt2" if tokenizer_type == "gpt2" else "llama3"

    print("Toktoktok vs tiktoken Correctness Test")
    print(f"Tokenizer: {model.upper()}")
    print(f"Quick mode: {args.quick}")
    print("=" * 70)

    errors = 0

    try:
        tik_encoding = load_tiktoken(model)

        hf_tokenizer_path = TEST_DIR / "tokenizers" / f"{tokenizer_type}-tokenizer.json"
        if not hf_tokenizer_path.exists():
            print(f"⚠ HF tokenizer file not found: {hf_tokenizer_path}")
            return 1

        tt_tokenizer = None
        converted_path = load_toktoktok(str(hf_tokenizer_path))
        tt_tokenizer = load_toktoktok_from_converted(converted_path)

        errors += run_encoding_tests(
            tokenizer_type, args.quick, tik_encoding, tt_tokenizer
        )
        errors += run_length_tests(args.quick, tik_encoding, tt_tokenizer)

        if errors == 0:
            print("\n✓ All tests passed!")
        else:
            print(f"\n✗ {errors} test(s) failed")

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    exit(main())
