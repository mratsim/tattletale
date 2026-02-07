#!/usr/bin/env python3
"""
Test Toktoktok against HuggingFace tokenizers.

Compares tokenization output using the `tokenizers` library
with local tokenizer JSON files.

Usage:
    python test_vs_hf_tokenizers.py --tokenizer gpt2 --quick
    python test_vs_hf_tokenizers.py --tokenizer llama --quick
"""

from pathlib import Path
from typing import List, Tuple, Optional
import time
import argparse

from tokenizers import Tokenizer
import pytoktoktok

TEST_DIR = Path(__file__).parent.resolve()


def load_hf_tokenizer(tokenizer_path: str):
    """Load HuggingFace tokenizer from JSON file."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(path))
    print(f"✓ HuggingFace tokenizer loaded ({path.name})")
    return tokenizer


def load_toktoktok(tokenizer_path: str):
    """Load toktoktok tokenizer."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = pytoktoktok.load_tokenizer(str(path))
    print(f"✓ Toktoktok loaded ({path.name})")
    return tokenizer


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
    text: str, hf_tokenizer, tt_tokenizer
) -> Tuple[Optional[List[int]], Optional[List[int]], float, float, bool]:
    """Compare encoding between HF tokenizer and toktoktok."""
    hf_tokens = None
    hf_time = 0.0
    tt_tokens = None
    tt_time = 0.0
    match = False

    try:
        start = time.perf_counter()
        output = hf_tokenizer.encode(text)
        hf_tokens = output.ids
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
    tokenizer_type: str, quick: bool, hf_tokenizer, tt_tokenizer
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
            text, hf_tokenizer, tt_tokenizer
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


def run_length_tests(quick: bool, hf_tokenizer, tt_tokenizer) -> int:
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
            text, hf_tokenizer, tt_tokenizer
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
        description="Toktoktok vs HuggingFace Correctness Test"
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

    args = parser.parse_args()

    tokenizer_type = args.tokenizer.lower()
    tokenizer_path = TEST_DIR / "tokenizers" / f"{tokenizer_type}-tokenizer.json"

    print("Toktoktok vs HuggingFace Correctness Test")
    print(f"Tokenizer: {tokenizer_type.upper()}")
    print(f"Quick mode: {args.quick}")
    print("=" * 70)

    errors = 0

    try:
        hf_tokenizer = load_hf_tokenizer(str(tokenizer_path))

        tt_tokenizer = None
        if tokenizer_path.exists():
            tt_tokenizer = load_toktoktok(str(tokenizer_path))
        else:
            print(f"⚠ Tokenizer file not found: {tokenizer_path}")

        errors += run_encoding_tests(
            tokenizer_type, args.quick, hf_tokenizer, tt_tokenizer
        )
        errors += run_length_tests(args.quick, hf_tokenizer, tt_tokenizer)

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
