#!/usr/bin/env python3
"""
Test HF to tiktoken conversion against HuggingFace tokenizers library.

This test verifies that our HF tokenizer parsing and conversion produces
identical token IDs to the HuggingFace tokenizers library.

Usage:
    python test_vs_hf_tokenizers.py --tokenizer gpt2
    python test_vs_hf_tokenizers.py --tokenizer exaone
"""

from pathlib import Path
from typing import List, Tuple
import argparse

from tokenizers import Tokenizer

TEST_DIR = Path(__file__).parent.resolve()


def load_hf_tokenizer(tokenizer_path: str):
    """Load HuggingFace tokenizer from JSON file."""
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(path))
    print(f"[OK] HuggingFace tokenizer loaded ({path.name})")
    return tokenizer


def get_test_samples() -> List[Tuple[str, str]]:
    """Get test samples from fixture files."""
    samples = []
    fixtures_dir = TEST_DIR / "fixtures"

    synthetic_samples = [
        ("hello_world", "Hello, world! This is a test."),
        ("unicode", "你好世界 🌍 Unicode test"),
        ("numbers", "123 456.789 @#$%"),
        ("code", "def foo():\n    return 42"),
        ("repetitive", "the " * 50),
    ]

    samples.extend(synthetic_samples)

    fixture_files = [
        ("verne", fixtures_dir / "large" / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt", 5000),
        ("shakespeare", fixtures_dir / "large" / "pg100-shakespeare.txt", 10000),
        (
            "sanguozhi",
            fixtures_dir / "large" / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt",
            5000,
        ),
    ]

    for name, path, max_chars in fixture_files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(max_chars)
            samples.append((name, sample))

    return samples


def run_nim_encoding_tests(hf_tokenizer_path: str) -> int:
    """Run encoding tests comparing Nim BPETokenizer against HuggingFace tokenizer."""
    print("\n" + "=" * 70)
    print("ENCODING TESTS (HuggingFace vs Nim BPETokenizer)")
    print("=" * 70)

    import pytoktoktok

    hf_tokenizer = load_hf_tokenizer(hf_tokenizer_path)
    nim_tokenizer = pytoktoktok.load_tokenizer_hf(hf_tokenizer_path)
    print(f"[OK] Nim BPETokenizer loaded ({nim_tokenizer})")

    samples = get_test_samples()
    errors = 0

    for i, (name, text) in enumerate(samples):
        display_text = text[:40].replace("\n", " ")
        print(
            f"\n[{i + 1}/{len(samples)}] {name}: {display_text}...", end=" ", flush=True
        )

        hf_output = hf_tokenizer.encode(text, add_special_tokens=False)
        hf_tokens = hf_output.ids
        nim_tokens = nim_tokenizer.encode_ordinary(text)

        hf_len = len(hf_tokens)
        nim_len = len(nim_tokens)
        match = hf_tokens == nim_tokens

        if match:
            print(f"[OK] MATCH (HF:{hf_len}, Nim:{nim_len})")
        else:
            print(f"[FAIL] MISMATCH (HF:{hf_len}, Nim:{nim_len})")
            errors += 1

    return errors


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="HF Tokenizer Test vs Nim BPETokenizer"
    )
    parser.add_argument(
        "--tokenizer",
        choices=[
            "gpt2",
            "llama3",
            "minimax-m2.1",
            "glm-4.7",
            "exaone",
            "step-3.5-flash",
        ],
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )

    args = parser.parse_args()

    tokenizer_type = args.tokenizer.lower()
    tokenizer_filename = f"{tokenizer_type}-tokenizer.json"
    tokenizer_path = TEST_DIR / "tokenizers" / tokenizer_filename

    print("HF Tokenizer Test vs Nim BPETokenizer")
    print(f"Tokenizer: {tokenizer_type.upper()}")
    print("=" * 70)

    errors = 0
    errors += run_nim_encoding_tests(str(tokenizer_path))

    if errors == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[FAIL] {errors} test(s) failed")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    exit(main())
