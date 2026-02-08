#!/usr/bin/env python3
"""
Test Nim BPETokenizer against tiktoken for HF tokenizers.
"""

import pytoktoktok
from pathlib import Path
from typing import List, Tuple
import tiktoken

TEST_DIR = Path(__file__).parent.resolve()


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


def run_nim_tests(tik_encoding, nim_tokenizer) -> int:
    """Run encoding comparison tests against the Nim BPETokenizer."""
    print("\n" + "=" * 70)
    print("NIM BPETOKENIZER ENCODING TESTS")
    print("=" * 70)

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
            print(f"[FAIL] MISMATCH (HF:{hf_len})")
            errors += 1

    return errors


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Nim BPETokenizer vs tiktoken Test")
    parser.add_argument(
        "--tokenizer",
        choices=["gpt2", "llama"],
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )

    args = parser.parse_args()

    tokenizer_type = args.tokenizer.lower()
    model = "gpt2" if tokenizer_type == "gpt2" else "llama3"

    print("Nim BPETokenizer vs tiktoken Test")
    print(f"Tokenizer: {model.upper()}")
    print("=" * 70)

    tik_encoding = tiktoken.encoding_for_model(model)
    print(f"[OK] tiktoken loaded ({model})")

    hf_tokenizer_path = TEST_DIR / "tokenizers" / f"{tokenizer_type}-tokenizer.json"
    nim_tokenizer = pytoktoktok.load_tokenizer(str(hf_tokenizer_path))
    print(f"[OK] Nim BPETokenizer loaded")

    errors = run_nim_tests(tik_encoding, nim_tokenizer)

    if errors == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[FAIL] {errors} test(s) failed")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    exit(main())
