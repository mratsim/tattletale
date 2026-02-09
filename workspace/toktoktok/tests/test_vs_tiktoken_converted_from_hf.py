#!/usr/bin/env python3
"""
Test Nim BPETokenizer against tiktoken for HF tokenizer JSON format (after conversion).
"""

import pytoktoktok
from pathlib import Path
from typing import List, Tuple
import json
import tiktoken

TEST_DIR = Path(__file__).parent.resolve()

HF_FILES = {
    "gpt2": "gpt2-tokenizer.json",
    "llama3": "llama3-tokenizer.json",
    "minimax-m2.1": "minimax-m2.1-tokenizer.json",
    "glm-4.7": "glm-4.7-tokenizer.json",
    "exaone": "exaone-tokenizer.json",
    "step-3.5-flash": "step-3.5-flash-tokenizer.json",
}


def load_tiktoken_encoding_from_json(json_str: str) -> tiktoken.Encoding:
    """Load tiktoken JSON format from string and create tiktoken.Encoding object."""
    import base64

    data = json.loads(json_str)

    mergeable_ranks: dict[bytes, int] = {}

    for key, rank in data["mergeable_ranks"].items():
        # Keys are base64-encoded bytes
        token_bytes = base64.b64decode(key)
        mergeable_ranks[token_bytes] = rank

    special_tokens = data.get("special_tokens", {})

    return tiktoken.Encoding(
        name="test",
        pat_str=data["pat_str"],
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def load_tiktoken_encoding(converted_path: str) -> tiktoken.Encoding:
    """Load tiktoken JSON format from file and create tiktoken.Encoding object."""
    with open(converted_path, "r", encoding="utf-8") as f:
        return load_tiktoken_encoding_from_json(f.read())


def get_test_samples() -> List[Tuple[str, str]]:
    """Get test samples from fixture files."""
    samples = []
    fixtures_dir = TEST_DIR / "fixtures"

    synthetic_samples = [
        ("hello_world", "Hello, world! This is a test."),
        ("numbers", "123 456.789 @#$%"),
        ("code", "def foo():\n    return 42"),
        ("unicode", "你好世界 🌍 Unicode test"),
        ("repetitive", "the " * 50),
    ]

    samples.extend(synthetic_samples)

    fixture_files = [
        ("verne", fixtures_dir / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt", 5000),
        ("shakespeare", fixtures_dir / "pg100-shakespeare.txt", 10000),
        (
            "sanguozhi",
            fixtures_dir / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt",
            5000,
        ),
    ]

    for name, path, max_chars in fixture_files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(max_chars)
            samples.append((name, sample))

    return samples


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

        tthf_tokens = tik_encoding.encode(text)
        nim_tokens = nim_tokenizer.encode(text)

        tthf_len = len(tthf_tokens)
        nim_len = len(nim_tokens)
        match = tthf_tokens == nim_tokens

        if match:
            print(f"[OK] MATCH (TTHF:{tthf_len}, Nim:{nim_len})")
        else:
            print(f"[FAIL] MISMATCH (TTHF:{tthf_len}, Nim:{nim_len})")
            errors += 1
            return errors  # Exit early on first failure

    return errors


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Nim BPETokenizer vs tiktoken (HF conversion) Test"
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
    errors = 0

    print("Nim BPETokenizer vs tiktoken (HF conversion) Test")
    print("=" * 70)

    hf_file = HF_FILES[tokenizer_type]
    hf_path = TEST_DIR / "tokenizers" / hf_file

    import hf_to_tiktoken

    tiktoken_json = hf_to_tiktoken.convert_hf_to_tiktoken(str(hf_path))
    tik_encoding = load_tiktoken_encoding_from_json(tiktoken_json)

    print(f"[OK] tiktoken.Encoding created from HF JSON")
    errors += run_hf_conversion_tests(tokenizer_type, hf_path, tik_encoding)

    if errors == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[FAIL] {errors} test(s) failed")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    exit(main())
