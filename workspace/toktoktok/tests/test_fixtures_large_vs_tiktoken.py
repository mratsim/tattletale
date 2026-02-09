#!/usr/bin/env python3
"""
Test Nim BPETokenizer against tiktoken for tiktoken file format (base64 line format).
"""

import pytoktoktok
from pathlib import Path
from typing import List, Tuple
import tiktoken

TEST_DIR = Path(__file__).parent.resolve()


TIKTOKEN_FILES = {
    "r50k_base": "r50k_base.tiktoken",
    "p50k_base": "p50k_base.tiktoken",
    "cl100k_base": "cl100k_base.tiktoken",
    "o200k_base": "o200k_base.tiktoken",
    "kimik2.5": "kimik2.5.tiktoken",
}
REGEX_MAP = {
    "r50k_base": "r50k",
    "p50k_base": "p50k",
    "cl100k_base": "cl100k",
    "o200k_base": "o200k",
    "kimik2.5": "kimik2.5",
}


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

        tt_tokens = tik_encoding.encode(text)
        nim_tokens = nim_tokenizer.encode(text)

        tt_len = len(tt_tokens)
        nim_len = len(nim_tokens)
        match = tt_tokens == nim_tokens

        if match:
            print(f"[OK] MATCH (TT:{tt_len}, Nim:{nim_len})")
        else:
            print(f"[FAIL] MISMATCH (TT:{tt_len}, Nim:{nim_len})")
            errors += 1
            return errors  # Exit early on first failure

    return errors


def main():
    """Main test runner."""
    import argparse
    import base64

    parser = argparse.ArgumentParser(
        description="Nim BPETokenizer vs tiktoken (file format) Test"
    )
    parser.add_argument(
        "--tokenizer",
        choices=["r50k_base", "p50k_base", "cl100k_base", "o200k_base", "kimik2.5"],
        default="r50k_base",
        help="Tokenizer to use (default: r50k_base)",
    )

    args = parser.parse_args()

    tokenizer_type = args.tokenizer.lower()
    errors = 0

    print("Nim BPETokenizer vs tiktoken (file format) Test")
    print("=" * 70)

    tiktoken_file = TIKTOKEN_FILES[tokenizer_type]
    tiktoken_path = TEST_DIR / "tokenizers" / tiktoken_file

    if tokenizer_type == "kimik2.5":
        # Kimi K2.5 uses custom tiktoken file with custom pattern
        from hf_to_tiktoken import KIMI_K25_PATTERN

        mergeable_ranks = {}
        with open(tiktoken_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    token_bytes = base64.b64decode(parts[0])
                    rank = int(parts[1])
                    mergeable_ranks[token_bytes] = rank
        tik_encoding = tiktoken.Encoding(
            name="kimik2.5",
            pat_str=KIMI_K25_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens={},
        )
        print(f"[OK] tiktoken loaded ({tokenizer_type})")
    else:
        tik_encoding = tiktoken.get_encoding(tokenizer_type)
        print(f"[OK] tiktoken loaded ({tokenizer_type})")
    errors += run_tiktoken_format_tests(
        tokenizer_type, tiktoken_path, tik_encoding, tokenizer_type
    )

    if errors == 0:
        print("\n[OK] All tests passed!")
    else:
        print(f"\n[FAIL] {errors} test(s) failed")

    return 1 if errors > 0 else 0


if __name__ == "__main__":
    exit(main())
