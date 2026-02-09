#!/usr/bin/env python3
"""
Benchmark Nim BPETokenizer against HuggingFace tokenizer (GLM-4.7).

Benchmarks encoding performance on large fixture files.
"""

from pathlib import Path
from typing import List, Tuple
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from tokenizers import Tokenizer

TEST_DIR = Path(__file__).parent.parent / "tests"
FIXTURES_DIR = TEST_DIR / "fixtures" / "large"


def get_fixtures() -> List[Tuple[str, Path, int]]:
    """Get fixture files to benchmark."""
    return [
        (
            "verne",
            FIXTURES_DIR / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt",
            10000,
        ),
        ("shakespeare", FIXTURES_DIR / "pg100-shakespeare.txt", 10000),
        (
            "sanguozhi",
            FIXTURES_DIR / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt",
            10000,
        ),
        ("sqlite", FIXTURES_DIR / "sqlite3.c", 10000),
    ]


def load_hf_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load HuggingFace tokenizer from JSON file."""
    path = Path(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(path))
    print(f"[OK] HuggingFace tokenizer loaded ({path.name})")
    return tokenizer


def benchmark_tokenizer(
    name: str,
    hf_tokenizer,
    nim_tokenizer,
    fixtures: List[Tuple[str, Path, int]],
) -> Tuple[float, float, int]:
    """Benchmark encoding performance."""
    total_chars = 0
    hf_time = 0.0
    nim_time = 0.0

    print(f"\n{'=' * 60}")
    print(f"Benchmarking {name}")
    print(f"{'=' * 60}")

    for fixture_name, path, max_chars in fixtures:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if max_chars:
                text = f.read(max_chars)
            else:
                text = f.read()

        print(f"\n  {fixture_name}: {len(text):,} chars", end="", flush=True)

        hf_start = time.perf_counter()
        hf_output = hf_tokenizer.encode(text, add_special_tokens=False)
        hf_tokens = hf_output.ids
        hf_end = time.perf_counter()
        hf_time += hf_end - hf_start

        nim_start = time.perf_counter()
        nim_tokens = nim_tokenizer.encode_ordinary(text)
        nim_end = time.perf_counter()
        nim_time += nim_end - nim_start

        total_chars += len(text)
        match = hf_tokens == nim_tokens
        print(
            f" | HF: {len(hf_tokens):,} tokens ({hf_end - hf_start:.4f}s) | nim: {len(nim_tokens):,} tokens ({nim_end - nim_start:.4f}s) | {'OK' if match else 'MISMATCH'}"
        )

    return hf_time, nim_time, total_chars


def main():
    import pytoktoktok

    print("\n" + "=" * 70)
    print("BENCHMARK: Nim BPETokenizer vs HuggingFace (GLM-4.7)")
    print("=" * 70)

    fixtures = get_fixtures()

    print("\n[INFO] Loading GLM-4.7 tokenizer...")
    hf_tokenizer = load_hf_tokenizer(
        str(TEST_DIR / "tokenizers" / "glm-4.7-tokenizer.json")
    )
    nim_tokenizer = pytoktoktok.load_tokenizer_hf(
        str(TEST_DIR / "tokenizers" / "glm-4.7-tokenizer.json")
    )
    hf_time, nim_time, chars = benchmark_tokenizer(
        "GLM-4.7", hf_tokenizer, nim_tokenizer, fixtures
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Chars: {chars:,}")
    print(f"HF: {hf_time:.4f}s")
    print(f"Nim: {nim_time:.4f}s")
    speedup = hf_time / nim_time if nim_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
