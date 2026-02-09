#!/usr/bin/env python3
"""
Benchmark Nim BPETokenizer against tiktoken for KimiK2.5 tokenizer.

Benchmarks encoding performance on large fixture files.
"""

import pytoktoktok
from pathlib import Path
from typing import List, Tuple
import time
import base64
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from hf_to_tiktoken import KIMI_K25_PATTERN
import tiktoken

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


def load_kimik25_tiktoken() -> tiktoken.Encoding:
    """Load KimiK2.5 tokenizer into tiktoken."""
    tiktoken_path = TEST_DIR / "tokenizers" / "kimik2.5.tiktoken"

    mergeable_ranks = {}
    with open(tiktoken_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                token_bytes = base64.b64decode(parts[0])
                rank = int(parts[1])
                mergeable_ranks[token_bytes] = rank

    return tiktoken.Encoding(
        name="kimik2.5",
        pat_str=KIMI_K25_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )


def benchmark_tokenizer(
    name: str,
    tik_encoding,
    nim_tokenizer,
    fixtures: List[Tuple[str, Path, int]],
) -> Tuple[float, float, int]:
    """Benchmark encoding performance."""
    total_chars = 0
    tik_time = 0.0
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

        tik_start = time.perf_counter()
        tik_tokens = tik_encoding.encode(text)
        tik_end = time.perf_counter()
        tik_time += tik_end - tik_start

        nim_start = time.perf_counter()
        nim_tokens = nim_tokenizer.encode(text)
        nim_end = time.perf_counter()
        nim_time += nim_end - nim_start

        total_chars += len(text)
        match = tik_tokens == nim_tokens
        print(
            f" | tiktoken: {len(tik_tokens):,} tokens ({tik_end - tik_start:.4f}s) | nim: {len(nim_tokens):,} tokens ({nim_end - nim_start:.4f}s) | {'OK' if match else 'MISMATCH'}"
        )

    return tik_time, nim_time, total_chars


def main():
    print("\n" + "=" * 70)
    print("BENCHMARK: Nim BPETokenizer vs tiktoken (KimiK2.5)")
    print("=" * 70)

    fixtures = get_fixtures()

    print("\n[INFO] Loading KimiK2.5 tokenizer...")
    tik_encoding = load_kimik25_tiktoken()
    nim_tokenizer = pytoktoktok.load_tokenizer_tiktoken(
        str(TEST_DIR / "tokenizers" / "kimik2.5.tiktoken"), "kimik2.5"
    )
    tik_time, nim_time, chars = benchmark_tokenizer(
        "KimiK2.5", tik_encoding, nim_tokenizer, fixtures
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Chars: {chars:,}")
    print(f"tiktoken: {tik_time:.4f}s")
    print(f"Nim: {nim_time:.4f}s")
    speedup = tik_time / nim_time if nim_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
