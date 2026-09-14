#!/usr/bin/env python3
"""
Benchmark Nim BPETokenizer against tiktoken and TokenDagger for KimiK2.5 tokenizer.

Benchmarks encoding performance on large fixture files, comparing three implementations:
1. tiktoken (OpenAI's reference implementation)
2. TokenDagger (C++17 high-performance implementation)
3. Nim BPETokenizer (Tattletale's implementation)
"""

from pathlib import Path
from typing import List, Tuple
import time
import base64
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from corpus_reader import read_corpus, read_corpus_prefix
import pytoktoktok

from hf_to_tiktoken import KIMI_K25_PATTERN
import tiktoken
import tokendagger

TEST_DIR = Path(__file__).parent.parent / "tests"
CORPUS_DIR = TEST_DIR / "corpus"


def get_fixtures() -> List[Tuple[str, Path, int]]:
    """Get fixture files to benchmark."""
    return [
        (
            "verne",
            CORPUS_DIR / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt".zst,
            10000,
        ),
        ("shakespeare", CORPUS_DIR / "pg100-shakespeare.txt".zst, 10000),
        (
            "sanguozhi",
            CORPUS_DIR / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt".zst,
            10000,
        ),
        ("sqlite", CORPUS_DIR / "sqlite3.c".zst, 10000),
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


def load_kimik25_tokendagger():
    """Load KimiK2.5 tokenizer into TokenDagger."""
    tiktoken_path = TEST_DIR / "tokenizers" / "kimik2.5.tiktoken"

    mergeable_ranks = {}
    with open(tiktoken_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                token_bytes = base64.b64decode(parts[0])
                rank = int(parts[1])
                mergeable_ranks[token_bytes] = rank

    return tokendagger.Encoding(
        name="kimik2.5",
        pat_str=KIMI_K25_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )


def benchmark_tokenizer(
    name: str,
    tik_encoding,
    td_encoding,
    nim_tokenizer,
    fixtures: List[Tuple[str, Path, int]],
) -> Tuple[float, float, float, int]:
    """Benchmark encoding performance against tiktoken, TokenDagger, and Nim."""
    total_chars = 0
    tik_time = 0.0
    td_time = 0.0
    nim_time = 0.0

    print(f"\n{'=' * 70}")
    print(f"Benchmarking {name}")
    print(f"{'=' * 70}")

    for fixture_name, path, max_chars in fixtures:
        if max_chars:
            text = read_corpus_prefix(path, max_chars)
        else:
            text = read_corpus(path)

        print(f"\n  {fixture_name}: {len(text):,} chars", end="", flush=True)

        tik_start = time.perf_counter()
        tik_tokens = tik_encoding.encode(text)
        tik_end = time.perf_counter()
        tik_time += tik_end - tik_start

        td_start = time.perf_counter()
        td_tokens = td_encoding.encode(text)
        td_end = time.perf_counter()
        td_time += td_end - td_start

        nim_start = time.perf_counter()
        nim_tokens = nim_tokenizer.encode(text)
        nim_end = time.perf_counter()
        nim_time += nim_end - nim_start

        total_chars += len(text)
        match_tik = tik_tokens == nim_tokens
        match_td = td_tokens == nim_tokens
        print(
            f" | tiktoken: {len(tik_tokens):,} tokens ({tik_end - tik_start:.4f}s) | "
            f"tokendagger: {len(td_tokens):,} tokens ({td_end - td_start:.4f}s) | "
            f"nim: {len(nim_tokens):,} tokens ({nim_end - nim_start:.4f}s) | "
            f"tik: {'OK' if match_tik else 'MISMATCH'} td: {'OK' if match_td else 'MISMATCH'}"
        )

    return tik_time, td_time, nim_time, total_chars


def main():
    """Benchmark Nim BPETokenizer against tiktoken and TokenDagger over the KimiK2.5 fixtures."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Nim BPETokenizer vs tiktoken vs TokenDagger (KimiK2.5)")
    print("=" * 70)

    fixtures = get_fixtures()

    print("\n[INFO] Loading KimiK2.5 tokenizer...")
    tik_encoding = load_kimik25_tiktoken()
    td_encoding = load_kimik25_tokendagger()
    nim_tokenizer = pytoktoktok.load_tokenizer_tiktoken(
        str(TEST_DIR / "tokenizers" / "kimik2.5.tiktoken"), "kimik2.5"
    )

    tik_time, td_time, nim_time, chars = benchmark_tokenizer(
        "KimiK2.5", tik_encoding, td_encoding, nim_tokenizer, fixtures
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Chars: {chars:,}")
    print(f"tiktoken: {tik_time:.4f}s")
    print(f"TokenDagger: {td_time:.4f}s")
    print(f"Nim: {nim_time:.4f}s")

    speedup_vs_tik = tik_time / nim_time
    speedup_vs_td = td_time / nim_time
    td_vs_tik = tik_time / td_time

    print(f"\nSpeedup (Nim vs tiktoken): {speedup_vs_tik:.2f}x")
    print(f"Speedup (Nim vs TokenDagger): {speedup_vs_td:.2f}x")
    print(f"TokenDagger vs tiktoken: {td_vs_tik:.2f}x")


if __name__ == "__main__":
    main()
