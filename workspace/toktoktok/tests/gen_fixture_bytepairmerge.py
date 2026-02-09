#!/usr/bin/env python3
"""
Generate fixture files for byte-pair merge algorithm testing.

This uses the reference Rust byte_pair_merge algorithm to ensure correct expected_tokens.
Format:
{
    "input_bytes": [228, 189, 160, ...],
    "ranks": {"bytes_literal": rank, ...},

    "description": "Human readable test description"
}
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Tuple

TEST_DIR = Path(__file__).parent.resolve()
FIXTURES_DIR = TEST_DIR / "fixtures" / "bytepairmerge"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def bytes_to_b64_key(d: Dict[bytes, int]) -> Dict[str, int]:
    """Convert bytes keys to base64 strings for JSON compatibility."""
    return {base64.b64encode(k).decode("ascii"): v for k, v in d.items()}


def byte_pair_merge(piece: List[int], ranks: Dict[bytes, int]) -> List[Tuple]:
    """
    Reference implementation matching Rust tiktoken exactly.

    Returns a list of (start_idx, rank) tuples representing merge boundaries.
    """
    parts: List[Tuple] = []
    min_rank = float("inf")
    min_rank_idx = 0

    # Initial pass: compute ranks for all adjacent byte pairs
    for i in range(len(piece) - 1):
        pair = bytes([piece[i], piece[i + 1]])
        rank = ranks.get(pair, float("inf"))
        if rank < min_rank:
            min_rank = rank
            min_rank_idx = i
        parts.append((i, rank))

    parts.append((len(piece) - 1, float("inf")))
    parts.append((len(piece), float("inf")))

    def get_rank(parts: List[Tuple], i: int):
        """Get rank for pair starting at parts[i], spanning to parts[i+3] boundary."""
        if i + 3 < len(parts):
            start_idx = parts[i][0]
            end_idx = parts[i + 3][0]
            pair = bytes(piece[start_idx:end_idx])
            return ranks.get(pair, float("inf"))
        return float("inf")

    # Main merge loop: repeatedly merge the lowest-ranked pair
    while min_rank != float("inf"):
        i = min_rank_idx

        # Update adjacent pair ranks before removal
        if i > 0:
            parts[i - 1] = (parts[i - 1][0], get_rank(parts, i - 1))

        parts[i] = (parts[i][0], get_rank(parts, i))
        parts.pop(i + 1)

        # Re-scan for new minimum rank
        min_rank = float("inf")
        min_rank_idx = 0
        for idx in range(len(parts) - 1):
            if parts[idx][1] < min_rank:
                min_rank = parts[idx][1]
                min_rank_idx = idx

    return parts


def byte_pair_encode(piece: List[int], ranks: Dict[bytes, int]) -> List[int]:
    """
    Reference implementation matching Rust tiktoken exactly.
    """
    if len(piece) == 1:
        pair = bytes([piece[0]])
        return [ranks[pair]] if pair in ranks else []

    merged = byte_pair_merge(piece, ranks)
    result: List[int] = []

    for i in range(len(merged) - 1):
        start_idx = merged[i][0]
        end_idx = merged[i + 1][0]
        token = bytes(piece[start_idx:end_idx])
        if token in ranks:
            result.append(ranks[token])
        elif len(token) == 1 and bytes([token[0]]) in ranks:
            result.append(ranks[bytes([token[0]])])

    return result


def get_test_cases() -> List[Dict[str, Any]]:
    """Get all byte pair merge test cases."""
    cases = []

    # Basic ASCII
    cases.append(
        {
            "description": "Simple ASCII text 'hello'",
            "input_bytes": list(b"hello"),
            "ranks": bytes_to_b64_key(
                {
                    b"h": 100,
                    b"e": 101,
                    b"l": 102,
                    b"o": 103,
                    b"he": 500,
                    b"el": 501,
                    b"ll": 502,
                    b"lo": 503,
                    b"hel": 1000,
                    b"ell": 1001,
                    b"llo": 1002,
                    b"hell": 2000,
                    b"ello": 2001,
                }
            ),
        }
    )

    cases.append(
        {
            "description": "Simple ASCII 'the'",
            "input_bytes": list(b"the"),
            "ranks": bytes_to_b64_key(
                {
                    b"t": 50,
                    b"h": 51,
                    b"e": 52,
                    b"th": 200,
                    b"he": 201,
                }
            ),
        }
    )

    # Multi-byte UTF-8 sequences (Chinese)
    cases.append(
        {
            "description": "Chinese '你好世界' -你好 world",
            "input_bytes": [228, 189, 160, 229, 165, 189, 228, 184, 150, 231, 149, 140],
            "ranks": bytes_to_b64_key(
                {
                    bytes([228, 189]): 19526,
                    bytes([229, 165]): 25001,
                    bytes([228, 184]): 10310,
                    bytes([231, 149]): 45911,
                    bytes([160]): 254,
                    bytes([189]): 121,
                    bytes([150]): 244,
                    bytes([140]): 234,
                }
            ),
        }
    )

    # Single 3-byte character (界)
    cases.append(
        {
            "description": "Single Chinese character '界' (one 3-byte UTF-8 char split across merges)",
            "input_bytes": [231, 149, 140],
            "ranks": bytes_to_b64_key(
                {
                    bytes([231, 149]): 45911,
                    bytes([149]): 244,
                    bytes([140]): 220,  # Add last byte
                }
            ),
        }
    )

    # Emoji (4-byte UTF-8)
    cases.append(
        {
            "description": "Emoji '🌍' - Earth globe",
            "input_bytes": [240, 159, 140, 141],
            "ranks": bytes_to_b64_key(
                {
                    bytes([240, 159]): 12520,
                    bytes([159, 140]): 234,
                    bytes([140, 141]): 235,
                    bytes([141]): 220,
                    bytes([240]): 300,  # First byte standalone
                }
            ),
        }
    )

    cases.append(
        {
            "description": "Multiple newlines '\\n\\n\\n'",
            "input_bytes": [10, 10, 10],
            "ranks": bytes_to_b64_key(
                {
                    bytes([10, 10]): 628,
                    bytes([10]): 220,
                }
            ),
        }
    )

    cases.append(
        {
            "description": "Code-like 'def foo():\\n    return 42'",
            "input_bytes": list(b"def foo():\n    return 42"),
            "ranks": bytes_to_b64_key(
                {
                    b"d": 70,
                    b"e": 71,
                    b"f": 72,
                    b"de": 300,
                    b"ef": 301,
                    b" ": 200,
                    b"f ": 400,
                    b"o": 80,
                    b"o ": 410,
                    b"(": 150,
                    b"()": 420,
                    b"():": 421,
                    b"\n": 198,
                    b"\n ": 199,
                    b"r": 82,
                    b"re": 310,
                    b"ret": 311,
                    b"retu": 312,
                    b"retur": 313,
                    b"return": 314,
                    b"4": 220,
                    b"42": 222,
                }
            ),
        }
    )

    cases.append(
        {
            "description": "Single byte 'a'",
            "input_bytes": list(b"a"),
            "ranks": bytes_to_b64_key(
                {
                    b"a": 33,
                }
            ),
        }
    )

    cases.append(
        {
            "description": "Two identical bytes '\\x00\\x00'",
            "input_bytes": [0, 0],
            "ranks": bytes_to_b64_key(
                {
                    bytes([0]): 100,
                    bytes([0, 0]): 500,
                }
            ),
        }
    )

    cases.append(
        {
            "description": "Bytes with no merge ranks available",
            "input_bytes": [1, 2, 3, 4],
            "ranks": {},
        }
    )

    cases.append(
        {
            "description": "High bytes (above 127)",
            "input_bytes": [128, 129, 255],
            "ranks": bytes_to_b64_key(
                {
                    bytes([128]): 1000,
                    bytes([129]): 1001,
                    bytes([255]): 1002,
                    bytes([128, 129]): 2000,
                }
            ),
        }
    )

    # Math symbols '𝜑² + 𝜑 + 1 ≡ 0' with complete rank coverage
    cases.append(
        {
            "description": "Math symbols '𝜑² + 𝜑 + 1 ≡ 0'",
            "input_bytes": [
                240,
                157,
                156,
                145,
                194,
                178,
                32,
                43,
                32,
                240,
                157,
                156,
                145,
                32,
                43,
                32,
                49,
                32,
                226,
                137,
                161,
                32,
                48,
            ],
            "ranks": bytes_to_b64_key(
                {
                    bytes([240]): 100,
                    bytes([157]): 101,
                    bytes([156]): 102,
                    bytes([145]): 103,
                    bytes([194]): 104,
                    bytes([178]): 105,
                    bytes([32]): 4,
                    bytes([43]): 1000,
                    bytes([49]): 10,
                    bytes([226]): 106,
                    bytes([137]): 107,
                    bytes([161]): 108,
                    bytes([48]): 11,
                    bytes([157, 156]): 200,
                    bytes([156, 145]): 201,
                }
            ),
        }
    )

    # Greek text - simplified with complete rank coverage
    cases.append(
        {
            "description": "Greek text 'Ελληνικά'",
            "input_bytes": [
                206,
                145,
                207,
                129,
                207,
                140,
                207,
                137,
                206,
                189,
                206,
                185,
                206,
                186,
                206,
                172,
            ],
            "ranks": bytes_to_b64_key(
                {
                    bytes([206, 145]): 30000,
                    bytes([207, 129]): 30001,
                    bytes([207, 140]): 30002,
                    bytes([207, 137]): 30003,
                    bytes([206, 189]): 30004,
                    bytes([206, 185]): 30005,
                    bytes([206, 186]): 30006,
                    bytes([206, 172]): 30007,
                    bytes([145, 207]): 40001,  # Missing pair needed
                    bytes([129, 207]): 40002,  # Missing pair needed
                    bytes([140, 207]): 40003,  # Missing pair needed
                    bytes([137, 206]): 40004,  # Missing pair needed
                    bytes([185, 206]): 40005,  # Missing pair needed
                    bytes([186, 206]): 40006,  # Missing pair needed
                }
            ),
        }
    )

    return cases


def b64_decode_to_bytes(b64_str: str) -> List[int]:
    """Convert base64 string back to byte values."""
    decoded = base64.b64decode(b64_str)
    return list(decoded)


def get_expected_tokens(case: Dict) -> List[int]:
    """Compute expected tokens using reference Rust algorithm."""
    input_bytes = case["input_bytes"]
    ranks_raw = case["ranks"]
    # Convert base64 keys back to bytes
    ranks = {}
    for b64_key, rank in ranks_raw.items():
        key_bytes = b64_decode_to_bytes(b64_key)
        ranks[bytes(key_bytes)] = rank
    return byte_pair_encode(input_bytes, ranks)


def main():
    """Generate all byte pair merge fixtures."""
    print("=" * 70)
    print("Generating byte-pair merge fixtures")
    print("=" * 70)

    cases = get_test_cases()
    # Compute expected_tokens using reference implementation
    for case in cases:
        case["expected_tokens"] = get_expected_tokens(case)
        print(f"Computed expected_tokens for: {case['description']}")
        print(f"  -> {case['expected_tokens']}")

    output_path = FIXTURES_DIR / "bytepairmerge.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(cases)} test cases to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
