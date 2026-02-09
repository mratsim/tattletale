#!/usr/bin/env python3
"""
Generate fixture files for byte-pair merge algorithm testing.

This focuses on the core BPE algorithm with minimal external dependencies.
Format:
{
    "input_bytes": [228, 189, 160, ...],
    "ranks": {"bytes_literal": rank, ...},
    "expected_tokens": [19526, 254, ...],
    "description": "Human readable test description"
}
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any

TEST_DIR = Path(__file__).parent.resolve()
FIXTURES_DIR = TEST_DIR / "fixtures" / "bytepairmerge"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def bytes_to_b64_key(d: Dict[bytes, int]) -> Dict[str, int]:
    """Convert bytes keys to base64 strings for JSON compatibility."""
    return {base64.b64encode(k).decode("ascii"): v for k, v in d.items()}


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
            "expected_tokens": [2000, 1001, 103],
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
            "expected_tokens": [200, 52],
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
            "expected_tokens": [19526, 254, 25001, 121, 10310, 244, 45911, 234],
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
                }
            ),
            "expected_tokens": [45911],
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
                }
            ),
            "expected_tokens": [12520, 220],
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
            "expected_tokens": [628, 220],
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
            "expected_tokens": [300, 72, 400, 80, 420, 421, 199, 313, 314, 222],
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
            "expected_tokens": [33],
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
            "expected_tokens": [500],
        }
    )

    cases.append(
        {
            "description": "Bytes with no merge ranks available",
            "input_bytes": [1, 2, 3, 4],
            "ranks": {},
            "expected_tokens": [],
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
            "expected_tokens": [2000, 1002],
        }
    )

    cases.append(
        {
            "description": "Math symbols '𝜑² + 𝜑 + 1 ≡ 0'",
            "input_bytes": list("𝜑² + 𝜑 + 1 ≡ 0".encode("utf-8")),
            "ranks": bytes_to_b64_key(
                {
                    bytes([240, 159, 148, 145]): 12345,
                    bytes([194, 178]): 54321,
                    bytes([32]): 4,
                    bytes([43]): 1000,
                    bytes([49]): 10,
                    bytes([226, 137, 145]): 11111,
                    bytes([48]): 11,
                }
            ),
            "expected_tokens": [12345, 54321, 4, 1000, 12345, 4, 10, 11111, 11],
        }
    )

    cases.append(
        {
            "description": "Greek text 'Ελληνικά'",
            "input_bytes": list("Ελληνικά".encode("utf-8")),
            "ranks": bytes_to_b64_key(
                {
                    bytes([206, 145]): 30000,
                    bytes([207, 129]): 30001,
                    bytes([207, 140]): 30002,
                    bytes([207, 137]): 30003,
                    bytes([206, 189]): 30004,
                    bytes([206, 173]): 30005,
                    bytes([206, 172]): 30006,
                }
            ),
            "expected_tokens": [30000, 30001, 30002, 30003, 30004, 30005, 30006],
        }
    )

    return cases


def main():
    """Generate all byte pair merge fixtures."""
    print("=" * 70)
    print("Generating byte-pair merge fixtures")
    print("=" * 70)

    cases = get_test_cases()
    output_path = FIXTURES_DIR / "bytepairmerge.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(cases)} test cases to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
