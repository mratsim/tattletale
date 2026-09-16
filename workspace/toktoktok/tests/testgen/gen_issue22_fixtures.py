#!/usr/bin/env python3
"""Generate fixtures for the models that fail issue #22 (special-pretokenization).

Issue #22 (mratsim/tattletale#22) reports that the tokenizer fails when
encoding Kimi-K2.5, K-EXAONE-236B-A23B or Step-3.5-Flash, on 三國志演義
and SQLite source, because their special pre-tokenization is not reproduced.

The models share a Sequence of `Split` steps with `behavior: Isolated`
plus a post-processor. The divergence is one zstd frame per family.
Rows use the {name, text, tokenIds, tokenizer} format.

| model              | failing texts | suspected gap                         |
| ------------------ | ------------- | ------------------------------------- |
| Kimi-K2.5          | 三國志演義         | the `。` + newline merge               |
| K-EXAONE-236B-A23B | 三國志演義, sqlite | the add_prefix_space post-processor   |
| Step-3.5-Flash     | 三國志演義, sqlite | the TemplateProcessing post-processor |

Run twice and sha256 both outputs, commit only when the hashes match.

Usage:
    uv run --no-sync python gen_issue22_fixtures.py
"""

import base64
import io
import json
from pathlib import Path

import compression.zstd
import tiktoken
from tokenizers import Tokenizer

TEST_DIR = Path(__file__).parent.resolve()
TOKTOKTOK_TESTS = TEST_DIR.parent.parent.parent / "toktoktok" / "tests"
TOKENIZERS_DIR = TOKTOKTOK_TESTS / "tokenizers"
CORPUS_DIR = TOKTOKTOK_TESTS / "corpus"
OUT_DIR = TEST_DIR.parent / "fixtures"

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}

# corpus prefix cuts, identical to the toktoktok reference harnesses
CORPORA = [
    ("sanguozhi5k", CORPUS_DIR / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 5000),
    ("verne5k", CORPUS_DIR / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 5000),
    ("shakespeare10k", CORPUS_DIR / "pg100-shakespeare.txt.zst", 10000),
    ("sqlite50k", CORPUS_DIR / "sqlite3.c.zst", 50000),
]

# issue #22 recorded reference counts (engine, vector) for the sanity
# check below, sourced verbatim from the issue body
EXPECTED_COUNTS = {
    ("exaone", "sanguozhi5k"): 4641,
    ("exaone", "sqlite50k"): 12312,
    ("step-3.5-flash", "sanguozhi5k"): 4118,
    ("step-3.5-flash", "sqlite50k"): 13625,
    ("kimik2.5", "sanguozhi5k"): 4697,
}

KIMI_K25_PATTERN = (
    r"[\p{Script=Han}]+"
    r"|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|"
    r"\p{N}{1,3}"
    r"|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|"
    r"\s*[\r\n]+"
    r"|"
    r"\s+(?!\S)"
    r"|"
    r"\s+"
)


def read_corpus_prefix(path, max_chars):
    """First max_chars characters of a corpus frame, utf-8 replacement
    and universal newlines, exactly like the corpus_reader.py module."""
    with compression.zstd.open(path, "rb") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as text:
            return text.read(max_chars)


def write_text_zst(path, payload):
    """One zstd frame around exact utf-8 payload bytes
    (zstd_frame conventions, level 19, content size and checksum in the header)."""
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))


def load_kimi_encoding():
    """tiktoken Encoding over the kimik2.5 rank file and pattern."""
    mergeable_ranks = {}
    with open(TOKENIZERS_DIR / "kimik2.5.tiktoken", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                mergeable_ranks[base64.b64decode(parts[0])] = int(parts[1])
    return tiktoken.Encoding(
        name="kimik2.5",
        pat_str=KIMI_K25_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )


def encode_family(family, texts):
    """Reference ids per corpus text for one family."""
    rows = []
    if family == "exaone" or family == "step-3.5-flash":
        tk = Tokenizer.from_file(str(TOKENIZERS_DIR / f"{family}-tokenizer.json"))
        for name, text in texts:
            ids = tk.encode(text, add_special_tokens=False).ids
            rows.append({"name": f"issue22_{name}", "text": text,
                         "tokenIds": ids, "tokenizer": f"{family}-tokenizer.json"})
    else:
        enc = load_kimi_encoding()
        for name, text in texts:
            ids = enc.encode_ordinary(text)
            rows.append({"name": f"issue22_{name}", "text": text,
                         "tokenIds": ids, "tokenizer": "kimik2.5.tiktoken"})
    return rows


def main():
    """Writes the issue22 regression fixtures for each served family over the corpus prefixes."""
    texts = [(name, read_corpus_prefix(path, max_chars)) for name, path, max_chars in CORPORA]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for family in ["exaone", "step-3.5-flash", "kimik2.5"]:
        rows = encode_family(family, texts)
        for row in rows:
            short = row["name"][len("issue22_"):]
            key = (family, short)
            if key in EXPECTED_COUNTS:
                expected = EXPECTED_COUNTS[key]
                got = len(row["tokenIds"])
                if got != expected:
                    raise SystemExit(
                        f"sanity gate failed: {family} {short} produced {got} ids,"
                        f" issue #22 recorded {expected}; corpus or engine drifted")
                print(f"sanity ok: {family} {short} = {got} ids (matches issue #22)")
        payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
        out_path = OUT_DIR / f"issue22_{family}.json.zst"
        write_text_zst(out_path, payload)
        print(f"wrote {out_path} ({len(rows)} rows, {len(payload)} json bytes)")


if __name__ == "__main__":
    main()
