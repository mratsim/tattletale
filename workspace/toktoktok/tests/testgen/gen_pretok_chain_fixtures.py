#!/usr/bin/env python3
"""HF reference chain segmentation fixtures. Each row carries piece
(lo,hi) byte offsets for the pre-tokenization stage over the shared
corpus prefixes plus adversarial rows.

Families covered are exaone, step-3.5-flash and gemma-4.

Rows follow the shared shape {name, text, pieces, tokenizer} where
pieces is a list of [lo, hi] BYTE offsets into text (half-open).

HF pre_tokenizer.pre_tokenize_str returns code-point offsets, so
the generator converts them to byte offsets via the utf-8 byte width
of each code point.

Nim suites compare these directly against Span byte views.
Offset conversion is exercised by the invariant check below, the byte
slices of every row must reassemble into text exactly.

Reference path per family (the authoritative HF engine):
- exaone and step-3.5-flash run Tokenizer.from_file(..).pre_tokenizer,
  a Sequence of Isolated Split steps with a ByteLevel(use_regex=false)
  tail (a byte remap that carries no split behavior).
- gemma-4 uses Split(String " ", MergedWithPrevious).

Byte-determinism rule, run this script TWICE, sha256 both outputs,
commit only when the hashes match run-to-run.

Usage:
    /path/to/venv/bin/python gen_pretok_chain_fixtures.py  # twice

Gemma-4 rows need TTT_GEMMA4_TOKENIZER in the environment, the absolute
path of the machine-local gemma-4-E2B-it tokenizer.json
(gitignored, never committed).
"""

import io
import json
import os
import sys
from pathlib import Path

import compression.zstd
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

# corpus prefix cuts, identical to the issue22 fixtures generator
# and the toktoktok reference harnesses
CORPORA = [
    ("sanguozhi5k", CORPUS_DIR / "pg23950-三國志演義-Romance_of_the_Three_Kingdoms.txt.zst", 5000),
    ("verne5k", CORPUS_DIR / "pg4791-Verne-Voyage_au_centre_de_la_Terre.txt.zst", 5000),
    ("shakespeare10k", CORPUS_DIR / "pg100-shakespeare.txt.zst", 10000),
    ("sqlite50k", CORPUS_DIR / "sqlite3.c.zst", 50000),
]

# adversarial rows, emoji, combining marks, mixed scripts, digit
# runs 1..10, space runs, CRLF runs, U+180E rows, specials-like byte seqs,
# other whitespace flavors (nbsp, ideographic space, line separator)
ADVERSARIAL = [
    ("adv_emoji", "hi 😀 there 🇺🇸!"),
    ("adv_emoji_lone", "x😀y \U0001F600z"),
    ("adv_combining", "e\u0301llo cafe\u0302 noir\u0303"),
    ("adv_combining_dense", "a\u0301\u0302\u0303b c\u0300d"),
    ("adv_mixed_scripts", "hello中文字world あいうカタカナ test Кот"),
    ("adv_digit_runs", "1 12 123 1234 12345 123456 1234567 12345678 123456789 1234567890"),
    ("adv_digit_cjk", "abc1234567中文ですカタカナ hello!!world"),
    ("adv_space_runs", "a  b   c    d"),
    ("adv_space_lead", " leading"),
    ("adv_space_trail", "trailing "),
    ("adv_tab_run", "\t\ttab run"),
    ("adv_nl_run", "\n\n\n"),
    ("adv_crlf_runs", "a\r\n\r\nb"),
    ("adv_crlf_alone", "\r\n\r\n\r\n"),
    ("adv_crlf_mix", "x\r\n y\r\n"),
    ("adv_u180e_letter", "a\u180Eb"),
    ("adv_u180e_alone", "\u180E \u180E"),
    ("adv_u180e_space_mix", " \u180Ex \u180E"),
    ("adv_specials_like", "<|endoftext|>user<|im_end|>"),
    ("adv_specials_like2", "<s> </s> a<|b|>c"),
    ("adv_other_ws", "\u00A0nbsp \u3000idsp \u2028ls"),
    ("adv_punct_marks", "it's a 'quoted' test (x) [y] {z}"),
    ("adv_slash_tail", "path/to/file.txt\r\n/etc/passwd"),
    ("adv_twin_letters", "Ünicode ÜBER uppercase"),
]

# - unit sanity rows with hand-recorded expected piece offsets
#   (code-point space, the raw pre_tokenize_str output), each row
#   {name, text, want}, want the (lo, hi) tuples recorded from the HF receipts
UNIT_EXPECTED = [
    ("gemma-4", "unit_gemma_hello", "hello world", [(0, 6), (6, 11)]),
    ("gemma-4", "unit_gemma_double_space", "a  b", [(0, 2), (2, 3), (3, 4)]),
    ("gemma-4", "unit_gemma_crlf", "a\n\nb  c", [(0, 5), (5, 6), (6, 7)]),
    ("gemma-4", "unit_gemma_lone_space", " ", [(0, 1)]),
    ("gemma-4", "unit_gemma_trailing", "a b  ", [(0, 2), (2, 4), (4, 5)]),
    ("gemma-4", "unit_gemma_tabs", "a\tb c", [(0, 4), (4, 5)]),
    ("step-3.5-flash", "unit_s35_digits", "1234567", [(0, 3), (3, 6), (6, 7)]),
    ("exaone", "unit_exaone_digits", "1234567",
     [(i, i + 1) for i in range(7)]),
    ("step-3.5-flash", "unit_s35_hello", "hello world", [(0, 5), (5, 11)]),
]


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


def char_to_byte_offsets(text, pairs):
    """HF pre_tokenize_str offsets are code-point offsets, convert to utf-8
    byte offsets (half-open), the unit the Nim Span views use."""
    widths = [1 if ord(c) < 0x80 else (2 if ord(c) < 0x800 else
              3 if ord(c) < 0x10000 else 4) for c in text]
    starts = [0]
    for w in widths:
        starts.append(starts[-1] + w)
    out = []
    for lo, hi in pairs:
        out.append([starts[lo], starts[hi]])
    return out


def tokenize_family(family, texts, cache={}):
    """Reference piece offsets per text for one family, byte offsets.
    Returns the `{text: pieces}` mapping. Tokenizer objects are cached
    per family (loading the gemma frame is the slow part, the cache keeps the run order-stable and quick)."""
    if family not in cache:
        if family == "gemma-4":
            gemma = os.environ.get("TTT_GEMMA4_TOKENIZER")
            if not gemma:
                sys.exit("TTT_GEMMA4_TOKENIZER must name the machine-local "
                         "gemma-4-E2B-it tokenizer.json (gitignored, not in repo)")
            tk = Tokenizer.from_file(gemma)
        else:
            tk = Tokenizer.from_file(str(
                TOKENIZERS_DIR / f"{family}-tokenizer.json"))
        cache[family] = tk
    tk = cache[family]
    rows = []
    for name, text in texts:
        pairs = [(a, b) for _, (a, b) in
                 tk.pre_tokenizer.pre_tokenize_str(text)]
        pieces = char_to_byte_offsets(text, pairs)
        # invariant check:
        #   byte slices must reassemble into text exactly
        rebuilt = "".join(text[lo:hi] for lo, hi in pieces)
        if rebuilt != text:
            raise SystemExit(
                f"coverage gate failed for {family} {name}: pieces do"
                f" not reassemble into text")
        rows.append({"name": name, "text": text, "pieces": pieces,
                     "tokenizer": f"{family}-tokenizer.json"})
    return rows


def main():
    """Entry point, emits the fixture frames into OUT_DIR."""
    texts = [("corpus_" + name, read_corpus_prefix(path, max_chars))
             for name, path, max_chars in CORPORA]
    texts += list(ADVERSARIAL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # unit sanity check against the hand-recorded HF test receipts
    for family, name, text, want in UNIT_EXPECTED:
        got = tokenize_family(family, [(name, text)])[0]["pieces"]
        if got != [list(p) for p in want]:
            raise SystemExit(
                f"unit gate failed for {family} {name}: got {got},"
                f" recorded {want}")
        print(f"unit ok: {family} {name} = {got}")

    for family in ["exaone", "step-3.5-flash", "gemma-4"]:
        rows = tokenize_family(family, texts)
        payload = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
        out_path = OUT_DIR / f"pretok_chain_{family}.json.zst"
        write_text_zst(out_path, payload)
        print(f"wrote {out_path} ({len(rows)} rows, {len(payload)} json bytes)")


if __name__ == "__main__":
    main()
