#!/usr/bin/env python3
"""Shared reader for the committed corpus frames
(workspace/toktoktok/tests/corpus/, provenance in that directory).

The corpus ships as zstd frames, streamed through the stdlib `compression.zstd` module.

Text semantics match the historical plain text reads exactly,
utf-8 decoding with `errors="replace"`, universal newline
translation and `read(max_chars)` returning max_chars characters.

The encoded token streams of the consumers stay identical to what
the plain text files produced.
"""

import io

import compression.zstd


def read_corpus_prefix(path, max_chars: int) -> str:
    """First max_chars characters of a corpus frame. Returns text-mode semantics
    (ZstdFile stream, utf-8 replacement, universal newlines)."""
    with compression.zstd.open(path, "rb") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as text:
            return text.read(max_chars)


def read_corpus(path) -> str:
    """Whole text of a corpus frame with the prefix reader's text semantics. Returns the full content."""
    with compression.zstd.open(path, "rb") as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as text:
            return text.read()
