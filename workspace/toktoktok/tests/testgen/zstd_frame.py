#!/usr/bin/env python3
"""Zstd frame writer for the toktoktok test fixtures.

One zstd frame, level 19, content size and checksum recorded in the
frame header, matching the recording convention of the workspace:
the JSON payload rides inside the zstd frame, the byte law is content-exact.
"""

import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}


def write_text_zst(path, payload):
    """One zstd frame around exact utf-8 payload bytes."""
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))


def read_text_zst(path):
    """Exact payload bytes of one zstd frame.
    A corrupt or content-size-unknown frame fails loudly."""
    with open(path, "rb") as f:
        return compression.zstd.decompress(f.read())
