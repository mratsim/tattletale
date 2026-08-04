# toktoktok

BPE tokenizer library for Nim: encoding, decoding, and serialization of byte-pair-encoding tokenizers, plus regex-based pre-tokenization.

## What it provides

Entry point: [`toktoktok.nim`](toktoktok.nim) re-exports [`src/bpe_codec.nim`](src/bpe_codec.nim), [`src/serialization.nim`](src/serialization.nim), and [`src/tokenizers_regexps.nim`](src/tokenizers_regexps.nim).

- **BPE codec** (`src/bpe_codec.nim`)
  - `BPETokenizer` ref type with encoder/decoder tables, special-token tables, and PCRE2-based pattern matchers.
  - `encode`, `encodeWithSpecialTokens`, `decodeToString`, and the raw `bytePairMerge` / `bytePairEncode` primitives.
  - Loaders: `loadFromTiktoken`, `loadHFTokenizer` (HuggingFace `tokenizer.json`), `loadTiktokenizer`, and `tokenCount`.
  - In-place encoding with `var seq[int]` accumulation (see the metering notes in the header); PCRE2 matching via `workspace/pcre2`.
- **Serialization** (`src/serialization.nim`)
  - `TiktokenFormat` and `HFTokenizer` deserialization (JSON via `jsony`), including mergeable ranks, special tokens, and base64-encoded byte decoding.
- **Regular-expression tokenizers** (`src/tokenizers_regexps.nim`)
  - Pre-tokenizer regular expressions for `R50k`/`Gpt2`/`P50k`, `Cl100k`, `O200k`, and `KimiK25`.

## Tests

- `tests/` — round-trip and serialization tests, plus fixture comparisons against HuggingFace `tokenizers` and OpenAI `tiktoken` (`tests/test_fixtures_*`).
- `bench/` — `meter_tokenizer.nim` (TTT_METER hot-path profiling) and Python comparisons against `tiktoken` / HF tokenizers.

## Status

Active development. Tokenizer encode/decode, special tokens, and HF/tiktoken loading are implemented; the tests/README.md is not yet filled in.

## Related

- Root project: [`../../README.md`](../../README.md)
- Package manifest: `toktoktok.nimble` (license: MIT or Apache 2.0, deps: `jsony`, `nim >= 2.2.0`)
