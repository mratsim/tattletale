# toktoktok test tree

Run contract and index. The `test_toktoktok` task compiles and runs every
suite file (`test_*` or `t_*`, non-recursive) in the three suite dirs, top
level first, then `tests/unit`, then `tests/fuzzing`:

```sh
nim test_toktoktok
# single suite:
nim cpp -r -d:release --stackTrace:on --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/toktoktok/tests/unit/t_unit_machine.nim
```

## Data staging

- `tests/tokenizers/` is gitignored, holding the recorded and downloaded
  tokenizer inputs every loading suite reads (see the contract below)
- a missing data file fails its suite loudly with the staging command
- when the directory is absent the task itself falls back
  to `nim download_test_tokenizers`, the chronos downloader

Stage the data once:

```sh
python3 workspace/toktoktok/tests/fetch_test_tokenizers.py
```

## Wall budget

- every suite prints a `wall <seconds>` receipt
- the budget is 60 s per suite and 2 min for the whole family including compiles
- the measured full run gives 24 suites, 52.7 s total suite wall, heaviest
  suite 9.4 s (`tests/fuzzing/t_cache_redundancy.nim`)

## Suite index

Top level, the fixture-replay and roundtrip suites. Each suite drains
the pipeline machine built from a checkpoint through bounded pulls:

- the machine surface lives in `tests/pytoktoktok.nim`, the nimpy
  extension module, built by `nim make_pytoktoktok`
- one-shot encode helpers loop over the same pull protocol

| suite | checks |
|---|---|
| `test_serialization.nim` | loader conversion rows (HF json spellings into `convertHfToTiktoken`): named rank and special-token expectations, byte-filler ranks for missing bytes |
| `test_fixtures_bytepairmerge.nim` | the 15 recorded bytepairmerge rows through the static merge structures (`bytePairEncodePQ`, `encodeSegmentPQ`, `encodeSegmentBt`), per-row engine-contract annotations |
| `test_fixtures_small_hf_tokenizers.nim` | recorded HF-tokenizer-library id streams over the 6 HF checkpoints, specials active |
| `test_fixtures_small_tiktoken_from_hf.nim` | recorded tiktoken-over-converted-json id streams over the same checkpoints, no specials |
| `test_fixtures_small_tiktoken.nim` | recorded tiktoken id streams over the 5 rank files (r50k, p50k, cl100k, o200k, kimik2.5), no specials |
| `test_roundtrip_hf_tokenizer.nim` | encode/decode roundtrips over the 6 HF checkpoints: per-language rows, the historical Chinese paragraph, the runic Verne passage |
| `test_roundtrip_tiktoken.nim` | encode/decode roundtrips over the 5 rank files: per-language rows and the Chinese historical paragraph |

`tests/unit/` carries explicit rows, named input against named expectation.

| suite | checks |
|---|---|
| `t_unit_machine.nim` | identity-machine protocol rows: stream == input bytes, resume after a named byte count, empty-input rows |
| `t_pretok_chain.nim` | chain fixture frames (exaone, step-3.5-flash, gemma-4) reproduced byte-identically. Per-feature chain rows (digit grouping, CJK split step, space splitter). Chain-vs-flat-join divergence isolated to the documented rows. Special-scan composition |
| `t_unit_bpe.nim` | BPE merge rows over one hand-built rank table: whole-piece hit beating the merge path, one- and two-merge pieces, the leftmost-pair tie rule, the 1-byte double-add quirk, unknown-byte BpeError |
| `t_unit_merge_tables.nim` | duplicate-rank dense id assignment (rank-ascending, byte-lexicographic tiebreak) and trie lookups over a named prefix chain |
| `t_unit_special_scan.nim` | special-scan decision rows: greedy leftmost-start selection, first-declared same-start tie rule, ordinary-region non-split, automaton-vs-two-level agreement, empty-dictionary passthrough, empty-pattern rejection |
| `t_bytelevel.nim` | the GPT-2 byte-to-unicode remap (deserializers procs/tables): table equality vs an independent computation, corpus remap strings unmap byte-identically, prefix-space position |
| `test_fixtures_special_pretok_vectors.nim` | the special-pretokenization recorded-vector frames (recorded upstream in issue #22, see https://github.com/mratsim/tattletale/issues/22): kimik2.5, step-3.5-flash asserted against the recorded ids, exaone rows skipped (HF added-token longest-match class) |

`tests/fuzzing/` holds the property and fuzz suites.

| suite | checks |
|---|---|
| `t_pq_reference.nim` | the priority-queue reference implementation (`pq_bpe_reference.nim`, test-side, no served path): ordinary path == the recorded tiktoken id streams per rank file, recorded colliding-rank tie rows, the 1-byte double-add structure, engine-history invariance, machine-walk rows, the PairIndex-vs-legacy structure twin, merge-table invariants |
| `t_bpe.nim` | backtracking encoder == priority-queue encoder == the naive-merge specification record, byte-identical id streams on every family over corpora, fuzz and adversarial rows |
| `t_cache_redundancy.nim` | engine-history invariance: warm == cold == cache-free recomputation under arbitrary engine warm/cold history |
| `t_dense_ids.nim` | `denseIdsOfSorted` neutrality vs the tuple-sort construction, duplicate ranks included |
| `t_vocab_trie.nim` | double-array trie equivalence vs the rank tables on every staged checkpoint, fuzz and corpus-slice keys, double-build determinism |
| `t_machine.nim` | identity + normalization machine streams over corpus prefixes, resume discipline, empty-input rows |
| `t_special_scan_stage.nim` | special-scan decision stream vs the greedy reference algorithm on seeded fuzz corpora, two-level vs automaton path equivalence, chunk-boundary invariance |
| `t_pipeline.nim` | chunk-boundary invariance (chunked feeds + finish == whole input for every split offset), machine stream rows (partial consumption + resume, drained stickiness, feed-restart guard) |
| `t_pretok_split.nim` | split scan == frontier-engine scan on 241,056 chain rows (12 families x 4 corpora + 84 fixture texts + 20,000 adversarial LCG rows), the 4,848-row whitespace adversarial matrix, hand-computed segmentation rows, the pat1 leftmost-vs-longest recorded verdicts |
| `t_pretok_dfa_fuzz.nim` | machine segmentation == standalone split scan per family over corpora, fixture texts and fuzz rows. Frontier-DFA vs NFA-simulation twin. Machine stream conformance. Build and scan timing receipts |

## tests/corpus/

zstd-compressed text frames (三國志演義, Verne, Shakespeare, sqlite3.c),
provenance in `PROVENANCE.md`. Suites read them through `workspace/zstd`
in Nim. No Python reader participates in any suite.

## tests/fixtures/

Recorded `.json.zst` frames, one JSON array per frame.

- each row carries `name`, `text` and the recorded expectation, either
  `tokenIds` id streams or `pieces` piece spans, recorded from the reference libraries
- groups are `small/` with per-checkpoint recorded id streams,
  `bytepairmerge/` with piece, rank and expected-token rows,
  `special_pretok_*` with special-pretokenization vectors, `pretok_chain_*`
  with recorded piece spans per chain family, plus
  `pq_reference_ties.json.zst` with the recorded colliding-rank tie rows
- frames are flat per-checkpoint files

## tests/tokenizers/

Gitignored recorded and downloaded inputs, plain files by contract.
These are the recorded source of truth the suites load, never wrapped.

- HF `tokenizer.json` files for gpt2, llama3, minimax-m2.1,
  glm-4.7, exaone, step-3.5-flash
- tiktoken rank files for r50k, p50k, cl100k, o200k, kimik2.5
- the GPT-2 vocab.bpe + encoder.json pair and the converted rank file
- staged once by the fetch script above
