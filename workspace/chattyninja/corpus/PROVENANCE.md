# Corpus provenance and ground-truth determination

Extracted 2026-09-17 read-only from the `_obsolete-20260910-chattyninja` branch of this repository, path
`workspace/chattyninja/tests/fixtures/`, via `git show` (repo main checkout, no worktree, no write-back).
243 blob files total:

- 18 `*.jinja` templates, 106 `*.json.zst` input frames, 106 `.meta.json` sidecars
- 5 `dsv4/test_input_*.json`, 5 `dsv4/test_output_*.txt`, `dsv4/rows.json`, `mistral7bv01/generated.json`
- `fixture_loader.nim`. All 106 `.json.zst` decompressed in place with `zstd -d -k` (originals kept)

## Ground-truth verdict: self-contained, no HF call needed

`.meta.json` holds input provenance only, exactly three keys on all 106 sidecars:

- `row`:
  the row stem name
- `kwargs`:
  the render kwargs passed to the template, like `bos_token` and `enable_thinking`
- `template_sha256`:
  sha256 of the suite `.jinja` file

No expected text, no generation spans, no device or dtype metadata.

The byte-level ground truth lives in the row payload itself, schema `chattyninja-chat-render-row-1`, one per
`*.json.zst` frame. Fields on all 106 rows are `schema`, `suite`, `template` (`dir`, `source`, `sha256`),
`row`, `add_generation_prompt`, `messages`, `tools`, `kwargs`, plus:

- `rendered`:
  the expected rendered text, present on 90 rows. This is the ground truth for byte comparison.
- `generation_spans`:
  list of `[start, end)` **codepoint** ranges into `rendered`, present on all rows, non-empty on 8 rows
  (`lagunaxs21` x3, `lfm25` x5)
- `expected_error`:
  `{exception, message}` on the 16 `err_*` rows, which carry no `rendered` key. The counts are `gemma3` 2,
  `gptoss20b` 4, `mistral7bv01` 1 and `qwen38flashnext` 9

Span semantics were verified on `lagunaxs21/second_system.json` span `[94,162)`, where a Python string
slice cuts the assistant turn exactly and a byte slice misaligns.

Token-injecting kwargs (`bos_token`, `eos_token`) are recorded verbatim in the row, so rendering needs
no tokenizer or model access. The engine supplies them as template context variables.

Integrity check. `template_sha256` matches the extracted `.jinja` bytes on 106/106 rows. The corpus is
internally consistent, engine-agnostic ground truth recorded from HF `render_jinja_template`, i.e.
Python jinja2, never from chattyninja's own output.

## Re-record procedure (only if a row must ever be regenerated)

Ground truth origin is HF transformers `render_jinja_template`. If regeneration ever becomes necessary,
the workspace-locked stack is Python 3.14 with transformers 5.16.1, run from the repository root:

```bash
uv run python <recorder>.py
```

with the row's `messages`, `tools`, `add_generation_prompt` and `kwargs` as call arguments and the suite `.jinja`
as `chat_template`. Lock transformers 5.16.1 (`pyproject.toml`) before recording, because span and whitespace
semantics are version-sensitive.

## Sidecar files

- `dsv4/` carries no `.jinja`. Its 5 `test_input_*.json` / `test_output_*.txt` pairs are external fixtures,
  byte-exact between the DeepSeek-V4 `encoding_dsv4.py` tests and the local-inference-lab vllm rust
  renderer tree, Apache-2.0, plain-text outputs (`rows.json` records this). Out of scope for the jinja engine
- `mistral7bv01/generated.json` records that `mistral7bv01.jinja` came from `mistral_common` 1.11.7 @
  `e224216f` (`build_chat_template`, config `v1, spm=True`), machine-generated rather than hand-written.
  It is a template-of-record provenance file, not a render row
- `fixture_loader.nim` is the v1 Nim loader, a zstd plus JSON bridge with insertion-order-preserving
  dicts because dict order is observable through `tojson` /`items`. Reading reference only, never
  compiled here, because the engine ships its own loader
