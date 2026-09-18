# chattyninja

Jinja-family chat-template engine, pure Nim over `nim c`, no torch, no GPU,
no Python at test time.

Templates compile to a flat append-only `seq[Node]` arena, a POD of six
`int32` fields. Dispatch runs through `const steps: array[NodeKind, Step]`,
total over the enum.

Control state lives in a driver owned by the `items` iterator. `Machine`
is read-only after load, concurrently renderable.

## Layout

| path      | contents                                                           |
|-----------|--------------------------------------------------------------------|
| `src/`    | engine modules                                                     |
| `tests/`  | suites driven by `run_tests.sh`, plus the corpus integrity script  |
| `corpus/` | 18 model template suites, 106 recorded input frames, expected bytes |

`python3 tests/check_corpus.py` verifies the extracted fixtures before any
render test trusts them.

`corpus/MANIFEST.md` records the feature burden and build targets of each
suite. `corpus/PROVENANCE.md` records how the ground truth was determined.

## Build

```bash
nim c --experimental:views --hints:off --warnings:off --path:src \
  --outdir:../../build/chattyninja/bin \
  --nimcache:../../build/chattyninja/nimcache <file>
```

Build artifacts land in `build/chattyninja/` at the repo root, inside the gitignored `build/` tree.

## Correctness contract

A render is correct only when its bytes equal
the `rendered` field of the corpus row (codepoint ranges for `generation_spans`).
The corpus carries the ground truth, so no tokenizer or model call happens at test time.
