# chattyninja

Jinja-family chat-template engine, pure Nim over `nim c`, no torch, no GPU,
no Python at test time.

Templates compile to a flat append-only `seq[Node]` arena, a POD of six
`int32` fields. Dispatch runs through `const steps: array[NodeKind, Step]`,
total over the enum.

Control state lives in a driver owned by the `items` iterator. `Machine`
is read-only after load, concurrently renderable.

## Layout

| path            | contents                                                            |
|-----------------|---------------------------------------------------------------------|
| `src/`          | engine modules                                                      |
| `tests/`        | suites driven by the `test_chattyninja` task, plus the corpus       |
| `tests/corpus/` | 18 model template suites, 106 recorded input frames, expected bytes |

`python3 tests/check_corpus.py` verifies the extracted fixtures before any
render test trusts them.

`tests/corpus/MANIFEST.md` records the feature burden and build targets of each
suite. `tests/corpus/PROVENANCE.md` records how the ground truth was determined.

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

## Pull API

`pull(m, t, d, buf)` writes the render's next bytes into a caller-owned presized
buffer and returns the count written. `0` means the render is complete.

Resumption state lives in the `Driver`, so consumers stop after any call and resume
mid-piece from the same driver.

Delivery and scratch ownership:

- Window bytes written in a failing call are discarded and never re-delivered.
- Scratch is caller-owned through `attachScratch(d, buf)` and must outlive the render.
  Size it for the largest derived value, with the JSON of the largest context value
  as the upper bound.
- Underestimation is safe. `ScratchError` names the shortfall, so the caller grows
  the buffer, reattaches it and repulls.

Composability:

- One `Machine` serves many `Driver`s with byte-identical renders, the contract `tests/t_twodriver.nim` checks.
- Bounded consumption composes through a ring-window machine over `pull`, a shape demonstrated in `tests/t_compose.nim`.
