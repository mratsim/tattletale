# chattyninja

Jinja-family chat-template engine, pure Nim over `nim c`, no torch, no GPU,
no Python at test time.

Templates compile to a flat append-only `seq[Node]` arena, a POD of six
`int32` fields. Dispatch runs through the total `NodeKind`-indexed `Steps` table.

## Layout

| path            | contents                                                            |
|-----------------|---------------------------------------------------------------------|
| `src/`          | engine modules                                                      |
| `tests/`        | suites driven by the `test_chattyninja` task, plus the corpus       |
| `tests/corpus/` | 18 model template suites, 106 recorded input frames, expected bytes |

`nim test_chattyninja` verifies the extracted fixtures before any render test trusts them.

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

`parseTemplate(src)` compiles template text once into the shared render artifact,
the `(CompiledTemplate, CompiledSymbols)` pair. The artifact borrows the template
text and is read-only at render, so one pair serves any number of renders.

`startRender(tmpl, sym, root, clock)` returns a fresh `Context` over the artifact,
ready to render the context dict `root`. `clock` is the epoch `strftime_now` reads.

`pull(context, buf)` writes the render's next bytes into a caller-owned presized
buffer and returns the count written. `0` means the render is complete.

Resumption state lives in the context's `RenderState`, so consumers stop after any
call and resume mid-piece from the same context.

Delivery contract:

- A piece longer than the window drains across calls, delivery itself never raising
  for a small window.
- Window bytes written in a failing call are discarded and never re-delivered.
- A cursor append that does not fit its borrowed scratch window raises `JinjaError`
  with cause `ceScratch`, naming the shortfall.

`pullAll(context)` returns the whole render in one call.
