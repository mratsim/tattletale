# ADR 0002: official Tile IR text as the frontend, compile-time primary

Layer: `frontend`, status: accepted.

The frontend is official NVIDIA Tile IR text, the format cutile-python and cutile-rs emit per the cuda-tile spec.
Nim ingests it through `staticRead` at compile time.
Runtime parsing stays available as the dev and inspection path.

## Context

Tattletile needs a kernel source that is not Nim.
The port fixed the target: parse real Tile IR text, run layout-algebra passes on it, hand the result to crucible.
The whole pipeline is value-level and VM-clean, so two ingestion times are possible.
Compile time runs through `staticRead` plus the Nim VM, runtime through `parseTileIr` on a string, both feed the CIRv2 bridge unchanged.

The subset-first rule governs growth.
Unsupported constructs refuse loudly, naming the construct and the kernel, and the subset grows per fixture family.

## Decision

- primary path: `parseTileIrFile(path)`, a template that `staticRead`s the file inside a `const` or `static` context
- `embedKernel` builds on it, the product entry compiles the whole pipeline in the VM (ADR `lowering/0008`)
- runtime path: `parseTileIr(text)` on a string, the dev and inspection mode
- subset-first: every refusal names the construct and the kernel, the subset grows per new fixture

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Nim DSL macro frontend (ceramic-style) | cutile-rs shows the cost: a two-track design that re-parses captured source at runtime, a linkme registry, and a shadow type system to satisfy two compilers |
| Python frontend | cutile-python needs a compiled C extension, an AST-extraction stack and a subprocess assembler per compile; none of that is reusable from Nim |
| Tile-IR assembler with binary bytecode interchange | tileiras's binary-only input forced every producer to write a byte-exact flavored writer; text with print/parse symmetry removes that friction class, binary is deferred |
| Reuse cutile-rs's bytecode format | versioned bytecode serves out-of-process toolchains, crucible is in-process, and the format drifts against reference encoders |

## Consequences

- the shipped binary contains only emitted artifacts, no parser and no runtime compiler (proven by `tests/manual_embed_axpy_gemm.nim`)
- the whole pipeline must stay VM-clean, proven by every embed suite const-evaluating `embedKernel`
- fixtures are hand-encoded against the official grammar with citations, cutile-rs emits
  the captured ones (`docs/captured-fixtures.md`)
- text round-trip is a contract: parse-print-parse reaches a fixed point on every fixture
