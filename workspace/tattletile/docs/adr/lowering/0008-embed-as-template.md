# ADR 0008: embedKernel is a template, the artifact is a const

Layer: `lowering`. Status: accepted, extended once for the second backend.

`embedKernel` is a template, not a compile-time proc.
It resolves the fixture path at the caller's site, runs the whole pipeline in the Nim VM, and returns a const-evaluable `EmbeddedKernel`.
Engine ingest stays runtime.

## Context

`parseTileIrFile` resolves a relative path against the instantiating module via `instantiationInfo(-1)`.
A `{.compileTime.}` proc wrapper relocates that resolution to `src/tattletile/`, and the fixture read fails.
The path must resolve in the frame of the module that wrote the string literal.

The product requirement is a shipped binary containing only the emitted artifact, no parser, no runtime compiler.
Everything from `staticRead` through the pass composition is VM-clean, so a `const` can hold the result.

## Decision

- `embedKernel(path, entry, threads, backendKind = gbMetal)` is a template returning `EmbeddedKernel`
- the template resolves the path at the caller's site, absolute paths pass through, then calls `parseTileIrFile` with the absolute path
- `backendKind` selects the artifact: `msl: string` or `words: seq[uint32]`, one shared bridge CIRv2 module feeds both compile entries
- engine ingest stays runtime: a real device step, not VM-hostable, outside the const boundary

Banked template semantics: a template body line is `instantiationInfo` frame 0, the caller's site is frame -1.
The template parameter must be named `backendKind`, `backend` collides with `EmbeddedKernel.backend`.

## Alternatives considered

| Alternative | Verdict |
|---|---|
| A `{.compileTime.}` proc returning `EmbeddedKernel` | the instantiationInfo relocation trap, the fixture read resolves against `src/tattletile/` |
| A macro | same resolution problem with more machinery, no benefit over the template |
| Runtime parse and compile per launch | the dev path, kept for inspection, wrong product surface: the shipped binary would carry the parser |
| A codegen CLI writing artifacts to disk | loses per-build type safety and the const object, and adds a build step between fixture and binary |

## Consequences

- the shipped-binary proof (`tests/manual_embed_axpy_gemm.nim`) reads the binary back
  through `getAppFilename`, the MSL text ships in the binary and the parser strings are absent
- both fixtures compile end to end in the VM through `embedKernel`, proven const-context by the embed suites
- the `words` and `msl` artifacts come from one shared bridge CIRv2 module, the backend choice fixes the case object field at compile time
- `backendKind` is a template parameter, so the backend choice is fixed at compile time
