# ADR 0009: crucible and ceramic stay untouched

Layer: root (umbrella ruling). Status: accepted, held across the whole port.

Tattletile changes zero lines in crucible and ceramic.

- tattletile consumes crucible's construction surface (`cirv2_build`)
- tattletile re-creates the per-target pass composition locally
- the shipped backends print the artifacts unchanged

## Context

The boundary ruling made crucible untouched the default: any touch needs an ADR stating why and what else was considered.
The motivation is crucible's moat: the pass registry, the lowering bundles, and the backend printers are the shared tested surface.
All other projects in the workspace compile against that surface.

Post-#170, crucible's GPU compiler runs on CIRv2:

- a frontend emits a `CirModule` through `cirv2_build`
- crucible declares the per-target pass compositions in `gpu_compiler.nim` macro bodies
  (`workspace/crucible/src/codegen/gpu_compiler.nim`, `codegenMetal`, `codegenSpirvShader`, ...)
- those macros compile a Nim kernel body, which tattletile does not have, tattletile cannot
  enter through them

Every tattletile need was satisfiable from the outside:

- the module is built directly on `cirv2_build`, no ingest step, no context re-creation
- the pass composition is a public list of `register*` calls, the printers are public
- the two tattletile-specific needs (atom asm, workgroup size bake) fit the existing CIRv2
  vocabulary

## Decision

- zero crucible and ceramic changes across the whole port, checkable:
  `git diff 051ba2f29..HEAD --stat -- workspace/crucible workspace/ceramic` is empty
- the pass composition lives in `low_tile.nim` (`runMetalPipeline`, `runSpirvPipeline`),
  by the port's boundary ruling: crucible gained no public ingest entry, its consumed
  API never widened
- tattletile imports the pass modules, `cirv2_build`, `backends_api`, `crucible_codegen` directly.
  The composition it re-creates names exactly the modules it needs

The source tree recorded two re-export gaps against the then-umbrella
`crucible_internal_compiler.nim` (`backends_api` and `passes_lowering_spirv` absent from it).
Both are moot post-#170: the umbrella is gone, the compile entries are macro bodies over
the pass modules, and no re-export contract remains to bypass.

Further crucible-side observations, recorded and not acted on:

- a `CirProc` declares its kernel-ness and workgroup size as fields (`isKernel`, `workgroup`),
  set directly at construction, no attribute machinery involved (ADR `ir/0002`)

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Fork crucible | abandons the shared pipeline and printers, the exact assets tattletile exists to reuse |
| Patch crucible additively (public ingest entry) | the touch is small, but it accumulates into a tattletile-shaped crucible; the boundary ruling requires an ADR per touch and none was load-bearing |
| Build a parallel backend stack | duplicates the moat; the second-backend dividend comes precisely from the stack being shared |

## Consequences

- tattletile owns the pass composition, so a crucible composition change is a tattletile change.
  `runMetalPipeline` and `runSpirvPipeline` name the source of their pass order constraints
- the umbrella ruling is checkable: `git diff` over crucible and ceramic paths is empty across the port
- the second-backend dividend persists on the shared stack: both backends consume the one
  bridge `CirModule`, zero tattletile lowering runs per backend
