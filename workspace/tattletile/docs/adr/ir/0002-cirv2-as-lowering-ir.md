# ADR 0002: CIRv2 is the lowering IR, the bridge builds the module directly

Layer: `ir`. Status: accepted. Supersedes `0001-keep-gpuast-as-lowering-ir.md`, which is not ported: crucible deleted GpuAst.

Tattletile lowers its resolved Tile IR programs straight into crucible's CIRv2 (`workspace/crucible/src/codegen/ir/`).
The bridge constructs the `CirModule` through `cirv2_build`'s constructors and runs crucible's pass pipeline itself.

## Context

The tile passes need four things the backend IR cannot express:

- tile-shaped typed values, static shapes plus element types
- layouts, index maps and thread-value maps, attached to ops as data
- token-ordered memory ops, `load_view_tko` and `store_view_tko`
- an abstract `mmaf` op, before atom selection

Tattletile keeps its own Tile IR (`src/tattletile/tile_ir.nim`) above the backend IR and lowers once, at the bridge.
GpuAst filled the backend-IR role before: a statement tree over scalar types, address spaces and builtin symbols.

Crucible removed GpuAst in PR #170 and moved the whole GPU compiler onto CIRv2 (`crucible_codegen`).
CIRv2 is a typed statement IR with a builder vocabulary (`cirv2_build`), a nanopass pipeline and per-target printers.

## Decision

One bridge, built directly on CIRv2's public construction surface.

### Bridge output

- Tile IR: tattletile-owned, unchanged by this decision, one op enum with per-op payload branches, print/parse round-trip symmetry
- CIRv2: the bridge emits one `CirModule` per kernel through `cirv2_build`, a `procD` kernel with `isKernel` and `workgroup` set as fields
- buffer parameters are `tAptr(caDevice, elem)`, scalars are value types,
  pointer reads go through `patE`
- coordinate builtins are `builtin` symbols carrying their `CirCoordBuiltinKind`, the inline-asm simdgroup mma goes through `asmS`

### Pipeline and boundary

- no Nim kernel body and no `lowerCirv2`: the compile-time pipeline runs entirely in the Nim VM over hand-built CIRv2
- pass composition: the bridge re-creates the per-target composition of `gpu_compiler.nim`, normalization plus shared
  lowering plus the Metal bundle plus validation, then calls `runPasses` and `codegen(m, cgMetal)` or `codegenSpirvShader(m)`
- a pass refusal surfaces as a `SubsetError` naming the kernel
- crucible is untouched: tattletile consumes the API, no public crucible
  ingest entry is added and no crucible file changes

## Alternatives considered

| Alternative                                          | Verdict                                                                                                                                                        |
|------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A minimal AST tier between Tile IR and CIRv2         | rejected: `cirv2_build` already covers the bridge's whole vocabulary (types, statements, expressions, asm) with less code, a second tier would only re-spell it |
| Extend CIRv2 with tile node and layout kinds         | forks the enum every backend and pass dispatches on, imports tile types into crucible against doctrine                                                          |
| Macro path (`metal:`) with a generated Nim kernel body | drags the Nim frontend's type resolution and pragma machinery in for a body tattletile already lowered, the direct construction drops both                    |
| Nimony NVJL or an SSA IR as the backend IR           | inapplicable: the frontend is Tile IR text and the printers are built                                                                                           |

## Consequences

- the bridge owns the pass composition, so a crucible composition change is a tattletile change
- `runMetalPipeline` and `runSpirvPipeline` are the only two composition sites, each re-creates the per-target composition
  of `gpu_compiler.nim` and names the source of its pass order constraints
- the ivec3 struct type behind the coordinate builtins is a tattletile-owned literal (`tt_ivec3`), the stable replacement
  for the Nim frontend's mangled tuple name
- the `weak` memory-ordering qualifier drops at the bridge, program order is the only order, same contract as before

The loop idiom is while-form: counter declaration, init assignment, `whileS` with the increment trailing the body.
