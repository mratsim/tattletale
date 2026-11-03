# ADR 0005: abstract mmaf stays abstract, atom selection is a pass

Layer: `tile-passes`, status: accepted.

`mmaf` in Tile IR is abstract, no atom, per the cuda-tile spec.
Tattletile lowers it one of two ways.
The abstract per-thread FMA loop covers every backend and shape, a hardware atom covers eligible shapes on Metal.
No selection is a fallback signal, not an error.

## Context

The cuda-tile spec fixes the public surface: `%acc = mmaf %a, %b, %acc` over tile operands, no atom named.
NVIDIA's closed `tileiras` decides the distribution, attaching the atom later as a witness attribute in `nv_tileaa`.
Tattletile owns that missing layer, openly.

The abstract mmaf has an exact per-thread decomposition, the one the python reference executes.
It is a FMA loop over the owned accumulator coordinates, reading the operands through their frozen view layouts.
The Apple simdgroup atom has a hardware-verified fragment record, bit-linear over all 32 lanes (`tests/test_mma_atom.nim`).

## Decision

- pass 4 (`selectMmaAtoms`) asks the atom layer per mmaf: `atomFor(elem, M, N, K, threads)`
- an atom applies when the element is f32, the thread count is 32 (one simdgroup), and M, N, K are positive multiples of the cell dims
- the one registered atom is `apple8x8x8F32()`, fragment layouts ported from ceramic's `h_mma_registry`, verified on hardware
- no selection records no selection: the abstract FMA lowering stays in charge, zero refusals, existing shapes untouched

The Metal bridge lowers a selected mmaf to the fragment program.
Nested repetition loops stage the operands per fragment, one simdgroup mma (`asmS`) per cell,
accumulator stores per coordinate.
The SPIR-V entry refuses atom selections, the simdgroup intrinsics have no SPIR-V form, so every SPIR-V shape takes the abstract fallback.

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Closed-backend atom choice, tileiras-style | the distribution becomes uninspectable and untunable, exactly the layer tattletile exists to open |
| TileLang-style op-local proposers (`GemmNode::InferLayout` + warp-partition policy) | the right algorithm shape, but a per-instruction registry with cost models is more machinery than one atom and one backend need today |
| ceramic's macro-time `gemm_mma` (asm at Nim-macro time) | emits `nnkAsmStmt` the compiler can never re-schedule, fuse or pipeline, the mma lowering anti-pattern |
| One universal FMA atom everywhere (ceramic's CUDA pick) | portable and trivially correct, but it forgoes the tensor core, tattletile keeps it as the abstract fallback instead |

## Consequences

- the fallback contract keeps every pre-atom shape green: the same gemm fixture runs abstract at 4 threads and on the atom at 32 threads
- crucible's rename pass uniquifies repeated declarations but does not rewrite inline-asm operand references
- atom emission therefore gives every fragment scalar one declaration site and unique names, the reused-name rebinding hazard is closed
- D and C may alias, the simdgroup call accumulates in place, matching the accumulator operand

Density baseline in `tests/test_gemm_atom_metal.nim`: 25 CIRv2 nodes (22 statements + 3 loops)
and 60 scheduled ops per thread per public mmaf (16x16x16 tile).
That is about 2x what tileiras hides per public op, with fragment-reload hoisting recorded as future work.
- No runtime fragment predication: the selector gives out-of-cell tile dims no atom
  selection and the abstract FMA path lowers them, every fragment predicate is statically true.
