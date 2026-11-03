# ADR 0007: the store target is kernel parameter slot 0

Layer: `lowering`. Status: accepted, unchanged since the first end-to-end fixture.

`lowerParams` emits the store target's pointer and shape arguments first, then the remaining entry arguments in entry order.
The launch convention binds the output buffer to slot 0, so kernel parameter order and launch order coincide.

## Context

The repo launch convention is `engine.launchKernel(name, outputBuffer, args)` (`tests/test_axpy_metal.nim`).
The output buffer is the second argument, bound to slot 0, the `args` tuple fills the remaining slots.

An entry-order parameter list puts the first entry argument, typically an input pointer, at slot 0.
Every launch then shifts by one slot against the convention, a misbinding caught only by wrong numerics at runtime.

## Decision

- `storeTargetOrder(prog)` collects the store target's pointer plus its SSA shape argument names
- `lowerParams` emits those first, then the remaining entry arguments in entry order
- the axpy fixture therefore launches as `(int32(N), aDev, int32(N), bDev, int32(N))`, output `cDev` bound at slot 0
- the gemm fixture has no dynamic shape args, the order reduces to `ptr_c, ptr_a, ptr_b`

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Entry-order parameters, caller remaps at launch | every call site carries the shift by hand, the misbinding is one edited tuple away |
| Named parameter binding at launch | the engine API binds positionally, adding names would touch crucible |
| Explicit output parameter on top of entry order | duplicates the buffer argument and makes the SPIR-V push-block shape differ from the Metal signature |

## Consequences

- kernel parameter order is canonical across backends, one signature per kernel
- the MSL goldens assert the signature slots, `tests/test_axpy_msl.nim` and `tests/test_gemm_msl.nim`
- the SPIR-V structure goldens assert the push block against the same order, `tests/test_embed_spirv_words.nim`
- a kernel with no store (not in the subset yet) would need a convention decision, recorded here as the open case
