# ADR 0006: swizzle is a codomain functor layer, concretization deferred

Layer: `tile-passes`. Status: accepted (layer + tests), smem wiring deferred.

Swizzle support is a functor layer over the algebra, not a stride kind inside it.
`layout_algebra/swizzle.nim` ships F2 arithmetic, the `SW_<bits>_<base>_<shift>` functor, and `SwizzledLayout` with property tests.
`SwizzledLayout` is the functor composed over any algebra `Layout`.
No fixture stages through shared memory yet, so no pass consumes the layer.

## Context

A swizzle permutes the offsets a layout produces, spreading shared-memory accesses across banks.
Three placements exist in the references:

- pycute: `Swizzle` is an offset functor by itself, the caller builds the composition, no
  `ComposedLayout` type
- tileiras: swizzle sits inside attributes on scheduled ops (`#cute_nvgpu.umma_k_layout<swizzle=128B>`)
- TileLang: swizzle is chosen inside per-instruction layout proposers, granularity by continuity (128B/64B/32B)

The seed proposal folds F2 strides into composed layouts, ceramic-native.
The pycute notes record its cost: composed layouts cover all functions but lose `complement` and the inverses.
In tattletile, F2 as a core `StrideScalar` kind would also ripple every algebra op.

## Decision

- the swizzle is a codomain functor: `SwizzledLayout.call(crd) = swizzle.apply(layout.call(crd))`, the core `Layout` type untouched
- F2 arithmetic stays beside the algebra: `f2Add` (XOR), `f2Mul` (carry-less), not a `StrideScalar` kind
- `initSwizzle` refuses loudly (`LayoutError`) on overlapping fields, a negative shift with `abs(shift) < bits` is no involution
- property tests ship with the layer, smem staging waits for a consumer

## Alternatives considered

| Alternative | Verdict |
|---|---|
| F2 as a core `StrideScalar` kind | ripples every algebra op with a third codomain, the whole layer pays for a consumer that does not exist yet |
| Composed layouts fusing swizzle into indexing | covers all functions but loses `complement` and the inverses, and pycute itself ships no `ComposedLayout` |
| tileiras-style attribute on scheduled ops | requires the scheduled-op tier tattletile does not have, the subset has no smem ops |
| TileLang-style proposer-integrated swizzle | the right home once smem staging exists, deferred with the wiring |

## Consequences

- the layer is verified without a consumer: pycute F2 docstring values, the coshape table from pycute `test_layout.py`
- functor equals F2-strided elementwise in both orientations, checked by `tests/test_layout_swizzle.nim`
- the bank-conflict-free property is checked: an 8x64 f32 tile under `SW3_3_3` hits all 32
  banks once per aligned 32-element segment of consecutive addresses
- the module doc records the integration point: a staging pass emits swizzled offsets, `SwizzledLayout.call` lowers like `Layout.call`

Compositional swizzling of zipped-divided staging tiles is tested, the future staging pass
builds the composition instead of hand-baking.
No pass references the layer today, so wiring it later is one consumer away, not a redesign.
