# ADR 0004: tattletile owns a value-level layout algebra

Layer: `tile-passes`, status: accepted.

Tattletile ports the CuTe layout algebra value-level from the vendored pycute snapshot into `src/tattletile/layout_algebra/`.
The port is the pass machinery's transformation vocabulary, byte-exact against the python reference implementation.

## Context

The tile passes resolve layouts: the thread-value partition of every tile value, the atom operand layouts, the tile decompositions.
That needs the algebra ops (compose, divides, complement, inverses, coalesce, products) callable on layouts built at pass time.
The inputs are runtime values: parsed shapes with `?` dims resolved at launch, thread counts fixed per launch.

Ceramic already ships a layout algebra, but as compile-time macros over static `Int` tuple types.
The conformance contract needs structural equality against the python reference implementation.
The macros host neither requirement.

The runtime-shapes ruling: the port is value-level, on runtime `IntTuple` values, usable in the Nim VM and at runtime.
Ceramic and crucible stay untouched.

## Decision

- port the ops value-level from the vendored pycute, the python reference implementation itself
- ported: `coalesce`/`coalesceZ`, `complement` (+ extend), `compose` (+ by-mode), `layoutAdd`, `greatestCommonDomain`
- ported continued: `rightInverse`, `logicalDivide`/`zippedDivide`, `logicalProduct`, `tilerToLayout`
- not ported, out of scope: `leftInverse`, `nullspace`, mode-subscript forms, F2 codomains in the core (ADR `tile-passes/0006`)

`cosize` ports as the reference's `_coshape`, pycute exports no `cosize`.

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Reuse ceramic's macro-time algebra | compile-time only over static `Int` types, cannot host runtime parsed shapes, results cannot be structurally compared against the python implementation |
| Extend ceramic with runtime layouts | touches ceramic against the boundary ruling, and the macro API shape (static tuples) is the wrong surface for runtime values |
| Call pycute from python only, keep Nim hand-rolled | hand-rolled index math is what the algebra exists to replace, and the conformance contract would have no Nim side |
| Port from CuTe C++ instead of pycute | pycute is the readable, testable form of the same algebra, and it is the python implementation's own source |

## Consequences

- byte-exact conformance: `tests/test_layout_algebra_conformance.nim` replicates the python partition identity checks 1:1
- byte-exact ownership tables and structural dumps of every intermediate, verified against live python runs
- Nim constraints shaped the helpers: declaration before use, no capture of mutable locals in nested procs
- subtle ports are banked in the module docs: complement sorts stride leaves in colex order with stable ties

composition's shape-1 branch evaluates at the stride itself, `coalesceZ` checks linearity at two points.
Out-of-scope ops refuse loudly or are absent, a missing op is a loud error, never a wrong number.
