# ADR 0003: refusals name the construct and the kernel, at parse time

Layer: `frontend`. Status: accepted, kept by every later layer.

Every refusal in tattletile raises `SubsetError`, naming the construct and the kernel where
it appears. Refusals fire at the earliest layer that can detect the problem.
The parser rejects grammar violations, the passes and the bridge reject semantic ones.

## Context

The subset-first contract makes refusals the growth mechanism, the subset grows per new fixture family.
Everything outside the subset must fail loudly instead of silently miscompiling.
A refusal naming the construct and the kernel turns the subset boundary into an actionable checklist.

The parser sees the text before any semantic pass runs.
Grammar-level checks (arities, type forms, token placement) are cheapest to reject there.

## Decision

- every refusal message reads like the input text, the construct name comes from `opSpelling(kind)`
- result arities are enforced at parse (`checkArity`, `refuseArity`), the official ops have fixed arities
- `?` is a dim and stride value only: refused in constant value lists and in the splat position, accepted in shape and stride lists
- dynamic shape or stride entries force the `tile<i32> -> ` operand-type prefix on `make_tensor_view`, parsed and preserved

Further parse-time refusals, all covered in `tests/test_tile_ir_refusals.nim`:

- unknown element types and unknown pointer pointees refuse by name
- explicit constant lists must fill the tile shape exactly, dynamic dims refuse
- `partition_view` tile dims must be static
- downstream refusals reuse the same contract, `refuse` in the passes and `refuseBridge` in the bridge produce identical shapes

## Alternatives considered

| Alternative | Verdict |
|---|---|
| Collect all errors, report at module end | heavier machinery for no gain at kernel scale, the first refusal is the actionable one |
| Late semantic checking only, no parse-time arity or type checks | pushes trivially detectable text errors into pass code, every pass then re-defends itself |
| MLIR-style diagnostics infrastructure | diagnostic severity, notes and spans are unjustified weight for one in-process compiler |
| Silent skips with defaults, cutile-style per-arch hints | hides the subset boundary, the miscompile class the tileiras drop-policy matrix warns about |

## Consequences

- 25 refusal unit cases cover the parser contract in `tests/test_tile_ir_refusals.nim`
- the same contract propagates, `refuse` in `tile_passes.nim` and `refuseBridge` in `low_tile.nim` produce identical message shapes
- growing the subset is one parse branch plus its refusal cases, per construct
- refusal text is part of the test surface, tests assert on message substrings
