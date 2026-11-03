# tattletile — Tile IR compiler on crucible internals

tattletile compiles official NVIDIA Tile IR, resolving layouts through its own transformation-oriented algebra.
The lowering bridges into crucible's CIRv2 (`workspace/crucible/src/codegen/ir/`).
The result runs on every crucible backend unchanged.

## Ingestion

- Source is Tile IR text emitted by cutile-python and cutile-rs per the cuda-tile spec.
- Ingestion is compile-time primary through `staticRead` and the Nim VM.
- Runtime parsing stays the dev and inspection path.

## Layout

| Path | Contents |
|---|---|
| `src/tattletile/` | Tile IR datatypes, subset parser, tile passes, lowering to crucible's CIRv2 |
| `tests/` | Suite driving the public compile entry, Tile IR text fixtures, the manual demo host |

## Rules

- No nimble. Modules import crucible and ceramic directly by relative path.
- The supported Tile IR subset grows per tracing bullet.
- Constructs outside the subset refuse loudly, naming the construct and the kernel.
- Crucible and ceramic stay untouched. Any change there requires an ADR stating why and what else was considered.

## References

- Fixtures are captured cutile-python emissions plus hand-encoded minimal snippets.
- Reference trees are read-only, no git operations inside.
- Trees live in `_references_layout_tile_algebra/` plus tileiras in `_references_research/nvopen-tools/`.
- Op records live in `.scratchspace/20261025-0922-C15D01-tattletile-compiler/`.
