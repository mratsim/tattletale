# ADR 0001: CIRv2 type representation — a closed lattice derived once at the boundary

Status: implemented (M1, CIRv2 tracer)

## Decision

CIRv2 owns its own closed type lattice (`CirType` in `src/cirv2/cirv2_lang.nim`).
The lattice covers what GPU kernels need:

| Kind    | Form                                                                                                                                         |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------|
| scalars | `int`/`uint`/`float` by bit width (8/16/32/64, -1 = machine word), `bool`, `void`                                                            |
| vector  | `vector(N, elem)`                                                                                                                            |
| pointer | the Leng pointer split: `ptr(addrSpace, elem)` (single object) vs `aptr(addrSpace, elem)` (array of objects), each with an explicit address space (thread / threadgroup / constant / device) |
| struct  | nominal struct types by name + field list                                                                                                    |
| array   | `array(N, elem)` value types                                                                                                                 |
| proc    | proc signature types                                                                                                                         |

The frontend (M2) translates Nim's typed `NimNode` into `CirType` exactly once, at the boundary.
Passes never re-derive types and never consult Nim's type system again.
Every typed operation node states its type in the Leng discipline (`(add Type a b)`), comparisons state their operand type and yield `bool`.

The `CirType` node keeps the origin `NimNode` (`origin` field) for fidelity and debugging.
The field is macro-time only: hand-built fixtures leave it nil, and no pass reads it.

## Alternatives considered

- **Nim's type system as the IR's types** (shady's choice): impossible outside the macro boundary.
  Pass-produced nodes have no typed `NimNode`, and re-deriving is exactly the double type-derivation cost the tracer removes
  (crucible today re-derives types twice, 495 + 357 lines).
- **Reusing the existing codegen `GpuType`**: couples the new IR to the old pipeline vocabulary and its 40+ kinds.
  CIRv2 must be self-contained and Leng-vocabulary-shaped.
- **NIF/Nimony Final IR directly**: no fork. The compiler runs in Nim v2 macros holding `NimNode`.
  There is no vendored NimNode→NIF translator, and serialization is nimony's own concern (C15D05 addendum 2).

## Consequence

- `sameType` is structural on the lattice (nominal structs compare by name),
  so the verifier can check `asgn` side agreement, call argument agreement,
  and literal fit without Nim's typechecker.
- Address spaces are CIRv2's own enum (`CirAddrSpace`), value-compatible by order with the codegen `AddressSpace`.
  No deep import of codegen internals reaches into cirv2, and none reach out of it.
- The lattice is closed: a construct the lattice cannot express (a raw `emit`, an oversized case selector) is a loud refusal.
  The refusal is a verifier or shape error, never a silent approximation.
