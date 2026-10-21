# Lean frontend POC (C10D09): toolchain gate + hand-translated add kernel

Verification carrier POC: compile Nim GPU kernels to Lean 4 as a verification target.
M1 is hand-written (no Nim frontend yet).
The crucible add kernel's per-thread body is modelled and proved in `AddKernel.lean`.

## Check commands

```bash
lake build                    # must exit 0 (P0 + P1 modules)
lake env lean Negative.lean   # must exit nonzero (negative control rejects)
```

`lake env lean` is required: plain `lean Negative.lean` cannot find the built `.olean` files.
`Negative.lean` is deliberately outside the lake glob, so the library build stays green.
The negative check runs the file directly and expects a rejection.

## Files

| File | Role |
|------|------|
| `lean-toolchain` | pinned `leanprover/lean4:v4.28.0` |
| `lakefile.lean` | package + lib with explicit roots (no glob) |
| `LeanPoc.lean` | P0 smoke: theorem + `example` + `#eval` |
| `AddKernel.lean` | P1: per-thread add kernel model + postcondition theorems |
| `Negative.lean` | deliberately false claim, Lean must reject it |

Known check hazard: a `sorry` only warns (exit 0). A future automated check
must build with `set_option warningAsError true`,
or grep for `declaration uses 'sorry'`.
