# Lean frontend POC (C10D09): Lean toolchain, hand-written add kernel, Nim-to-Lean emitter

Verification carrier POC: compile Nim GPU kernels to Lean 4 as a verification target.

- M1 is hand-written: the crucible add kernel's per-thread body is modelled
  and proved in `AddKernel.lean`, no Nim frontend involved
- M2 adds `nim_to_lean.nim`, an emitter over crucible's post-pass GpuAst
  producing the same module

## Check commands

```bash
lake build                    # must exit 0 (P0 + P1 modules)
lake env lean Negative.lean   # must exit nonzero (negative control rejects)
```

`lake env lean` is required: plain `lean Negative.lean` cannot find the built `.olean` files.
`Negative.lean` is deliberately outside the lake glob, so the library build stays green.
The negative check runs the file directly and expects a rejection.

## Emitter test (P2, run from `workspace/crucible`)

```bash
nim c -r --hints:off --warnings:off \
  -o:build/wip/t_p2_emit --nimcache:nimcache/wip poc/lean/t_p2_emit.nim
```

The test:

- emits `generated/add.lean` from the tracer kernel's post-pass GpuAst
- elaborates it with `lake env lean -DwarningAsError=true generated/add.lean`
- checks the emitted statements against the hand-written P1 statements,
  byte-identical

`generated/` is gitignored: it is a build artifact the test regenerates.

## Files

| File | Role |
|------|------|
| `lean-toolchain` | pinned `leanprover/lean4:v4.28.0` |
| `lakefile.lean` | package + lib with explicit roots (no glob) |
| `LeanPoc.lean` | P0 smoke: theorem + `example` + `#eval` |
| `AddKernel.lean` | P1: per-thread add kernel model + postcondition theorems |
| `nim_to_lean.nim` | P2: GpuAst → Lean emitter (`irVulkan` deep-import pattern) |
| `t_p2_emit.nim` | P2 test: emission + elaboration + statement match |
| `Negative.lean` | deliberately false claim, Lean must reject it |

Known check hazard: a `sorry` only warns (exit 0). Any automated check
must build with `set_option warningAsError true`,
or grep for `declaration uses 'sorry'`.
