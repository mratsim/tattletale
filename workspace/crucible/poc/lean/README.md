# Lean frontend POC (C10D09): Lean toolchain, hand-written add kernel, Nim-to-Lean emitter

Verification carrier POC: compile Nim GPU kernels to Lean 4 as a verification target.

- M1 is hand-written: the crucible add kernel's per-thread body is modelled
  and proved in `AddKernel.lean`, no Nim frontend involved
- M2 adds `nim_to_lean.nim`, an emitter over crucible's post-pass GpuAst
  producing the same module
- M3 adds the verification pragmas (`lean_pragmas.nim`) and their extraction
  into Lean spec scaffolds with a proven/unproven extraction manifest

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

## Pragma test (P3, run from `workspace/crucible`)

```bash
nim c -r --hints:off --warnings:off \
  -o:build/wip/t_p3_pragmas --nimcache:nimcache/wip poc/lean/t_p3_pragmas.nim
```

The test applies the three pragmas (defined in `lean_pragmas.nim`, poc-local)
to the tracer add kernel via `irVulkanLean` and emits `generated/add_spec.lean`,
the pragma-free module plus the three spec scaffolds.

Elaboration runs with `lake env lean -DwarningAsError=true`.
The guard invariant and the add-shape postcondition close, the module is sorry-free.

A wrong postcondition (`output tid = a tid + b tid + 1`) and a wrong invariant (`tid ≤ n`) fail their proof attempt.
The failed proof becomes a `sorry`-backed scaffold, recorded unproven in the manifest JSON
beside the module, and fails the warningAsError elaboration.

Out-of-scope payloads are refused loudly: unparseable payload, unknown identifier,
wrong precondition shape, buffer-touching invariant, unknown pragma name.

Verification pragmas are captured at proc level only: a statement-level
`{.invariant: ...}.` inside the kernel body is refused loudly.

Crucible's frontend discards statement pragmas, so capturing statement-level
attachment would need a re-emission layer outside the tracer's scope.

A precondition is an assumption, so it carries no Lean declaration.
It is emitted as a comment and appears as the hypothesis `h` in the scaffolds.

The manifest lives beside the module (`add_spec.manifest.json`), one entry per pragma:

| field | content |
|------|------|
| `kind` | precondition, invariant or postcondition |
| `payload` | the pragma payload verbatim |
| `name` | the emitted theorem name, null for the precondition |
| `proven` | `true`, `false`, or null for the assumed precondition |
| `error` | Lean error excerpt, null when proven |

## Files

| File | Role |
|------|------|
| `lean-toolchain` | pinned `leanprover/lean4:v4.28.0` |
| `lakefile.lean` | package + lib with explicit roots (no glob) |
| `LeanPoc.lean` | P0 smoke: theorem + `example` + `#eval` |
| `AddKernel.lean` | P1: per-thread add kernel model + postcondition theorems |
| `nim_to_lean.nim` | P2+P3: GpuAst → Lean emitter (`irVulkan`/`irVulkanLean`) + pragma scaffold extraction |
| `lean_pragmas.nim` | P3: `{.precondition.}`/`{.invariant.}`/`{.postcondition.}` + payload grammar |
| `t_p2_emit.nim` | P2 test: emission + elaboration + statement match |
| `t_p3_pragmas.nim` | P3 test: pragma extraction, proven/unproven manifest, refusals |
| `Negative.lean` | deliberately false claim, Lean must reject it |

Known check hazard: a `sorry` only warns (exit 0). Any automated check
must build with `set_option warningAsError true`,
or grep for `declaration uses 'sorry'`.
