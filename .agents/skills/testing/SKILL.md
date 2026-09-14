---
name: testing
description: How tests are written in this repo — fixture suites, analytic suites, the assert surface, the linters
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: testing
---

## What I do

I define how tests are written in this repo:

- the two test kinds (fixture-replay, analytic) and when each applies
- the assert surface and the suite shape
- where fixtures come from, how a new model gets its test tree
- the linters that verify all of it before anything commits

## When to use me

Use this skill when doing any of these:

- writing or extending a test under `workspace/*/tests/`
- adding a new model, quantization scheme, or layer ([ARCHITECTURE.md](../../workspace/transformers/ARCHITECTURE.md))
- recording a new fixture family ([FIXTURE_GENERATION.md](../../workspace/transformers/tests/testgen/FIXTURE_GENERATION.md))

Related docs, each with its own job:

| doc | job |
|---|---|
| [`tests/README.md`](../../workspace/transformers/tests/README.md) | test tree, check ladder, the stakes |
| [`tests/harness/README.md`](../../workspace/transformers/tests/harness/README.md) | assert API: assertStats, assertArgMax, the record |
| [`ARCHITECTURE.md` Extension Points](../../workspace/transformers/ARCHITECTURE.md) | where a new model or layer plugs in, test side included |
| [`FIXTURE_GENERATION.md`](../../workspace/transformers/tests/testgen/FIXTURE_GENERATION.md) | recording: tiers, payload rules, provenance |
| [`writing-docs` skill](../writing-docs/SKILL.md) | doc comments, together with the global writing-code-doc skill |

## Test kinds

| kind | expected side | home |
|---|---|---|
| fixture-replay | recorded sidecar statistics + decision records, produced once by the python reference | `q_bf16/`, `q_exl3/` |
| analytic | truth computed in-process: closed forms, exact order statistics, in-process reference ops | `kvcache/`, `samplers/`, `batch_invariance/` |

A fixture-replay suite can never compare bit-exactly.

Honest rounding between recording and replay is the checked quantity.

An analytic suite compares against values computed in the same process,
where bit-exact is available and expected whenever the math is exact.

## Assert surface

Two functions, that is all, full contract in [the harness README](../../workspace/transformers/tests/harness/README.md).

- `assertStats(actual, record, kind)` checks the statistical property:
  quantiles, histogram, boundary elements, mean and tail bands
- `assertArgMax(actual, record)` checks the decision: argmax id,
  top-k pair, margin, tail

`kind` names a statistical property, never an operation:

| kind | error model |
|---|---|
| `kElementwise` | reduction-free sequence, tight band, zero mismatch |
| `kReduction` | one accumulation point, moderate band, small mismatch fraction, histogram L1 |
| composed chains | `kReduction` plus the depth argument |

Ban list, enforced by the linters:

- ad-hoc `rtol` or `atol` at call sites, the band comes from the kind
- raw `doAssert` for value comparisons in suites
- `try`/`except`/`discard` that eats a failed check
- `check*` enforcement procs outside `tests/harness/`
- PASS emission and verdict printing, a failed assert raises, rc=0 is green

## Suite shape

- one file, one flat `main`, setup then asserts, no framework sections
- every random tensor is seeded or replaced by a closed form
- `const` blocks carry filepaths only, model geometry arrives over
  `config.json` through the loader, never a literal
- setup helpers import `tests/layer_utils.nim` for layer setup,
  `tests/stateful_utils.nim` for the stateful context
- ulp math imports `tests/ulp_utils.nim`
- headers carry the run command and the contract, never the journey
- `std/unittest` (`suite`/`test`/`check`) is retired for suites,
  existing conversions remove it

## Adding a new model, the short version

1. the model module + registry entry ([ARCHITECTURE.md](../../workspace/transformers/ARCHITECTURE.md#extension-points))
2. one generator per fixture tier, named `gen_<dtype>_<model>_<NN>_<tier>.py`
3. record the families ([`FIXTURE_GENERATION.md`](../../workspace/transformers/tests/testgen/FIXTURE_GENERATION.md))
4. the suites per tier, per the suite-shape rules above
5. register the granular task inside `config.nims`

## Linters, run before commit, always

| linter | checks |
|---|---|
| `.agents/skills/writing-docs/tools/lint_docs.py` | doc comments everywhere |
| `tests/linters/lint_gen_scripts.py` | generators: docs, config over consts, entry shape, directory allowlist |
| `tests/linters/lint_fixtures.py` | fixtures: size caps, dir tiers, symlinks, record schema, provenance |
| `tests/linters/lint_nim_fixtures_consumers.py` | suites: assert allowlist, flat main, consts, shared helpers, docs |

Each linter header carries its rules table, the golden doc-comment
rules, and pointers to both doc skills.

Read them, they are always forgotten.

A finding is fixed, never silenced and never widened into acceptance.

## Nim mechanics that bite (libtorch FFI)

- wrap test code in procs: module-scope `TorchTensor` variables fail
  compilation, `cppNonPod` types reject brace initialization
- every branch of a `case` must assign `result`
- a parameter shadowing an accessed field is a compile error, rename it
- import `workspace/libtorch_testutils` for tensor test utilities
- `runCppTest` stays for the transitional suites only

## Kernel tests against a reference

A kernel test against a reference has three parts, so a reader always
sees which side is which:

1. a proc that computes with the function under check
2. a proc that computes the reference: an independent implementation,
   libtorch math over the same rounded inputs, or a closed form
3. the comparison, naming both sides at the assert

Rules:

- name the two results `actual` and `expected` at the assert site
- both sides derive from the same seeded inputs, rounding applied
  identically, never round the reference from the kernel's output
- a Nim reimplementation of the kernel's arithmetic as the reference
  is banned, the duplication can carry the same misunderstanding twice
- a reference sharing the kernel's code path proves nothing, it must
  be able to diverge
- data-preparation helpers that mirror a storage format are allowed,
  once, in a shared helper, never copied per test
- the ceiling for a kernel test is around 80 lines
