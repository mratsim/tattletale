# Tattletale Agent Guidelines

AI coding tool guidelines for Tattletale repo.

## Project

AI inference library in Nim (C++ backend) wrapping libtorch. Tensor ops, safetensors loading, tokenizers, transformer layers.

Python only for test-vector generation (`uv run`). Dev: prefer PyTorch from `uv` / `.venv` (Python 3.14 per `pyproject.toml`) for reference vectors, not system Python.

## Build / Test / Lint

### Dependencies (from `config.nims`)

```bash
nim install_deps          # runtime: nimpy, jsony, stew, packedjson, iface
nim install_deps_dev      # dev: zip (vendor libtorch), chronos (download test tokenizers)
```

### Tests

```bash
nim test_libtorch
nim test_safetensors
nim test_transformers
nim test_toktoktok
```

Single file:
```bash
nim cpp -r --verbosity:0 --hints:off --warnings:off \
  --outdir:build/tests/test_name --nimcache:nimcache/tests/test_name \
  workspace/path/to/test_file.nim
```

### Vendoring

```bash
nim install_libtorch          # download libtorch (needs zip)
nim download_test_tokenizers  # gpt-2 + llama3 fixtures (needs chronos)
nim make_pytoktoktok          # build pytoktoktok.so for Python import
```

### Python

Always `uv run`. `.venv` managed by `uv`:

```bash
uv run --group test-vectors python workspace/module/tests/testgen/generate_vectors.py
```

NEVER `--break-system-packages`.

### Code Analysis

`nim cpp` only. `nim check` in C mode lies about C++ exceptions. Hard error on libtorch imports:
> "You are running 'nim check' in C mode. It will misreport that C++ exceptions can't be caught because they aren't ref objects."

## Architecture: Tensor Layer

### `Tensor` — Ref Wrapper

`TorchTensor` (raw `importcpp: "torch::Tensor"`) never public. Wrapped in `ref object` `Tensor` (`tensors.nim:29`):

```nim
type Tensor* = ref object
  raw: TorchTensor
```

Avoids Nim C++ FFI issues with value types in containers (copy-ctor, `=wasMoved`, `{}` default init).

**Contract:** No C++ types (`TorchTensor`, `CppVector`, `CppTuple`, `IntArrayRef`, `ArrayRef`, `CppString`) leak past `tensors.nim` / `tensors_nn.nim`. `workspace/libtorch` (`libtorch.nim:1-2`) only re-exports those two.

### `wrapLibtorch:` Macro — Auto-Forwarding

`tensors.nim:177`. Inside `wrapLibtorch:` block, every `proc`/`func` gets:

1. `{.inline.}` auto-added
2. C++ exceptions → `LibTorchDefect` via `convertLibTorchExceptions` (`tensors.nim:71`)
3. Return wrapped via `wrapTorchTensor` (`tensors.nim:43`) — no-op for non-Tensor returns
4. Inputs unwrapped via `unwrapArg` (`tensors.nim:121`) — `Tensor` → `.raw`, `varargs[int]` → `asTorchView`, `typedesc` → `toScalarKind`

**Bare signatures** (~95% of procs). `autoForward` (`tensors.nim:138`) generates forwarding body:

```nim
wrapLibtorch:
  func zeros*(size: varargs[int]): Tensor
  func zeros*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor
  func dim*(a: Tensor): int
```

`autoForward` generates: `{.inline.} convertLibTorchExceptions: wrapTorchTensor: F.zeros(asTorchView(size))`

**Explicit bodies** bypass auto-forward. Write wrapping manually (`tensors.nim:505`):

```nim
func `<.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
  convertLibTorchExceptions:
    wrapTorchTensorImpl:
      F.lt(a.raw, b.raw)
```

Templates always explicit (`tensors.nim:392`):

```nim
template sizes*(a: Tensor): openArray[int] =
  F.sizes(a.raw).asNimView()
```

**`wrapTorchTensor`** (`tensors.nim:43`): `when typeof(...)` dispatch — `TorchTensor` → `Tensor`, `CppVector[TorchTensor]` → `seq[Tensor]`, `CppTuple2` → tuple. No-op else.

### C++ Exception Handling

`LibTorchDefect` inherits `Defect` (`tensors.nim:69`). Keeps `raises: []` clean.

`convertLibTorchExceptions` (`tensors.nim:71`) catches `TorchError` (`c10::Error`), re-raises as `LibTorchDefect` with `.what()` message.

`LibTorchDefect` exported — test harnesses can `except LibTorchDefect`.

### Module Structure (libtorch)

```
workspace/libtorch/libtorch.nim        — re-export: tensors + tensors_nn [line 1-2]

PUBLIC (no C++ types leak):
  src/tensors.nim          — Core wrapper (Tensor + ~500 procs)
  src/tensors_nn.nim       — NN functional (activations, losses, SDPA)
  src/tensors_py.nim       — Nim ↔ Python Tensor bridge

INTERNAL (not exported):
  src/raw/abi/torch_tensors.nim   — Pure C++ FFI (~265 procs)
  src/raw/abi/neural_nets.nim     — C++ FFI neural net API
  src/raw/abi/c10.nim             — TorchError, IntArrayRef
  src/raw/abi/std_cpp.nim         — CppStdException, what()
  src/raw/torch_tensors_sugar.nim — asTorchView, indexing macros
  src/raw_libtorch.nim            — Re-exports raw modules
```

## Testing

### `runTest` from `libtorch_testutils`

Use for ALL tests. Catches C++ exceptions. `quit(1)` on failure.

```nim
import
  workspace/libtorch/src/tensors,
  workspace/libtorch_testutils

proc runTests*() =
  runTest "my test":
    proc(): bool =
      let t = zeros(2, 3, kFloat32)
      doAssert t.dim() == 2
      doAssert t.size(0) == 2
      true

when isMainModule:
  runTests()
```

**NEVER `std/unittest`.** Cannot catch C++ exceptions. Happy-path-only tests still need `runTest` for error coverage.

### `runTest` API (`libtorch_testutils.nim`)

- `runTest(name, body)` (`:70`): PASS/FAIL print, `quit(1)` on fail
- `catchExceptions(body)` (`:26`): `true`/`false` return. Catches: `TorchError`, `CppStdException`, `LibTorchDefect`, `CatchableError`, `Defect`
- `assertAllClose(actual, expected, rtol, abstol)` (`:90`): Tensor tolerance comparison
- `assertShape(tensor, expectedShape)` (`:119`): Shape check
- `printTensor(t, label)` / `printTensorShape(t, label)` (`:159/167`): Debug
- `dataPtrHex(t)` / `shapePtrHex(t)` (`:186/190`): Pointer aliasing debug
- `traceExec(body)` (`:146`): Statement execution tracer

### Test Org

- CI: `tests/test_*.nim` or `tests/t_*.nim`
- Manual: `tests/manual_test_*.nim` (multi-GB models)
- Fixtures: `tests/fixtures/`
- Fixtures generator: `tests/testgen/`
- `const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures"`

### Test Discipline

- Don't change tests to pass. Proof needed. Python verification script if test wrong.
- Comment-out OK for focus. Re-enable before finish.
- Compile ≠ work. Tests prove work.

## Code Style

### General

- Nim 2.2.0+. C++ backend (`nim cpp`).
- **Comments are welcome** — Do not delete comments unless wrong or outdated. Technical explanations, algorithmic rationale, and edge case reasoning should be preserved. For tensor-heavy code, include expected shapes in comments to ease debugging.

### File Org

- Re-export at root: `workspace/mylib/mylib.nim` exports `src/mylib`
- Tests: `workspace/module/tests/`
- Fixtures: `workspace/module/tests/fixtures/`
- Tasks: `config.nims` only. **NEVER `.nimble` files.**

### Imports

```nim
import std/tables
import std/sequtils  # mapIt

import pkg/jsony

import workspace/libtorch             # Tensor + NN API
import workspace/libtorch as F        # Short alias
import workspace/libtorch/src/tensors # Direct (Tensor, LibTorchDefect)
import workspace/libtorch_testutils   # runTest, assertAllClose
import workspace/safetensors
import workspace/toktoktok

import ./internal {.all.}             # Friend modules: test internals
```

**libtorch imports:**
- `workspace/libtorch` — Public API
- `workspace/libtorch as F` — Short alias
- `workspace/libtorch/src/tensors` — Direct
- **NEVER** `src/raw/abi/` or `src/raw_liborch` from app code

**Friend modules:** `import ./internal {.all.}` exposes private symbols for testing. Use from sibling test modules only.

### Naming

- Types: PascalCase, `*` public (`BPETokenizer*`)
- Procs: camelCase, `*` public (`compilePcre2*`)
- Consts: PascalCase (`MaxInt`, `MAX_HEADER_SIZE`)
- Vars: camelCase (`errorCode`)

### Type System

- `{.final.}` for sealed types
- Value > ref for perf
- `distinct` for type safety
- `enum` for type-safe constants

### Error Handling

- Exceptions for unrecoverable (logic bug, key dependency missing, preconditions/invariant failures, ...)
- Document error model in headers
- `Result[T, Err]` pattern for fallible ops (bad user input, network connectivity, ...)
- `newException(Type, message)` for raising

### Patterns

**Case** — every branch assigns `result`:

```nim
proc foo(x: int): int =
  case x
  of 1: result = 10
  of 2: result = 20
  else: result = 0
```

**Resources:**

```nim
var mf = memfiles.open(path, mode = fmRead)
defer: mf.close()

proc `=destroy`(code: Pcre2Code) =
  if code.code != nil:
    code_free(code.code)
```

### Tensor Equality

| Type | Method | Desc |
|------|--------|------|
| Referential | `a == b` (Nim `ref`) | Same Nim ref |
| LibTorch same | `is_same(a, b)` | Same internal ptr |
| Value | `equal(a, b)` → `bool` | Same values + shape |
| Tolerance | `allClose(a, b, rtol, abstol)` → `bool` | Float comparison |
| Element-wise | `eq(a, b)` → `Tensor` | Bool Tensor |

No `==` overload. Referential = default `ref` behavior.

### Adding Wrappers

Inside existing `wrapLibtorch:` block. Two modes:

**Bare** (auto-forward, ~95%):

```nim
wrapLibtorch:
  func myNewOp*(a: Tensor, b: Tensor): Tensor
```

Generates: `{.inline.} convertLibTorchExceptions: wrapTorchTensor: F.myNewOp(a.raw, b.raw)`

**Explicit** (custom logic):

```nim
wrapLibtorch:
  func myComplexOp*(a: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensor:
        F.myComplexOp(a.raw, someComputedArg)
```

`tensors_nn.nim` uses same `wrapLibtorch:` (`:32`). Imports `./tensors {.all.}` for infra. `privateAccess(Tensor)` for `.raw` direct access.

### Common Pitfalls

1. Missing `import std/sequtils` for `mapIt`
2. Wrong workspace path (use `workspace/module`)
3. Module-level FFI vars → C++ brace init error
4. Missing `result =` in case branches
5. Shadowing `result` special variable
6. `nim check` instead of `nim cpp` → false C++ exception warnings
7. `python` instead of `uv run`
8. `std/unittest` → use `runTest` from `libtorch_testutils`
9. Import `TorchTensor` from app code → always `Tensor`
10. Missing `convertLibTorchExceptions:` in explicit `wrapLibtorch:` bodies

### License Header

```nim
# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
```
