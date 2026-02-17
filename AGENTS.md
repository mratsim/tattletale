# Tattletale Agent Guidelines

This document provides guidelines for AI agentic coding tools operating in the Tattletale repository.

## Project Overview

Tattletale is an AI inference framework primarily written in Nim (targeting C++ backend) with Python for test vector generation. The project uses libtorch for tensor operations and implements tokenizers, safetensors loaders, and transformer models.

## Build, Test, and Lint Commands

### Running Tests
```bash
# Test all modules
nim test_libtorch
nim test_safetensors
nim test_transformers
nim test_toktoktok

# Run a single test file (use --build and --nimcache to avoid polluting directory)
nim cpp -r --verbosity:0 --hint:off --warnings:off \
  --build:build/tests/test_name --nimcache:nimcache/tests/test_name \
  workspace/path/to/test_file.nim
```

### Building
```bash
nim make_pytoktoktok    # Build Python extension
nim download_test_tokenizers
nim install_libtorch
```

### Python Testing
Always use `uv run` for Python testing:
```bash
uv run --group test-vectors python workspace/module/tests/testgen/generate_vectors.py
```
NEVER use `--break-system-packages` flag.

### Code Analysis: Prefer Compilation over `nim check`

Use `nim cpp` (compilation) rather than `nim check` for code analysis. The C backend misreports C++ exceptions: it says "only ref objects can raise" but C++ exceptions ARE NOT ref object and CAN be caught with C++ backend.

## Code Style Guidelines

### General Principles
- **Nim 2.2.0+ required** - Use modern Nim features
- **C++ backend** - All code compiles with `nim cpp`
- **Comments are welcome** - Do not delete comments unless wrong or outdated. Technical explanations, algorithmic rationale, and edge case reasoning should be preserved. For tensor-heavy code, include expected shapes in comments to ease debugging.

### File Organization
- Re-export modules at package root: `workspace/mylib/mylib.nim` exports `src/mylib`
- Tests in `workspace/module/tests/` directory
- Fixtures in `workspace/module/tests/fixture/`

### Import Convention
```nim
# Standard library - use std/ prefix
import std/tables
import std/os
import std/sequtils  # Required for mapIt

# Third-party packages - use pkg/ prefix
import pkg/jsony

# First-party workspace libraries - use workspace/ prefix
import workspace/toktoktok
import workspace/safetensors

# Local import (relative)
import ./serialization
```

### Naming Convention
- **Types**: PascalCase, `*` for public (e.g., `BPETokenizer*`, `Dtype*`)
- **Procedures/Functions**: camelCase, `*` for public (e.g., `compilePcre2*`)
- **Constants**: PascalCase (e.g., `MaxInt`, `MAX_HEADER_SIZE`)
- **Variables**: camelCase (e.g., `errorCode`, `filename`)

### Type System
- Use `{.final.}` for sealed types when appropriate
- Prefer value types over ref types for performance
- Use `distinct` for type safety
- Use `enum` for type-safe constants
- For FFI types (TorchTensor), use value type, avoid brace initialization

### Error Handling
- Use exceptions primarily for unrecoverable errors (file not found, invalid format)
- Document the error model in the file header (e.g., "raises on invalid format, returns result on success")
- Use `result` return pattern for fallible operations
- Use `newException(Type, message)` for raising

### C++ Exceptions

C++ exceptions (like `torch::Error`) cannot be caught by `std/unittest`. Use custom test runner with `catchTorchExceptions` template (see `workspace/transformers/tests/common_utils.nim`):
```nim
proc runTest*(name: string, body: proc(): bool) =
  let passed = catchTorchExceptions(body())
  if passed: echo "✅ PASS | ", name
  else: echo "❌ FAIL | ", name
```
Import C++ exceptions:
```nim
type
  CStdException {.importcpp: "std::exception", header: "<exception>", inheritable.} = object
  CRuntimeError {.importcpp: "std::runtime_error", header: "<stdexcept>".} = object of CStdException
proc fn() =
  try: raise initRuntimeError("foo2")
  except CStdException as e:
    doAssert e is CStdException
  # Note: getCurrentException() not available for imported exceptions
```

### Code Patterns
**Testing (Critical)**: Wrap test code in a proc to avoid C++ brace init with FFI types:
```nim
import std/unittest, workspace/libtorch
proc runTests*() =
  suite "tensor tests":
    test "generate tensor":
      let tensor = arange(10, kFloat32)
      check tensor.numel() == 10
when isMainModule:
  runTests()
```
**Never** declare module-level variables with FFI types.

**Case Statements**: Every branch must assign to `result`:
```nim
proc foo(x: int): int =
  case x
  of 1: result = 10
  of 2: result = 20
  else: result = 0
```

### Resource Management
- Use `defer` for cleanup (files, handles):
  ```nim
  var mf = memfiles.open(fixturePath, mode = fmRead)
  defer: mf.close()
  ```
- Use `=destroy` hook for FFI resource cleanup:
  ```nim
  proc `=destroy`(code: Pcre2Code) =
    if code.code != nil:
      code_free(code.code)
  ```

### Test Organization
- **CI tests**: Files in `tests/` starting with `test_` or `t_`
- **Manual tests**: Files starting with `manual_test_` (require multi-GB model downloads, not run in CI)
- Use `std/unittest` with `suite`, `test`, `check` OR custom `runTest` for C++ exceptions
- Define test constants as `const` at module level
- Reference fixture using `const FIXTURES_DIR = currentSourcePath().parentDir() / "fixture"`

### Working with Tests
- **Do not change tests to make them pass** unless there is overwhelming evidence the test is wrong. Create a Python verification script to generate that evidence.
- **Commenting out tests to focus on something else is fine**, but you MUST re-enable them before finishing.
- **Claims need proof** - Don't oversell features completion or quality. Use nuance like "should work" or "may need verification" when uncertain. Compiling is the bare minimum; working is demonstrated with tests.

### Common Pitfalls
1. Missing `import std/sequtils` when using `mapIt`
2. Using wrong workspace import path (use `workspace/module`)
3. Module-level FFI variable declarations causing C++ brace init error
4. Forgetting `result =` in case branches
5. Parameter shadowing field access
6. Using `nim check` instead of `nim cpp` - gives false warning about C++ exceptions
7. Using `python` instead of `uv run` for Python scripts

### License Header
Include in new files:
```nim
# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
```
