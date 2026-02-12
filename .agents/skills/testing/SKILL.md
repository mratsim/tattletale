---
name: testing
description: Nim testing conventions, unittest framework, and C++ compatibility patterns
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: testing
---

## What I do

I provide guidance for writing tests in Nim that:
- Work correctly with the `std/unittest` framework
- Avoid C++ compilation errors with complex FFI types like `TorchTensor`
- Follow project conventions for test organization

## When to use me

Use this skill when:
- Writing new tests for any workspace module
- Debugging C++ compilation errors in test code
- Organizing test fixtures and test data

## The unittest framework

Nim's standard library provides a simple testing framework:

```nim
import std/unittest

suite "my module tests":
  test "addition works":
    check 1 + 1 == 2

  test "string handling":
    let result = "hello".toUpperAscii()
    check result == "HELLO"
```

**Key procs:**
- `suite(name, body)` - Group related tests
- `test(name, body)` - Define a single test
- `check(expr)` - Assert expression is true, prints failed value on failure
- `doAssert(expr)` - Like `check` but raises on failure (use for invariants)
- `submitTest(result)` - Submit test result from a procedure

## Critical pattern: Wrap tests in a proc

### The problem

When you declare variables at module scope (top-level) in Nim tests, the generated C++ code uses `= {}` initialization:

```cpp
TorchTensor expectedTensor = {};  // This fails!
expectedTensor = myFunction(a, b);
```

The C++ `torch::Tensor` type (and other FFI types with `cppNonPod`) does not accept brace initialization. This causes:
```
error: ambiguous overload for 'operator=' (operand types are 'at::Tensor' and '<brace-enclosed initializer list>')
```

### The solution

Always wrap test code in a `proc main()`:

```nim
import std/unittest, workspace/libtorch

proc generateTensor(): TorchTensor =
  # This works - Nim generates:
  # auto result = myFunction(a, b);
  arange(10, kFloat32)

proc runTests*() =
  suite "tensor tests":
    test "generate tensor":
      let tensor = generateTensor()
      check tensor.numel() == 10

when isMainModule:
  runTests()
```

This generates proper C++:
```cpp
auto tensor = generateTensor();  // No {} initialization
```

## Test organization

### Fixture file pattern

Tests that load files should follow this pattern:

```nim
import std/unittest, std/os, workspace/safetensors, workspace/libtorch

const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures"

proc main() =
  suite "safetensors loading":
    test "load fixture":
      let fixturePath = FIXTURES_DIR / "model.safetensors"
      check fileExists(fixturePath)

      var mf = memfiles.open(fixturePath, mode = fmRead)
      defer: mf.close()

      let (st, offset) = safetensors.load(mf)
      check st.tensors.len > 0

when isMainModule:
  main()
```

Key points:
- Use `currentSourcePath().parentDir() / "fixtures"` for fixture paths
- Use `memfiles.open` with `defer: mf.close()`
- Return early or use `continue` for missing fixtures

### Test constants

Define test parameters as `const` at module level:

```nim
const Patterns = ["gradient", "alternating", "repeating"]
const Shapes: array[4, seq[int64]] = [
  @[int64 8],
  @[int64 4, 4],
  @[int64 2, 3, 4],
  @[int64 3, 2, 2, 2]
]
const TestedDtypes = [F64, F32, F16, I64, I32, I16, I8, U64, U32, U16, U8]
```

### Helper procedures

Extract reusable logic into `proc` with `*` export:

```nim
proc generateExpectedTensor*(pattern: string, shape: seq[int64], dtype: ScalarKind): TorchTensor =
  let shapeRef = shape.asTorchView()
  let numel = shape.product()

  case pattern
  of "gradient":
    arange(numel, dtype).reshape(shapeRef).to(dtype)
  of "alternating":
    let flat = arange(numel, kInt64)
    let modVal = (flat % 2).to(kFloat64)
    modVal.reshape(shapeRef).to(dtype)
  else:
    raise newException(ValueError, "Unknown pattern: " & pattern)
```

Note: Each branch of a `case` must assign to `result`.

## Running tests

Each module has a task defined in `config.nims` for running its tests:

```bash
# Test toktoktok
nim test_toktoktok

# Test libtorch
nim test_libtorch

# Test safetensors
nim test_safetensors
```

The command `nim test_toktoktok` compiles and runs all test files in `workspace/toktoktok/tests/` that start with `test_` or `t_`.

### Compilation settings

The project uses:
- `--path:.` - Makes `workspace/module` imports work
- Tests compiled with: `nim cpp -r` plus flags for output and cache directories

### Fixture files

For this project, fixtures are in:
```bash
workspace/toktoktok/tests/tokenizers/
```

Reference fixtures using:
```nim
const FIXTURES_DIR = currentSourcePath().parentDir() / "tokenizers"
```

## Common errors and fixes

### "undeclared field" with parameter shadowing

If you have a parameter named `shape` and access a field `info.shape`:
```nim
proc generateExpectedTensor*(pattern: string, shape: seq[int64], ...): TorchTensor =
  for info in tensors:  # error: 'shape' shadows info.shape
    check info.shape == shape
```

Fix: Rename parameter to avoid shadowing:
```nim
proc generateExpectedTensor*(pattern: string, shapeSeq: seq[int64], ...): TorchTensor =
  for info in tensors:
    check info.shape == shapeSeq  # Now works
```

### Case statement not returning

Each branch of a case must explicitly assign to `result`:
```nim
proc foo(x: int): int =
  case x
  of 1: result = 10  # Must use 'result ='
  of 2: 20             # ERROR: doesn't assign!
```

## How to add a new test

### Step 1: Create the test file

Follow the naming convention: `test_*.nim` or `t_*.nim` in the module's `tests/` directory.

```nim
# workspace/my_module/tests/test_myfeature.nim

import std/unittest, std/os
import workspace/my_module

proc runMyFeatureTests*() =
  suite "my feature tests":
    test "basic functionality":
      let result = myModule.function()
      check result == expectedValue

when isMainModule:
  runMyFeatureTests()
```

### Step 2: Run the test

The test will be discovered automatically by the test command:

```bash
# If it's in my_module:
nim c -r --task:test_my_module
```

Or run all tests for the module:
```bash
nim test_my_module
```

### Step 3: Add test fixtures (if needed)

Create a `fixtures/` directory and add test data:

```bash
workspace/my_module/tests/fixtures/
```

Reference in test code:
```nim
const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures"
let fixturePath = FIXTURES_DIR / "test_data.bin"
```

### Checklist for new tests

- [ ] File name starts with `test_` or `t_`
- [ ] Located in `workspace/module/tests/`
- [ ] Test code wrapped in `proc runTests*()`
- [ ] Has `when isMainModule: runTests()` at the end
- [ ] Uses `defer` for resource cleanup (files, etc.)
- [ ] Helper procedures exported with `*`
- [ ] Constants defined at module level with `const`

## Python test vector generation

For AI/ML modules, test vectors are generated via Python scripts using torch and safetensors.

### Directory structure

```
workspace/module/
├── tests/
│   ├── test_module.nim          # Nim tests
│   ├── fixtures/                # Generated fixture files
│   │   ├── model.safetensors
│   │   └── tokenizer.json
│   └── testgen/                 # Python test vector generators
│       └── generate_vectors.py
```

### Convention

- **Python 3.12** standard for all test vector generation (matches vLLM/SGLang)
- **Single root `pyproject.toml`** with `[dependency-groups]` for shared dependencies:
  ```toml
  [dependency-groups]
  test-vectors = [
      "torch>=2.0.0",
      "safetensors>=0.7.0",
      "transformers>=4.40.0",
      "numpy>=2.4.2",
  ]
  ```
- **Run generators** with: `uv run --group test-vectors python workspace/module/tests/testgen/generate_vectors.py`

### Example testgen script

```python
import torch
import numpy as np
from safetensors.numpy import save_file
import os

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "fixtures",
)

def generate_vandermonde():
    x = torch.arange(1, 6, dtype=torch.float32)
    vandermonde = torch.vander(x, increasing=True).T
    return vandermonde.to(torch.bfloat16).view(torch.uint16).numpy()

def main():
    fixtures = {
        "BF16_vandermonde_5x5": generate_vandermonde(),
    }
    save_file(fixtures, os.path.join(FIXTURES_DIR, "vandermonde.safetensors"))
    print("Fixtures generated")

if __name__ == "__main__":
    main()
```

### Fixture regeneration

When adding new test vectors, regenerate the fixture files:

```bash
uv run --group test-vectors python workspace/module/tests/testgen/generate_vectors.py
```
