---
name: python-interop
description: Nim to Python interoperability including nimpy for calling Python from Nim and exporting Nim to Python, nimporter for packaging Nim modules as Python packages, and cffi/ctypes for calling Nim from Python
---

# Nim to Python Interoperability Skill

This skill covers the main approaches for integrating Nim with Python: nimpy (call Python from Nim and export Nim to Python), nimporter (package Nim as Python modules), and cffi/ctypes (call Nim from Python).

## Overview

There are two primary integration directions:

1. **Nim → Python**: Use nimpy to call Python libraries from Nim code
2. **Python → Nim**: Use nimpy with `{.exportpy.}` to expose Nim functions to Python, or use nimporter for packaging

Common use cases:
- Extending Nim with Python scientific computing (Scipy, Numpy)
- Adding scripting capability to Nim applications
- Speeding up Python code by rewriting hot paths in Nim
- Packaging Nim modules for Python distribution

---

## nimpy: Calling Python from Nim

nimpy allows calling Python code and libraries directly from Nim. It provides ABI compatibility - compiled modules don't depend on a particular Python version.

### Basic Usage

```nim
import nimpy

let os = pyImport("os")
echo "Current dir is: ", os.getcwd().to(string)

# sum(range(1, 5))
let py = pyBuiltinsModule()
let s = py.sum(py.range(0, 5)).to(int)
assert s == 10
```

### Type Conversion

- **Nim → Python**: Automatic via Nimpy templates
- **Python → Nim**: Manual conversion using `.to(T)` API

```nim
let np = pyImport("numpy")
let arr = np.array(@[1.0, 2.0, 3.0].toNdArray)
discard py.print(arr)
```

---

## nimpy: Exporting Nim to Python

nimpy can export Nim functions as a Python module. This is how you create Python extensions in Nim.

### Basic Export

```nim
# mymodule.nim - filename MUST match the module name
import nimpy

proc greet*(name: string): string {.exportpy.} =
  return "Hello, " & name & "!"
```

### Compilation

```bash
# Windows:
nim c --app:lib --out:mymodule.pyd --threads:on --tlsEmulation:off --passL:-static mymodule

# Linux/macOS:
nim c --app:lib --out:mymodule.so --threads:on mymodule
```

### Using from Python

```python
# test.py
import mymodule
assert mymodule.greet("world") == "Hello, world!"
assert mymodule.greet(name="world") == "Hello, world!"
```

### Exporting Nim Types as Python Classes (Experimental)

```nim
# mymodule.nim
type TestType = ref object of PyNimObjectExperimental
  myField: string

proc setMyField*(self: TestType, value: string) {.exportpy.} =
  self.myField = value

proc getMyField*(self: TestType): string {.exportpy.} =
  self.myField

proc newTestType*(): TestType {.exportpy.} =
  TestType()
```

```python
# test.py
import mymodule
tt = mymodule.newTestType()
tt.setMyField("Hello")
assert tt.getMyField() == "Hello"
```

### Type Mapping for Exports

| Nim Type | Python Type |
|----------|-------------|
| int | int |
| float | float |
| string | str |
| seq[T] | list |
| tuple | tuple |
| bool | bool |
| ref object | Python class |

---

## nimporter: Packaging Nim as Python Modules

nimporter builds on nimpy to provide seamless import and packaging for distribution.

### Installation

```bash
pip install nimporter
```

### Quick Start

```python
# main.py
import nimporter  # Must import before Nim modules
import mymodule   # Compiled automatically!

result = mymodule.greet("world")
```

### Two Concepts: Modules vs Libraries

**Extension Modules** (simple, direct import):
- Single `.nim` file
- No dependencies other than nimpy
- Cannot customize compiler switches
- Cannot import other Nim modules in same directory

**Extension Libraries** (full project):
- Folder with `libname.nim`, `libname.nim.cfg`, `libname.nimble`
- Can have external Nim dependencies
- Full folder structure supported
- CLI switches can be customized

### Library Folder Structure

```
mylibrary/
    mylibrary.nim      # Must be present
    mylibrary.nim.cfg  # Must be present (can be empty)
    mylibrary.nimble   # Must contain `requires "nimpy"`
```

### Distribution

```python
# setup.py
import setuptools
from nimporter import get_nim_extensions, WINDOWS, MACOS, LINUX

setuptools.setup(
    name='mylibrary',
    install_requires=['nimporter'],
    py_modules=['mylibrary.py'],
    ext_modules=get_nim_extensions(platforms=[WINDOWS, LINUX, MACOS])
)
```

```bash
# Source distribution (end users need Nim compiler)
python setup.py sdist

# Binary distribution
python setup.py bdist_wheel
```

### CLI Commands

```bash
nimporter clean      # Remove cached builds
nimporter compile    # Precompile all extensions
nimporter list       # List detected extensions
```

### Docker Usage

```bash
# Precompile for Docker (no Nim needed in container)
nimporter compile
```

Ensure `__pycache__` directories are included in Docker image.

---

## cffi: Calling Nim from Python

cffi provides a Python interface to call compiled Nim shared libraries.

### Nim Side: Export Functions

```nim
# called.nim
proc nim_add*(num1: int, num2: int): int {.exportc.} =
    return num1 + num2
```

Compile as a shared library:
```bash
nim c --app:lib called.nim
# Creates libcalled.so (or .pyd on Windows, .dylib on macOS)
```

### Python Side: Call via cffi

```python
from cffi import FFI

ffi = FFI()

ffi.cdef("""
    int nim_add(int num1, int num2);
""")

lib = ffi.dlopen("./libcalled.so")
result = lib.nim_add(5, 10)
print(result)  # 15
```

### Type Mapping (Nim to C)

| Nim Type | C Type |
|----------|--------|
| int | long (c_long) |
| int8 | int8_t |
| int16 | int16_t |
| int32 | int32_t |
| int64 | int64_t |
| uint | unsigned long |
| float | double |
| cstring | char* |
| ptr T | T* |

---

## ctypes: Calling Nim from Python

ctypes is Python's built-in FFI library. Similar to cffi but uses stdlib only.

### Nim Side: Export with Dynamic Library

```nim
# partitions.nim
proc partitions*(cards: var array[0..9, int], subtotal: int): int {. exportc, dynlib .} =
    var total: int
    result = 0

    for i in 0..9:
        if cards[i] > 0:
            total = subtotal + i + 1
            if total < 21:
                result += 1
                cards[i] -= 1
                result += partitions(cards, total)
                cards[i] += 1
            elif total == 21:
                result += 1

    return result
```

Compile:
```bash
nim c --app:lib --dynlib:yes partitions.nim
```

### Python Side: Call via ctypes

```python
#!/usr/bin/env python
from ctypes import *
import os

lib = cdll.LoadLibrary(os.path.abspath("libpartitions.so"))
lib.partitions.argtypes = [POINTER(c_long), c_long]
lib.partitions.restype = c_long

deck = [4] * 9
deck.append(16)

for i in range(10):
    deck[i] -= 1
    p = 0
    for j in range(10):
        deck[j] -= 1
        nums_arr = (c_long * len(deck))(*deck)
        n = lib.partitions(nums_arr, c_long(j + 1))
        deck[j] += 1
        p += n
    print(f'Dealer showing {i} partitions = {p}')
    deck[i] += 1
```

---

## Comparison Reference

| Feature | nimpy | nimporter | cffi | ctypes |
|---------|-------|-----------|------|--------|
| Direction | Both | Nim→Python | Python→Nim | Python→Nim |
| Dependencies | nimpy | nimporter | cffi | stdlib |
| Ease of Use | Medium | Easy | Medium | Medium |
| Performance | Native | Native | Native | Native |
| Distribution | Manual | Wheels/Source | Source | Source |
| Type Safety | Nim | Nim | Manual | Manual |
| ABI Stable | Yes | Yes | N/A | N/A |

### When to Use Which

- **nimpy (call Python)**: Need Python libraries (Numpy, Scipy) in Nim application
- **nimpy (export to Python)**: Create Python extension in Nim
- **nimporter**: Distribute Nim code to Python users with easy packaging
- **cffi**: Simple Nim→Python calls, want lightweight solution
- **ctypes**: stdlib-only solution, no extra dependencies

---

## Common Patterns

### Performance Optimization Pattern

```python
# Identify hot path in Python
def slow_function():
    for i in range(1000000):
        # compute-intensive work

# Rewrite in Nim with {.exportpy.}, package with nimporter
```

### Scientific Computing Pattern (nimpy)

```nim
import nimpy
import arraymancer
import scinim/numpyarrays  # For efficient numpy interop

let np = pyImport("numpy")
let scipy = pyImport("scipy")

# Use numpy/scipy directly
let result = scipy.special.gamma(nim_array.toNdArray)
```

### Scripting Pattern

```nim
import nimpy

proc calculate*(x: float): float {.exportpy.} =
  result = x * 2.0
```

---

## Troubleshooting

### nimpy Import Error

If you get `dynamic module does not define module export function`:
- Ensure the Nim file name matches the Python module name exactly

### nimpy libpython Not Found

```bash
pip install find_libpython
python3 -c 'import find_libpython; print(find_libpython.find_libpython())'
```

Then set `nimpy.py_lib.pyInitLibPath`.

### GC Issues with Multiple Modules

For multiple nimpy modules, consider moving Nim runtime to a separate shared library. See [Nim docs on DLL generation](https://nim-lang.org/docs/nimc.html#dll-generation).

### Windows Threads with MinGW

Use `--tlsEmulation:off` and link statically with `--passL:-static` on Windows.

---

## Installation Quick Reference

```bash
# Nim compiler
# https://nim-lang.org/install.html

# Python packages
pip install nimporter    # For packaging Nim as Python modules
pip install cffi         # For cffi approach (optional, ctypes is stdlib)
pip install find_libpython  # For debugging libpython issues

# Nim packages
nimble install nimpy    # For calling Python from Nim and exporting to Python
```

---

## Resources

- [nimpy GitHub](https://github.com/yglukhov/nimpy)
- [nimporter GitHub](https://github.com/Pebaz/Nimporter)
- [Nim for Python Programmers](https://github.com/nim-lang/Nim/wiki/Nim-for-Python-Programmers)
- [SciNim/numpyarrays](https://github.com/SciNim/scinim) for performance-critical numpy interop
