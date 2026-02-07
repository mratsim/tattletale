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

### Returning Complex Types (Tables, JsonNode)

nimpy supports returning Nim complex types that Python can use:

```nim
import nimpy, tables, json

proc getTable*(): Table[string, int] {.exportpy.} =
  result = {"Hello": 0, "SomeKey": 10}.toTable

proc getJsonAsDict*(): JsonNode {.exportpy.} =
  result = %*{
    "SomeKey": 1.0,
    "Another": 5,
    "Foo": [1, 2, 3.5, {"InArray": 5}],
    "Bar": {"Nested": "Value"}
  }
```

```python
import mymodule

# Table becomes dict
table = mymodule.getTable()
assert table["Hello"] == 0
assert table["SomeKey"] == 10

# JsonNode becomes dict
json_obj = mymodule.getJsonAsDict()
assert json_obj["SomeKey"] == 1.0
assert json_obj["Foo"] == [1, 2, 3.5, {"InArray": 5}]
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
| Table[K, V] | dict |
| JsonNode | dict |
| ref object of PyNimObjectExperimental | Python class |

### Exporting Nim Types as Python Classes (Experimental)

nimpy can export Nim types as Python classes. This is useful for creating Python-native objects that wrap Nim state and behavior.

#### Requirements

- An exported type must be a `ref object` that inherits from `PyNimObjectExperimental` (directly or indirectly)
- At least one exported proc must have `self` as first argument to trigger type export
- Procs with `self` as first argument become methods on the Python type
- The type is only exported if requirements are met

#### Simple Example

```nim
# mymodule.nim
import nimpy

type
  TestType* = ref object of PyNimObjectExperimental
    myField*: string

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

#### `__init__` and `__repr__`

nimpy provides special handling for initialization and representation:

**`__init__`** - Export a proc as Python `__init__` method when:
1. Proc name matches pattern `init##TypeName`
2. Has at least one argument
3. First argument named `self`
4. First argument type is `##TypeName`
5. No return type

**`__repr__`** - Export a proc as Python `__repr__` method when:
1. Proc name is `$`
2. Has exactly one argument
3. First argument named `self`
4. First argument type is `##TypeName`
5. Return type is `string`

**Documentation** - Set module and type docstrings:
```nim
setModuleDocString("This is a test module")
setDocStringForType(MyType, "This is a test type")
```

#### Complete Example

```nim
# simple.nim
import nimpy
import strformat

pyExportModule("simple")  # Only needed if filename differs from module name

type
  SimpleObj* = ref object of PyNimObjectExperimental
    a*: int

## Export as __init__ (tp_init)
proc initSimpleObj*(self: SimpleObj, a: int = 1) {.exportpy.} =
  echo "Calling initSimpleObj for SimpleObj"
  self.a = a

## Export as __repr__ (tp_repr)
proc `$`*(self: SimpleObj): string {.exportpy.} =
  &"SimpleObj(a={self.a})"

setModuleDocString("This is a test module")
setDocStringForType(SimpleObj, "This is a test type")
```

Compile:
```bash
nim c --app:lib -o:./simple.so ./simple.nim
```

Use in Python:
```python
import simple

print(simple.__doc__)          # This is a test module
print(simple.SimpleObj.__doc__)  # This is a test type

obj = simple.SimpleObj(a=2)
print(obj)                    # SimpleObj(a=2)
```

#### Full Test Example

```nim
# texport_pytype.nim
import nimpy
import nimpy/py_types
import nimpy/py_lib as lib
import strformat

type
  PyCustomType* = ref object of PyNimObjectExperimental
    a*: int
    b*: float
    c*: string

proc initPyCustomType*(self: PyCustomType, aa: int = 1, bb: float = 2.0, cc: string = "default") {.exportpy} =
  self.a = aa
  self.b = bb
  self.c = cc

proc destroyPyCustomType*(self: PyCustomType) {.exportpy} =
  discard  # Cleanup if needed

proc `$`*(self: PyCustomType): string {.exportpy} =
  &"a: {self.a}, b: {self.b}, c: {self.c}"

proc get_a*(self: PyCustomType): int {.exportpy} =
  self.a

proc set_a*(self: PyCustomType, val: int) {.exportpy} =
  self.a = val

setModuleDocString("Test module for exported Python types")
setDocStringForType(PyCustomType, "Custom type with a, b, c fields")

import unittest

suite "Test Exported Python Types":
  let m = pyImport("texport_pytype")

  test "Test __doc__":
    check getAttr(m, "__doc__").`$` == "Test module for exported Python types"
    check getAttr(getAttr(m, "PyCustomType"), "__doc__").`$` == "Custom type with a, b, c fields"

  test "Test __init__ and methods":
    let constructor = getAttr(m, "PyCustomType")
    let obj = callObject(constructor, 99, 3.14, "hello")
    check obj.get_a().to(int) == 99
    obj.set_a(42)
    check obj.get_a().to(int) == 42

  test "Test __repr__":
    let constructor = getAttr(m, "PyCustomType")
    let obj = callObject(constructor, 99, 3.14, "hello")
    check obj.`$` == "a: 99, b: 3.14, c: hello"
```

#### Method Signatures

| Usage | Proc Signature | Becomes |
|-------|----------------|---------|
| Constructor | `proc newType*(): Type` | Class method (factory) |
| `__init__` | `proc initType*(self: Type, args...)` | `Type.__init__(self, args...)` |
| `__repr__` | `proc `$`*(self: Type): string` | `Type.__repr__(self)` |
| Method | `proc method*(self: Type, args...): R` | `Type.method(self, args...)` |

### Type Mapping for Exports

| Nim Type | Python Type |
|----------|-------------|
| int | int |
| float | float |
| string | str |
| seq[T] | list |
| tuple | tuple |
| bool | bool |
| ref object of PyNimObjectExperimental | Python class |

**Note**: For Python class exports, the Nim type must inherit from `PyNimObjectExperimental`. See "Exporting Nim Types as Python Classes" above.

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
