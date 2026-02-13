---
name: torch_tensors
description: Nim bindings to libtorch for tensor operations with high-level sugar
license: MIT
compatibility: opencode
metadata:
  audience: ml-developers
  workflow: tensor-computing
---

## What I do

Torch tensors provide bindings to PyTorch's libtorch C++ library:
- Low-level bindings in `torch_tensors.nim` (raw FFI to C++)
- High-level sugar in `torch_tensors_sugar.nim` (Nim-friendly API)
- Support for CPU and CUDA tensors
- Full tensor operations (creation, indexing, math, FFT, etc.)

## When to use me

Use torch tensors when you need to:
- Load and manipulate tensors
- Perform tensor operations (reshape, transpose, matmul, etc.)
- Work with ML models and neural network layers
- Bridge between Nim and PyTorch ecosystems

## Core types

### TorchTensor

```nim
type TorchTensor* {.importcpp: "torch::Tensor", cppNonPod, bycopy.} = object
```

### ScalarKind - Data types

```nim
type ScalarKind* {.importc: "torch::ScalarType", size: sizeof(int8).} = enum
  kUint8 = 0
  kInt8 = 1
  kInt16 = 2
  kInt32 = 3
  kInt64 = 4
  kFloat16 = 5
  kFloat32 = 6
  kFloat64 = 7
  kComplexF32 = 9
  kComplexF64 = 10
  kBool = 11
  kBfloat16 = 15
```

### Device - Computation device

```nim
type DeviceKind* {.importc: "c10::DeviceType", size: sizeof(int16).} = enum
  kCPU = 0
  kCUDA = 1
  # ... other devices

type Device* {.importc: "c10::Device", bycopy.} = object
  kind: DeviceKind
  index: DeviceIndex
```

## Tensor creation

```nim
# From blob (no copy, shares memory)
let tensor = from_blob(data_ptr, sizes, dtype)

# Empty tensor
let tensor = empty(size, dtype)

# Clone tensor
let tensor2 = tensor.clone()

# From existing data
let tensor = toTorchTensor(my_seq)
```

## Type conversion helpers (torch_tensors_sugar.nim)

```nim
func toScalarKind*(T: typedesc[SomeTorchType]): static ScalarKind
func toTypedesc*(scalarKind: ScalarKind): typedesc

# Example: Convert between Nim types and Torch ScalarKind
let dtype = int32.toScalarKind()
let nimType = kFloat32.toTypedesc()  # Returns typedesc[float32]
```

## ArrayRef conversions (torch_tensors_sugar.nim)

Torch uses `ArrayRef[T]` for shape/size parameters. These templates convert between Nim openArrays and Torch ArrayRef:

```nim
# Convert Nim openArray to Torch ArrayRef (for shape/size parameters)
template asTorchView*[T](oa: openArray[T]): ArrayRef[T]
template asTorchView*(meta: Metadata): ArrayRef[int64]

# Convert Torch ArrayRef back to Nim openArray
template asNimView*[T](ar: ArrayRef[T]): openArray[T]

# Example: Creating tensors with shape
let shape = @[3, 4, 5].asTorchView()
let tensor = empty(shape, kFloat32)

# Reading tensor shape
let sizes = tensor.sizes()  # Returns IntArrayRef
echo sizes.asNimView()      # @[3, 4, 5]

# ArrayRef helpers
let len = sizes.len()       # 3
for val in sizes.items():   # Iterate values
  echo val
let first = sizes[0]        # Index access
```

Note: `asTorchView` creates a temporary copy because ArrayRef requires a pointer to data.

## Tensor metadata

```nim
let dim = tensor.dim()
let shape = tensor.sizes()        # Returns IntArrayRef
let strides = tensor.strides()
let ndim = tensor.ndimension()
let numel = tensor.numel()       # Total elements
let nbytes = tensor.nbytes()     # Bytes size
```

## Data access

```nim
# Raw data pointer
let data_ptr = tensor.data_ptr(float32)

# Get scalar from 0-dim tensor
let value: float32 = tensor.item(float32)
```

## Indexing

### Basic indexing

```nim
# Get value at index
let value = tensor[0, 1]

# Set value at index
tensor[0, 1] = 42.0
```

### Slicing

```nim
# Inclusive range (end included) - rarely used
let slice = tensor[0..5, 3]     # elements 0,1,2,3,4,5

# Exclusive range (end excluded) - matches Python semantics
let slice = tensor[0..<5, 3]    # elements 0,1,2,3,4

# Span slice (whole dimension, equivalent to Python ":")
let span = tensor[_, 3]          # all of dimension 0, index 3 of dimension 1

# Full span (all dimensions)
let all = tensor[_.._]           # all dimensions, equivalent to Slice()

# With step
let stepped = tensor[0..10|2]
```

### Negative indexing with `-` (Python-style)

**IMPORTANT**: The syntax uses `-N` (negative numbers) following Python conventions, NOT `^N` (hat notation).

Negative indexing rules:
- `-1` = last element (excludes last in slicing)
- `-2` = second-to-last element
- `-3` = third-to-last element
- etc.

Combined with `..-` for end-relative slicing:

```nim
# Python equivalents:
# a[:-1] -> all but last element
# a[-3:] -> last 3 elements
# a[-3:-1] -> elements from 3rd-from-end to before last

# Nim equivalents (using - for negative, ..- for end-relative):
let slice = tensor[_..-1, _]    # all but last (Python: a[:-1])
let slice = tensor[-3.._, _]     # last 3 elements (Python: a[-3:])
let slice = tensor[-3..-1, _]    # 3rd-from-end to before last (Python: a[-3:-1])

# Combined with start index:
let slice = tensor[1..-1, _]     # from index 1 to before last
```

### Why `-` instead of `^`

The change from `^` (Nim inclusive) to `-` (Python exclusive) was intentional:

| Old (^) | New (-) | Reason |
|----------|----------|--------|
| Exclusive mental model | Matches Python exactly | Easier to port Python algorithms |
| Off-by-one confusion | No off-by-one | Just subtract 1 from negative index |
| `a[^1..^2]` | `a[-1..-2]` | Direct Python translation |

**Python to Nim translation cheat sheet:**
- `a[:-1]` → `a[_..-1]` (subtract 1 from stop)
- `a[-3:]` → `a[-3.._]` (use `_` for "to end")
- `a[-3:-1]` → `a[-3..-1]` (stop is exclusive, so -1 means "before last")

### Span (`_`) vs Ellipsis (`...`)

The `_` symbol and `...` ellipsis have different meanings:

- **`_` (Span)**: Selects the entire dimension. Equivalent to Python's `:` or libtorch's `Slice()`.
  - `tensor[_, 3]` = `tensor[:, 3]` = all of dim 0, specific index of dim 1
  - `tensor[_.._]` = `tensor[:, :]` = all dimensions fully selected

- **`...` (Ellipsis)**: Expands to fill remaining dimensions. Equivalent to Python's `...` and libtorch's `torch::indexing::Ellipsis`.
  - `tensor[..., 0]` = `tensor[:, :, :, 0]` = ellipsis expands to match tensor rank
  - `tensor[0, ...]` = `tensor[0, :, :, :]` = leading index, ellipsis fills rest
  - `tensor[1, ..., 0]` = `tensor[1, :, :, 0]` = indices at start and end, ellipsis in middle

### Set via slice

```nim
# Assign single value
tensor[0..5, 3] = 999.0

# Assign array values
tensor[0..1, 0..1] = [[111, 222], [333, 444]]

# Assign from another tensor
tensor[-2..-1, 2..<5] = other_tensor  # last 2 elements, cols 2-4
```

## Tensor operations

### Arithmetic

```nim
let c = a + b
let d = a - b
let e = a * b
let f = a * 2.0

a += b
a -= 5.0
```

### Reshaping

```nim
let reshaped = tensor.reshape([3, 4])
let transposed = tensor.transpose(0, 1)
let permuted = tensor.permute([2, 0, 1])
```

### Aggregation

```nim
let sum = tensor.sum()
let mean = tensor.mean()
let max_val = tensor.max()
let min_val = tensor.min()
```

## Type and device conversion

```nim
# Change dtype
let float_tensor = int_tensor.to(kFloat32)

# Change device
let cpu_tensor = gpu_tensor.cpu()
let gpu_tensor = cpu_tensor.cuda()
```

## Random tensors

```nim
let rand_tensor = rand([3, 4])           # Uniform [0, 1)
let randn_tensor = randn([3, 4])         # Normal N(0, 1)
let zeros_tensor = zeros([3, 4])
let ones_tensor = ones([3, 4])
```

## Memory management

Important notes:
- `from_blob` creates views sharing original memory
- `clone()` creates an independent copy
- Sliced tensors share memory with parent
- Use `.contiguous()` before direct data access

## Error handling

```nim
try:
  check_index(tensor, idx0, idx1)
  let value = tensor[idx0, idx1]
except IndexDefect as e:
  echo "Out of bounds: ", e.msg
```

## Test patterns from test_safetensors.nim

### Shape conversion with asTorchView

Convert Nim sequences to Torch ArrayRef for shape parameters:

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

### Dtype conversion with toTorchType

Convert between Dtype enum and ScalarKind:

```nim
const TestedDtypes = [F64, F32, F16, I64, I32, I16, I8, U64, U32, U16, U8]

proc runTests*() =
  for dtype in TestedDtypes:
    let torchType = dtype.toTorchType()
    let tensor = arange(8, torchType)
    check tensor.scalarType() == torchType
```

### Tensor creation and comparison

Create expected tensors and compare with loaded ones:

```nim
proc compareTensors*(expected, actual: TorchTensor) =
  check expected.shape == actual.shape
  check expected.scalarType() == actual.scalarType()
  check actual == expected  # Uses equal() under the hood

# Usage in tests
let expectedTensor = generateExpectedTensor(pattern, shape, dtype.toTorchType())
let actualTensor = safetensorsLoader.getTensor(key)
check actualTensor == expectedTensor
```

### Tensor chaining pattern

Many tensor operations can be chained:

```nim
let tensor = arange(numel, kInt64)
  .reshape(shapeRef)
  .to(kFloat64)
  .cpu()
```

## Advanced slicing syntax (indexing_macros.nim)

The indexing macros provide Python-like slicing with Nim syntax:

```nim
# Basic slices (inclusive start, exclusive end like Python)
let slice = tensor[0..5]        # elements 0,1,2,3,4
let slice = tensor[1..3, 0..2]  # 2D slice

# Exclusive end with ..<
let slice = tensor[0..<5]       # elements 0,1,2,3,4 (same as 0..4)

# Negative indexing with - (Python-style, EXCLUSIVE upper bound)
# -1 = last element, -2 = second-to-last, etc.
let slice = tensor[-2..-1]      # last 2 elements (Python: a[-2:])
let slice = tensor[0..-1]       # all but last element (Python: a[:-1])
let slice = tensor[0..-3]       # all but last 2 elements (Python: a[:-3])

# Stepped slices with |
let slice = tensor[0..10|2]     # every 2nd element
let slice = tensor[_.._|2]      # entire dim, every 2nd
let slice = tensor[|2]          # NEW: cleaner syntax for every 2nd (Python [::2])
let slice = tensor[|2, 3]       # every 2nd of dim 0, index 3 of dim 1

# Negative steps (reversing) - NOT supported, use flip()
let reversed = tensor.flip(@[0])  # Reverse along dimension 0

# Span slices (whole dimension, equivalent to Python ":")
let span = tensor[_, 3]         # all of dim 0, index 3 of dim 1
let span = tensor[1.._, _]      # dim 0 from 1, all of dim 1
let span = tensor[_.._]         # entire dimension (Slice)

# Ellipsis for multi-dimensional expansion (Python "...")
let result = tensor[IndexEllipsis, 0]  # last dimension index 0
let result = tensor[0, IndexEllipsis]  # first dimension index 0, rest all
let result = tensor[1, IndexEllipsis, 0]  # first dim index 1, last dim index 0

# Combined
let slice = tensor[-2..-1, 0..<5]  # last 2 elements, first 5 of dim 1
```

### Key distinctions

| Nim syntax | Python equivalent | libtorch equivalent | Notes |
|-----------|------------------|-------------------|-------|
| `_` | `:` | `Slice()` | Full span of one dimension |
| `_.._` | `:` | `Slice()` | Full span (not Ellipsis!) |
| `|step` | `::step` | `Slice(None, None, step)` | Stepped full span |
| `...` | `...` | `Ellipsis` | Expands to fill remaining dims |
| `-1` | `-1` | N/A | Last element (exclusive in slicing) |
| `-2` | `-2` | N/A | Second-to-last element |

### Python Slice to Nim Translation Reference (Updated for Python semantics)

**Python slices are EXCLUSIVE on the end.** Nim uses `..` (inclusive) or `..<` (exclusive), but for negative indices we use `..-` following Python semantics directly.

| Python syntax | Nim syntax | Result indices | Description |
|---------------|-----------|---------------|-------------|
| `t[:]` | `t[_.._]` | 0,1,2,3,4 | All elements |
| `t[:2]` | `t[_..<2]` | 0,1 | First 2 elements |
| `t[2:]` | `t[2.._]` | 2,3,4 | From index 2 to end |
| `t[1:4]` | `t[1..<4]` | 1,2,3 | Elements 1,2,3 |
| `t[::2]` | `t[_.._|2]` or `t[|2]` | 0,2,4 | Every 2nd element |
| `t[1::2]` | `t[1.._|2]` | 1,3 | From 1, every 2nd |
| `t[:4:2]` | `t[_..<4|2]` | 0,2 | Elements 0,2 |
| `t[1:4:2]` | `t[1..<4|2]` | 1,3 | Elements 1,3 |
| `t[:-1]` | `t[_..-1]` | 0,1,2,3 | All but last |
| `t[-3:]` | `t[-3.._]` | 2,3,4 | Last 3 elements |
| `t[-3:-1]` | `t[-3..-1]` | 2,3 | 3rd-from-end to before last |
| `t[::-1]` | `t.flip([dim])` | - | Reverse (use flip()) |

### Negative indexing rules

For a 5-element array (indices 0, 1, 2, 3, 4):

| Python | Nim | Actual indices | Explanation |
|--------|-----|---------------|-------------|
| -1 | -1 | 4 | Last element |
| -2 | -2 | 3 | Second-to-last |
| -3 | -3 | 2 | Third-to-last |
| `a[:-1]` | `a[_..-1]` | 0,1,2,3 | Stop at -1 (4), exclusive → 0-3 |
| `a[-3:]` | `a[-3.._]` | 2,3,4 | Start at -3 (2), go to end |
| `a[-3:-1]` | `a[-3..-1]` | 2,3 | Start at -3 (2), stop before -1 (4) |

**Key insight**: In Python slicing, `-1` as stop means "up to but NOT including the last element". The libtorch `Slice` constructor follows Python's exclusive upper bound semantics.

### Ellipsis (`...`) vs Span (`_`)

The `_` symbol and `...` ellipsis have different meanings:

- **`_` (Span)**: Selects the entire dimension. Equivalent to Python's `:` or libtorch's `Slice()`.
  - `tensor[_, 3]` = `tensor[:, 3]` = all of dim 0, specific index of dim 1
  - `tensor[_.._]` = `tensor[:, :]` = all dimensions fully selected

- **`...` (Ellipsis)**: Expands to fill remaining dimensions. Equivalent to Python's `...` and libtorch's `torch::indexing::Ellipsis`.
  - `tensor[..., 0]` = `tensor[:, :, :, 0]` = ellipsis expands to match tensor rank
  - `tensor[0, ...]` = `tensor[0, :, :, :]` = leading index, ellipsis fills rest
  - `tensor[1, ..., 0]` = `tensor[1, :, :, 0]` = indices at start and end, ellipsis in middle

### Step type and operators

Internal workaround for operator precedence:

```nim
type Step = object
  b: int       # end of range
  step: int    # step size

# Operators:
# `|` - step operator (positive only)
# `..` - inclusive range
# `..<` - exclusive range
# `..-` - end-relative range (Python exclusive semantics)
# `|-` - negative step (NOT SUPPORTED - use flip() instead)
```

Note: Negative steps (`|-`) are not supported because libtorch's Slice() doesn't support them.
To reverse a dimension, use `tensor.flip(@[dim])` instead.

### Runtime negative index normalization

Negative indices are normalized at runtime using `pythonSliceToTorchSlice` function:

```nim
func pythonSliceToTorchSlice*(
  start: int | Nullopt_t,
  stop: int | Nullopt_t,
  size: int
): TorchSlice {.inline.} =
  ## Convert Python-style slice to libtorch Slice with proper negative index handling.
  ##
  ## Python semantics:
  ##   - Exclusive upper bound (stop is not included)
  ##   - Negative indices are normalized: -N → size + (-N) = size - N
  ##
  ## Example for size=5, stop=-1:
  ##   -1 + 5 = 4 → Slice(start, 4) which gives elements [start, 4)
  ##   Since stop=4 is exclusive, we get elements up to index 3 (all but last)
```

This function is automatically called during tensor indexing when negative indices are detected.

### FancySelectorKind - Indexing dispatch

Types for dispatching different indexing modes:

```nim
type FancySelectorKind* = enum
  FancyNone          # No fancy indexing
  FancyIndex        # Integer array indexing
  FancyMaskFull     # Boolean mask on full tensor
  FancyMaskAxis     # Boolean mask on specific axis
  FancyUnknownFull  # Unknown selector full
  FancyUnknownAxis  # Unknown selector axis
```

### Slicing macros

```nim
# Desugar all syntactic sugar to TorchSlice
desugarSlices*(args: untyped)

# Typed dispatch for read operations
slice_typed_dispatch*(t: typed, args: varargs[typed])

# Typed dispatch for write operations
slice_typed_dispatch_mut*(t: typed, args: varargs[typed], val: typed)
```

## Negative indexing tests (test_indexing.nim)

Test patterns for negative indexing:

```nim
suite "Python Slice Syntax to Nim Translation Reference":
  ## For a 5x5 Vandermonde matrix (indices 0-4 on each axis):
  ## [[1,1,1,1,1], [2,4,8,16,32], [3,9,27,81,243], [4,16,64,256,1024], [5,25,125,625,3125]]

  test formatName("Python a[:-1] -> Nim a[_..-1]", "a[:-1]"):
    ## Nim: a[_..-1] gets all but last (stop=-1 is exclusive)
    ## Python: a[:-1] gets all but last element
    let t = genShiftedVandermonde5x5(kFloat64)
    let sliced = t[_..-1, _]
    check: sliced.shape[0] == 4  # indices 0,1,2,3 (not 4)
    check: sliced[3, 0].item(float64) == 4.0  # Row 3, not row 4

  test formatName("Python a[-3:] -> Nim a[-3.._]", "a[-3:]"):
    ## Nim: a[-3.._] gets last 3 (start=-3 means 3rd from end)
    ## Python: a[-3:] gets last 3 indices (2,3,4)
    let t = genShiftedVandermonde5x5(kFloat64)
    let sliced = t[-3.._, _]
    check: sliced.shape[0] == 3  # indices 2,3,4
    check: sliced[0, 0].item(float64) == 3.0  # Row 2
    check: sliced[2, 0].item(float64) == 5.0  # Row 4

  test formatName("Python a[-3:-1] -> Nim a[-3..-1]", "a[-3:-1]"):
    ## Nim: a[-3..-1] gets 3rd-from-end to before last
    ## Python: a[-3:-1] gets indices 2, 3 (exclusive stop)
    let t = genShiftedVandermonde5x5(kFloat64)
    let sliced = t[-3..-1, _]
    check: sliced.shape[0] == 2  # indices 2, 3
    check: sliced[0, 0].item(float64) == 3.0  # Row 2
    check: sliced[1, 0].item(float64) == 4.0  # Row 3
```

### FFT test patterns (test_torchtensors.nim)

Test FFT operations with roundtrip verification:

```nim
suite "FFT1D":
  setup:
    let shape = [8'i64]
    let f64input = rand(shape.asTorchView(), kfloat64)
    let c64input = rand(shape.asTorchView(), kComplexF64)

  test "fft, ifft":
    let fftout = fft(c64input)
    let ifftout = ifft(fftout)
    let max_input = max(abs(ifftout)).item(float64)
    var rel_diff = abs(ifftout - c64input)
    rel_diff /= max_input
    check mean(rel_diff).item(float64) < 1e-12

  test "rfft, irfft":
    let fftout = rfft(f64input)
    let ifftout = irfft(fftout)
    # ... similar verification
```

### Tensor creation test patterns

```nim
suite "Tensor creation":
  test "eye":
    let t = eye(2, kInt64)
    check t == [[1, 0], [0, 1]].toTorchTensor()

  test "zeros":
    let shape = [2'i64, 3]
    let t = zeros(shape.asTorchView(), kFloat32)
    check t == [[0.0'f32, 0.0, 0.0], [0.0'f32, 0.0, 0.0]].toTorchTensor()

  test "linspace":
    let steps = 120'i64
    let reft = toSeq(0..<120).map(x => float64(x)/float64(steps-1)).toTorchTensor()
    let t = linspace(0.0, 1.0, steps, kFloat64)
    let rel_error = mean(t - reft)
    check rel_error.item(float64) <= 1e-12

  test "arange":
    let steps = 130'i64
    let step = 1.0/float64(steps)
    let t = arange(0.0, 1.0, step, float64)
    for i in 0..<130:
      let val = t[i].item(float64)
      let refval: float64 = i.float64 / 130.0
      check (val - refval) < 1e-12
```

### Sort and argsort tests

```nim
test "sort, argsort":
  let t = [2, 3, 4, 1, 5, 6].toTorchTensor()
  let
    s = t.sort()
    args = t.argsort()
  check s.get(0) == [1, 2, 3, 4, 5, 6].toTorchTensor()
  check s.get(1) == args
  check args == [3, 0, 1, 2, 4, 5].toTorchTensor()
```

### Operator precedence tests

```nim
suite "Operator precedence":
  test "+ and *":
    let a = [[1, 2], [3, 4]].toTorchTensor()
    let b = -a
    check b * a + b == [[-2, -6], [-12, -20]].toTorchTensor()
    check b * (a + b) == [[0, 0], [0, 0]].toTorchTensor()

  test "+ and .abs":
    let a = [[1, 2], [3, 4]].toTorchTensor()
    let b = -a
    check a + b.abs == [[2, 4], [6, 8]].toTorchTensor()
    check (a + b).abs == [[0, 0], [0, 0]].toTorchTensor()
```

### Key test patterns

1. Always wrap test code in `proc runTests*()` to avoid C++ `= {}` initialization
2. Use `shape.asTorchView()` for shape parameters
3. Use `dtype.toTorchType()` for dtype conversion
4. Use `==` for tensor comparison (uses `equal()` internally)
5. Each branch of `case` must assign to `result`
6. For FFT tests, verify roundtrip with `abs()` and `mean()` of difference
7. Use `max(abs(tensor)).item(float64)` for normalization in error calculations
