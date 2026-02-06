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
let sizes = tensor.sizes()  # Returns ArrayRef[int64]
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
# Range slice
let slice = tensor[0..5, 3]

# Span slice (whole dimension)
let span = tensor[_, 3]

# With step
let stepped = tensor[0..10|2]
```

### Set via slice

```nim
# Assign single value
tensor[0..5, 3] = 999.0

# Assign array values
tensor[0..1, 0..1] = [[111, 222], [333, 444]]

# Assign from another tensor
tensor[^2..^1, 2..4] = other_tensor
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

### Key test patterns

1. Always wrap test code in `proc runTests*()` to avoid C++ `= {}` initialization
2. Use `shape.asTorchView()` for shape parameters
3. Use `dtype.toTorchType()` for dtype conversion
4. Use `==` for tensor comparison (uses `equal()` internally)
5. Each branch of `case` must assign to `result`
