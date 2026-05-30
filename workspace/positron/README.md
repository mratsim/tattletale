# Positron

GPU kernel compiler framework for Nim. Transpiles Nim code to multiple GPU backends.

## Backends

- **CUDA** (via NVRTC JIT compilation)
- **OpenCL** (via clCreateProgramWithSource)
- **Vulkan** (GLSL → SPIR-V via glslangValidator)
- **WebGPU** (WGSL via wgpu-native)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Nim source  │───→│   IR layer    │───→│  Backend emitter   │
│              │     │              │     │                   │
│  cuda:{ ... }│     │  gpu_types    │     │ cuda_lang.nim     │
│  webgpu:{...}│     │  nim_to_gpu   │     │ wgsl_lang.nim     │
│  opencl:{...}│     │              │     │ opencl_lang.nim    │
│  vulkan:{...}│     │              │     │ vulkan_lang.nim    │
└─────────────┘     └──────────────┘     └───────────────────┘
```

### Key modules

| Module | Purpose |
|--------|---------|
| `src/codegen/ir/gpu_types.nim` | IR type system (`GpuType`, `GpuNodeKind`) |
| `src/codegen/ir/nim_to_gpu.nim` | Nim → GPU AST transpiler |
| `src/codegen/targets/*.nim` | Backend-specific code generation |
| `src/codegen/exec/*.nim` | Runtime execution helpers |
| `src/codegen/nvrtc.nim` | NVRTC JIT compilation wrapper |
| `src/codegen/builtins/` | Backend builtin functions |
| `src/abis/*.nim` | C ABI bindings (Vulkan, OpenCL, CUDA, etc.) |

## Usage

```nim
import workspace/positron/src/codegen/nvrtc

# Write kernels in Nim DSL (functions marked {.global.} become GPU kernels)
const kernelCode = cuda:
  proc add(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    let tid = blockIdx.x * blockDim.x + threadIdx.x
    if tid < 256:
      output[tid] = a[tid] + b[tid]

# Compile and load
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
nv.load()

# Execute kernel
var a, b, result: array[256, uint32]
nv.execCuda(res = result, inputs = (a, b))
```

## Testing

```bash
nim cpp -r workspace/positron/tests/codegen/nvrtc/test_*.nim
nim cpp -r workspace/positron/tests/codegen/opencl/test_*.nim
nim cpp -r workspace/positron/tests/codegen/vulkan/test_*.nim
nim cpp -r workspace/positron/tests/codegen/webgpu/test_*.nim
```

## License

Dual-licensed under MIT and Apache 2.0. See LICENSE in the root directory.
