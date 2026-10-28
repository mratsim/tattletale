# Positron

Portable GPU kernel specs and prebuilt CUDA kernels for Nim.
Kernel code generation lives in [`workspace/crucible`](../crucible/README.md).

## Backends

The kernel specs compile, through crucible, to these backends:

- **Metal** (MSL)
- **OpenCL** (via clCreateProgramWithSource)
- **Vulkan** (GLSL → SPIR-V via glslangValidator)
- **WebGPU** (WGSL via wgpu-native)

The `cuda:` DSL macro refuses at compile time on CIRv2 (no CUDA printer yet).
Prebuilt CUDA kernels ship as a static library instead (see below).

## Architecture

Positron hosts the kernel layer. The compiler is crucible.
It lowers Nim to CIRv2, a backend-neutral intermediate representation, then prints per-backend source.

```
┌──────────────┐     ┌────────────────────────────┐     ┌──────────────────┐
│  Nim source  │───→ │  CIRv2 (crucible)          │───→ │  Backend printer │
│              │     │                            │     │                  │
│  metal:{...} │     │  nim_to_crucible           │     │  printer_msl     │
│  opencl:{...}│     │  nim_type_resolvers        │     │  printer_opencl  │
│  glsl:{...}  │     │  cirv2_* under codegen/ir  │     │  printer_glsl …  │
└──────────────┘     └────────────────────────────┘     └──────────────────┘
```

### Key modules

| Module | Purpose |
|--------|---------|
| `src/kernels/portable/` | Portable kernel specs (`activations`, `hadamard_transforms`) |
| `src/kernels/cuda/` | CUDA kernel sources, built into the static library by `make_libpositron_cuda.cu` |
| `libpositron_cuda.nim` | Host wiring for the prebuilt CUDA static library (Hadamard FWHT-128 + RMSNorm) |
| `../crucible/src/codegen/frontend/nim_to_crucible.nim` | Nim → CIRv2 lowering |
| `../crucible/src/codegen/frontend/nim_type_resolvers.nim` | Canonical type identity, base58 struct names (`Vec2_float32_DsoW2JH`) |
| `../crucible/src/codegen/ir/cirv2_*.nim` | CIRv2 module graph (`cirv2_build`, `cirv2_lang`, `cirv2_pretty`, `cirv2_verify`) |
| `../crucible/src/codegen/passes/` | Structuring, legalization, optimization, lowering and validation passes |
| `../crucible/src/codegen/backends/*.nim` | Per-backend printers |

## Usage

```nim
import workspace/crucible

# Write kernels in the Nim DSL (procs marked {.global.} become GPU kernels)
const kernelCode = metal:
  proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    let tid = thread_position_in_grid.x
    output[tid] = a[tid] + b[tid]

# Compile, load and launch
var engine = newEngine(runMetal)
engine.ingest(kernelCode)
var a: array[2, uint32] = [10'u32, 20'u32]
var b: array[2, uint32] = [1'u32, 2'u32]
let aDev = engine.allocDevice(uint32, a.len)
let bDev = engine.allocDevice(uint32, b.len)
let oDev = engine.allocDevice(uint32, 2)
engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
engine.launchKernel("addKernel", oDev, (aDev, bDev))
```

## Testing

Kernel code generation is tested through crucible's per-backend suites.
Every test launches the kernel and asserts its output:

```bash
nim c -r workspace/crucible/tests/codegen/metal/types/t_metal_array_length_arith.nim
```

Positron's own kernel tests live under [`tests/`](tests/): `manual_*` runnable kernels, `properties/` and `ceramic/` suites.

## License

Dual-licensed under MIT and Apache 2.0. See LICENSE in the root directory.
