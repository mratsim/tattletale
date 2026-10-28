# Crucible

GPU kernel code generator: translates a subset of Nim into native GPU source for six backends:
CUDA, OpenCL, Vulkan (GLSL), Vulkan (HLSL), WebGPU (WGSL), and Metal (MSL).
The allowed subset is everything except `string`, `seq`, memory allocations, and IO/syscalls.

## What it is / what it isn't

Crucible is primarily a **compile-time code generator**.
Its entry points are Nim macros (`cuda`, `opencl`, `glsl`, `hlsl`, `webgpu`, `metal`).
Each macro consumes a kernel body written in that subset of Nim and emits a string of native GPU source code.

- It does not ship a device runtime or a scheduler.
- It does include minimal host-side runtimes under [`src/runtime/backends/`](src/runtime/backends/):
  they compile, load, and launch the generated code on each backend.

It is the base layer of the tattletale GPU stack: kernels, layout algebra, tile algebra all call Crucible to compile to GPUs.

- The generated source is handed to an external toolchain to run:
  NVRTC compiles CUDA to PTX, glslangValidator compiles GLSL to SPIR-V.
- The host-side runtimes under [`src/runtime/backends/`](src/runtime/backends/), backed by the ABIs under
  [`src/abis/`](src/abis/), load and launch the results.
- The Metal runtime compiles MSL on-device via `newLibraryWithSource`, it needs no external CLI.

Because it is a code generator, "correctness" of a feature means the emitted code compiles
*and* computes the right result, which is why every test must call `engine.launchKernel()`,
not just `engine.ingest()` (see [`AGENTS.md`](AGENTS.md)).

## Architecture in brief

- **One IR, six backends.** [`nim_to_crucible.nim`](src/codegen/frontend/nim_to_crucible.nim) lowers Nim AST to one
  backend-neutral IR, each backend under [`src/codegen/backends/`](src/codegen/backends/) prints it:
  `printer_opencl.nim`, `printer_glsl.nim`, `printer_hlsl.nim`, `printer_wgsl.nim`, `printer_msl.nim`,
  plus `printer_spirv.nim`, the direct SPIR-V encoder. The CUDA printer is not rebuilt on the CIRv2 IR yet, `cuda:` blocks refuse loudly.
- **A pass pipeline.** Semantic work lives in named passes under [`src/codegen/passes/`](src/codegen/passes/),
  run through a [`PassRegistry`](src/codegen/passes/pass_registry.nim).
  Pass families: normalizations, legalizations, preprocessing, lowering, and validations.
- **Compile-time codegen.** The `cuda`/`opencl`/`glsl`/`hlsl`/`webgpu`/`metal` macros emit
  source at compile time through [`src/codegen/gpu_compiler.nim`](src/codegen/gpu_compiler.nim).
- **Tests.** IR-level tests under [`tests/cirv2/`](tests/cirv2/) plus auto-runnable suites per backend
  (`cuda/`, `opencl/`, `vulkan_via_glsl/`, `vulkan_via_hlsl/`, `webgpu/`, `metal/`).
  Tests must call `engine.launchKernel()`, not just `engine.ingest()` (see [`AGENTS.md`](AGENTS.md)).

## Builtins: one vocabulary, six backends

Kernel bodies write coordinates and synchronization in one canonical vocabulary, the MSL names:

- `thread_position_in_grid`, `threadgroup_position_in_grid`, `thread_position_in_threadgroup`
- `threads_per_threadgroup`, `threadgroups_per_grid`, `thread_index_in_threadgroup`
- `threadgroup_barrier`

Every other backend name is a template alias expanding to the canonical name during Nim semantic analysis,
so the IR only ever contains canonical names, with no alias tables or name-to-kind maps in the compiler.

- Any backend's idiom works in any backend's kernel: a `cuda:` kernel may write `get_global_id(0)`,
  a `metal:` kernel may write `blockIdx.x`.
- The catalog, [`src/builtins/builtins_catalog.nim`](src/builtins/builtins_catalog.nim),
  defines the canonical dummies and the alias set.
- The printers, [`src/codegen/backends/`](src/codegen/backends/), emit the native names below.

| Canonical (MSL) | CUDA | OpenCL | Vulkan (GLSL) | WGSL |
| --- | --- | --- | --- | --- |
| `thread_position_in_grid` | `(blockIdx.d*blockDim.d+threadIdx.d)` | `get_global_id(d)` | `gl_GlobalInvocationID` | `global_id` |
| `threadgroup_position_in_grid` | `blockIdx` | `get_group_id(d)` | `gl_WorkGroupID` | `workgroup_id` |
| `thread_position_in_threadgroup` | `threadIdx` | `get_local_id(d)` | `gl_LocalInvocationID` | `local_invocation_id` |
| `threads_per_threadgroup` | `blockDim` | `get_local_size(d)` | `gl_WorkGroupSize` | deferred (no WGSL builtin) |
| `threadgroups_per_grid` | `gridDim` | `get_num_groups(d)` | `gl_NumWorkGroups` | `num_workgroups` |
| `thread_index_in_threadgroup` | `(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)` | `(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))` | `gl_LocalInvocationIndex` | `local_invocation_index` |
| `threadgroup_barrier` | `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | `barrier()` | `workgroupBarrier()` |
| `reduceAdd` | `warpReduceAdd` (emitted `__forceinline__` template) | `sub_group_reduce_add` | `subgroupAdd` | `subgroupAdd` (no directive; subgroups device feature, see note) |
| `reduceMax` | `warpReduceMax` (emitted `__forceinline__` template) | `sub_group_reduce_max` | `subgroupMax` | `subgroupMax` (no directive; subgroups device feature, see note) |

Notes:

- **CUDA and OpenCL name the vector coordinates per component**, `d` is the accessed component
  (`x`/`y`/`z`). OpenCL emits a literal dimension (`get_global_id(0)`), CUDA folds it into
  `(blockIdx.d*blockDim.d+threadIdx.d)`.
  Whole-value uses of the coordinate vectors are out of scope.
- **WGSL injects one `@builtin(...)` kernel param per canonical coordinate the body references**,
  in first-use order: `global_id`, `workgroup_id`, `local_invocation_id`, `num_workgroups`,
  `local_invocation_index`, only referenced builtins are injected, so generated WGSL text
  legitimately changes with the kernel body. Do not assert WGSL emitted text, assert results.
- **`threads_per_threadgroup` is deferred on WGSL.**
  WGSL has no `workgroup_size` builtin, only the `@workgroup_size` attribute, so referencing it
  in a `webgpu:` kernel is a loud compile error. `thread_index_in_threadgroup` maps to WGSL's
  native `local_invocation_index` instead of a composed formula.

- **Alias families**, the Nim names that expand to the canonical name:

| Backend | Alias names |
| --- | --- |
| CUDA | `blockIdx`, `blockDim`, `gridDim`, `threadIdx`, `syncthreads` |
| OpenCL | `get_global_id(d)`, `get_group_id(d)`, `get_local_id(d)`, `get_local_size(d)`, `get_num_groups(d)`, `barrier(flags)`, `workgroup_barrier(flags)` |
| GLSL | `gl_GlobalInvocationID`, `gl_WorkGroupID`, `gl_LocalInvocationID`, `gl_WorkGroupSize`, `gl_NumWorkGroups`, `gl_LocalInvocationIndex` |
| WGSL | `global_id`, `workgroup_id`, `local_invocation_id`, `num_workgroups`, `local_invocation_index`, `workgroupBarrier` |
| MSL | the canonical names themselves |
| HLSL | the canonical names themselves |

  `workgroup_barrier(flags)` and `workgroupBarrier()` are one declaration with a defaulted flag.
  `get_global_size` is excluded from the vocabulary, it is OpenCL-only native.
- **No OpenCL `get_*` registration.**
  [`NimGpuFnBuiltins`](src/builtins/builtins_functions.nim) registers only the Nim
  function-style magic builtins (`toOpenArray`, `len`).
- The OpenCL work-item names are template aliases in `builtins_catalog.nim`,
  they expand to canonical names during Nim semantic analysis. The name-only registration path never consults the list,
  so no codegen registration exists for them.

## Source layout

```
workspace/crucible/
  crucible.nim                    Top-level API: re-exports the curated surface and the GPU builtins
  AGENTS.md                       Test conventions and run commands
  ARCHITECTURE.md                 Pipeline, data-flow walkthrough, extension points
  src/
    crucible_public_api.nim       Curated surface: engines, device memory, launch sugar, macros
    crucible_gpu_builtins.nim     Builtin vocabulary hub, callable from kernel bodies
    builtins/
      builtins.nim              Re-exports the catalog and pragmas to the entry macros
      builtins_catalog.nim      Canonical coordinate/sync builtins + per-backend idiom aliases
      builtins_functions.nim    Device math builtins (exp2/rsqrt) + Nim function name tables
      builtins_gpu_types.nim    float16, bfloat16, fp8e4m3, fp8e5m2
      builtins_warp_sync_reductions.nim  reduceAdd, reduceMax
      builtins_atomics.nim      atomicFetchAdd
      builtins_pragmas.nim      Backend pragmas: cudaName, workgroup, builtin, smem, ...
      builtins_compilermagic.nim  CompileTarget, ccGetBackend
    codegen/
      gpu_compiler.nim            Entry point: `cuda:`/`opencl:`/`glsl:`/`hlsl:`/`webgpu:`/`metal:` macros
      frontend/
        nim_to_crucible.nim            Frontend: pure 1:1 Nim AST -> IR translator
        nim_type_resolvers.nim  Type/overload resolution
      ir/
        cirv2_lang.nim        IR types: the CIRv2 module/proc/inst graph
        cirv2_build.nim       IR constructors
        cirv2_pretty.nim      IR pretty-printer
        cirv2_verify.nim      IR verifier
      passes/
        pass_datatypes.nim        Pass/registry types, walk, runPasses
        pass_registry.nim         PassRegistry
        passes_normalizations.nim  Frontend-extraction passes
        passes_legalizations.nim  Legalizations
        passes_norm_aliases.nim   Builtin-alias normalization
        passes_norm_shell_fold.nim  Constant shell folding
        passes_norm_structure.nim Structure normalization
        passes_low_kernel_params.nim  Kernel-parameter lowering
        passes_low_reserved.nim   Reserved-identifier lowering
        passes_lowering.nim       Lowering
        passes_validations.nim    Pre/post validation passes
        passes_optimizations.nim  Optimizations
        passes_lowering_cuda.nim  Per-language lowering passes
        passes_lowering_glsl.nim
        passes_lowering_hlsl.nim
        passes_lowering_msl.nim
        passes_lowering_opencl.nim
        passes_lowering_wgsl.nim
      backends/
        targets.nim               BackendKind, the backend identifier shared by
                                  printers, engines and launch serialization
        printer_opencl.nim        Printers: OpenCL C, GLSL, HLSL, WGSL, MSL
        printer_glsl.nim
        printer_hlsl.nim          HLSL printer: dxc-compiled, D3D12 register bindings, ConstantBuffer scalars
        printer_wgsl.nim
        printer_msl.nim
        printer_spirv.nim         Direct SPIR-V emission over spirv_encoder.nim
    runtime/
      engines_api.nim is under engines/: the HwEngine concept, newEngine, re-exports
      engines/
        engines_api.nim           The HwEngine concept, the newEngine dispatch, re-exports
        engines_common.nim        DeviceMem, Dim3, LaunchConfig, dim3
        cuda_api.nim              Engine facades (CudaEngine, OpenCLEngine, VulkanEngine,
        opencl_api.nim            WebgpuEngine, MetalEngine)
        vulkan_api.nim
        webgpu_api.nim
        metal_api.nim
      backends/
        cuda_runtime.nim          Platform sessions: library loading, device discovery,
        opencl_runtime.nim        source -> artifact compilation, buffers, dispatch
        vulkan_common.nim
        vulkan_descriptor_runtime.nim
        vulkan_address_runtime.nim
        vulkan_compile.nim
        webgpu_runtime.nim
        metal_runtime.nim         MSL compile/dispatch
        runtime_utils.nim
      launch/
        launch_kernel.nim         launchKernel + chevron sugar
        launch_device_fn.nim      launchDeviceFn: typed Nim device functions
        launch_serialization.nim  Host -> device argument packing
    abis/
      c_abi.nim                   Raw C FFI bindings
      opencl_abi.nim
      vulkan_abi.nim
      nvidia_abi.nim              CUDA driver + NVRTC bindings
      nvidia_paths.nim            CUDA library discovery
      objc_abi.nim                libobjc msgSend bridge (Metal's host API)
    macros/ast_rebuilder.nim
  tests/
    cirv2/                        IR-level tests (pure Nim, no engine)
    codegen/
      cuda/ opencl/ vulkan_via_glsl/ vulkan_via_hlsl/ webgpu/ metal/
                                  Per-backend auto-runnable tests
    runtime/                      Engine/runtime tests (t_launch_device_fn, ...)
  vendor/wgpu/                    Vendored wgpu-native headers/libs
```

## Build and run

Run from the repo root (`tattletale`).

Requirements:

- a working Nim
- NVRTC/glslangValidator for compilation
- a CUDA/OpenCL/Vulkan/WebGPU/Metal device for `engine.launchKernel()`

The Metal tests run on any Apple Silicon Mac
(macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, tested 2026-08-17).

```bash
# CUDA (NVRTC, primary target)
nim cpp -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/cuda/*/t_cuda_*.nim

# OpenCL
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/opencl/*/t_opencl_*.nim

# Vulkan via GLSL
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/vulkan_via_glsl/*/t_vulkan_*.nim

# Vulkan via HLSL (dxc + spirv-val)
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/vulkan_via_hlsl/*/t_hlsl_*.nim

# WebGPU
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/webgpu/*/t_webgpu_*.nim

# Metal (macOS, on-device MSL compilation)
# tested ABI: macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, 2026-08-17
nim test_crucible_metal

# IR-level tests (pure Nim)
nim c -r --hints:off --warnings:off \
  --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
  workspace/crucible/tests/cirv2/*.nim
```

## Related docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md): pipeline, data-flow walkthrough, extension points.
- [`AGENTS.md`](AGENTS.md): test naming, `engine.launchKernel()` requirement, minimal reproductions, debugging.
- [`workspace/ceramic/`](../ceramic/): layout algebra consumed by Crucible.
- [`workspace/positron/`](../positron/): kernel specifications consumed by Crucible.

## License

Dual-licensed under either of the MIT License or the Apache License, Version 2.0, at your option.
See the license headers in each source file.
