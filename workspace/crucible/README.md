# Crucible

GPU kernel code generator: translates a subset of Nim (everything allowed except `string`, `seq`, memory allocations, and IO/syscalls) into native GPU source for five backends — CUDA, OpenCL, Vulkan (GLSL), WebGPU (WGSL), and Metal (MSL).

## What it is / what it isn't

Crucible is primarily a **compile-time code generator**: its entry points are Nim macros (`cuda`, `opencl`, `vulkan`, `webgpu`, `metal`) that consume a kernel body written in that subset of Nim and emit a string of native GPU source code. It does not ship a device runtime or a scheduler, but it does include minimal host-side runtimes under [`src/runtime/exec/`](src/runtime/exec/) that compile, load, and launch the generated code on each backend (NVRTC, OpenCL, Vulkan, WebGPU, Metal).

It sits on top of the rest of the tattletale stack:

- It **consumes** the layout algebra produced by `workspace/ceramic/` and the kernel specifications produced by `workspace/positron/`. Per the header in [`workspace/crucible/crucible.nim`](crucible.nim): "takes Ceramic layout algebra and Positron kernel specifications and emits native GPU code (CUDA, OpenCL, Vulkan, WebGPU, Metal)".
- The generated source is handed to an external toolchain to run: NVRTC compiles CUDA to PTX, glslangValidator compiles GLSL to SPIR-V, and the host-side runtimes under [`src/runtime/exec/`](src/runtime/exec/) (backed by the ABIs under [`src/abis/`](src/abis/)) load and launch the results. The Metal runtime compiles MSL on-device via `newLibraryWithSource`, so it needs no external CLI.

Because it is a code generator, "correctness" of a feature means the emitted code compiles *and* computes the right result — which is why every test must call `engine.run()`, not just `engine.ingest()` (see [`AGENTS.md`](AGENTS.md)).

## Architecture in brief

- **One IR, five backends.** [`src/codegen/ir/nim_to_gpu.nim`](src/codegen/ir/nim_to_gpu.nim) lowers Nim AST to a backend-neutral IR; each target under [`src/codegen/targets/`](src/codegen/targets/) (`cuda_lang.nim`, `opencl_lang.nim`, `vulkan_lang.nim`, `wgsl_lang.nim`, `metal_lang.nim`) prints it.
- **A pass pipeline.** Semantic work lives in named passes under [`src/codegen/passes/`](src/codegen/passes/) (normalizations, legalizations, preprocessing, lowering, validations), run through a [`PassRegistry`](src/codegen/passes/pass_registry.nim).
- **Compile-time and runtime codegen.** The `cuda`/`opencl`/`vulkan`/`webgpu`/`metal` macros emit source at compile time; `codegen(gen, ast, ...)` in [`src/codegen/gpu_compiler.nim`](src/codegen/gpu_compiler.nim) clones the IR and regenerates for a backend at runtime.
- **Tests.** IR-level tests under [`tests/codegen/ir/`](tests/codegen/ir/) plus auto-runnable suites per backend (`nvrtc/`, `opencl/`, `vulkan/`, `webgpu/`, `metal/`). Tests must call `engine.run()`, not just `engine.ingest()` (see [`AGENTS.md`](AGENTS.md)).

## Builtins: one vocabulary, five backends

Kernel bodies write coordinates and synchronization in one canonical vocabulary, the MSL names:
`thread_position_in_grid`, `threadgroup_position_in_grid`,
`thread_position_in_threadgroup`, `threads_per_threadgroup`,
`threadgroups_per_grid`, `thread_index_in_threadgroup`, `threadgroup_barrier`.
Every other backend spelling is a template alias that expands to the canonical name during sem,
so the IR only ever contains canonical names, with no alias tables or name→kind maps in the compiler.
Any backend's idiom works in any backend's kernel: a `cuda:` kernel may write
`get_global_id(0)`, a `metal:` kernel may write `blockIdx.x`. The catalog,
[`src/codegen/builtins/builtins_catalog.nim`](src/codegen/builtins/builtins_catalog.nim),
defines the canonical dummies and the alias set. The printers,
[`src/codegen/targets/`](src/codegen/targets/),
emit the native spellings below.

| Canonical (MSL) | CUDA | OpenCL | Vulkan (GLSL) | WGSL | MSL |
|---|---|---|---|---|---|
| `thread_position_in_grid` | `(blockIdx.d*blockDim.d+threadIdx.d)` | `get_global_id(d)` | `gl_GlobalInvocationID` | `global_id` | `thread_position_in_grid` |
| `threadgroup_position_in_grid` | `blockIdx` | `get_group_id(d)` | `gl_WorkGroupID` | `workgroup_id` | `threadgroup_position_in_grid` |
| `thread_position_in_threadgroup` | `threadIdx` | `get_local_id(d)` | `gl_LocalInvocationID` | `local_invocation_id` | `thread_position_in_threadgroup` |
| `threads_per_threadgroup` | `blockDim` | `get_local_size(d)` | `gl_WorkGroupSize` | deferred (no WGSL builtin) | `threads_per_threadgroup` |
| `threadgroups_per_grid` | `gridDim` | `get_num_groups(d)` | `gl_NumWorkGroups` | `num_workgroups` | `threadgroups_per_grid` |
| `thread_index_in_threadgroup` | `(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)` | `(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))` | `gl_LocalInvocationIndex` | `local_invocation_index` | `thread_index_in_threadgroup` |
| `threadgroup_barrier` | `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | `barrier()` | `workgroupBarrier()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |

Notes:

- **CUDA and OpenCL spell the vector coordinates per component.** `d` is the accessed component (`x`/`y`/`z`).
  OpenCL emits a literal dimension (`get_global_id(0)`), CUDA folds it into the parenthesized
  `(blockIdx.d*blockDim.d+threadIdx.d)` expression. Whole-value uses of the coordinate vectors are out of scope.
- **WGSL injects one `@builtin(...)` kernel param per canonical coordinate the body references**, in first-use order:
  the five possible params are `global_id`, `workgroup_id`, `local_invocation_id`,
  `num_workgroups` and `local_invocation_index`; only referenced builtins are injected,
  so generated WGSL text legitimately changes with the kernel body. Do not
  assert WGSL emitted text. Assert execution results.
- **`threads_per_threadgroup` is deferred on WGSL.** WGSL has no
  `workgroup_size` builtin, only the `@workgroup_size` attribute, so referencing
  it in a `webgpu:` kernel is a loud compile error, by design. GEMM never
  references it, and `thread_index_in_threadgroup` maps to WGSL's native
  `local_invocation_index` instead of a composed formula.
- **Alias families**, the Nim spellings that expand to the canonical name:
  - CUDA: `blockIdx`, `blockDim`, `gridDim`, `threadIdx`, `syncthreads`
  - OpenCL: `get_global_id(d)`, `get_group_id(d)`, `get_local_id(d)`,
    `get_local_size(d)`, `get_num_groups(d)`, `barrier(flags)`,
    `workgroup_barrier(flags)`
  - GLSL: `gl_GlobalInvocationID`, `gl_WorkGroupID`, `gl_LocalInvocationID`,
    `gl_WorkGroupSize`, `gl_NumWorkGroups`, `gl_LocalInvocationIndex`
  - WGSL: `global_id`, `workgroup_id`, `local_invocation_id`,
    `num_workgroups`, `local_invocation_index`, `workgroupBarrier`
  - MSL: the canonical names themselves
  `workgroup_barrier(flags)` and `workgroupBarrier()` are one declaration with a defaulted flag.
  `get_global_size` is excluded from the vocabulary: it is OpenCL-only native.
- **Inert registration list.** `NimGpuFnBuiltins` in [`src/codegen/builtins/nim_builtins.nim`](src/codegen/builtins/nim_builtins.nim)
  still lists the OpenCL `get_*` names. The list is a dead contract: the alias
  templates expand during sem, before codegen's name-only registration path
  would consult it, so calls never reach it.

## Status

Crucible is under active development. The pass-architecture refactor (single `FnTable`, Symbol ref identity, `GpuRangeKind`, base58 mangling) is merged on `master` (commit `a076641`). NVRTC/CUDA is the primary, most-tested target; OpenCL, Vulkan, WebGPU, and Metal share the same IR and pipeline but have smaller suites and some backend-specific lowering (e.g. WGSL `injectAddressOf`, Vulkan SSBO/push-constant lowering in [`src/codegen/passes/passes_preprocessing.nim`](src/codegen/passes/passes_preprocessing.nim), Metal's dispatch-time threadgroup size and compiler-enforced 31-binding limit). WIP is tracked per-feature; gaps are labelled in the source with `XXX`/`WIP` comments.

## Source layout

```
workspace/crucible/
  crucible.nim                    Entry module (doc-comment only)
  AGENTS.md                       Test conventions and run commands
  src/
    codegen/
      gpu_compiler.nim            Compiler entry point: the `cuda:`/`opencl:`/`vulkan:`/`webgpu:` macros
      ir/
        gpu_types.nim             IR types: GpuAst, GpuType, GpuContext, Symbol, FnTable
        gpu_type_constructors.nim Type constructors
        nim_to_gpu.nim            Frontend: pure 1:1 Nim AST -> IR translator
        resolvers.nim             Type/overload resolution
      passes/
        pass_datatypes.nim        Pass/registry types, walk
        pass_registry.nim         PassRegistry
        passes_normalizations.nim Phase 4: frontend-extraction passes
        passes_legalizations.nim  Legalizations
        passes_preprocessing.nim  Preprocessing (incl. mangleNames, backend passes)
        passes_lowering.nim       Lowering (gtSpan -> ptr+len, etc.)
        passes_validations.nim    Pre/post validation passes
        passes_optimizations.nim  Optimizations
      targets/
        targets_lang.nim          Dispatches codegenCuda/OpenCL/Vulkan/WebGPU/Metal
        cuda_lang.nim opencl_lang.nim vulkan_lang.nim wgsl_lang.nim metal_lang.nim   Printers
        lang_utils.nim            Shared printer helpers
      builtins/                   Unified builtin vocabulary
        builtins_catalog.nim     Canonical coordinate/sync builtins + per-backend idiom aliases
        builtins_pragmas.nim     Backend pragmas: cudaName, workgroup, builtin, shared, ...
        builtins.nim             Re-exports the catalog and pragmas to the entry macros
        nim_builtins.nim         Nim-level builtins: operators, min/max/abs, inert OpenCL get_* list
    runtime/
      engines.nim                HwEngine — the sole public runtime API (init/ingest/getArtifact/run/chevrons)
      engines/                   CudaEngine (nvrtc.nim), OpenCLEngine (cl.nim), VulkanEngine (vk.nim), WgpuEngine (wgpu.nim), MetalEngine (metal.nim)
      exec/                      Low-level drivers (cuda/opencl/vulkan/wgpu runtimes; metal_runtime.nim: MSL compile/dispatch)
    abis/                         C/OpenCL/NVIDIA/Vulkan ABI bindings (objc_abi.nim: libobjc msgSend bridge)
    macros/ast_rebuilder.nim
  tests/codegen/
    ir/                           IR-level tests (pure Nim, no engine)
    nvrtc/ opencl/ vulkan/ webgpu/ metal/  Per-backend auto-runnable tests
  specs/                          (empty) planned spec files
  vendor/wgpu/                    Vendored wgpu headers/libs
```

## Build and run

Run from the repo root (`tattletale`). Requires a working Nim and the relevant toolchains (NVRTC/glslangValidator for compilation, or a CUDA/OpenCL/Vulkan/WebGPU/Metal device for `engine.run()`). The Metal tests run on any Apple Silicon Mac (macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, tested 2026-08-17).

```bash
# NVRTC (primary target)
nim cpp -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/nvrtc/test_*.nim

# OpenCL
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/opencl/test_*.nim

# Vulkan
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/vulkan/test_*.nim

# WebGPU
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/webgpu/test_*.nim

# Metal (macOS, on-device MSL compilation)
# tested ABI: macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, 2026-08-17
nim test_crucible_metal

# IR-level tests (pure Nim, e.g.)
nim c -r --hints:off --warnings:off \
  --debugger:native \
  --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
  workspace/crucible/tests/codegen/ir/test_base58.nim
```

## Related docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline, data-flow walkthrough, extension points.
- [`AGENTS.md`](AGENTS.md) — test naming, `engine.run()` requirement, minimal reproductions, debugging.
- [`workspace/ceramic/`](../ceramic/) — layout algebra consumed by Crucible.
- [`workspace/positron/`](../positron/) — kernel specifications consumed by Crucible.

## License

Dual-licensed under either of the MIT License or the Apache License, Version 2.0, at your option. See the license headers in each source file.
