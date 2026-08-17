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
      builtins/                   Per-backend and Nim builtins
        metal_builtins.nim        Metal index-builtin dummies (five MSL attribute names, one param per referenced builtin)
    runtime/
      engines.nim                HwEngine — the sole public runtime API (init/ingest/getArtifact/run/chevrons)
      engines/                   CudaEngine (nvrtc.nim), OpenCLEngine (cl.nim), VulkanEngine (vk.nim), WgpuEngine (wgpu.nim), MetalEngine (metal.nim)
      exec/                      Low-level drivers (cuda/opencl/vulkan/wgpu runtimes; metal_runtime.nim: MSL compile/dispatch)
    abis/                         C/OpenCL/NVIDIA/Vulkan ABI bindings (objc_abi.nim: libobjc msgSend bridge)
    macros/ast_rebuilder.nim
  tests/codegen/
    ir/                           IR-level tests (22 files)
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
