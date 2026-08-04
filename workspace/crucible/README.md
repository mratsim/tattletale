# Crucible

GPU kernel code generator: translates a GPU DSL written in Nim into native GPU source for four backends — CUDA, OpenCL, Vulkan (GLSL), and WebGPU (WGSL).

## What it is / what it isn't

Crucible is a **compile-time code generator**, not a runtime. Its entry points are Nim macros (`cuda`, `opencl`, `vulkan`, `webgpu`) that consume a GPU DSL body and emit a string of native GPU source code. It does not ship a device runtime, a scheduler, or a host-side execution model of its own.

It sits on top of the rest of the tattletale stack:

- It **consumes** the layout algebra produced by `workspace/ceramic/` and the kernel specifications produced by `workspace/positron/`. Per the header in [`workspace/crucible/crucible.nim`](crucible.nim): "takes Ceramic layout algebra and Positron kernel specifications and emits native GPU code (CUDA, OpenCL, Vulkan, WebGPU)".
- The generated source is handed to an external toolchain to run: NVRTC compiles CUDA to PTX, shaderc compiles GLSL to SPIR-V, and the host-side ABIs under [`src/abis/`](src/abis/) load and launch the results (see `NvrtcContext`, `ShadercContext`, `VulkanContext`).

Because it is a code generator, "correctness" of a feature means the emitted code compiles *and* computes the right result — which is why every test must call `execute()`, not just `compile()` (see [`AGENTS.md`](AGENTS.md)).

## Capability proof

All claims below are source-linked.

- **Four backends from one IR.** The frontend [`src/codegen/ir/nim_to_gpu.nim`](src/codegen/ir/nim_to_gpu.nim) lowers Nim AST to a backend-neutral IR once; each backend is a "dumb syntax printer" under [`src/codegen/targets/`](src/codegen/targets/) (`cuda_lang.nim`, `opencl_lang.nim`, `vulkan_lang.nim`, `wgsl_lang.nim`). The four macros in [`src/codegen/gpu_compiler.nim`](src/codegen/gpu_compiler.nim) share a common pass pipeline (`registerCommonPasses`) and diverge only in backend-specific passes and keyword checks.
- **A real pass pipeline.** The frontend is a pure 1:1 AST translator; semantic work lives in named, testable passes under [`src/codegen/passes/`](src/codegen/passes/) (normalizations, legalizations, preprocessing, lowering, validations). Passes are declared with a name, kind, phase, and dependency list and run through a [`PassRegistry`](src/codegen/passes/pass_registry.nim). This refactor landed in commit `a076641` ("feat(crucible): Complete pass-architecture refactoring (#43)").
- **Single `FnTable`, Symbol ref identity.** `GpuContext` carries one unified function table keyed by immutable symbol fingerprint (`fnTable` in [`src/codegen/ir/gpu_types.nim`](src/codegen/ir/gpu_types.nim), replacing five separate function tables). Symbols are `ref` objects with a stable `iSym` fingerprint and a mutatable `name` (see `Symbol` in the same file).
- **Explicit range semantics.** `GpuRangeKind` distinguishes `rkInclusive` (`a..b`, emitted `i <= end`) from `rkExclusive` (`a..<b`, emitted `i < end`), replacing a `+1` hack in the backend printers.
- **Collision-safe name mangling.** `mangleNames` applies a `NamePolicy`; `npHashSuffix` appends a 7-character base58 hash of the 64-bit signature (58^7 ≈ 2.2e12 namespace) so distinct signatures survive mangling. See `mangleNamesImpl` in [`src/codegen/passes/passes_preprocessing.nim`](src/codegen/passes/passes_preprocessing.nim) and the encoder in [`src/codegen/ir/gpu_types.nim`](src/codegen/ir/gpu_types.nim).
- **IR and backend test suites.** 22 IR-level tests under [`tests/codegen/ir/`](tests/codegen/ir/) (roundtrip, scope, symbols, fntable, base58, per-pass tests), plus auto-runnable suites per backend under [`tests/codegen/`](tests/codegen/): `nvrtc/`, `opencl/`, `vulkan/`, `webgpu/`. See [`AGENTS.md`](AGENTS.md) for the naming convention and the `manual_*` exception.
- **Compile-time and runtime codegen.** `cuda`/`opencl`/`vulkan`/`webgpu` macros emit code at compile time; the runtime `codegen(gen, ast, ...)` proc in [`src/codegen/gpu_compiler.nim`](src/codegen/gpu_compiler.nim) clones the IR and regenerates for a chosen backend at runtime.

## Status

Crucible is under active development. The pass-architecture refactor (single `FnTable`, Symbol ref identity, `GpuRangeKind`, base58 mangling) is merged on `master` (commit `a076641`). NVRTC/CUDA is the primary, most-tested target; OpenCL, Vulkan, and WebGPU share the same IR and pipeline but have smaller suites and some backend-specific lowering (e.g. WGSL `injectAddressOf`, Vulkan SSBO/push-constant lowering in [`src/codegen/passes/passes_preprocessing.nim`](src/codegen/passes/passes_preprocessing.nim)). WIP is tracked per-feature; gaps are labelled in the source with `XXX`/`WIP` comments.

## Source layout

```
workspace/crucible/
  crucible.nim                    Entry module (doc-comment only)
  AGENTS.md                       Test conventions and run commands
  src/
    codegen/
      gpu_compiler.nim            Compiler entry point: macros + runtime codegen
      nvrtc.nim cl.nim vk.nim wgpu.nim   Toolchain wrappers
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
        targets_lang.nim          Dispatches codegenCuda/OpenCL/Vulkan/WebGPU
        cuda_lang.nim opencl_lang.nim vulkan_lang.nim wgsl_lang.nim   Printers
        lang_utils.nim            Shared printer helpers
      builtins/                   Per-backend and Nim builtins
      exec/                       Host-side runtimes (cuda/opencl/vulkan/wgpu)
    abis/                         C/OpenCL/NVIDIA/shaderc/Vulkan ABI bindings
    macros/ast_rebuilder.nim
  tests/codegen/
    ir/                           IR-level tests (22 files)
    nvrtc/ opencl/ vulkan/ webgpu/  Per-backend auto-runnable tests
  specs/                          (empty) planned spec files
  vendor/wgpu/                    Vendored wgpu headers/libs
```

## Build and run

Run from the repo root (`tattletale`). Requires a working Nim and the relevant toolchains (NVRTC/shaderc for compilation, or a CUDA/OpenCL/Vulkan/WebGPU device for `execute()`).

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

# IR-level tests (pure Nim, e.g.)
nim c -r --hints:off --warnings:off \
  --debugger:native \
  --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
  workspace/crucible/tests/codegen/ir/test_base58.nim
```

## Related docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline, data-flow walkthrough, extension points.
- [`AGENTS.md`](AGENTS.md) — test naming, `execute()` requirement, minimal reproductions, debugging.
- [`workspace/ceramic/`](../ceramic/) — layout algebra consumed by Crucible.
- [`workspace/positron/`](../positron/) — kernel specifications consumed by Crucible.

## License

Dual-licensed under either of the MIT License or the Apache License, Version 2.0, at your option. See the license headers in each source file.
