# Crucible Agent Guidelines

AI coding tool guidelines for the Crucible codegen backend.

## TDD Workflow

**Always: test first, fix second.**

1. Create a minimal reproduction with **zero external dependencies** (no `workspace/ceramic/`, only crucible types + Nim stdlib)
2. Verify the repro **triggers the bug** — compile and capture the error
3. Only then implement the fix
4. Write tests for **all targets** (NVRTC, OpenCL, Vulkan, WebGPU) before considering the fix complete

## Test Organization

```
tests/codegen/
  ir/           — IR-level tests (INTERNAL codegen API — deep imports, no engine)
  nvrtc/        — CUDA via NVRTC (primary target)
  opencl/       — OpenCL
  vulkan/       — Vulkan
  webgpu/       — WebGPU
  metal/        — Metal (macOS)
```

**IR tests are the exception to the public import:** they test compiler internals
(passes, IR, type constructors) and keep deep `workspace/crucible/src/codegen/...`
imports. Only engine/runtime tests use the public API (below).

### Naming Convention

| Prefix | When to use |
|--------|-------------|
| `test_ir_<desc>.nim` | IR-level tests — INTERNAL codegen API (deep imports), no engine |
| `test_nvrtc_<desc>.nim` | Auto-runnable on CUDA — `import workspace/crucible`; `bkCuda.init()` + `engine.ingest()` + `engine.run()` (or `engine.run<<(grid, blk)>>(...)`) |
| `test_opencl_<desc>.nim` | Auto-runnable on OpenCL — `import workspace/crucible`; `bkOpenCL.init()` + `engine.ingest()` + `engine.run()` |
| `test_vulkan_<desc>.nim` | Auto-runnable on Vulkan — `import workspace/crucible`; `bkVulkan.init()` + `engine.ingest()` + `engine.run()` |
| `test_webgpu_<desc>.nim` | Auto-runnable on WebGPU — `import workspace/crucible`; `bkWGSL.init()` + `engine.ingest()` + `engine.run()` |
| `test_metal_<desc>.nim` | Auto-runnable on Metal (macOS) — `import workspace/crucible`; `bkMetal.init()` + `engine.ingest()` + `engine.run()` (or `engine.run<<(grid, blk)>>(...)`; tested ABI: macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, 2026-08-17) |
| `manual_<backend>_<desc>.nim` | **Manual only** — tests that expect a compile-time error (e.g., `proc that won't compile`). These cannot auto-run. |

### Test Requirements

**One import for everything:** `import workspace/crucible` re-exports the DSL
macros (`cuda:`/`opencl:`/`vulkan:`/`webgpu:`/`metal:`) and the engine API (`bkCuda`/
`bkOpenCL`/`bkVulkan`/`bkWGSL`/`bkMetal`, `init`, `ingest`, `getArtifact`, `run`, `check`,
`deviceName`). IR tests are the exception — they keep deep internal imports.

**All tests must call `engine.run()`** (or the chevron form `engine.run<<(grid, blk)>>(...)` — no space around `<<`/`>>`), not just `getArtifact()`/`ingest()`. A compile check is not enough — it can produce valid CUDA that computes wrong results.

**Wrap the engine-execution section in a private `proc runTest()`** and call it
from `when isMainModule: runTest()`. The uniform name makes tests grep-able, and
proc scope deterministically destroys the engine at return — which is what tests
the RAII lifecycle (`=destroy` never runs on module-scope engines). See
`tests/codegen/vulkan/test_vulkan_scalar_args.nim` for the pattern.

**Never declare backend-resource objects at module scope** — blocks may separate
sections within a proc, but a test made of only module-level `block`s (no proc)
is a no-go. Never declare backend-resource objects (engines, contexts, modules,
OpenCL/Vulkan handles) at module scope via template helpers:
- Nim templates do **not** scope `var` declarations — a `template runKernel(...) = var engine = bkCuda.init()` expanded at module level keeps every `engine` alive until program exit.
- CUDA contexts are expensive: one live context holds a large device-memory reservation that is only returned on destruction. N module-scope contexts can exhaust the GPU (`CUDA_ERROR_OUT_OF_MEMORY` on tiny allocs — see the `test_nvrtc_if_expr` fix).
- A `proc` scopes its locals naturally: resources are released when the proc returns, so N kernels in one test file use one context at a time.
- Module-scope `const` kernel sources are fine — only runtime objects (engines created via `bkCuda.init()`/`bkOpenCL.init()` etc. from `runtime/engines`) must be function-local.

Exception: tests for backends not supported in the current environment. For those:
1. Do a **sanity check** by inspecting the generated code text
2. Report the run command to the user so they can execute it manually

## Engine API — per-target notes

The engine lifecycle: `init()` → `ingest(source)` → `getArtifact()` →
`run` (re-ingest replaces the context; RAII releases the old one). Args tuple
follows kernel-param order minus the output; scalars are by-value (4-byte),
pointers become device buffers.

- **CUDA / OpenCL** — `grid`/`blk` are host-side launch config; `blk` is free
  (not shader-baked).
- **Metal** — `grid`/`blk` are host-side launch config like CUDA/OpenCL.
  `blk` is dispatch-time, validated against the Apple Silicon 1024-thread limit.
  Kernel args bind as buffers (output at index 0, then inputs in order).
  Scalars pack into one shared constant buffer at 16-byte slots.
  The engine enforces the 31-binding limit at ingest.
  Output reads back directly from `contents()` after `waitUntilCompleted`,
  with no staging or map path.
- **Vulkan / WebGPU** — the workgroup size is baked into the shader at codegen
  time; the engine validates the run `blk` against the baked size and quits
  loudly on mismatch.
- **Vulkan scalar args** — by-value params are emitted as push constants
  (`layout(push_constant) uniform KernelParams`); the engine packs 4-byte
  scalars into the push-constant range. Only 4-byte scalars are supported
  (loud quit otherwise; vec/struct by-value args need a real std430 layout).
- **One kernel per `vulkan:` source when using scalars** — a multi-kernel
  source unions all kernels' scalar params into one file-scope block and
  misaligns them (kernel 2's params land after kernel 1's). Pointer-only
  multi-kernel sources are fine.

### Run commands

```bash
cd tattletale

# NVRTC
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

# Metal (macOS)
# tested ABI: macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, 2026-08-17
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/metal/test_*.nim
# whole suite, including the intentional-quit -d:case* binaries:
nim test_crucible_metal
```

## File Header

Every test file must have a header comment:

```nim
## <Concise description of what this tests>
##
## Run:
##   cd tattletale
##   <exact run command>
```

## Minimal Reproductions

A "minimal repro" for a crucible bug:
- Uses ONLY types defined in the test file (no ceramic, no external lib)
- Has a `cuda:` / `opencl:` / `vulkan:` / `webgpu:` / `metal:` block
- Demonstrates the failing pattern in the simplest possible way
- Runs the kernel via `engine.run()` to verify correctness

## When the environment lacks a backend

If running on a machine without CUDA (and thus can't NVRTC):
1. Compile the test up to `engine.ingest()` or inspect the codegen output
2. Print the generated code with `echo kernel` or inspect `engine.getArtifact()`
3. Pass the exact run command to the user for manual execution

## Debugging

### Inspecting Nim AST during macro execution

Inside `toGpuAst` (or any function called from a macro), you have access to
NimNode objects from the typed AST. Use `std/macros` procs to inspect them:

```nim
# .repr — prints the Nim source representation
echo node.repr
# "for i in items(Slice[int](a: 0, b: 127)):"

# .treeRepr — prints the AST tree structure (kind, children, leaf values)
echo node.treeRepr
# ForStmt
#   Sym "i"
#   Call
#     Sym "items"
#     ObjConstr ...
```

Add these directly inside `toGpuAst`, `resolveType`, or any other macro-
context function. The output appears during compilation.

### Stack trace navigation

During macro execution (the common case for Crucible bugs), the crash is
pure Nim — no C++ code involved. The stack trace line numbers come from
the Nim source as the compiler sees it. If the reported line seems off,
the file probably changed between compilations (edits shift lines).

To keep traces precise:

- **One statement per line.** Never `;` on the same line that contains other
  code.

- **Loop bodies on separate lines.** A for-loop with a body on the same line
  crashes trace to the `for` line, hiding which iteration or expression
  triggered the failure.

## Coding Style

**One statement per line.**
Never put multiple statements on the same line with `;`.
Every `let`, `var`, `const`, `if`, `for`, `while`, assignment, expression
statement must be on its own line.

Wrong:
```nim
let a = 1; let b = 2
C[0] = a; C[1] = b
```

Right:
```nim
let a = 1
let b = 2
C[0] = a
C[1] = b
```

Rationale: `;` kills debuggability — you can't set breakpoints per-statement,
stack traces lose line-level precision, and generated CUDA errors are harder
to map back to source.
