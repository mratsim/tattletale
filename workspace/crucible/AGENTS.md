# Crucible Agent Guidelines

AI coding tool guidelines for the Crucible codegen backend.

## Modifying Crucible

**Always check what exists before adding a new helper.**
The codebase carries a large vocabulary of type constructors, resolvers, and IR utilities.
Search `src/codegen/frontend/`, `src/codegen/ir/`, and `src/codegen/passes/` for an existing
helper before writing a new one.

**Nanopass architecture.**

- The frontend (`src/codegen/frontend/`) and the printers (`src/codegen/targets/`) stay
  as simple as possible, ideally 1:1 with the source and the target.
- All smartness and complexity is pushed into a pass under `src/codegen/passes/`.
- A pass is independently testable: build the input IR, run the pass, diff the IR
  against the expected output (`tests/ir/`).
- A transform placed in the frontend or a printer defeats that: nothing can observe
  the delta it produces.

## TDD Workflow

**Always: test first, fix second.**

1. Create a minimal reproduction with **zero external dependencies** (no `workspace/ceramic/`, only crucible types + Nim stdlib)
2. Verify the repro **triggers the bug**: compile and capture the error
3. Only then implement the fix
4. Write tests for **all targets** (CUDA, OpenCL, Vulkan, WebGPU, Metal) before considering the fix complete

## Test Organization

```
tests/
  ir/             : IR-level tests (INTERNAL codegen API, deep imports, no engine)
    mangling/ types/ functions/ control_flow/
    legalizations/ optimizations/
  codegen/
    cuda/         : CUDA via NVRTC (primary target)
    opencl/       : OpenCL
    vulkan/       : Vulkan
    webgpu/       : WebGPU
    metal/        : Metal (macOS)
    each backend: app/ compiletime/ control_flow/ device/ functions/
                  generics/ serialization/ types/ (+ inline_asm/ on nvrtc)
```

**IR tests are the exception to the public import:** they test compiler internals
(passes, IR, type constructors) and keep deep `workspace/crucible/src/codegen/...` imports.
Only engine/runtime tests use the public API (below).

### Naming Convention

| Prefix | When to use |
|--------|-------------|
| `t_ir_<desc>.nim` | IR-level tests under `tests/ir/`, the INTERNAL codegen API (deep imports), no engine |
| `t_cuda_<desc>.nim` | Auto-runnable on CUDA: `import workspace/crucible`, `bkCuda.init()` + `engine.ingest()` + `engine.launchKernel()` (or the chevron form) |
| `t_opencl_<desc>.nim` | Auto-runnable on OpenCL: `import workspace/crucible`, `bkOpenCL.init()` + `engine.ingest()` + `engine.launchKernel()` |
| `t_vulkan_<desc>.nim` | Auto-runnable on Vulkan: `import workspace/crucible`, `bkVulkan.init()` + `engine.ingest()` + `engine.launchKernel()` |
| `t_webgpu_<desc>.nim` | Auto-runnable on WebGPU: `import workspace/crucible`, `bkWebGPU.init()` + `engine.ingest()` + `engine.launchKernel()` |
| `t_metal_<desc>.nim` | Auto-runnable on Metal (macOS): `import workspace/crucible`, `bkMetal.init()` + `engine.ingest()` + `engine.launchKernel()` (or the chevron form; tested ABI: macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, 2026-08-17) |
| `manual_<backend>_<desc>.nim` | **Manual only**: tests expecting a compile-time error, they cannot auto-run. |

### Test Requirements

**Header shape:** every test file opens with the same shape, nothing else:

- line 1: the title, one line, it names the surface under test
- next: `Test that <the single behavior the test asserts>`
- a blank line, then the run command verbatim to the end
- no mechanism narration, no journey, the linter rule
  `test-header-testthat` enforces the shape

**One import for everything:** `import workspace/crucible` re-exports the public surface.
IR tests are the exception, they keep deep internal imports.

- DSL macros: `cuda:`/`opencl:`/`vulkan:`/`webgpu:`/`metal:`
- Engines: `bkCuda`/`bkOpenCL`/`bkVulkan`/`bkWebGPU`/`bkMetal`, `init`, `ingest`,
  `getArtifact`, `check`, `deviceName`
- Launch: `launchKernel`, `launchDeviceFn`

**All tests must call `engine.launchKernel()`**, the chevron form
`engine.launchKernel<<(grid, blk)>>(...)` with no space around `<<`/`>>`,
or `engine.launchDeviceFn`, not just `getArtifact()`/`ingest()`.
A compile check is not enough, it can produce valid CUDA that computes wrong results.

**Wrap the engine-execution section in a private `proc runTest()`** and call it
from `when isMainModule: runTest()`.

- The uniform name makes tests grep-able.
- Proc scope deterministically destroys the engine at return, which tests the RAII
  lifecycle (`=destroy` never runs on module-scope engines).
- See `tests/codegen/vulkan/serialization/t_vulkan_scalar_args.nim` for the pattern.

**Never declare backend-resource objects at module scope.** Blocks may separate
sections within a proc, but a test made of only module-level `block`s (no proc) is a no-go.
Never declare backend-resource objects (engines, contexts, modules, OpenCL/Vulkan handles)
at module scope via template helpers:

- Nim templates do **not** scope `var` declarations: a module-level `template runKernel(...)`
  containing `var engine = bkCuda.init()` keeps every `engine` alive until program exit.
- CUDA contexts are expensive: one live context holds a large device-memory reservation,
  only returned on destruction. N module-scope contexts can exhaust the GPU,
  `CUDA_ERROR_OUT_OF_MEMORY` on tiny allocs, `t_cuda_if_expr` documents the fix.
- A `proc` scopes its locals naturally: resources are released when the proc returns, so N
  kernels in one test file use one context at a time.
- Module-scope `const` kernel sources are fine: only runtime objects, engines created via
  `bkCuda.init()`/`bkOpenCL.init()` etc., must be function-local.

Exception: tests for backends not supported in the current environment.
For those:

1. Do a **sanity check** by inspecting the generated code text
2. Report the run command to the user so they can execute it manually

### What counts as a test

A test exercises compiler functionality: an AST transformation, an emission, a runtime result.

Hand-built AST fed to an internal accessor with the expected kind hardcoded asserts
the type of the test, not the compiler.

In scope: passes, rewrites, folds, lowerings, codegen.
The assertion targets the delta between input and output.

- echo tests are banned: hand-build a node, stuff a field, assert the accessor returns it
  (`scopeAdd` then `scopeGet`), tests nothing that can regress and ossifies internals
- printer labels are not contracts, a `pretty()` output containing "Var" is a rendering detail
- a pass test whose every block asserts "nothing changed" never exercises the rewrite

Every pass test asserts the positive case plus a hazard negative: an input that must pass through untouched.

- derived accessors are tested through their consumer pass: `insertPointerDeref` deref depth,
  the atomic pointee check accepting `addr s.counter`
- value assertions beat shape assertions: a hand-built gtInt64 index defeats a hardcoded-default regression, a structural check does not

Linter rule of record: `test-header-testthat` caps the abstract
at 2 lines and 3 one-liner bullets.

## Engine API, per-target notes

The engine lifecycle: `init()` → `ingest(source)` → `getArtifact()` → `launchKernel`.
Re-ingest replaces the context, RAII releases the old one.
The args tuple order matches the kernel params minus the output, scalars are by-value
(4-byte), pointers become device buffers.

| Target | `grid`/`blk` |
| --- | --- |
| CUDA / OpenCL | host-side launch config, `blk` free (not shader-baked) |
| Metal | host-side like CUDA/OpenCL, `blk` dispatch-time, validated against the Apple Silicon 1024-thread limit |
| Vulkan / WebGPU | workgroup size baked into the shader at codegen, the engine validates the run `blk` against it and quits loudly on mismatch |

| Target | Scalar args and readback |
| --- | --- |
| CUDA / OpenCL | by-value 4-byte scalars, pointers become device buffers |
| Metal | buffers, output at index 0 then inputs in order, scalars in one shared constant buffer at 16-byte slots |
| Metal | more than 31 bindings rejected at ingest via NSError, readback from `contents()` after `waitUntilCompleted`, no staging or map path |
| Vulkan | by-value params are push constants (`layout(push_constant) uniform KernelParams`), 4-byte only, loud quit otherwise |
| Vulkan | one kernel per `vulkan:` source with scalars, a multi-kernel source unions params file-scope and misaligns them, pointer-only multi-kernel is fine |

### Run commands

```bash
cd tattletale

# CUDA (NVRTC)
nim cpp -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/cuda/*/t_cuda_*.nim

# OpenCL
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/opencl/*/t_opencl_*.nim

# Vulkan
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/vulkan/*/t_vulkan_*.nim

# WebGPU
nim c -r --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/crucible/tests/codegen/webgpu/*/t_webgpu_*.nim

# Metal (macOS)
# tested ABI: macOS 26.6.1, Nim 2.2.10, CLT SDK 26.5, MSL 4.0, 2026-08-17
nim test_crucible_metal
```

## Builtins: canonical names and the per-backend names

Kernel bodies write coordinates and barriers in the canonical MSL vocabulary:

- `thread_position_in_grid`, `threadgroup_position_in_grid`, `thread_position_in_threadgroup`
- `threads_per_threadgroup`, `threadgroups_per_grid`, `thread_index_in_threadgroup`
- `threadgroup_barrier`

Every other backend name is a template alias expanding to the canonical name during Nim semantic analysis,
so the IR only ever contains canonical names.

- [`src/builtins/builtins_catalog.nim`](src/builtins/builtins_catalog.nim)
  is the source of truth for the alias set.
- The printers under `src/codegen/targets/` emit the native names below.

| Canonical (MSL) | CUDA | OpenCL | Vulkan (GLSL) | WGSL |
|---|---|---|---|---|
| `thread_position_in_grid` | `(blockIdx.d*blockDim.d+threadIdx.d)` | `get_global_id(d)` | `gl_GlobalInvocationID` | `global_id` |
| `threadgroup_position_in_grid` | `blockIdx` | `get_group_id(d)` | `gl_WorkGroupID` | `workgroup_id` |
| `thread_position_in_threadgroup` | `threadIdx` | `get_local_id(d)` | `gl_LocalInvocationID` | `local_invocation_id` |
| `threads_per_threadgroup` | `blockDim` | `get_local_size(d)` | `gl_WorkGroupSize` | deferred (no WGSL builtin) |
| `threadgroups_per_grid` | `gridDim` | `get_num_groups(d)` | `gl_NumWorkGroups` | `num_workgroups` |
| `thread_index_in_threadgroup` | `(threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x)` | `(get_local_id(2)*get_local_size(0)*get_local_size(1) + get_local_id(1)*get_local_size(0) + get_local_id(0))` | `gl_LocalInvocationIndex` | `local_invocation_index` |
| `threadgroup_barrier` | `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | `barrier()` | `workgroupBarrier()` |
| `reduceAdd` | `warpReduceAdd` (emitted `__forceinline__` template) | `sub_group_reduce_add` | `subgroupAdd` | `subgroupAdd` (no directive; subgroups device feature, see note) |
| `reduceMax` | `warpReduceMax` (emitted `__forceinline__` template) | `sub_group_reduce_max` | `subgroupMax` | `subgroupMax` (no directive; subgroups device feature, see note) |

- CUDA and OpenCL name the vector coordinates per component (`d` is the accessed component).
  Whole-value uses of the coordinate vectors are out of scope.
- WGSL injects one `@builtin(...)` kernel param per referenced canonical coordinate,
  and the body references the param name. Generated WGSL text therefore changes
  with each kernel body: do not assert WGSL param naming, ordering, or injection-set text,
  assert execution results (`engine.launchKernel()`).
- `threads_per_threadgroup` is deferred on WGSL: WGSL has no `workgroup_size` builtin,
  only the `@workgroup_size` attribute. Referencing it in a `webgpu:` kernel is a loud
  compile error. `thread_index_in_threadgroup` maps to the WGSL native name
  `local_invocation_index`.

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
- Runs the kernel via `engine.launchKernel()` to verify correctness

## When the environment lacks a backend

If running on a machine without CUDA (and thus can't NVRTC):

1. Compile the test up to `engine.ingest()` or inspect the codegen output
2. Print the generated code with `echo kernel` or inspect `engine.getArtifact()`
3. Pass the exact run command to the user for manual execution

## Debugging

### Inspecting Nim AST during macro execution

Inside `toGpuAst` (or any function called from a macro), you have access to NimNode
objects from the typed AST. Use `std/macros` procs to inspect them:

```nim
# .repr: prints the Nim source representation
echo node.repr
# "for i in items(Slice[int](a: 0, b: 127)):"

# .treeRepr: prints the AST tree structure (kind, children, leaf values)
echo node.treeRepr
# ForStmt
#   Sym "i"
#   Call
#     Sym "items"
#     ObjConstr ...
```

Add these directly inside `toGpuAst`, `resolveType`, or any other macro-context function.
The output appears during compilation.

### Stack trace navigation

During macro execution (the common case for Crucible bugs), the crash is pure Nim,
no C++ code involved.

- The stack trace line numbers come from the Nim source as the compiler sees it.
- If the reported line seems off, the file probably changed between compilations
  (edits shift lines).

To keep traces precise:

- **One statement per line**: never `;` on the same line that contains other code.
- **Loop bodies on separate lines**: a for-loop with a body on the same line crashes trace
  to the `for` line, hiding which iteration or expression triggered the failure.

## Coding Style

**One statement per line.**
Never put multiple statements on the same line with `;`.
Every `let`, `var`, `const`, `if`, `for`, `while`, assignment, expression statement
must be on its own line.

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

Rationale: `;` kills debuggability, you can't set breakpoints per-statement,
stack traces lose line-level precision, and generated CUDA errors are harder
to map back to source.
