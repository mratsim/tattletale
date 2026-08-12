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
  nvrtc/        — CUDA via NVRTC (primary target)
  opencl/       — OpenCL
  vulkan/       — Vulkan
  webgpu/       — WebGPU
```

### Naming Convention

| Prefix | When to use |
|--------|-------------|
| `test_nvrtc_<desc>.nim` | Auto-runnable on CUDA — uses `bkCuda.init()` + `engine.ingest()` + `engine.run()` |
| `test_opencl_<desc>.nim` | Auto-runnable on OpenCL — uses `bkOpenCL.init()` + `engine.ingest()` + `engine.run()` |
| `test_vulkan_<desc>.nim` | Auto-runnable on Vulkan — uses `bkVulkan.init()` + `engine.run()` |
| `test_webgpu_<desc>.nim` | Auto-runnable on WebGPU — uses `bkWGSL.init()` + `engine.run()` |
| `manual_<backend>_<desc>.nim` | **Manual only** — tests that expect a compile-time error (e.g., `proc that won't compile`). These cannot auto-run. |

### Test Requirements

**All tests must call `engine.run()`** (or the chevron form `engine.run<<(grid, blk)>>(...)`), not just `getArtifact()`/`ingest()`. A compile check is not enough — it can produce valid CUDA that computes wrong results.

**Run each test inside a `proc`** — blocks may separate sections within a proc, but a test made of only module-level `block`s (no proc) is a no-go. Never declare backend-resource objects (engines, contexts, modules, OpenCL/Vulkan handles) at module scope via template helpers:
- Nim templates do **not** scope `var` declarations — a `template runKernel(...) = var engine = bkCuda.init()` expanded at module level keeps every `engine` alive until program exit.
- CUDA contexts are expensive: one live context holds a large device-memory reservation that is only returned on destruction. N module-scope contexts can exhaust the GPU (`CUDA_ERROR_OUT_OF_MEMORY` on tiny allocs — see the `test_nvrtc_if_expr` fix).
- A `proc` scopes its locals naturally: resources are released when the proc returns, so N kernels in one test file use one context at a time.
- Module-scope `const` kernel sources are fine — only runtime objects (engines created via `bkCuda.init()`/`bkOpenCL.init()` etc. from `runtime/engines`) must be function-local.

Exception: tests for backends not supported in the current environment. For those:
1. Do a **sanity check** by inspecting the generated code text
2. Report the run command to the user so they can execute it manually

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
- Has a `cuda:` / `opencl:` / `vulkan:` / `webgpu:` block
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
