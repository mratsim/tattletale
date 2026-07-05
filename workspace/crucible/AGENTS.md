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
| `test_nvrtc_<desc>.nim` | Auto-runnable on CUDA — uses `nvrtc.compile()` + `nvrtc.execute()` |
| `test_opencl_<desc>.nim` | Auto-runnable on OpenCL — uses `execOpenCL()` |
| `test_vulkan_<desc>.nim` | Auto-runnable on Vulkan |
| `test_webgpu_<desc>.nim` | Auto-runnable on WebGPU |
| `manual_<backend>_<desc>.nim` | **Manual only** — tests that expect a compile-time error (e.g., `proc that won't compile`). These cannot auto-run. |

### Test Requirements

**All tests MUST call `execute()`**, not just `getPtx()`/`compile()`. A compile check is not enough — it can produce valid CUDA that computes wrong results.

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
- Uses `execute()` to verify correctness

## When the environment lacks a backend

If running on a machine without CUDA (and thus can't NVRTC):
1. Compile the test up to `nvrtc.compile()` or inspect the codegen output
2. Print the generated code with `echo kernel` or inspect `nv.getPtx()`
3. Pass the exact run command to the user for manual execution
