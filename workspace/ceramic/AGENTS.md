# Ceramic Agents Guidelines

## Running tests

Ceramic tests are run through `tattletale/config.nims` tasks — from the
`tattletale/` dir (NOT this dir):

```bash
nim test_ceramic          # all tests in workspace/ceramic/tests
```

Mechanics (see `config.nims`):
- The task scans `tests/` for files named `test_*` or `t_*` and compiles
  each with `nim c` (C backend) via `testerCmd` into
  `build/tests/<file>` — a new `test_*.nim` in `tests/` is picked up
  automatically, no registration needed.
- **Non-recursive**: `tests/gpu/` is NOT included (run those directly,
  e.g. `nim cpp -r tests/gpu/test_*.nim`).
- The task aborts at the first failure.

## Test structure: run sections inside procs

- Wrap each test section in a `proc` (e.g. `proc sec1() = ...`) and call it at
  module scope — never run test logic at module scope directly. Blocks may
  separate sections *inside* a proc, but a test made of only module-level
  `block`s (no proc) is a no-go: a proc is what scopes locals (and frees
  resources) when it returns.
- Module-scope `var`/objects live until **program exit** (Nim templates do NOT
  scope `var` either). For CPU-only tests this is mostly harmless, but any
  object holding a backend resource (CUDA context, NVRTC program, OpenCL/Vulkan
  handles, GPU buffers) must be function-local — contexts hold a large
  device-memory reservation that is only returned on destruction, so N
  module-scope contexts can exhaust the GPU (`CUDA_ERROR_OUT_OF_MEMORY` on
  tiny allocs).
- `const` fixtures and `const` kernel sources at module scope are fine — only
  runtime resource-holding objects need the proc scope.

Single file (C backend, mirrors `testerCmd`):
```bash
nim c -r --hints:off --warnings:off \
  --outdir:build/tests/test_name --nimcache:nimcache/tests/test_name \
  workspace/ceramic/tests/test_file.nim
```

## Tensor access: `[]` vs `()`

- `<tensor>(<args>)` — dual dispatch:
  - If any arg is `_` (underscore/joker): returns a **sub-view** (TensorView). This is the primary use of `()`.
  - No `_`: returns a **scalar element**. Assignment via `()` is discouraged (`{.warning.}` emitted). Use `[]` / `[]=` instead.
- `<tensor>[<args>]` / `<tensor>[<args>] = val` — **preferred for scalar access**. Use this whenever there's no `_` in the args.

**Rule:** No `_` joker → prefer `[]=` for writes and `[]` for reads. Only use `()` when creating sub-views with `_`.

## Parameter ordering convention

Output/mutable parameters come first, except where the function name itself encodes the order:

| Operation | Ceramic | CuTe | Note |
|-----------|---------|------|------|
| copy | `copyFrom(dst, src)` | `copy(src, dst)` | output first |
| gemm | `gemm(C, A, B)` | `gemm(A, B, C)` | output first |
| axpby | `axpby(alpha, X, beta, Y)` | `axpby(alpha, x, beta, y)` | **exception**: name `axpby` = ordering |

## Kernel file structure

GPU-suitable kernels use flat-index iteration (`for i: tv(i) = ...`), acceptable on GPU (divmod hidden by parallelism). CPU kernels may use contiguity-fused paths.

| Category | File |
|----------|------|
| Fill (GPU) | `kernel_fillwith_gpu.nim` |
| Fill (CPU) | `kernel_fillwith_cpu.nim` |
| Copy (GPU) | `kernel_copy_gpu.nim` |
| Copy (CPU) | `kernel_copy_cpu.nim` |
| GEMM | `kernel_gemm_gpu.nim` |
| AXPBY | `kernel_axpby_gpu.nim` |
