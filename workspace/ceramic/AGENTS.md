# Ceramic Agents Guidelines

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
