# Ceramic

CuTe's layout algebra, ported to Nim and portable to any GPU and CPU.

## What it is

Ceramic is a layout-algebra and tensor library in Nim. It is a port of
[NVIDIA CuTe](https://github.com/NVIDIA/cutlass)'s layout algebra — the
compile-time `Layout[Shape, Stride]` model — with the same core types and
transformations, expressed as Nim types and macros.

Unlike CuTe/Cutlass, which target NVIDIA CUDA exclusively, Ceramic's GPU
code generation is backend-agnostic. GPU kernels are emitted by
[Crucible](../crucible/) (`workspace/crucible/AGENTS.md`), which lowers Nim
to CUDA, OpenCL, Vulkan, and WebGPU. Ceramic ships CPU kernels alongside the
GPU kernels so the same layout algebra drives both paths.

The public entry point is `ceramic.nim`, which re-exports the layout algebra
and the CPU/GPU kernels.

## Capability proof

All capabilities below are backed by source in this directory.

### Layout algebra

- Core types `Layout[Shape, Stride]` and `Int[N]` — `src/layouts_datatypes.nim`,
  `src/int_tuples.nim`.
- Layout construction — `src/layout_constructors.nim` (`make_layout`,
  `col_major_strides`, `compact_order`, `LayoutCT`).
- Transformations and selectors — `src/layouts.nim` (mode, isCompact,
  filter_zeros, padRight/padLeft, mapLeavesWith, upcast/downcast, zipModes,
  groupModes, takeModes, selectModes, replaceMode).
- Layout algebra — `src/layout_algebra.nim` (coalesce, complement, compose,
  logical_divide, zipped_divide, right_inverse, left_inverse,
  blocked_product, raked_product, tile_to_shape).
- Indexing — `src/layout_indexing.nim`, `src/layout_indexing_gpu.nim`
  (crd2idx, idx2crd, slice, dice, X/Y markers).
- Tests: `tests/test_layout_algebra.nim`, `tests/test_layout_operators.nim`,
  `tests/test_layout_indexing.nim`, `tests/test_int_tuples.nim`.

### CPU + GPU kernels

Ceramic ships both a CPU and a GPU kernel for the same operations, so a tile
described by a layout can be run on either path:

| Op | CPU | GPU |
|----|-----|-----|
| fill | `src/kernel_fillwith_cpu.nim` | `src/kernel_fillwith_gpu.nim` |
| copy | `src/kernel_copy_cpu.nim` | `src/kernel_copy_gpu.nim` |
| GEMM | (see microkernels below) | `src/kernel_gemm_gpu.nim` |
| epilogue (D = α·AB + β·C) | fused into the GEMM | `src/tile_algebra/epilogues.nim` |

CPU kernels fuse contiguous suffixes into `copyMem`/`zeroMem` and use
stride-sorted nested loops otherwise; GPU kernels use flat-index `crd2idx`
iteration, which is acceptable on GPU where divmod is cheap. See the codegen
flow comments in `src/kernel_fillwith_cpu.nim` and `src/kernel_copy_cpu.nim`.

### CPU GEMM within ~3% of OpenBLAS

`benchmark/bench_ex02_matmul_cpu_simd.nim` benches three CPU GEMM variants —
`examples/ex02a_matmul_handtuned.nim` (hand-tuned), `examples/ex02b_matmul_layout_algebra.nim`
(layout-algebra), and `benchmark/laser_matmul/gemm.nim` (Laser reference) — against
OpenBLAS/cblas on square float32 matrices at sizes 128/512/1024/1920. It asserts
numerical correctness against both the Laser reference and cblas via `maxAbsErr`
(`allClose`, tol `5e-4`). The layout-algebra GEMM is reported within ~3% of
OpenBLAS. Run with `-d:cblas` to include the OpenBLAS comparison.

GEMM microkernels (AVX/AVX2/AVX512/SSE) live in
`benchmark/laser_matmul/gemm_ukernel_*.nim` and
`examples/ex02_matmul_microkernels/`.

## Status

Layout algebra, tensor types, and CPU fill/copy kernels are functional and
tested. GPU code generation is done by Crucible.

**WIP — GPU batch GEMM:** the faithful CuTe `sgemm_1` port
(`experiments/nvidia_cutlass_cute_tutorial/sgemm_1.nim`) is not fully
working. Individual NVRTC kernel tests pass, but the full `sgemm_1` kernel
does not. See `experiments/nvidia_cutlass_cute_tutorial/sgemm_1.nim` and
`tests/gemm/test_kernel_gemm.nim`. Do not treat the GPU batch-GEMM path as
production-ready.

## Source layout

```
workspace/ceramic/
  ceramic.nim                 — public entry point (re-exports)
  AGENTS.md                   — conventions (access, parameter order, kernel naming)
  src/
    int_tuples.nim            — Int[N] tuples, scans, reductions
    layouts_datatypes.nim     — Layout[Shape, Stride] type, predicates
    layout_constructors.nim   — make_layout, col_major_strides, LayoutCT
    layouts.nim               — transforms and selectors
    layout_algebra.nim        — coalesce, complement, compose, divide, products
    layout_indexing*.nim      — crd2idx, idx2crd, slice, dice
    tensor_datatypes.nim      — Tensor / TensorView
    tensors.nim               — []/() access, slice, local_tile, partition
    kernel_*_cpu.nim          — CPU kernels (contiguity-fused)
    kernel_*_gpu.nim          — GPU kernels (flat-index)
  benchmark/
    bench_ex02_matmul_cpu_simd.nim  — CPU GEMM vs OpenBLAS benchmark
    laser_matmul/                   — Laser strided-GEMM reimplementation
  examples/
    ex02a_matmul_handtuned.nim      — hand-tuned GEMM
    ex02b_matmul_layout_algebra.nim — layout-algebra GEMM
    ex02_matmul_microkernels/       — AVX/AVX512 microkernels
  experiments/
    nvidia_cutlass_cute_tutorial/sgemm_1.nim  — GPU batch-GEMM port (WIP)
  tests/                        — layout, kernel, and tensor tests
```

## Build / run

All commands run from the tattletale repo root (`tattletale/`).

```bash
# Run a single test (C++ backend)
nim cpp -r --verbosity:0 --hints:off --warnings:off \
  --outdir:build/tests --nimcache:nimcache/tests \
  workspace/ceramic/tests/test_layout_algebra.nim

# CPU GEMM benchmark (add -d:cblas to include OpenBLAS)
nim cpp -r --outdir:build \
  workspace/ceramic/benchmark/bench_ex02_matmul_cpu_simd.nim
```

GPU kernels are emitted by Crucible and run through NVRTC (CUDA, OpenCL,
Vulkan, WebGPU). See `workspace/crucible/AGENTS.md` for the codegen test
commands.

## Related docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — position in the stack, Mermaid diagram, data flow.
- [AGENTS.md](AGENTS.md) — tensor `[]` vs `()` access, output-first parameter ordering, kernel file naming.
- [Crucible](../crucible/AGENTS.md) — the GPU codegen backend (CUDA/OpenCL/Vulkan/WebGPU).
- [tattletale README](../README.md) — repository goals and motivation.

## License

Dual-licensed under MIT or Apache-2.0, at your option. See the license terms
in the repository root.
