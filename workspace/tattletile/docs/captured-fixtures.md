# tattletile

## Captured fixtures: compiler-emitted Tile IR

Two fixtures are compiler-emitted rather than hand-encoded:

| Fixture | Kernel | Source |
|---|---|---|
| `fixtures/captured_vec_add_axpy.tileir` | saxpy, tile 4x8 | `cutile-examples/examples/saxpy.rs` |
| `fixtures/captured_gemm.tileir` | gemm, f32 16x16x8, K=64 | `cutile-examples/examples/gemm.rs` |

Production path:

- compiler: cutile-rs at `cuda-rust` fc74719, compile-only API
  `cutile::compile_api::KernelCompiler`
- text: the `DumpStage::Ir` pretty-printer (`Module::to_mlir_text`)
- no GPU, no driver, no `tileiras`: IR-level compilation is pure Rust
- specialization bits mimic the launcher for contiguous tensors
  (`compute_spec`), target `sm_120`

`tests/test_tile_ir_captured.nim` checks:

- parse + print round-trip fixed point on both fixtures
- shape assertions for the constructs only real IR has

```
nim c -r --hints:off --warnings:off \
  --nimcache:nimcache/tattletile --outdir:build/tattletile \
  workspace/tattletile/tests/test_tile_ir_captured.nim
```

## Grammar evidence the captures add

Hand-encoded fixtures and real compiler output agree on every shared construct:
operand order, type forms, the `weak` qualifier, token plumbing.

Real kernels additionally use six constructs the hand fixtures never show, all now in the subset:

| Construct | Captured form |
|---|---|
| entry attribute | `entry @saxpy_entry(...) di_name = "saxpy" {` |
| assume | `%17 = assume bounded<0, ?>, %1 : tile<i32>`, `%18 = assume div_by<4>, %17 : tile<i32>` |
| padding | `partition_view<tile=(4x8), padding_value = zero, tensor_view<...>>` |
| scalar to tile | `%42 = reshape %9 : tile<f32> -> tile<1x1xf32>`, then `broadcast` |
| loop | `%95 = for %83 in (%79 to %81, step %82) : tile<i32> iter_values(%84 = %77) -> (tile<16x16xf32>)` with a `continue`-closed body |
| terminator | entries close with a bare `return` |

## Build recipe (macOS, no CUDA toolkit installed)

`cutile` needs CUDA headers at build time only: `cuda-bindings` runs bindgen over
`cuda.h` + `curand.h`, runtime loading is dlopen.

- assemble a headers-only CUDA 13.0 toolkit at `/tmp/cuda-toolkit` from NVIDIA
  redistrib component tarballs: include/ of cuda_cudart, cuda_nvcc, cuda_cccl,
  cuda_crt and libcurand
- point `CUDA_TOOLKIT_PATH` at the toolkit, `LIBCLANG_PATH` at the Xcode CLT
  libclang
- build with `CARGO_TARGET_DIR` outside the reference tree

Full record: the C15D04 fixture-capture scratchspace operation (outside the repository tree),
where `CAPTURE.md` has commands and kernel sources and `DIVERGENCE.md` compares
hand-encoded fixtures against compiler-emitted text.
