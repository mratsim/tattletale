# Fixture grammar evidence

The fixtures are hand-encoded Tile IR text.
cutile-python cannot run on this machine, see `STATE.md` in the op directory for the recorded attempt.
Every construct below is transcribed from official NVIDIA sources.

| Construct | Evidence |
|---|---|
| `cuda_tile.module @kernels { ... }` | `cuda-tile/test/Dialect/CudaTile/ops.mlir:4` |
| `entry @name(%arg: tile<ptr<f32>>, ...) { ... }` | `cuda-tile/test/Dialect/CudaTile/ops.mlir:1601-1602`, shorthand type form per `cutile-rs/cutile-ir/src/ir/fmt.rs:9` |
| `constant <i8: 42> : tile<i8>` | `cuda-tile/test/Dialect/CudaTile/ops.mlir:48-49` |
| splat constant into a shaped tile | `cuda-tile/lib/Dialect/CudaTile/IR/CudaTile.cpp:600-601` ("scalar ... splat behavior") |
| `make_token : token` | `cuda-tile/test/Dialect/CudaTile/memory_consistency_ops.mlir:9-10` |
| `make_tensor_view %ptr, shape = [..], strides = [..] : tensor_view<..>` | `cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Ops.td`, MakeTensorViewOp mlirExamples |
| dynamic shape/strides form `%s0 ... : tile<i32> -> tensor_view<?xf32, strides=[?]>` | MakeTensorViewOp mlirExamples, dynamic case |
| `make_partition_view %tv : partition_view<tile=(..), tensor_view<..>>` | MakePartitionViewOp mlirExamples |
| `get_tile_block_id : tile<i32>` returning 3 results | `cuda-tile/test/Dialect/CudaTile/ops.mlir:1370-1371` |
| `load_view_tko weak %view[%i] token = %tok : .. -> tile<..>, token` | LoadViewTkoOp mlirExamples, token-input case |
| `store_view_tko weak %tile, %view[%i] token = %tok : .. -> token` | StoreViewTkoOp mlirExamples, token-input case |
| `addf` / `mulf` elementwise | `cuda-tile/test/Dialect/CudaTile/arith.mlir` (addf, mulf families) |
| `mmaf %a, %b, %acc : tile<MxKxf32>, tile<KxNxf32>, tile<MxNxf32>` | `cuda-tile/test/Dialect/CudaTile/ops.mlir:1083-1085`, MmaFOp mlirExamples |
| `yield` closing an entry | `cuda-tile/test/Dialect/CudaTile/ops.mlir:80-81` |

Kernel structure per the `vec_add_kernel_1d` sample in `cutile-python/samples/VectorAddition.py`:

- bid, tiled load, elementwise op, tiled store
