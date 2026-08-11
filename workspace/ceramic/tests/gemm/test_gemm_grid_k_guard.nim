## Compile-time guard: gemm_grid must reject problem K != TileShape.K in v1.
##
## RID HIDN-A-003/HPC-A-002 (High 0.90): gemm_grid had no link between the
## problem K and the tile K — a caller passing K > tileK got a silent
## partial-K GEMM. Fixed by adding the explicit problem-K parameter and a
## static `doAssert tileK == K` (the CTA tile spans the whole problem K in
## v1; the k-tile loop later relaxes this).
##
## Positive case must compile; the K-mismatch case must not. CPU-runnable:
## the guard is a static doAssert (gemm_mma's asm body is never invoked
## here, matching the deleted test_nvidia_tensor_cores pattern).

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_gemm_epilogues

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

proc main() =
  # Concrete problem (M, N, K) = (64, 32, 32), tile (32, 16, 32), 128 threads.
  var bufA = newSeq[uint32](64 * 32)
  var bufB = newSeq[uint32](32 * 32)
  var bufC = newSeq[float32](64 * 32)
  let A = cast[ptr UncheckedArray[uint32]](addr bufA[0])
  let B = cast[ptr UncheckedArray[uint32]](addr bufB[0])
  var tC = make_view(cast[ptr UncheckedArray[float32]](addr bufC[0]) +% 0, (32, 16), (1, 64))
  let thr = tiled.get_slice(0)
  var tCv = tiled.partition_C(thr, tC)
  var epi = initEpiAXPBY(1.0'f32, 0.0'f32, tCv)

  doAssert compiles(gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), 0, 0, 0)),
    "gemm_grid with problem K == TileShape.K must compile"
  doAssert not compiles(gemm_grid(tiled, tCv, A, 64, B, 32, epi, 64, 32, 64, (32, 16, 32), 0, 0, 0)),
    "gemm_grid must reject problem K (64) != TileShape.K (32) in v1 — no k-tile loop"
  echo "  [OK] gemm_grid K guard: K == tileK compiles, K != tileK rejected"

main()
