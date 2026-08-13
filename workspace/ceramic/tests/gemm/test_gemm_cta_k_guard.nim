## Compile-time guard: gemm_cta requires problem K to be a multiple of
## TileShape.K (the k-tile depth BLK_K).
##
## gemm_cta slices the CTA tile (which spans the whole problem K) into
## K div BLK_K k-tiles (K as a tiled mode, the local_tile pattern) and
## loops them, so a problem K that is not a multiple of BLK_K cannot be
## tiled: the static `doAssert K mod tileK == 0` rejects it loudly.
##
## Positive cases must compile (K == BLK_K: one k-tile, and K = 2·BLK_K:
## two k-tiles); the non-multiple case must not. CPU-runnable: the guard
## is a static doAssert (the gemm_mma asm body is never invoked here).

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
  # Concrete problem (M, N) = (64, 32), tile (32, 16), BLK_K = 32, 128 threads.
  var bufA = newSeq[uint32](64 * 64)
  var bufB = newSeq[uint32](32 * 64)
  var bufC = newSeq[float32](64 * 32)
  let A = cast[ptr UncheckedArray[uint32]](addr bufA[0])
  let B = cast[ptr UncheckedArray[uint32]](addr bufB[0])
  var tC = make_view(cast[ptr UncheckedArray[float32]](addr bufC[0]) +% 0, (32, 16), (1, 64))
  let thr = tiled.get_slice(0)
  var tCv = tiled.partition_C(thr, tC)
  var epi = initEpiAXPBY(1.0'f32, 0.0'f32, tCv)

  doAssert compiles(gemm_cta(tiled, tCv, A, 64, B, 32, epi, 64, 32, 32, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with problem K == TileShape.K (one k-tile) must compile"
  doAssert compiles(gemm_cta(tiled, tCv, A, 64, B, 32, epi, 64, 32, 64, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with problem K = 2·TileShape.K (two k-tiles) must compile"
  doAssert not compiles(gemm_cta(tiled, tCv, A, 64, B, 32, epi, 64, 32, 48, (32, 16, 32), 0, 0, 0)),
    "gemm_cta must reject problem K (48) not a multiple of TileShape.K (32). The k-tile loop needs K = kTiles·BLK_K"
  echo "  [OK] gemm_cta K guard: K == BLK_K and K = 2·BLK_K compile, K not a multiple of BLK_K rejected"

main()
