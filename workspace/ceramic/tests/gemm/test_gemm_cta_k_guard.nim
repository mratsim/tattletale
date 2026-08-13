## Compile-time guard: gemm_cta's K contract: the VIEW K (kView, the
## allocated extent) must be a multiple of TileShape.K (the k-tile depth
## tileK); the PROBLEM K is runtime.
##
## gemm_cta slices the CTA tile (which spans the whole problem K) into
## ceil(K/tileK) k-tiles (the K dimension chunked into tileK-deep tiles)
## and loops them; the last k-tile may be partial (ragged K), its
## k >= validK coordinates gather zeros (the residue, runtime
## predication). The view K must still tile evenly: local_tile needs an
## even (M, kView) tile grid, so the static `doAssert kView mod tileK
## == 0` rejects a mis-allocated view K loudly. A problem K that is not
## a multiple of tileK is legal: the residue k-tile is predicated at
## runtime.
##
## Positive cases must compile: kView == tileK (one k-tile, K = kView),
## kView = 2·tileK with K = kView (two exact k-tiles), kView = 2·tileK
## with K = tileK + 16 (two k-tiles, ragged residue), and ragged
## M/N + padded (non-compact) leading
## strides with a runtime K. The non-multiple view K and the A/B view K
## mismatch must not compile. CPU-runnable: the guard is a static
## doAssert (the gemm_mma asm body is never invoked here).

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
  # Concrete problem (M, N) = (64, 32), tile (32, 16), tileK = 32, 128 threads.
  var bufA = newSeq[uint32](64 * 64)
  var bufB = newSeq[uint32](32 * 64)
  var bufC = newSeq[float32](64 * 32)
  let A = cast[ptr UncheckedArray[uint32]](addr bufA[0])
  let B = cast[ptr UncheckedArray[uint32]](addr bufB[0])
  var tC = make_view(cast[ptr UncheckedArray[float32]](addr bufC[0]) +% 0, (32, 16), (1, 64))
  let thr = tiled.get_slice(0)
  var tCv = tiled.partition_C(thr, tC)
  var epi = initEpiAXPBY(1.0'f32, 0.0'f32, tCv)

  # kView == tileK with problem K == kView: one k-tile
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 32), (1, 64)), make_view(B, (32, 32), (1, 32)), 64, 32, 32, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with kView == TileShape.K and problem K == kView (one k-tile) must compile"
  # kView = 2·tileK with problem K == kView: two exact k-tiles
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 64), (1, 64)), make_view(B, (32, 64), (1, 32)), 64, 32, 64, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with kView = 2·TileShape.K and problem K == kView (two k-tiles) must compile"
  # kView = 2·tileK with problem K = 48: two k-tiles, the second partial
  # (ragged K, residue 16): runtime predication
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 64), (1, 64)), make_view(B, (32, 64), (1, 32)), 64, 32, 48, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with a problem K (48) not a multiple of TileShape.K must compile: the residue is runtime predication"
  # kView not a multiple of tileK: local_tile cannot tile the view K
  doAssert not compiles(gemm_cta(tiled, tCv, make_view(A, (64, 48), (1, 64)), make_view(B, (32, 48), (1, 32)), 64, 32, 48, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta must reject a view K (48) not a multiple of TileShape.K (32). local_tile needs the allocated K to tile evenly"
  doAssert not compiles(gemm_cta(tiled, tCv, make_view(A, (64, 64), (1, 64)), make_view(B, (32, 32), (1, 32)), 64, 32, 64, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta must reject A and B views that disagree on the allocated K (A kView = 64, B kView = 32). The k-tile grid is sliced with the A view's K"

  # Ragged M/N and padded leading strides with a runtime K are runtime
  # predication, not compile-time rejections.
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (48, 32), (1, 48)), make_view(B, (32, 32), (1, 32)), 48, 32, 32, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with ragged M (48, tileM 32) must compile: boundary tiles are predicated at runtime"
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 32), (1, 64)), make_view(B, (48, 32), (1, 48)), 64, 48, 32, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with ragged N (48, tileN 16) must compile: boundary tiles are predicated at runtime"
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 64), (1, 80)), make_view(B, (32, 64), (1, 48)), 64, 32, 48, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with padded leading strides (ldA 80, ldB 48) and a ragged runtime K must compile: compactness and the K residue are runtime properties"

  echo "  [OK] gemm_cta K guard: kView == tileK and kView = 2·tileK compile, ragged runtime K (residue) compiles, view K not a multiple of tileK rejected, A/B kView mismatch rejected, ragged M/N and padded strides compile"

main()
