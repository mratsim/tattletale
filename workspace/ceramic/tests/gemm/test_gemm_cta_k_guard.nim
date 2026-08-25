## Compile-time guard: the view K (kView, the allocated size) must be a
## multiple of TileShape.K (tileK). Input K is runtime.
##
## gemm_cta slices the CTA tile (which spans the whole input K) into
## ceil(K/tileK) tileK-sized slices of K and loops them. The last slice
## may be partial (ragged K), with its k >= validK coordinates
## zero-filled at the load. The static `doAssert kView mod tileK == 0`
## rejects a mis-allocated view K. An input K not a multiple of tileK is
## legal. The residue slice is predicated at runtime.
##
## CPU-runnable: the guard is a static doAssert, the gemm_mma asm body
## is never invoked here.

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry
import workspace/ceramic/src/hardware/h_properties
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/crucible

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

proc main() =
  # Input (M, N) = (64, 32), tile (32, 16), tileK = 32, 128 threads.
  var bufA = newSeq[uint32](64 * 64)
  var bufB = newSeq[uint32](32 * 64)
  var bufC = newSeq[float32](64 * 32)
  let A = cast[ptr UncheckedArray[uint32]](addr bufA[0])
  let B = cast[ptr UncheckedArray[uint32]](addr bufB[0])
  var tC = make_view(cast[ptr UncheckedArray[float32]](addr bufC[0]) +% 0, (32, 16), (1, 64))
  let thr = tiled.get_slice(0)
  var tCv = tiled.partition_C(thr, tC)
  var epi = initEpiAXPBY(1.0'f32, 0.0'f32, tCv)

  # kView == tileK with input K == kView: one slice of K
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 32), (1, 64)), make_view(B, (32, 32), (1, 32)), 64, 32, 32, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with kView == TileShape.K and problem K == kView (one slice of K) must compile"
  # kView = 2·tileK with input K == kView: two exact slices
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 64), (1, 64)), make_view(B, (32, 64), (1, 32)), 64, 32, 64, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with kView = 2·TileShape.K and problem K == kView (two slices) must compile"
  # kView = 2·tileK with input K = 48: two slices, the second partial
  # (ragged K, residue 16): runtime predication
  doAssert compiles(gemm_cta(tiled, tCv, make_view(A, (64, 64), (1, 64)), make_view(B, (32, 64), (1, 32)), 64, 32, 48, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta with a problem K (48) not a multiple of TileShape.K must compile: the residue is runtime predication"
  # kView not a multiple of tileK: local_tile cannot tile the view K
  doAssert not compiles(gemm_cta(tiled, tCv, make_view(A, (64, 48), (1, 64)), make_view(B, (32, 48), (1, 32)), 64, 32, 48, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta must reject a view K (48) not a multiple of TileShape.K (32). local_tile needs the allocated K to tile evenly"
  doAssert not compiles(gemm_cta(tiled, tCv, make_view(A, (64, 64), (1, 64)), make_view(B, (32, 32), (1, 32)), 64, 32, 64, epi, (32, 16, 32), 0, 0, 0)),
    "gemm_cta must reject A and B views that disagree on the allocated K (A kView = 64, B kView = 32). The K slicing uses the A view's K"

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
