## Phase 4 — tiled + k-loop tensor-core GEMM.
##
## C(32×16) = A(32×32) · B(16×32): 2×2 tiled m16n8k8 tf32 mma (128
## threads), k-loop over 4 k-tiles, accumulator persists across tiles.
## Direct gmem→register fills per k-tile, one mma.sync per k-tile,
## direct register→gmem epilogue. No smem, no copies.
##
## The fragment coords are the partition_A/B/C math emitted in-kernel
## (crd2idx on the atom's layouts + the get_slice decomposition), the
## same math validated host-side against tensor-layouts tile_mma_grid
## (test_partition_host) and on-GPU (test_fragment_dump_gpu).
##
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm/test_gemm_tiled_gpu.nim

import std/[strformat, random]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/crucible/src/codegen/nvrtc

# ═════════════════════════════════════════════════════════════════════════
#  Kernel — 2×2 tiled, K-parameterized
# ═════════════════════════════════════════════════════════════════════════

const kernelCode = cuda:
  proc mmaGemmTiled(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, K: int32) {.global.} =
    ## C(32×16) = A(32×K) · B(16×K), 2×2 tiled m16n8k8 tf32, 128 threads.
    ## gmem col-major: A[m + k·M], B[n + k·N], C[m + n·M].
    let t = int(threadIdx.x)
    let
      T = 32            # threads per atom
      mAtom = 16
      nAtom = 8
      kAtom = 8
      thrM = 2
      thrN = 2

    # get_slice: flat = tv + T·(tm + ThrM·(tn + ThrN·tk))
    let atomIdx = t div T
    let tm = atomIdx mod thrM
    let tn = (atomIdx div thrM) mod thrN
    let localT = t mod T

    let aLayout = make_layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    let bLayout = make_layout(((4, 8), 2), ((8, 1), 32))
    let cLayout = make_layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))

    # accumulator (C fragment)
    var cFrag: array[4, float32]
    for v in 0 .. 3:
      cFrag[v] = 0.0'f32

    # k-loop
    let kTiles = int(K) div kAtom
    for kTile in 0 ..< kTiles:
      # A fragment: (tm, lk) within the A atom tile, gmem at kTile offset
      var aFrag: array[4, uint32]
      for v in 0 .. 3:
        let off = crd2idx(aLayout, localT + T * v)
        aFrag[v] = A[(tm * mAtom + (off mod mAtom)) + (kTile * kAtom + (off div mAtom)) * int(M)]
      # B fragment
      var bFrag: array[2, uint32]
      for v in 0 .. 1:
        let off = crd2idx(bLayout, localT + T * v)
        bFrag[v] = B[(tn * nAtom + (off mod nAtom)) + (kTile * kAtom + (off div nAtom)) * int(N)]
      # mma.sync (accumulates into cFrag)
      asm "\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\" : \"+f\"(cFrag[0]), \"+f\"(cFrag[1]), \"+f\"(cFrag[2]), \"+f\"(cFrag[3]) : \"r\"(aFrag[0]), \"r\"(aFrag[1]), \"r\"(aFrag[2]), \"r\"(aFrag[3]), \"r\"(bFrag[0]), \"r\"(bFrag[1])"

    # epilogue: registers → gmem
    for v in 0 .. 3:
      let off = crd2idx(cLayout, localT + T * v)
      C[(tm * mAtom + (off mod mAtom)) + (tn * nAtom + (off div mAtom)) * int(M)] = cFrag[v]

# ═════════════════════════════════════════════════════════════════════════
#  Host side
# ═════════════════════════════════════════════════════════════════════════

func tf32ify(x: float32): uint32 =
  (cast[uint32](x)) and 0xFFFFE000'u32

proc tf32Reference(C: var openArray[float32];
                   A: openArray[uint32], B: openArray[uint32],
                   M, N, K: int) =
  ## C[m,n] = Σ_k tf32(A[m,k]) · tf32(B[n,k]) — exact for small ints.
  for m in 0 ..< M:
    for n in 0 ..< N:
      var sum = 0.0'f32
      for k in 0 ..< K:
        let av = cast[float32](A[m + k * M])
        let bv = cast[float32](B[n + k * N])
        sum = sum + av * bv
      C[m + n * M] = sum

when isMainModule:
  const M = 32
  const N = 16
  const K = 32
  var rng = initRand(0xBEEF)
  for trial in 0 ..< 16:
    var A = newSeq[uint32](M * K)
    var B = newSeq[uint32](N * K)
    for i in 0 ..< A.len: A[i] = tf32ify(float32(rng.rand(0 .. 15)))
    for i in 0 ..< B.len: B[i] = tf32ify(float32(rng.rand(0 .. 15)))

    var refC = newSeq[float32](M * N)
    refC.tf32Reference(A, B, M, N, K)

    var gpuC = newSeq[float32](M * N)
    let m32 = int32(M)
    let n32 = int32(N)
    let k32 = int32(K)
    var nv = initNvrtc(kernelCode)
    nv.compile()
    nv.getPtx()
    nv.execute("mmaGemmTiled", dim3(1), dim3(128), gpuC, (A, B, m32, n32, k32))

    for j in 0 ..< M * N:
      doAssert gpuC[j] == refC[j],
        &"trial {trial} [{j mod M},{j div M}]: gpu {gpuC[j]} != ref {refC[j]}"
  echo "  OK — 2×2 tiled k-loop mma GEMM bit-exact vs reference (16 trials)"
