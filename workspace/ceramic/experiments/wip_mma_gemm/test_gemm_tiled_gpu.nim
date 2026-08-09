## Phase 4 — tiled + k-loop tensor-core GEMM.
##
## C(32×16) = A(32×32) · B(16×32): 2×2 tiled m16n8k8 tf32 mma (128
## threads), k-loop over 4 k-tiles, accumulator persists across tiles.
## Direct gmem→register fills per k-tile, one mma.sync per k-tile,
## direct register→gmem epilogue. No smem, no copies.
##
## REWRITE (uses the blessed primitives): the fragment offsets come from
## partition_A/B/C over the TiledMma (atom + 2×2 thread layout),
## flattened host-side to constant shape/stride tuples folded in-kernel
## via make_layout + crd2idx. The thread decomposition is idx2crd on the
## thread layout; the k-loop steps the partition's (RestM, RestK) mode.
## What stays hand-written: the (T,V) flat register index (t + T·v),
## the mma.sync asm, and the epilogue.
##
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm/test_gemm_tiled_gpu.nim

import std/[strformat, random]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/crucible/src/codegen/nvrtc

# ═════════════════════════════════════════════════════════════════════════
#  Blessed derivation — atom + tiling + partitions, flattened to constants
# ═════════════════════════════════════════════════════════════════════════

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled2 = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

const pA = partition_A(tiled2, 32, 32)   # rest (1, 4) — the k-loop steps RestK
const pB = partition_B(tiled2, 16, 32)
const pC = partition_C(tiled2, 32, 16)
const pAFlatShape  = flatten(pA.shape)
const pAFlatStride = flatten(pA.stride)
const pBFlatShape  = flatten(pB.shape)
const pBFlatStride = flatten(pB.stride)
const pCFlatShape  = flatten(pC.shape)
const pCFlatStride = flatten(pC.stride)

const T      = toIntVal(atom.threadCount(opA))
const VA     = toIntVal(atom.valuesPerThread(opA))
const VB     = toIntVal(atom.valuesPerThread(opB))
const VC     = toIntVal(atom.valuesPerThread(opC))
const kAtom  = atom.mnk.k
const thrM   = toIntVal(tiled2.threadLayout.shape[0])
const thrN   = toIntVal(tiled2.threadLayout.shape[1])
const thrK   = toIntVal(tiled2.threadLayout.shape[2])
const thrShape = flatten(tiled2.threadLayout.shape)

# The DSL cannot size arrays from host consts (the kernel sees only
# literals), so the register-array sizes are written inline below and
# pinned to the atom here — drift fails the build.
static:
  doAssert VA == 4 and VB == 2 and VC == 4, "kernel register arrays must match the atom's V counts"

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
    ## Single-block POC: the tile origin is 0, so the partition's tile
    ## offset is the gmem index (tileM = M = 32, tileN = N = 16).
    let t = int(threadIdx.x)

    # thread decomposition: atom index via idx2crd on the thread layout,
    # tv = t mod T — the atom's LOCAL T-mode coordinate (the flat (T,V)
    # register index uses tv, not the global t)
    let tv = t mod T
    let coords = idx2crd(make_layout(thrShape), t div T)
    let tm = coords[0]
    let tn = coords[1]
    let tk = coords[2]

    # accumulator (C fragment)
    var cFrag: array[4, float32]
    for v in 0 ..< VC:
      cFrag[v] = 0.0'f32

    # k-loop over the partition's (RestM, RestK) mode
    let kTiles = int(K) div kAtom
    for rk in 0 ..< kTiles:
      # A fragment: flat (T,V) index + thread/rest coords
      var aFrag: array[4, uint32]
      for v in 0 ..< VA:
        let aOff = crd2idx(make_layout(pAFlatShape, pAFlatStride),
                           (tv + T * v) + T * VA * (tm + thrM * (tk + thrK * rk)))
        aFrag[v] = A[aOff]
      # B fragment
      var bFrag: array[2, uint32]
      for v in 0 ..< VB:
        let bOff = crd2idx(make_layout(pBFlatShape, pBFlatStride),
                           (tv + T * v) + T * VB * (tn + thrN * (tk + thrK * rk)))
        bFrag[v] = B[bOff]
      # mma.sync (accumulates into cFrag)
      asm "\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\" : \"+f\"(cFrag[0]), \"+f\"(cFrag[1]), \"+f\"(cFrag[2]), \"+f\"(cFrag[3]) : \"r\"(aFrag[0]), \"r\"(aFrag[1]), \"r\"(aFrag[2]), \"r\"(aFrag[3]), \"r\"(bFrag[0]), \"r\"(bFrag[1])"

    # epilogue: registers → gmem
    for v in 0 ..< VC:
      let cOff = crd2idx(make_layout(pCFlatShape, pCFlatStride),
                         (tv + T * v) + T * VC * (tm + thrM * tn))
      C[cOff] = cFrag[v]

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
