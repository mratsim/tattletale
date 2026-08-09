## Manual GPU test: the tiled GEMM (gemm_tiled) via NVRTC/CUDA.
##
## C(32×16) = α·A(32×16)·B(16×16) + β·C — 1×1 grid, single k-block
## (K = BLK_K = 16), config (α, β) = (1.0, 0.0), 128 threads.
##
## gemm_tiled(tma, threadIdx, alpha, A, B, beta, C) = tiling + thread
## decomposition + k-block loop over gemm_ukernel, with a fused
## α·(A·B) + β·C epilogue. The C buffer is pre-filled with NaN: the β=0
## branch must skip the C read, so a spurious read (or a dropped store)
## produces NaN != expected.
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm80_gemm_tiled_cuda.nim

import std/[strformat, random]
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
import workspace/crucible/src/codegen/nvrtc

# ═════════════════════════════════════════════════════════════════════════
#  Blessed derivation — atom + tiling + partitions, flattened to constants
# ═════════════════════════════════════════════════════════════════════════

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled2 = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

const BLK_K = 16                 # k-block size in elements
const VA = toIntVal(atom.valuesPerThread(opA))   # 4 A registers per thread
const VB = toIntVal(atom.valuesPerThread(opB))   # 2 B registers per thread
const VC = toIntVal(atom.valuesPerThread(opC))   # 4 C registers per thread
const T  = toIntVal(atom.threadCount(opA))       # 32 threads per atom
const kAtom = atom.mnk.k                          # 8 k elements per slice
const thrM = toIntVal(tiled2.threadLayout.shape[0])
const thrN = toIntVal(tiled2.threadLayout.shape[1])
const thrK = toIntVal(tiled2.threadLayout.shape[2])
const thrShape = flatten(tiled2.threadLayout.shape)

const BLK_M = thrM * toIntVal(atom.mnk.m)         # 2·16 = 32
const BLK_N = thrN * toIntVal(atom.mnk.n)         # 2·8  = 16
const blockSize = T * thrM * thrN * thrK          # 32·2·2·1 = 128

# host-side partition flattening — the same derivation gemm_tiled performs
# in-template; pinned below so a drift in the atom/tiling fails the build
const pA = partition_A(tiled2, BLK_M, BLK_K)
const pB = partition_B(tiled2, BLK_N, BLK_K)
const pC = partition_C(tiled2, BLK_M, BLK_N)
const pAFlatShape  = flatten(pA.shape)
const pBFlatShape  = flatten(pB.shape)
const pCFlatShape  = flatten(pC.shape)

static:
  doAssert BLK_M == 32 and BLK_N == 16 and blockSize == 128,
    "tiled geometry pinned to the m16n8k8 tf32 atom with (2,2,1) tiling"
  doAssert T == 32 and VA == 4 and VB == 2 and VC == 4,
    "kernel register arrays pinned to the m16n8k8 tf32 atom"
  doAssert kAtom == 8 and thrK == 1,
    "k-slicing pinned to the atom k-depth (threads never along K)"
  doAssert pAFlatShape === (4, 8, 2, 2, 2, 1, 1, 2) and
         pBFlatShape === (4, 8, 2, 2, 1, 1, 2) and
         pCFlatShape === (4, 8, 2, 2, 2, 2, 1, 1),
    "partition shapes pinned: atom (T,V) leaves, (Thr·), (Rest·) per operand"

# ═════════════════════════════════════════════════════════════════════════
#  Baked geometry (Phase 1: all shapes/strides static)
# ═════════════════════════════════════════════════════════════════════════

const M = 32
const N = 16
const K = 16

# ═════════════════════════════════════════════════════════════════════════
#  Kernel
# ═════════════════════════════════════════════════════════════════════════

const kernelCode = cuda:
  proc gemmTiledKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    ## C(32×16) = α·A(32×16)·B(16×16) + β·C — 1×1 tiled m16n8k8 tf32,
    ## 128 threads, single k-block (K = BLK_K = 16), fused epilogue.
    ## gmem col-major: A[m + k·32], B[n + k·16], C[m + n·32].
    let bx = int(blockIdx.x)   # M-tile index (test scaffolding, REQ-009)
    let by = int(blockIdx.y)   # N-tile index
    let tA = make_view(A +% (bx * BLK_M),
                       make_layout((Int[BLK_M](), Int[K]()), (Int[1](), Int[M]())))
    let tB = make_view(B +% (by * BLK_N),
                       make_layout((Int[BLK_N](), Int[K]()), (Int[1](), Int[N]())))
    var tC = make_view(C +% (bx * BLK_M + by * BLK_N * M),
                       make_layout((Int[BLK_M](), Int[BLK_N]()), (Int[1](), Int[M]())))
    gemm_tiled(tiled2, int(threadIdx.x), alpha, tA, tB, beta, tC, BLK_K)

# ═════════════════════════════════════════════════════════════════════════
#  Host side
# ═════════════════════════════════════════════════════════════════════════

func tf32ify(x: float32): uint32 =
  ## f32 → tf32 bit pattern: truncate the low 13 mantissa bits.
  ## For small-integer inputs the mantissa is zero → exact.
  (cast[uint32](x)) and 0xFFFFE000'u32

when isMainModule:
  var rng = initRand(0xC0FFEE)

  # compile once, execute many
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()

  for trial in 0 ..< 16:
    # A/B in f32 domain 0..15 (tf32 truncation is identity); GPU buffers
    # carry the tf32 bit patterns of the same values
    var A = newSeq[float32](M * K)
    var B = newSeq[float32](N * K)
    var A_gpu = newSeq[uint32](M * K)
    var B_gpu = newSeq[uint32](N * K)
    for i in 0 ..< M * K:
      let v = float32(rng.rand(0 .. 15))
      A[i] = v
      A_gpu[i] = tf32ify(v)
    for i in 0 ..< N * K:
      let v = float32(rng.rand(0 .. 15))
      B[i] = v
      B_gpu[i] = tf32ify(v)

    # oracle: zeroed accumulator → reference fragment gemm on f32 views →
    # C_ref = α·acc + β·C_init (β term only when β != 0 — mirrors the
    # kernel's β=0 skip-read; β·NaN would poison the oracle)
    var acc = newSeq[float32](M * N)
    var accV = make_view(acc, (M, N), (1, M))
    let aV = make_view(A, (M, K), (1, M))
    let bV = make_view(B, (N, K), (1, N))
    gemm_fragment(accV, aV, bV)
    const alpha = 1.0'f32
    const beta = 0.0'f32
    var C_ref = newSeq[float32](M * N)
    var C_init = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      C_init[i] = 0x7FC00000'f32    # NaN sentinel — a spurious C read fails
      C_ref[i] = if beta == 0.0'f32: alpha * acc[i]
                 else: alpha * acc[i] + beta * C_init[i]

    var gpuC = newSeq[float32](M * N)
    for i in 0 ..< M * N:
      gpuC[i] = 0x7FC00000'f32
    nv.execute("gemmTiledKernel", dim3(1), dim3(blockSize), gpuC,
               (A_gpu, B_gpu, alpha, beta))

    for j in 0 ..< M * N:
      doAssert gpuC[j] == C_ref[j],
        &"trial {trial} [{j mod M},{j div M}]: gpu {gpuC[j]} != ref {C_ref[j]}"

  echo "  OK — gemm_tiled 1×1 single-k-block bit-exact vs reference (16 trials, (1,0), NaN C)"
