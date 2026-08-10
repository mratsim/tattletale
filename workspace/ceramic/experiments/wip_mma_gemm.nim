## Phase 3 — wip_mma_gemm: single-tile tensor-core GEMM, bit-exact.
##
## C(16×8) = A(16×8) · B(8×8) — one m16n8k8 tf32 atom, 32 threads, no
## smem, no copies: direct gmem→register fragment fills, one mma.sync,
## direct register→gmem stores.
##
## REWRITE (uses the blessed primitives): the fragment offsets come from
## partition_A/B/C over the TiledMma (atom + thread layout), flattened
## host-side to constant shape/stride tuples that the kernel folds via
## make_layout + crd2idx. The thread decomposition is idx2crd on the
## thread layout. What stays hand-written: the (T,V) flat register index
## (t + T·v), the mma.sync asm, and the epilogue.
##
## tf32 path: input buffers are uint32 holding tf32 bit patterns
## (host-side f32 → tf32 truncation of the low 13 mantissa bits). The
## DSL's cast emission is a value conversion, not a bitcast, so bit
## patterns travel as u32, not via cast.
##
## Reference: exact for small-integer inputs (products and partial sums
## fit f32's 24-bit mantissa), so the test is bit-exact.
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm.nim

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
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

const pA = partition_A(tiled, 16, 8)
const pB = partition_B(tiled, 8, 8)
const pC = partition_C(tiled, 16, 8)
const pAFlatShape  = flatten(pA.shape)
const pAFlatStride = flatten(pA.stride)
const pBFlatShape  = flatten(pB.shape)
const pBFlatStride = flatten(pB.stride)
const pCFlatShape  = flatten(pC.shape)
const pCFlatStride = flatten(pC.stride)

const T  = toIntVal(atom.threadCount(opA))       # threads per atom
const VA = toIntVal(atom.valuesPerThread(opA))   # A registers per thread
const VB = toIntVal(atom.valuesPerThread(opB))   # B registers per thread
const VC = toIntVal(atom.valuesPerThread(opC))   # C registers per thread

# The DSL cannot size arrays from host consts (the kernel sees only
# literals), so the register-array sizes are written inline below and
# pinned to the atom here — drift fails the build.
static:
  doAssert VA == 4 and VB == 2 and VC == 4, "kernel register arrays must match the atom's V counts"

# ═════════════════════════════════════════════════════════════════════════
#  Kernel
# ═════════════════════════════════════════════════════════════════════════

const kernelCode = cuda:
  proc mmaGemm16x8x8(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32]) {.global.} =
    ## C(16×8) = A(16×8) · B(8×8), tf32, one m16n8k8 atom, 32 threads.
    ## gmem is col-major (CuTe convention): A[m + k·16], B[n + k·8].
    ## 1×1 tiled: the fragment offset IS the gmem index.
    let t = int(threadIdx.x)

    # ── fragments: direct gmem → registers ──
    #   flat (T,V) index = t + T·v — the atom's register order.
    var aFrag: array[4, uint32]
    for v in 0 ..< VA:
      aFrag[v] = A[crd2idx(make_layout(pAFlatShape, pAFlatStride), t + T * v)]
    var bFrag: array[2, uint32]
    for v in 0 ..< VB:
      bFrag[v] = B[crd2idx(make_layout(pBFlatShape, pBFlatStride), t + T * v)]
    var cFrag: array[4, float32]
    for v in 0 ..< VC:
      cFrag[v] = 0.0'f32

    # ── one mma.sync ──
    asm "\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\" : \"+f\"(cFrag[0]), \"+f\"(cFrag[1]), \"+f\"(cFrag[2]), \"+f\"(cFrag[3]) : \"r\"(aFrag[0]), \"r\"(aFrag[1]), \"r\"(aFrag[2]), \"r\"(aFrag[3]), \"r\"(bFrag[0]), \"r\"(bFrag[1])"

    # ── epilogue: registers → gmem ──
    for v in 0 ..< VC:
      let cOff = crd2idx(make_layout(pCFlatShape, pCFlatStride), t + T * v)
      C[cOff] = cFrag[v]

# ═════════════════════════════════════════════════════════════════════════
#  Host side
# ═════════════════════════════════════════════════════════════════════════

func tf32ify(x: float32): uint32 =
  ## f32 → tf32 bit pattern: truncate the low 13 mantissa bits.
  ## For small-integer inputs the mantissa is zero → exact.
  (cast[uint32](x)) and 0xFFFFE000'u32

proc tf32Reference(C: var openArray[float32],
                   A: openArray[uint32], B: openArray[uint32]) =
  ## C[m,n] = Σ_k tf32(A[m,k]) · tf32(B[n,k]) — exact for small ints.
  for m in 0 ..< 16:
    for n in 0 ..< 8:
      var sum = 0.0'f32
      for k in 0 ..< 8:
        let av = cast[float32](A[m + k * 16])
        let bv = cast[float32](B[n + k * 8])
        sum = sum + av * bv
      C[m + n * 16] = sum

when isMainModule:
  var rng = initRand(0xC0FFEE)
  # the kernel source is a module-level const — compile once, execute per trial
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()
  for trial in 0 ..< 16:
    var A = newSeq[uint32](16 * 8)
    var B = newSeq[uint32](8 * 8)
    for i in 0 ..< A.len:
      A[i] = tf32ify(float32(rng.rand(0 .. 15)))   # small ints → tf32-exact
    for i in 0 ..< B.len:
      B[i] = tf32ify(float32(rng.rand(0 .. 15)))

    var refC = newSeq[float32](16 * 8)
    refC.tf32Reference(A, B)

    var gpuC = newSeq[float32](16 * 8)
    nv.execute("mmaGemm16x8x8", dim3(1), dim3(32), gpuC, (A, B))

    for j in 0 ..< 16 * 8:
      doAssert gpuC[j] == refC[j],
        &"trial {trial} [{j mod 16},{j div 16}]: gpu {gpuC[j]} != ref {refC[j]}"
  echo "  OK — single-tile mma GEMM bit-exact vs reference (16 trials)"
