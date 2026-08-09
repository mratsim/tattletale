## Manual GPU test: the sm80 tensor-core microtile via NVRTC/CUDA — one register-level MMA.
##
## gemm_fragment(atom.instr, cFrag, aFrag, bFrag) replaces the hand-written mma.sync asm:
## one m16n8k8 tf32 atom, 32 threads, direct gmem → register fragments,
## one gemm call, direct register → gmem stores. Plus the explicit-output
## 5-arg form: gemm_fragment(atom.instr, dFrag, aFrag, bFrag, cFrag).
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm80_tensor_cores_cuda.nim
##
## The register map IS the atom's own (T,V) layouts (1×1 tiling): the flat
## (T,V) index t + T·v → gmem offset via the flattened layout consts.

import std/[strformat, random]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/crucible/src/codegen/nvrtc

# ═════════════════════════════════════════════════════════════════════════
#  Blessed derivation — atom consts flattened to kernel-foldable tuples
# ═════════════════════════════════════════════════════════════════════════

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const T  = toIntVal(atom.threadCount(opA))       # 32 threads per atom
const VA = toIntVal(atom.valuesPerThread(opA))   # 4 A registers per thread
const VB = toIntVal(atom.valuesPerThread(opB))   # 2 B registers per thread
const VC = toIntVal(atom.valuesPerThread(opC))   # 4 C registers per thread

const aShape  = flatten(atom.aLayout.shape)
const aStride = flatten(atom.aLayout.stride)
const bShape  = flatten(atom.bLayout.shape)
const bStride = flatten(atom.bLayout.stride)
const cShape  = flatten(atom.cLayout.shape)
const cStride = flatten(atom.cLayout.stride)

# the kernel register arrays below are literals — pin them to the atom
static:
  doAssert T == 32 and VA == 4 and VB == 2 and VC == 4,
    "kernel arrays pinned to the m16n8k8 tf32 atom"

# ═════════════════════════════════════════════════════════════════════════
#  Kernels
# ═════════════════════════════════════════════════════════════════════════

const kernelCode = cuda:
  proc mmaMicrotile(C: ptr UncheckedArray[float32],
                    A, B: ptr UncheckedArray[uint32]) {.global.} =
    ## C(16×8) = A(16×8) · B(8×8) — one m16n8k8 tf32 atom, 32 threads.
    ## gmem is col-major: A[m + k·16], B[n + k·8], C[m + n·16].
    let t = int(threadIdx.x)

    # fragments: direct gmem → registers (flat (T,V) index = t + T·v)
    var aFrag: array[4, uint32]
    for v in 0 ..< VA:
      aFrag[v] = A[crd2idx(make_layout(aShape, aStride), t + T * v)]
    var bFrag: array[2, uint32]
    for v in 0 ..< VB:
      bFrag[v] = B[crd2idx(make_layout(bShape, bStride), t + T * v)]
    var cFrag: array[4, float32]
    for v in 0 ..< VC:
      cFrag[v] = 0.0'f32

    gemm_fragment(atom.instr, cFrag, aFrag, bFrag)   # one mma.sync — in-place accumulate

    # epilogue: registers → gmem
    for v in 0 ..< VC:
      let cOff = crd2idx(make_layout(cShape, cStride), t + T * v)
      C[cOff] = cFrag[v]

const kernelCodeExplicit = cuda:
  proc mmaMicrotileExplicit(C: ptr UncheckedArray[float32],
                            A, B: ptr UncheckedArray[uint32]) {.global.} =
    ## C(16×8) = A(16×8) · B(8×8) + 1 — the 5-arg form: dFrag out, cFrag in.
    let t = int(threadIdx.x)

    var aFrag: array[4, uint32]
    for v in 0 ..< VA:
      aFrag[v] = A[crd2idx(make_layout(aShape, aStride), t + T * v)]
    var bFrag: array[2, uint32]
    for v in 0 ..< VB:
      bFrag[v] = B[crd2idx(make_layout(bShape, bStride), t + T * v)]
    var cFrag: array[4, float32]
    for v in 0 ..< VC:
      cFrag[v] = 1.0'f32            # nonzero accumulator input
    var dFrag: array[4, float32]

    gemm_fragment(atom.instr, dFrag, aFrag, bFrag, cFrag)   # dFrag = aFrag·bFrag + cFrag

    for v in 0 ..< VC:
      let cOff = crd2idx(make_layout(cShape, cStride), t + T * v)
      C[cOff] = dFrag[v]

# ═════════════════════════════════════════════════════════════════════════
#  Host side
# ═════════════════════════════════════════════════════════════════════════

func tf32ify(x: float32): uint32 =
  ## f32 → tf32 bit pattern: truncate the low 13 mantissa bits.
  ## For small-integer inputs the mantissa is zero → exact.
  (cast[uint32](x)) and 0xFFFFE000'u32

proc tf32Reference(C: var openArray[float32],
                   A: openArray[uint32], B: openArray[uint32],
                   cInit: float32) =
  ## C[m,n] = cInit + Σ_k tf32(A[m,k]) · tf32(B[n,k]) — exact for small ints.
  for m in 0 ..< 16:
    for n in 0 ..< 8:
      var sum = cInit
      for k in 0 ..< 8:
        let av = cast[float32](A[m + k * 16])
        let bv = cast[float32](B[n + k * 8])
        sum = sum + av * bv
      C[m + n * 16] = sum

when isMainModule:
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    var A = newSeq[uint32](16 * 8)
    var B = newSeq[uint32](8 * 8)
    for i in 0 ..< A.len:
      A[i] = tf32ify(float32(rng.rand(0 .. 15)))
    for i in 0 ..< B.len:
      B[i] = tf32ify(float32(rng.rand(0 .. 15)))

    # in-place (4-arg)
    var refC = newSeq[float32](16 * 8)
    refC.tf32Reference(A, B, 0.0'f32)
    var gpuC = newSeq[float32](16 * 8)
    var nv = initNvrtc(kernelCode)
    nv.compile()
    nv.getPtx()
    nv.execute("mmaMicrotile", dim3(1), dim3(32), gpuC, (A, B))
    for j in 0 ..< 16 * 8:
      doAssert gpuC[j] == refC[j],
        &"in-place trial {trial} [{j mod 16},{j div 16}]: gpu {gpuC[j]} != ref {refC[j]}"

    # explicit-output (5-arg), cFrag = 1.0
    refC.tf32Reference(A, B, 1.0'f32)
    var gpuD = newSeq[float32](16 * 8)
    var nv5 = initNvrtc(kernelCodeExplicit)
    nv5.compile()
    nv5.getPtx()
    nv5.execute("mmaMicrotileExplicit", dim3(1), dim3(32), gpuD, (A, B))
    for j in 0 ..< 16 * 8:
      doAssert gpuD[j] == refC[j],
        &"explicit trial {trial} [{j mod 16},{j div 16}]: gpu {gpuD[j]} != ref {refC[j]}"

  echo "  OK — m16n8k8 tf32 microtile bit-exact vs reference (16 trials, in-place + explicit)"
