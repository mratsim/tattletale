## Manual GPU test: the sm80 GEBB microkernel via NVRTC/CUDA — the loop over K.
##
## gemm_ukernel(mma, cFrag, aFrag, bFrag) = one gemm_fragment per k-slice,
## accumulated in cFrag (CuTe dispatch [5] analog). One m16n8k8 tf32 atom,
## 32 threads, K = 2 k-slices: A is 16×16, B is 16×8, C is 16×8. The
## k-loop is unrolled at macro time — constant fragment indices keep the
## asm operands register-resident.
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm80_gemm_ukernel_cuda.nim

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

# one k-slice of A/B is a (M,K_atom)/(N,K_atom) tile; slice s adds s·K_atom
# columns → s·(K_atom · k-column-stride) elements of gmem offset (col-major:
# A[m + k·M], B[n + k·N])
const aSliceStride = atom.mnk.k * atom.mnk.m   # 8·16 = 128
const bSliceStride = atom.mnk.k * atom.mnk.n   # 8·8 = 64

# the kernel register arrays below are literals — pin them to the atom
static:
  doAssert T == 32 and VA == 4 and VB == 2 and VC == 4,
    "kernel arrays pinned to the m16n8k8 tf32 atom"
  doAssert aSliceStride == 128 and bSliceStride == 64,
    "k-slice strides pinned to the col-major (M,K)/(N,K) layouts"

# ═════════════════════════════════════════════════════════════════════════
#  Kernel
# ═════════════════════════════════════════════════════════════════════════

const kernelCode = cuda:
  proc gemmUkernelMicrotile(C: ptr UncheckedArray[float32],
                            A, B: ptr UncheckedArray[uint32]) {.global.} =
    ## C(16×8) = A(16×16) · B(16×8) — two m16n8k8 k-slices via gemm_ukernel.
    ## gmem is col-major: A[m + k·16], B[n + k·8], C[m + n·16].
    let t = int(threadIdx.x)

    # fragments: direct gmem → registers; aFrag[s] = the s-th k-slice
    var aFrag: array[2, array[4, uint32]]
    for s in 0 ..< 2:
      for v in 0 ..< VA:
        aFrag[s][v] = A[crd2idx(make_layout(aShape, aStride), t + T * v) + s * aSliceStride]
    var bFrag: array[2, array[2, uint32]]
    for s in 0 ..< 2:
      for v in 0 ..< VB:
        bFrag[s][v] = B[crd2idx(make_layout(bShape, bStride), t + T * v) + s * bSliceStride]
    var cFrag: array[4, float32]
    for v in 0 ..< VC:
      cFrag[v] = 0.0'f32

    gemm_ukernel(atom, cFrag, aFrag, bFrag)   # two mma.sync, accumulating

    # epilogue: registers → gmem
    for v in 0 ..< VC:
      let cOff = crd2idx(make_layout(cShape, cStride), t + T * v)
      C[cOff] = cFrag[v]

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
  ## C[m,n] = cInit + Σ_{k=0..15} tf32(A[m,k]) · tf32(B[n,k]) — exact for small ints.
  for m in 0 ..< 16:
    for n in 0 ..< 8:
      var sum = cInit
      for k in 0 ..< 16:
        let av = cast[float32](A[m + k * 16])
        let bv = cast[float32](B[n + k * 8])
        sum = sum + av * bv
      C[m + n * 16] = sum

when isMainModule:
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    var A = newSeq[uint32](16 * 16)
    var B = newSeq[uint32](16 * 8)
    for i in 0 ..< A.len:
      A[i] = tf32ify(float32(rng.rand(0 .. 15)))
    for i in 0 ..< B.len:
      B[i] = tf32ify(float32(rng.rand(0 .. 15)))

    var refC = newSeq[float32](16 * 8)
    refC.tf32Reference(A, B, 0.0'f32)

    var gpuC = newSeq[float32](16 * 8)
    var nv = initNvrtc(kernelCode)
    nv.compile()
    nv.getPtx()
    nv.execute("gemmUkernelMicrotile", dim3(1), dim3(32), gpuC, (A, B))
    for j in 0 ..< 16 * 8:
      doAssert gpuC[j] == refC[j],
        &"trial {trial} [{j mod 16},{j div 16}]: gpu {gpuC[j]} != ref {refC[j]}"

  echo "  OK — m16n8k8 tf32 gemm_ukernel bit-exact vs reference (2 k-slices, 16 trials)"
