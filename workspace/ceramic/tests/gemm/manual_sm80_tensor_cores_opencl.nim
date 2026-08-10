## Manual GPU test: the sm80 tensor-core microtile via the OpenCL backend —
## one register-level MMA.
##
## Same microtile as manual_sm80_tensor_cores_cuda.nim (one m16n8k8 tf32
## atom, 32 work-items), but emitted by the `opencl:` macro and executed on
## the OpenCL device. The mma.sync inline PTX travels inside the OpenCL C
## kernel as `asm(...)` — the same trick leela-zero uses for tensor cores
## through its OpenCL backend: src/kernels/hgemm_tensorcore.opencl embeds
## wmma.load/mma/store PTX in .opencl kernels, relying on NVIDIA's OpenCL
## compiler accepting inline PTX asm.
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run with:
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm80_tensor_cores_opencl.nim

import std/[strformat, random]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_nvidia
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible/src/codegen/cl

# ═════════════════════════════════════════════════════════════════════════
#  Blessed derivation — atom consts flattened to kernel-foldable tuples
# ═════════════════════════════════════════════════════════════════════════

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const T  = toIntVal(atom.threadCount(opA))       # 32 work-items per atom
const VA = toIntVal(atom.valuesPerThread(opA))   # 4 A registers per thread
const VB = toIntVal(atom.valuesPerThread(opB))   # 2 B registers per thread
const VC = toIntVal(atom.valuesPerThread(opC))   # 4 C registers per thread

const aShape  = flatten(atom.aLayout.shape)
const aStride = flatten(atom.aLayout.stride)
const bShape  = flatten(atom.bLayout.shape)
const bStride = flatten(atom.bLayout.stride)
const cShape  = flatten(atom.cLayout.shape)
const cStride = flatten(atom.cLayout.stride)

static:
  doAssert T == 32 and VA == 4 and VB == 2 and VC == 4,
    "kernel arrays pinned to the m16n8k8 tf32 atom"

# ═════════════════════════════════════════════════════════════════════════
#  Kernel
# ═════════════════════════════════════════════════════════════════════════

const kernelCode = opencl:
  proc mmaMicrotile(A, B: ptr UncheckedArray[uint32],
                    C: ptr UncheckedArray[float32]) {.global.} =
    ## C(16×8) = A(16×8) · B(8×8) — one m16n8k8 tf32 atom, 32 work-items.
    ## OpenCL binds kernel args in order: inputs (A, B) first, output (C) last.
    let t = int(get_local_id(0))

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

# ═════════════════════════════════════════════════════════════════════════
#  Host side
# ═════════════════════════════════════════════════════════════════════════

proc runMicrotile(A, B: openArray[uint32]): seq[float32] =
  ## Launch the microtile with one 32-thread work-group (one warp) and
  ## read back the 16×8 C fragment.
  var ctx = initOpenCL()
  defer: ctx.shutdown()
  var aBuf = ctx.allocBuffer(A.len * 4)
  var bBuf = ctx.allocBuffer(B.len * 4)
  var cBuf = ctx.allocBuffer(16 * 8 * 4)
  aBuf.writeBuffer(A)
  bBuf.writeBuffer(B)
  var kernel = ctx.compileKernel("mmaMicrotile", kernelCode)
  defer: kernel.destroyKernel()
  kernel.setArg(0, aBuf)
  kernel.setArg(1, bBuf)
  kernel.setArg(2, cBuf)
  kernel.runKernel([csize_t(32)], [csize_t(32)])
  # the seq-returning readBuffer overload cannot be instantiated from the
  # same module (pre-existing; "cannot instantiate: T") — use the pointer form
  result = newSeq[float32](16 * 8)
  cBuf.readBuffer(cast[ptr UncheckedArray[float32]](result[0].addr))
  aBuf.dealloc()
  bBuf.dealloc()
  cBuf.dealloc()

when isMainModule:
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    let A = tf32Fixture(rng, 16, 8)
    let B = tf32Fixture(rng, 8, 8)

    var refC = newSeq[float32](16 * 8)
    tf32Reference(refC, A, B, 16, 8, 8, 0.0'f32)

    let gpuC = runMicrotile(A, B)
    # exact == (not allClose): the fixture domain is exact-representable, so
    # the OpenCL path must match the reference bit-for-bit
    for j in 0 ..< 16 * 8:
      doAssert gpuC[j] == refC[j],
        &"trial {trial} [{j mod 16},{j div 16}]: gpu {gpuC[j]} != ref {refC[j]}"

  echo "  OK — m16n8k8 tf32 microtile bit-exact vs reference via OpenCL (tf32-exact fixture, 16 trials)"
