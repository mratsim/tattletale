## Manual GPU test: the sm80 tensor-core microtile via the OpenCL backend —
## one register-level MMA.
##
## Same microtile as manual_sm80_tensor_cores_cuda.nim (one m16n8k8 tf32
## atom, 32 work-items), emitted by the `opencl:` macro and executed on the
## OpenCL device. The kernel body is the CUDA twin's verbatim: partition
## views (partition_A/B/C, make_fragment_A/B, copyFrom, fillWith), one
## mma.sync, axpby epilogue — no flattened consts, no hand-rolled gathers.
##
## NVIDIA-OpenCL only: the mma.sync inline PTX travels inside the OpenCL C
## kernel as `asm(...)` with GCC-style constraints, which only NVIDIA's
## OpenCL compiler accepts (Intel/AMD/POCL reject it at build time). The
## host also verifies the device vendor before launching. mma.sync is
## warp-synchronous, so the work-group is pinned to exactly one warp
## (32 work-items) — see WORK_GROUP below.
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run with:
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm80_tensor_cores_opencl.nim

import std/[strformat, strutils, random]
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
import workspace/ceramic/src/kernel_copy_gpu
import workspace/ceramic/src/kernel_fillwith_gpu
import workspace/ceramic/src/kernel_axpby_gpu
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible/src/codegen/cl

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

# mma.sync is warp-synchronous: all 32 lanes of one warp must execute it
# convergently. Pin the work-group to exactly one warp — a non-32-multiple
# group would split work-items across warps with inactive lanes (hang or
# illegal-instruction risk for mma.sync).
const WORK_GROUP = toIntVal(atom.threadCount(opA))   # 32 work-items = one warp
static:
  doAssert WORK_GROUP == 32,
    "mma.sync needs exactly one warp (32 work-items), got " & $WORK_GROUP

func mmaMicrotile(tma: static TiledMma; t: int;
                  C: ptr UncheckedArray[float32];
                  A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## C(16×8) = A(16×8)·B(8×8) — one m16n8k8 tf32 atom, 32 work-items, in-place.
  ## Fragment gathering: thr_mma.partition_A/B/C, fragment registers as owning
  ## tensors (make_tensor_like), copyFrom/fillWith/axpby — all layout
  ## algebra, no loops, no offsets, no raw-addr views.
  const
    M = tma.atom.mnk.m
    N = tma.atom.mnk.n
    K = tma.atom.mnk.k
    VA = toIntVal(tma.atom.valuesPerThread(opA))
    VB = toIntVal(tma.atom.valuesPerThread(opB))
    VC = toIntVal(tma.atom.valuesPerThread(opC))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, make_view(A, make_layout((M, K), (1, M))))
  let tBv = tma.partition_B(thr, make_view(B, make_layout((N, K), (1, N))))
  var tCv = tma.partition_C(thr, make_view(C, make_layout((M, N), (1, M))))
  # the fragment registers as owning tensors shaped like the partitions
  # (CuTe make_fragment_A/B/C) — one declaration, no raw-addr views
  var aFrag = make_fragment_A(tma.atom, tAv)
  aFrag.copyFrom(tAv)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(tBv)
  # the accumulator is identity-shaped (the register order — a compact
  # make_tensor_like would scramble it: 0,2,1,3)
  var cFrag = make_tensor(float32, (VC,))
  cFrag.fillWith(0.0'f32)

  gemm_fragment(tma.atom.instr, cFrag.data, aFrag.data, bFrag.data)   # one mma.sync — in-place accumulate

  axpby(1.0'f32, cFrag, 0.0'f32, tCv)

const kernelCode = opencl:
  proc mmaMicrotileKernel(C: ptr UncheckedArray[float32],
                          A, B: ptr UncheckedArray[uint32]) {.global.} =
    ## C(16×8) = A(16×8)·B(8×8) — one m16n8k8 tf32 atom, 32 work-items.
    ## Output-first (C, A, B), matching the CUDA twins and CONVENTIONS.md
    ## §1 (Context → Out → InOut → In); setArg binds positionally.
    mmaMicrotile(tiled, int(get_local_id(0)), C, A, B)

# ═════════════════════════════════════════════════════════════════════════
#  Host side
# ═════════════════════════════════════════════════════════════════════════

proc runMicrotile(ctx: var OpenCLContext; kernel: var OpenCLKernel;
                  A, B: openArray[uint32]): seq[float32] =
  ## One launch of the precompiled microtile kernel with one 32-work-item
  ## work-group (exactly one warp) and read-back of the 16×8 C fragment.
  ## The context and the compiled kernel are owned by the caller and reused
  ## across trials — only the buffers are per-launch.
  var aBuf = ctx.allocBuffer(A.len * 4)
  var bBuf = ctx.allocBuffer(B.len * 4)
  var cBuf = ctx.allocBuffer(16 * 8 * 4)
  defer:
    aBuf.dealloc()
    bBuf.dealloc()
    cBuf.dealloc()
  aBuf.writeBuffer(A)
  bBuf.writeBuffer(B)
  kernel.setArg(0, cBuf)
  kernel.setArg(1, aBuf)
  kernel.setArg(2, bBuf)
  kernel.runKernel([csize_t(WORK_GROUP)], [csize_t(WORK_GROUP)])
  # the seq-returning readBuffer overload cannot be instantiated from the
  # same module (pre-existing; "cannot instantiate: T") — use the pointer form
  result = newSeq[float32](16 * 8)
  cBuf.readBuffer(cast[ptr UncheckedArray[float32]](result[0].addr))

when isMainModule:
  var ctx = initOpenCL()
  doAssert ctx.device.vendor().contains("NVIDIA"),
    "this kernel embeds NVIDIA inline PTX (asm mma.sync) — only NVIDIA's " &
    "OpenCL compiler accepts it; got device vendor: " & ctx.device.vendor()
  var kernel = ctx.compileKernel("mmaMicrotileKernel", kernelCode)

  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    let A = tf32Fixture(rng, 16, 8)
    let B = tf32Fixture(rng, 8, 8)

    var refC = newSeq[float32](16 * 8)
    refC.tf32Reference(A, B, 16, 8, 8, 0.0'f32)

    let gpuC = runMicrotile(ctx, kernel, A, B)
    # exact == (not allClose): the fixture domain is exact-representable, so
    # the OpenCL path must match the reference bit-for-bit
    for j in 0 ..< 16 * 8:
      doAssert gpuC[j] == refC[j],
        &"trial {trial} [{j mod 16},{j div 16}]: gpu {gpuC[j]} != ref {refC[j]}"

  kernel.destroyKernel()
  ctx.shutdown()
  echo "  OK — m16n8k8 tf32 microtile bit-exact vs reference via OpenCL (tf32-exact fixture, 16 trials)"
