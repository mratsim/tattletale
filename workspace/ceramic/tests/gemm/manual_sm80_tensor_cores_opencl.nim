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

  gemm_atom(tma.atom, cFrag, aFrag, bFrag)   # one mma.sync — in-place accumulate

  axpby(1.0'f32, cFrag, 0.0'f32, tCv)

func mmaMicrotileExplicit(tma: static TiledMma; t: int;
                          C: ptr UncheckedArray[float32];
                          A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## C(16×8) = A(16×8)·B(8×8) + 1 — explicit destination: dFrag starts as a
  ## copy of cFrag, then accumulates in place.
  ## (port of the CUDA twin; added with the backend-engine rework so the
  ## OpenCL path exercises both MMA forms like the _cuda file).
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
  var aFrag = make_fragment_A(tma.atom, tAv)
  aFrag.copyFrom(tAv)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(tBv)
  var cFrag = make_tensor(float32, (VC,))
  cFrag.fillWith(1.0'f32)                        # nonzero accumulator input
  var dFrag = make_tensor(float32, (VC,))

  dFrag.copyFrom(cFrag)                        # seed the accumulator input
  gemm_atom(tma.atom, dFrag, aFrag, bFrag)   # dFrag = aFrag·bFrag + cFrag

  axpby(1.0'f32, dFrag, 0.0'f32, tCv)

const kernelCode = opencl:
  proc mmaMicrotileKernel(A, B: ptr UncheckedArray[uint32];
                          C: ptr UncheckedArray[float32]) {.global.} =
    ## C(16×8) = A(16×8)·B(8×8) — one m16n8k8 tf32 atom, 32 work-items.
    ## Inputs-first (A, B, C): the engine's OpenCL `run` uses execOpenCL,
    ## which binds args 0..N-1 = inputs, arg N = output (vs the CUDA twins'
    ## output-first, which match execCuda's res-first convention).
    mmaMicrotile(tiled, int(get_local_id(0)), C, A, B)

  proc mmaMicrotileExplicitKernel(A, B: ptr UncheckedArray[uint32];
                                  C: ptr UncheckedArray[float32]) {.global.} =
    ## C(16×8) = A(16×8)·B(8×8) + 1 — the explicit-destination form.
    mmaMicrotileExplicit(tiled, int(get_local_id(0)), C, A, B)

# ═════════════════════════════════════════════════════════════════════════
#  Host side — thin shell: the oracle + trial loop + report live in
#  gemm_test_lib (testMicrotile); the engine's `run` dispatches to
#  execOpenCL. No buffer management here.
# ═════════════════════════════════════════════════════════════════════════

proc main() =
  var engine = "opencl".getEngine(kernelCode)
  doAssert engine.ctx.device.vendor().contains("NVIDIA"),
    "this kernel embeds NVIDIA inline PTX (asm mma.sync) — only NVIDIA's " &
    "OpenCL compiler accepts it; got device vendor: " & engine.ctx.device.vendor()
  testMicrotile(engine, atom, "SM80")
  echo "  OK: m16n8k8 tf32 microtile matches reference via OpenCL (tf32-exact fixture, 16 trials)"

when isMainModule:
  main()
