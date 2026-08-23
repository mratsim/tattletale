## Manual GPU test: sm80 tensor-core microtile via the OpenCL backend.
## NVIDIA-OpenCL only: the inline PTX mma.sync needs NVIDIA's compiler,
## and the work-group is pinned to one warp (warp-synchronous).
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_sm80_tensor_cores_opencl.nim \
##     --nimcache:nimcache/tests/manual_sm80_tensor_cores_opencl.nim \
##     workspace/ceramic/tests/atoms_mma/manual_sm80_tensor_cores_opencl.nim

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
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

const WORK_GROUP = toIntVal(atom.threadCount(opA))   # 32 work-items = one warp
static:
  doAssert WORK_GROUP == 32,
    "mma.sync needs exactly one warp (32 work-items), got " & $WORK_GROUP

func mmaMicrotile(tma: static TiledMma; t: int;
                  C: ptr UncheckedArray[float32];
                  A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## One m16n8k8 tf32 atom (C = A·B), in-place, via the library path.
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
  # Accumulator is identity-shaped: make_tensor_like would scramble register order to 0,2,1,3.
  var cFrag = make_tensor(float32, (VC,))
  cFrag.fillWith(0.0'f32)

  gemm_atom(tma.atom, cFrag, aFrag, bFrag)   # one mma.sync, in-place accumulate

  for i in 0 ..< size(tCv.layout):
    tCv(i) = cFrag(i)

func mmaMicrotileExplicit(tma: static TiledMma; t: int;
                          C: ptr UncheckedArray[float32];
                          A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## Same atom, explicit destination (C = A·B + cFrag, cFrag = 1.0).
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

  dFrag.copyFrom(cFrag)
  gemm_atom(tma.atom, dFrag, aFrag, bFrag)   # dFrag = aFrag·bFrag + cFrag

  for i in 0 ..< size(tCv.layout):
    tCv(i) = dFrag(i)

const kernelCode = opencl:
  proc mmaMicrotileKernel(C: ptr UncheckedArray[float32],
                          A, B: ptr UncheckedArray[uint32]) {.global.} =
    ## Output-first: the engine binds the output at binding 0, inputs at 1..N.
    mmaMicrotile(tiled, int(get_local_id(0)), C, A, B)

  proc mmaMicrotileExplicitKernel(C: ptr UncheckedArray[float32],
                                  A, B: ptr UncheckedArray[uint32]) {.global.} =
    mmaMicrotileExplicit(tiled, int(get_local_id(0)), C, A, B)

proc runTest() =
  var engine = bkOpenCL.init(kernelCode)
  doAssert engine.deviceName().contains("NVIDIA"),
    "this kernel embeds NVIDIA inline PTX (asm mma.sync) — only NVIDIA's " &
    "OpenCL compiler accepts it; got device name: " & engine.deviceName()
  testMicrotile(engine, atom, "SM80")
  echo "  OK: m16n8k8 tf32 microtile matches reference via OpenCL (tf32-exact fixture, 16 trials)"

when isMainModule:
  runTest()
