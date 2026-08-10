## Manual GPU test: the sm80 tensor-core microtile via NVRTC/CUDA — one register-level MMA.
##
## gemm_fragment(atom.instr, cFrag, aFrag, bFrag) replaces the hand-written mma.sync asm:
## one m16n8k8 tf32 atom, 32 threads. Fragment gathering is CuTe layout algebra
## (sgemm_2.cu): partition_A/B/C once (thr_mma.partition), the fragment
## registers as identity views, and copyFrom/fillWith do the gather/clear —
## no for loops, no offset arithmetic. The epilogue is axpby (gemm_device
## convention). Plus the explicit-output 5-arg form:
## gemm_fragment(atom.instr, dFrag, aFrag, bFrag, cFrag).
##
## The atom is the parameter — SM80_16x8x8_F32TF32TF32F32_TN; the tiling is
## 1×1×1 (single atom); geometry is derived inside the driver func. The
## oracle harness lives in gemm_test_lib.
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm80_tensor_cores_cuda.nim

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
import workspace/crucible/src/codegen/nvrtc

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

func mmaMicrotile(tma: static TiledMma; t: int;
                  C: ptr UncheckedArray[float32];
                  A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## C(16×8) = A(16×8)·B(8×8) — one m16n8k8 tf32 atom, 32 threads, in-place.
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

func mmaMicrotileExplicit(tma: static TiledMma; t: int;
                          C: ptr UncheckedArray[float32];
                          A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## C(16×8) = A(16×8)·B(8×8) + 1 — the 5-arg form: dFrag out, cFrag in.
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

  gemm_fragment(tma.atom.instr, dFrag.data, aFrag.data, bFrag.data, cFrag.data)   # dFrag = aFrag·bFrag + cFrag

  axpby(1.0'f32, dFrag, 0.0'f32, tCv)

const kernelCode = cuda:
  proc mmaMicrotileKernel(C: ptr UncheckedArray[float32],
                          A, B: ptr UncheckedArray[uint32]) {.global.} =
    mmaMicrotile(tiled, int(threadIdx.x), C, A, B)

  proc mmaMicrotileExplicitKernel(C: ptr UncheckedArray[float32],
                                  A, B: ptr UncheckedArray[uint32]) {.global.} =
    mmaMicrotileExplicit(tiled, int(threadIdx.x), C, A, B)

when isMainModule:
  var nv = initNvrtc(kernelCode)
  nv.compile()
  nv.getPtx()
  testMicrotile(nv, atom, "SM80")
  # Row-major operands are not exercised here: partition_A/B/C's thread
  # (T-mode) offsets collapse for row-major views (col-major thread-1
  # offset 16 vs row-major 1 — all threads read nearly the same data), a
  # pre-existing bug in the partition's coordinate composition. The
  # fragment layer itself is correct (GPU dumps match A[m·8+k]); the
  # partition fix and the row-major acceptance test are tracked as a
  # follow-up.
