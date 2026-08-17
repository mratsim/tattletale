## Manual GPU test: the sm80 warp-level GEMM (gemm_warp)
## via NVRTC/CUDA.
##
## C(16×8) = A(16×16)·B(16×8): two m16n8k8 k slices through
## gemm_warp, one gemm_atom per slice accumulated in cFrag,
## 32 threads. Epilogue is a direct identity copy to C.
##
## Atom is the parameter, SM80_16x8x8_F32TF32TF32F32_TN. Tiling is
## 1×1×1 (single atom). Geometry derives inside the driver func.
## Reference harness lives in gemm_test_lib.
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_sm80_gemm_warp_cuda.nim --nimcache:nimcache/tests/manual_sm80_gemm_warp_cuda.nim \
##     workspace/ceramic/tests/gemm/manual_sm80_gemm_warp_cuda.nim

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

func gemmWarpMicrotile(tma: static TiledMma; t: int;
                          C: ptr UncheckedArray[float32];
                          A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## C(16×8) = A(16×16)·B(16×8): two m16n8k8 k slices via gemm_warp.
  ## Fragment gathering: partition_A/B of the full (M, 2K)/(N, 2K) views,
  ## fragments as owning tensors (make_fragment_A/B). No loops, no
  ## offsets, no raw-addr views.
  const
    kSlices = 2
    M = tma.atom.mnk.m
    N = tma.atom.mnk.n
    K = tma.atom.mnk.k
    VA = toIntVal(tma.atom.valuesPerThread(opA))
    VB = toIntVal(tma.atom.valuesPerThread(opB))
    VC = toIntVal(tma.atom.valuesPerThread(opC))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, make_view(A, make_layout((M, kSlices * K), (1, M))))
  let tBv = tma.partition_B(thr, make_view(B, make_layout((N, kSlices * K), (1, N))))
  var tCv = tma.partition_C(thr, make_view(C, make_layout((M, N), (1, M))))
  # fragments as owning tensors shaped like the partitions:
  # V flattened to atom register order, the k slices in the partition's
  # RepeatK mode, so one copyFrom gathers all slices in the flat order
  # gemm_warp indexes per k slice
  var aFrag = make_fragment_A(tma.atom, tAv)
  aFrag.copyFrom(tAv)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(tBv)
  # accumulator: flat (VC,), zeroed, gemm_warp accumulates in place
  var cFrag = make_tensor(float32, (VC,))
  cFrag.fillWith(0.0'f32)

  gemm_warp(tma.atom, cFrag, aFrag, bFrag)   # two mma.sync, accumulating

  for i in 0 ..< size(tCv.layout):
    tCv(i) = cFrag(i)

const kernelCode = cuda:
  proc gemmWarpKernel(C: ptr UncheckedArray[float32],
                         A, B: ptr UncheckedArray[uint32]) {.global.} =
    gemmWarpMicrotile(tiled, int(threadIdx.x), C, A, B)

proc runTest() =
  var engine = bkCuda.init(kernelCode)
  testWarp(engine, atom, "SM80")

when isMainModule:
  runTest()
