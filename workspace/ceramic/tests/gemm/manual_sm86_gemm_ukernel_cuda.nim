## Manual GPU test: the sm86 GEBB microkernel via NVRTC/CUDA — the loop over K.
##
## C(16×8) = A(16×16)·B(16×8) — two m16n8k8 k_blocks through gemm_ukernel
## (one gemm_atom per slice, accumulated in cFrag — CuTe sgemm_2.cu's ukernel K-loop
## analog). 32 threads. Fragment gathering is CuTe layout algebra (sgemm_2.cu): the
## partition of the full (M, 2K)/(N, 2K) views carries the k_blocks in its
## RestK mode; the register blocks are identity views and copyFrom does the
## whole gather — no for loops, no offset arithmetic. The epilogue is axpby.
##
## The atom is the parameter — SM86_16x8x8_F32TF32TF32F32_TN; the tiling is
## 1×1×1 (single atom); geometry is derived inside the driver func. The
## oracle harness lives in gemm_test_lib.
##
## Requires an sm_80+ GPU. Run with:
##   CUDA_HOME=/usr/local/cuda-12 LD_LIBRARY_PATH=/usr/local/cuda-12/lib64 \
##   nim cpp -r workspace/ceramic/tests/gemm/manual_sm86_gemm_ukernel_cuda.nim

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

const atom = SM86_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

func gemmUkernelMicrotile(tma: static TiledMma; t: int;
                          C: ptr UncheckedArray[float32];
                          A, B: ptr UncheckedArray[uint32]) {.inline.} =
  ## C(16×8) = A(16×16)·B(16×8) — two m16n8k8 k_blocks via gemm_ukernel.
  ## Fragment gathering: partition_A/B of the full (M, 2K)/(N, 2K) views, the
  ## fragment blocks as owning tensors (make_fragment_A/B), one copyFrom
  ## gathers all slices — no loops, no offsets, no raw-addr views.
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
  # the fragment blocks as owning tensors shaped like the partitions:
  # V flattened to atom register order, the k_blocks are the partition's
  # RestK mode. One copyFrom gathers the whole block through each
  # tensor's layout (coordinate semantics: dst(i)/src(i) decode i through
  # their own shapes, identical flat enumeration, matching gemm_ukernel's
  # coordinate slices)
  var aFrag = make_fragment_A(tma.atom, tAv)
  aFrag.copyFrom(tAv)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(tBv)
  # the accumulator: make_tensor, passed directly to gemm_ukernel
  var cFrag = make_tensor(float32, (VC,))
  cFrag.fillWith(0.0'f32)

  gemm_ukernel(tma.atom, cFrag, aFrag, bFrag)   # two mma.sync, accumulating

  axpby(1.0'f32, cFrag, 0.0'f32, tCv)

const kernelCode = cuda:
  proc gemmUkernelKernel(C: ptr UncheckedArray[float32],
                         A, B: ptr UncheckedArray[uint32]) {.global.} =
    gemmUkernelMicrotile(tiled, int(threadIdx.x), C, A, B)

proc main() =
  var engine = "cuda".getEngine(kernelCode)
  testUkernel(engine, atom, "SM86")

when isMainModule:
  main()
