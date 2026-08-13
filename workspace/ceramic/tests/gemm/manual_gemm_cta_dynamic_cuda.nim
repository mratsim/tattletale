## Manual GPU test: gemm_cta with runtime M/N and runtime leading strides
## via NVRTC/CUDA.
##
## The kernel receives M, N and ldA, ldB, ldC as runtime arguments (the
## launcher's problem shape) and builds the (M, K), (N, K), (M, N)
## problem views from them: the M/N shape leaves and the leading strides
## are runtime ints, K stays static (32). gemm_cta covers the
## ceil(M/tileM) × ceil(N/tileN) CTA grid (tile 32×16, K = tileK = 32),
## 128 threads.
##
## The harness (testGemmCtaDynamic in gemm_test_lib, shared with the
## OpenCL twin) runs:
##   - the dynamic-strides case: the 64×32×32 anchor problem with padded
##     leading strides (ldA = 80, ldB = 48, ldC = 80) — bit-exact vs the
##     static anchor
##   - ragged M/N cases: M = 48 (ragged M), N = 48 (ragged N) and
##     M = N = 48 (both ragged) with compact strides — the boundary tiles
##     gather zeros outside the problem and store only the valid elements
##     (no OOB)
##
## The atom is the parameter, SM80_16x8x8_F32TF32TF32F32_TN tiled
## (2,2,1). Requires an sm_80+ GPU. Run with:
##   nim cpp -r workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic_cuda.nim

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
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/ceramic/tests/gemm/gemm_test_lib
import workspace/crucible

{.experimental: "callOperator".}

const atom = SM80_16x8x8_F32TF32TF32F32_TN
const tiled = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

const kernelCode = cuda:
  # Dynamic-shape gemm_cta driver: M, N and the leading strides arrive as
  # kernel arguments; the views are built from them at runtime. The 1D
  # grid covers ceil(M/32) × ceil(N/16) CTAs (same decomposition as the
  # OpenCL twin's get_global_id path).
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global.} =
    let blk = int(blockIdx.x)
    let gridM = (int(M) + 31) div 32
    let mCTA = blk mod gridM
    let nCTA = blk div gridM
    let pA = make_view(A, (int(M), 32), (1, int(ldA)))
    let pB = make_view(B, (int(N), 32), (1, int(ldB)))
    let pC = make_view(C, (int(M), int(N)), (1, int(ldC)))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, int(M), int(N), epi, (32, 16, 32),
             mCTA, nCTA, int(threadIdx.x))

proc runTest() =
  # the kernel strings hardcode the tile literals (32, 16, 32) and the
  # 128-thread block; pin them to the atom's coverage so a config change
  # cannot desync the kernel from the harness silently
  static:
    doAssert 32 === toIntVal(tiled.threadLayout.shape[0]) * toIntVal(tiled.atom.mnk.m) and
      16 === toIntVal(tiled.threadLayout.shape[1]) * toIntVal(tiled.atom.mnk.n) and
      32 mod (toIntVal(tiled.threadLayout.shape[2]) * toIntVal(tiled.atom.mnk.k)) == 0 and
      128 === toIntVal(tiled.atom.threadCount(opA)) * toIntVal(product(tiled.threadLayout.shape)),
      "manual_gemm_cta_dynamic: the kernel's tile/block literals (32, 16, 32, 128)" &
      " must match the atom's coverage"

  var engine = bkCuda.init(kernelCode)
  # dynamic M/N + non-compact strides: the anchor problem with padding
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     64, 32, 80, 48, 80)
  # ragged M/N: compact strides, boundary tiles predicated
  # (48 = 1.5 tiles in M, 40 = 2.5 tiles in N: both ragged dims covered)
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     48, 32, 48, 32, 48)
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     32, 40, 32, 40, 32)
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     48, 40, 48, 40, 48)
  # ragged both dims + padded strides + β != 0: the masked C read is
  # exercised (the other trials run β = 0 where C is never read)
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     48, 40, 64, 48, 64, alpha = 1.0'f32, beta = 1.0'f32)

when isMainModule:
  runTest()
