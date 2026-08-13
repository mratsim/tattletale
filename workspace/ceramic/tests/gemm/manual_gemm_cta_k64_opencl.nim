## Manual GPU test: the k-tile loop through gemm_cta via the OpenCL
## backend.
##
## Same problem as manual_gemm_cta_k64_cuda.nim: C(64×32) = α·A(64×64)·
## B(32×64) + β·C over a 2×2 CTA grid with TWO k-tiles (K = 64 =
## 2·BLK_K, BLK_K = 32), 128 work-items per CTA. The kernel body is the
## CUDA twin's verbatim; the launch geometry is linearized: the engine
## launches grid.x·grid.y·blockSize work-items and the kernel decomposes
## the linear work-item id into (mCTA, nCTA, threadIdx). The oracle,
## trial loop and report live in gemm_test_lib (testGemmCtaK64),
## shared with the CUDA twin.
##
## NVIDIA-OpenCL only: the mma.sync inline PTX travels inside the OpenCL C
## kernel as asm(...), which only NVIDIA's OpenCL compiler accepts. The
## host verifies the device vendor before launching.
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run with:
##   nim cpp -r workspace/ceramic/tests/gemm/manual_gemm_cta_k64_opencl.nim

import std/[strformat, strutils]
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

# Work-group spans the full CTA: mma.sync is warp-synchronous, each
# 32-lane warp must execute it convergently, and 128 = 4 warps.
const blockSize = 128
static:
  doAssert blockSize == toIntVal(tiled.atom.threadCount(opA)) * 2 * 2 * 1

const kernelCode = opencl:
  proc gemmCtaK64Kernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      alpha, beta: float32) {.global.} =
    ## Same 2×2 CTA grid decomposition with K = 64 = 2·BLK_K: gemm_cta
    ## hands each CTA a full-K (32, 64) / (16, 64) tile view and loops
    ## the two k-tiles into the persistent accumulator (gemm_tiled
    ## computes one k-tile each).
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let m0 = mCTA * 32
    let n0 = nCTA * 16
    var tC = make_view(C +% (m0 + n0 * 64), (32, 16), (1, 64))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, A, 64, B, 32, epi, 64, 32, 64, (32, 16, 32), mCTA, nCTA, threadIdx)

proc runTest() =
  var engine = bkOpenCL.init(kernelCode)
  doAssert engine.deviceName().contains("NVIDIA"),
    "gemm_cta OpenCL needs NVIDIA's OpenCL compiler for the mma.sync asm; got: " &
    engine.deviceName()
  testGemmCtaK64(engine, tiled, "SM80")

when isMainModule:
  runTest()
