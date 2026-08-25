## Manual GPU test: gemm_cta's loop over the K dimension via the OpenCL
## backend.
##
## C(64×32) = α·A(64×64)·B(32×64) + β·C over a 2×2 CTA grid with two
## tileK-sized slices of K (K = 64 = 2·tileK, tileK = 32), 128
## work-items per CTA. Launch geometry is linearized: the engine
## launches grid.x·grid.y·blockSize work-items and the kernel
## decomposes the linear work-item id into (mCTA, nCTA, threadIdx).
## Reference, trial loop and report live in gemm_test_lib
## (testGemmCtaK64).
##
## NVIDIA-OpenCL only: the mma.sync inline PTX is embedded in the
## OpenCL C kernel as asm(...), which only NVIDIA's OpenCL compiler
## accepts. Host verifies the device vendor before launching.
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_cta_k64_opencl.nim --nimcache:nimcache/tests/manual_gemm_cta_k64_opencl.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_cta_k64_opencl.nim

import std/[strformat, strutils]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/hardware/h_configgen
import workspace/ceramic/src/hardware/h_registry
import workspace/ceramic/src/hardware/h_properties

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
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let mCTA = blk mod 2
    let nCTA = blk div 2
    let pA = make_view(A, (64, 64), (1, 64))
    let pB = make_view(B, (32, 64), (1, 32))
    let pC = make_view(C, (64, 32), (1, 64))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, 64, 32, 64, epi, (32, 16, 32), mCTA, nCTA, threadIdx)

proc runTest() =
  var engine = bkOpenCL.init(kernelCode)
  doAssert engine.deviceName().contains("NVIDIA"),
    "gemm_cta OpenCL needs NVIDIA's OpenCL compiler for the mma.sync asm; got: " &
    engine.deviceName()
  testGemmCtaK64(engine, tiled, "SM80")

when isMainModule:
  runTest()
