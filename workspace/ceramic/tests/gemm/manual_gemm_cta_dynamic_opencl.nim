## Manual GPU test: gemm_cta with runtime M/N and runtime leading strides
## via the OpenCL backend.
##
## Same kernel and problems as manual_gemm_cta_dynamic_cuda.nim: the
## kernel receives M, N, ldA, ldB, ldC as runtime arguments, builds the
## problem views from them (K stays static 32), and gemm_cta covers the
## ceil(M/32) × ceil(N/16) CTA grid, 128 work-items per CTA. The launch
## geometry is linearized: the engine launches gridM·gridN·blockSize
## work-items and the kernel decomposes the linear id into
## (mCTA, nCTA, threadIdx). Work-groups span the full CTA (blk.x = 128):
## mma.sync works in multi-warp groups on NVIDIA's OpenCL (verified by
## experiment, the old one-warp pin was over-cautious).
##
## The oracle, trial loop and report live in gemm_test_lib
## (testGemmCtaDynamic), shared with the CUDA twin.
##
## NVIDIA-OpenCL only: the mma.sync inline PTX travels inside the OpenCL C
## kernel as asm(...), which only NVIDIA's OpenCL compiler accepts. The
## host verifies the device vendor before launching.
##
## Requires an sm_80+ GPU with NVIDIA's OpenCL. Run with:
##   nim c -r workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic_opencl.nim

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
  # Dynamic-shape gemm_cta driver: M, N and the leading strides arrive as
  # kernel arguments; the views are built from them at runtime. The
  # linear work-item id decomposes into (mCTA, nCTA, threadIdx) over the
  # ceil(M/32) × ceil(N/16) CTA grid. Output-first (C): the OpenCL
  # engine's run binds the output at binding 0, inputs at 1..N in
  # signature order.
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global.} =
    let gid = int(get_global_id(0))
    let threadIdx = gid mod 128
    let blk = gid div 128
    let gridM = (int(M) + 31) div 32
    let mCTA = blk mod gridM
    let nCTA = blk div gridM
    let pA = make_view(A, (int(M), 32), (1, int(ldA)))
    let pB = make_view(B, (int(N), 32), (1, int(ldB)))
    let pC = make_view(C, (int(M), int(N)), (1, int(ldC)))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(threadIdx)
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, int(M), int(N), epi, (32, 16, 32),
             mCTA, nCTA, threadIdx)

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

  var engine = bkOpenCL.init(kernelCode)
  doAssert engine.deviceName().contains("NVIDIA"),
    "gemm_cta dynamic OpenCL needs NVIDIA's OpenCL compiler for the mma.sync asm; got: " &
    engine.deviceName()
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
