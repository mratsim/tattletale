## Manual GPU test: unified gemm_cta entry with runtime M/N/K and runtime
## leading strides: one kernel source for all five Crucible backends.
##
## Kernel receives M, N, K, ldA, ldB, ldC as runtime arguments and builds
## the input views (M, kView), (N, kView), (M, N) from them.
## View K is the allocated kView = 64 (pinned in runTest). Input K is
## runtime, at most kView. gemm_cta covers the ceil(M/32) ×
## ceil(N/16) CTA grid (tile 32×16, tileK = 32), 128 threads. Harness
## (testGemmCtaDynamic in gemm_test_lib) runs ragged K, M, and N
## shapes, padded strides, and β != 0.
##
## NaN K-pad (columns K ..< kView of A/B) proves the residue load
## zero-fills: a false-positive load reads NaN into the accumulator,
## so the allClose check fails.
##
## The full gemm_cta body is written once in canonical names
## (thread_position_in_grid, threadgroup_barrier) and expanded for CUDA
## and OpenCL. Vulkan/Metal/WebGPU expand a portable slice instead,
## covering only the coordinates and synchronization (see below).
## Dispatch shape: 1D workgroups (128, 1, 1) and 1D linearized grids.
## The flat global id decomposes into (tid, blk), then (mCTA, nCTA).
## The {.workgroup: (128, 1, 1).} size is baked where the backend needs
## compile-time workgroup sizes (Vulkan, WGSL). CUDA, OpenCL and Metal
## take the block size at dispatch time.
##
## The tensor-core atom (cp.async / mma.sync inline PTX) is
## NVIDIA-ISA-bound: the full body is expanded for CUDA and OpenCL, which
## execute it where the ISA exists (CUDA on an sm_80+ GPU, OpenCL on
## NVIDIA's OpenCL compiler). Vulkan/Metal/WebGPU compile the portable
## coordinates + synchronization slice only, since the tensor-core atom
## is out of scope for this sprint. Compile-gates and source inspection live in the test_gemm_cta_dynamic_* files.
##
## Atom parameter: SM80_16x8x8_F32TF32TF32F32_TN tiled (2,2,1).
## Requires an sm_80+ GPU. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_cta_dynamic.nim --nimcache:nimcache/tests/manual_gemm_cta_dynamic.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic.nim

import std/strutils
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
const tiled* = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

# Work-group spans the full CTA: mma.sync is warp-synchronous, each
# 32-lane warp must execute it convergently, and 128 = 4 warps.
const blockSize = 128
static:
  doAssert blockSize == toIntVal(tiled.atom.threadCount(opA)) * 2 * 2 * 1

template gemmCtaDynamicDispatch(): (int, int, int, int, int, int) =
  ## The unified dispatch header: 1D linearized grid over 1D (128, 1, 1)
  ## workgroups. Returns (tid, blk, gid, gridM, mCTA, nCTA):
  ## the flat global id (thread_position_in_grid.x) decomposes into the flat thread
  ## id `tid` and the flat CTA id `blk`. The CTA grid is gridM × gridN CTAs,
  ## so (mCTA, nCTA) = (blk mod gridM, blk div gridM).
  ## blk, gid, and gridM are returned for the emitted-text shape only.
  ## The full-body kernels consume tid, mCTA, and nCTA. The slice kernels
  ## read only mCTA.
  ## Canonical names lower per backend (CUDA blockIdx/blockDim/threadIdx,
  ## OpenCL get_global_id, GLSL gl_GlobalInvocationID, WGSL global_id,
  ## MSL native).
  let gid = int(thread_position_in_grid.x)
  let tid = gid mod blockSize
  let blk = gid div blockSize
  let gridM = (int(M) + 31) div 32
  (tid, blk, gid, gridM, blk mod gridM, blk div gridM)

template gemmCtaKernelBody() {.dirty.} =
  ## Shared gemm_cta kernel body: view setup, epilogue init, and the gemm_cta
  ## call, written once in canonical names.
  ## Instantiated by the CUDA and OpenCL kernels. Per-backend coordinate
  ## and barrier lowerings come from the dispatch header and the builtin printers.
  ## The {.dirty.} pragma is load-bearing: template hygiene would gensym-rename
  ## the destructured names (tid, mCTA, nCTA), changing the emitted text that
  ## the CUDA inspect gate pins ("int mCTA = tmpTuple").
  let (tid, blk, gid, gridM, mCTA, nCTA) = gemmCtaDynamicDispatch()
  let pA = make_view(A, (int(M), 64), (1, int(ldA)))
  let pB = make_view(B, (int(N), 64), (1, int(ldB)))
  let pC = make_view(C, (int(M), int(N)), (1, int(ldC)))
  let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
  let thr = tiled.get_slice(tid)
  var tCv = tiled.partition_C(thr, tC)
  var epi = initEpiAXPBY(alpha, beta, tCv)
  gemm_cta(tiled, tCv, pA, pB, int(M), int(N), int(K), epi, (32, 16, 32),
           mCTA, nCTA, tid)

# ── Full body: the NVIDIA-ISA-bound atom is expanded for CUDA and OpenCL ──

const kernelCodeCuda* = cuda:
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, K, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global.} =
    gemmCtaKernelBody()

const kernelCodeOpenCL* = opencl:
  # Output-first (C): the OpenCL engine's run binds the output at binding 0,
  # inputs at 1..N in signature order.
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, K, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global.} =
    gemmCtaKernelBody()

# ── Portable slice: coordinates + synchronization, compiled and inspected ──
# The tensor-core atom cannot print on Vulkan/Metal/WebGPU (inline PTX),
# so those backends compile the dispatch header + barrier from the same
# unified source. Execution of the builtin layer on the runnable backends
# is proven by the cross-vocabulary kernels, not this slice.
# The slice kernels keep the full-entry signature: A, B, N, K, ldA, ldB,
# ldC, alpha, and beta are unused here by design. All five backends share
# one entry shape.
const kernelCodeVulkan* = vulkan:
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, K, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global, workgroup: (128, 1, 1).} =
    let (tid, blk, gid, gridM, mCTA, nCTA) = gemmCtaDynamicDispatch()
    threadgroup_barrier()
    C[0] = float32(mCTA)

const kernelCodeWebGPU* = webgpu:
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, K, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global, workgroup: (128, 1, 1).} =
    let (tid, blk, gid, gridM, mCTA, nCTA) = gemmCtaDynamicDispatch()
    threadgroup_barrier()
    C[0] = float32(mCTA)

const kernelCodeMetal* = metal:
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, K, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global.} =
    let (tid, blk, gid, gridM, mCTA, nCTA) = gemmCtaDynamicDispatch()
    threadgroup_barrier()
    C[0] = float32(mCTA)

proc runTest() =
  # Kernel strings hardcode the tile literals (32, 16, 32), the allocated-K
  # literal 64, and the 128-thread block. Pin them to the atom's tile shape:
  # a config change cannot silently desync the kernel from the harness.
  static:
    doAssert 32 === tiled.thrM * toIntVal(tiled.atom.mnk.m) and
      16 === tiled.thrN * toIntVal(tiled.atom.mnk.n) and
      32 mod (tiled.thrK * toIntVal(tiled.atom.mnk.k)) == 0 and
      64 mod 32 == 0 and
      128 === tiled.threadCount(),
      "manual_gemm_cta_dynamic: the kernel's tile/block literals (32, 16, 32, 64, 128)" &
      " must match the atom's coverage"

  # Allocated K the kernel views are built on: the kernel string's
  # literal must equal this. The tile pin above checks divisibility
  # only, so a literal change to another multiple of tileK would
  # silently desync without this exact-value pin.
  const kView = 64
  static:
    doAssert kView === 64,
      "manual_gemm_cta_dynamic: the allocated-K const (" & $kView &
      ") must equal the kernel string's view-K literal (64). The pin" &
      " below checks divisibility only"

  block:
    # CUDA: the execution gate (requires an sm_80+ GPU and runs on the Linux box).
    var engine = bkCuda.init(kernelCodeCuda)
    # K = 32 < kView = 64: one tileK-sized slice of K with padded strides,
    # runtime loop bound (ceil(K/tileK)), NaN K-pad untouched
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 32, 80, 48, 80, kView)
    # K = 64 = kView: two exact tileK-sized slices of K
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 64, 64, 32, 64, kView)
    # Ragged K = 48: two tileK-sized slices of K, residue 16, the last
    # slice's load zero-fills k >= 16
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 48, 64, 32, 64, kView)
    # K = 16 < tileK: one partial slice of K (validK = 16 < tileK = 32)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 16, 64, 32, 64, kView)
    # ragged N (40 = 2.5 tiles) + ragged K (40 = tileK + 8)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       32, 40, 40, 32, 40, 32, kView)
    # ragged M + ragged N + ragged K (48 = 1.5 tiles in M, 2.5 in N,
    # tileK + 16 in K)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       48, 40, 48, 48, 40, 48, kView)
    # ragged everything + padded strides + β != 0: the masked C read is exercised
    # (the other trials run β = 0 where C is never read)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       48, 40, 48, 64, 48, 64, kView, alpha = 1.0'f32, beta = 1.0'f32)
    # K = 0: no loop over the K dimension, the load never runs, the epilogue
    # stores beta·C (the A/B buffers are entirely NaN and must stay untouched)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 0, 64, 32, 64, kView, alpha = 1.0'f32, beta = 1.0'f32)

  block:
    # OpenCL: NVIDIA-OpenCL only (the mma.sync inline PTX), the bonus gate.
    var engine = bkOpenCL.init(kernelCodeOpenCL)
    doAssert engine.deviceName().contains("NVIDIA"),
      "gemm_cta dynamic OpenCL needs NVIDIA's OpenCL compiler for the mma.sync asm; got: " &
      engine.deviceName()
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 32, 80, 48, 80, kView)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 64, 64, 32, 64, kView)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 48, 64, 32, 64, kView)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 16, 64, 32, 64, kView)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       32, 40, 40, 32, 40, 32, kView)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       48, 40, 48, 48, 40, 48, kView)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       48, 40, 48, 64, 48, 64, kView, alpha = 1.0'f32, beta = 1.0'f32)
    testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                       64, 32, 0, 64, 32, 64, kView, alpha = 1.0'f32, beta = 1.0'f32)

when isMainModule:
  runTest()
