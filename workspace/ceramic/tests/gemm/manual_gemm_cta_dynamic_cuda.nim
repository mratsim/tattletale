## Manual GPU test: gemm_cta with runtime M/N/K and runtime leading
## strides via NVRTC/CUDA.
##
## Kernel receives M, N, K, ldA, ldB, ldC as runtime arguments and
## builds the input views (M, kView), (N, kView), (M, N) from them.
## View K is the allocated kView = 64 (pinned in runTest). Input K is
## runtime, at most kView. gemm_cta covers the ceil(M/32) ×
## ceil(N/16) CTA grid (tile 32×16, tileK = 32), 128 threads. Harness
## (testGemmCtaDynamic in gemm_test_lib) runs ragged K, M, and N
## shapes, padded strides, and β != 0.
##
## NaN K-pad (columns K ..< kView of A/B) proves the residue load
## zero-fills: a false-positive load reads NaN into the accumulator and
## the allClose check fails.
##
## Atom parameter: SM80_16x8x8_F32TF32TF32F32_TN tiled (2,2,1).
## Requires an sm_80+ GPU. Run with:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_gemm_cta_dynamic_cuda.nim --nimcache:nimcache/tests/manual_gemm_cta_dynamic_cuda.nim \
##     workspace/ceramic/tests/gemm/manual_gemm_cta_dynamic_cuda.nim

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
  proc gemmCtaDynamicKernel(
      C: ptr UncheckedArray[float32],
      A, B: ptr UncheckedArray[uint32],
      M, N, K, ldA, ldB, ldC: int32,
      alpha, beta: float32) {.global.} =
    let blk = int(blockIdx.x)
    let gridM = (int(M) + 31) div 32
    let mCTA = blk mod gridM
    let nCTA = blk div gridM
    let pA = make_view(A, (int(M), 64), (1, int(ldA)))
    let pB = make_view(B, (int(N), 64), (1, int(ldB)))
    let pC = make_view(C, (int(M), int(N)), (1, int(ldC)))
    let tC = local_tile(pC, (32, 16), (mCTA, nCTA))
    let thr = tiled.get_slice(int(threadIdx.x))
    var tCv = tiled.partition_C(thr, tC)
    var epi = initEpiAXPBY(alpha, beta, tCv)
    gemm_cta(tiled, tCv, pA, pB, int(M), int(N), int(K), epi, (32, 16, 32),
             mCTA, nCTA, int(threadIdx.x))

proc runTest() =
  # Kernel strings hardcode the tile literals (32, 16, 32), the
  # allocated-K literal 64, and the 128-thread block. Pin them to the
  # atom's tile shape so a config change cannot silently desync the
  # kernel from the harness.
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

  var engine = bkCuda.init(kernelCode)
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
  # ragged everything + padded strides + β != 0: the masked C read is
  # exercised (the other trials run β = 0 where C is never read)
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     48, 40, 48, 64, 48, 64, kView, alpha = 1.0'f32, beta = 1.0'f32)
  # K = 0: no loop over the K dimension, the load never runs, the
  # epilogue stores beta·C (the A/B buffers are entirely NaN and must
  # stay untouched)
  testGemmCtaDynamic(engine, tiled, "SM80", "gemmCtaDynamicKernel",
                     64, 32, 0, 64, 32, 64, kView, alpha = 1.0'f32, beta = 1.0'f32)

when isMainModule:
  runTest()
