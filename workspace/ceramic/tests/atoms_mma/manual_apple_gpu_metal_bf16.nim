## Manual GPU test: BF16 Apple simdgroup microtile, bit-exact.
##
## Requires an Apple GPU (bf16 simdgroup matrices need Metal 3.1+).
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_apple_gpu_metal_bf16.nim \
##     --nimcache:nimcache/tests/manual_apple_gpu_metal_bf16.nim \
##     workspace/ceramic/tests/atoms_mma/manual_apple_gpu_metal_bf16.nim

import std/[strformat, strutils, random, math]

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/atoms
import workspace/ceramic/src/kernel_gemm/atoms_apple
import workspace/ceramic/src/atoms_mma_partitioning
import workspace/ceramic/src/tensors
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/kernel_copy_gpu
import workspace/ceramic/src/kernel_fillwith_gpu
import workspace/ceramic/src/kernel_gemm_epilogues
import workspace/ceramic/src/kernel_gemm_gpu
import workspace/crucible

{.experimental: "callOperator".}

# ═════════════════════════════════════════════════════════════════════════
#  Host bf16 ↔ f32 conversions (bit patterns)
# ═════════════════════════════════════════════════════════════════════════
#  bf16 is the top 16 bits of f32: conversion is a shift, no table.

func bf16ToF32(h: uint16): float32 =
  ## Exact: subnormals, Inf and NaN included.
  cast[float32](uint32(h) shl 16)

func f32ToBf16(x: float32): uint16 =
  ## RNE on the dropped 16 bits (spec: float→bfloat rounds ties to even).
  let u = cast[uint32](x)
  uint16((u + 0x7FFF'u32 + ((u shr 16) and 1)) shr 16)

proc converterKat() =
  ## Known-answer vectors: RNE ties, subnormal, Inf/NaN.
  doAssert bf16ToF32(0x3F80'u16) == 1.0'f32
  doAssert bf16ToF32(0x8001'u16) == -9.183549615799121e-41'f32   # −2⁻¹³³
  doAssert classify(bf16ToF32(0x7F80'u16)) == fcInf
  doAssert classify(bf16ToF32(0x7FC0'u16)) == fcNan
  doAssert f32ToBf16(1.5'f32) == 0x3FC0'u16
  doAssert f32ToBf16(1.00390625'f32) == 0x3F80'u16               # 1+2⁻⁸, tie down
  doAssert f32ToBf16(1.01171875'f32) == 0x3F82'u16               # 1+2⁻⁷+2⁻⁸, tie up
  doAssert f32ToBf16(1.005859375'f32) == 0x3F81'u16              # 1+2⁻⁸+2⁻⁹, round up
  for i in -15 .. 15:
    doAssert bf16ToF32(f32ToBf16(float32(i))) == float32(i)
  echo "  OK: bf16↔f32 converters match known-answer vectors (RNE, subnormal, Inf/NaN)"

# ═════════════════════════════════════════════════════════════════════════
#  Microtile kernels (library path, one 8×8×8 atom per call)
# ═════════════════════════════════════════════════════════════════════════

const atom = APPLE_8x8x8_BF16
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

func bf16MmaMicrotile(tma: static TiledMma; t: int;
                      C: ptr UncheckedArray[float32];
                      A, B: ptr UncheckedArray[bfloat16]) {.inline.} =
  ## One 8×8×8 bf16 simdgroup atom (C = A·B), in-place, via the library path.
  const
    M = tma.atom.mnk.m
    N = tma.atom.mnk.n
    K = tma.atom.mnk.k
  let Aview = make_view(A, make_layout((M, K), (1, M)))
  let Bview = make_view(B, make_layout((N, K), (1, N)))
  var Cview = make_view(C, make_layout((M, N), (1, M)))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, Aview)
  let tBv = tma.partition_B(thr, Bview)
  var tCv = tma.partition_C(thr, Cview)
  var aFrag = make_fragment_A(tma.atom, tAv)
  simdgroupLoad(aFrag, Aview.data, uint32(M), 0'u32, true)
  var bFrag = make_fragment_B(tma.atom, tBv)
  simdgroupLoad(bFrag, Bview.data, uint32(N), 0'u32, false)
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(0.0'f32)

  gemm_atom(tma.atom, cFrag, aFrag, bFrag)   # one simdgroup_multiply_accumulate

  simdgroupStore(cFrag, Cview.data, uint32(M), 0'u32, true)

func bf16MmaMicrotileExplicit(tma: static TiledMma; t: int;
                              C: ptr UncheckedArray[float32];
                              A, B: ptr UncheckedArray[bfloat16]) {.inline.} =
  ## Same atom, explicit destination (C = A·B + cFrag, cFrag = 1.0).
  const
    M = tma.atom.mnk.m
    N = tma.atom.mnk.n
    K = tma.atom.mnk.k
  let Aview = make_view(A, make_layout((M, K), (1, M)))
  let Bview = make_view(B, make_layout((N, K), (1, N)))
  var Cview = make_view(C, make_layout((M, N), (1, M)))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, Aview)
  let tBv = tma.partition_B(thr, Bview)
  var tCv = tma.partition_C(thr, Cview)
  var aFrag = make_fragment_A(tma.atom, tAv)
  simdgroupLoad(aFrag, Aview.data, uint32(M), 0'u32, true)
  var bFrag = make_fragment_B(tma.atom, tBv)
  simdgroupLoad(bFrag, Bview.data, uint32(N), 0'u32, false)
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(1.0'f32)                        # nonzero accumulator input
  var dFrag = make_fragment_C(tma.atom, tCv)

  dFrag = cFrag
  gemm_atom(tma.atom, dFrag, aFrag, bFrag)   # dFrag = aFrag·bFrag + cFrag

  simdgroupStore(dFrag, Cview.data, uint32(M), 0'u32, true)

func bf16Fill(tma: static TiledMma; t: int;
              A: ptr UncheckedArray[bfloat16]) {.inline.} =
  ## Pins the bfloat make_filled spelling the f32 kernels never emit.
  const
    M = tma.atom.mnk.m
    K = tma.atom.mnk.k
  let Aview = make_view(A, make_layout((M, K), (1, M)))
  let thr = tma.get_slice(t)
  let tAv = tma.partition_A(thr, Aview)
  var aFrag = make_fragment_A(tma.atom, tAv)
  let h = A[0]
  aFrag.fillWith(h)

const kernelCode = metal:
  proc bf16MmaKernel(C: ptr UncheckedArray[float32],
                     A, B: ptr UncheckedArray[bfloat16]) {.global.} =
    bf16MmaMicrotile(tiled, int(thread_index_in_simdgroup), C, A, B)

  proc bf16MmaExplicitKernel(C: ptr UncheckedArray[float32],
                             A, B: ptr UncheckedArray[bfloat16]) {.global.} =
    bf16MmaMicrotileExplicit(tiled, int(thread_index_in_simdgroup), C, A, B)

  proc bf16FillKernel(A: ptr UncheckedArray[bfloat16]) {.global.} =
    bf16Fill(tiled, int(thread_index_in_simdgroup), A)

# ═════════════════════════════════════════════════════════════════════════
#  Fixtures, reference and the run loop
# ═════════════════════════════════════════════════════════════════════════

proc bf16Fixture(rng: var Rand; M, K: int): seq[uint16] =
  ## (M, K) col-major bf16 fixture as uint16 bit patterns, −15..15.
  result = newSeq[uint16](M * K)
  for i in 0 ..< result.len:
    result[i] = f32ToBf16(float32(rng.rand(-15 .. 15)))

proc verifyBf16(atom: static MmaAtom; gpuC: openArray[float32];
                A, B: openArray[uint16]; cInit: float32; context: string) =
  ## Reference: exact f32 GEMM over bf16ToF32 values.
  ## Gate: maxAbsErr == 0.
  const M = atom.mnk.m
  const N = atom.mnk.n
  const K = atom.mnk.k
  var refC = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var sum = cInit
      for k in 0 ..< K:
        sum += bf16ToF32(A[m + k * M]) * bf16ToF32(B[n + k * N])
      refC[m + n * M] = sum
  var maxAbsErr = 0.0'f32
  for i in 0 ..< M * N:
    let absErr = abs(gpuC[i] - refC[i])
    if absErr > maxAbsErr: maxAbsErr = absErr
    doAssert absErr == 0.0'f32,
      &"{context} [{i mod M},{i div M}]: got {gpuC[i]}, expected {refC[i]}"
  echo &"    PASS (maxAbsErr={maxAbsErr:.2e})"

proc runTest() =
  var engine = bkMetal.init()
  engine.ingest(kernelCode)
  let msl = engine.getArtifact()
  doAssert msl.contains("simdgroup_bfloat8x8"),
    "bf16 fragments must emit the simdgroup_bfloat8x8 matrix type"
  doAssert msl.contains("make_filled_simdgroup_matrix<bfloat, 8>"),
    "bf16-fragment fill must emit the bfloat make_filled spelling"
  const
    M = atom.mnk.m
    N = atom.mnk.n
    K = atom.mnk.k
  var rng = initRand(0xB16)
  for trial in 0 ..< 16:
    # Partial sums ≤ 1800 need 11 bits — inexact in bf16 — so the f32
    # accumulator discriminates bf16 accumulation.
    let A = bf16Fixture(rng, M, K)
    let B = bf16Fixture(rng, N, K)

    var gpuC = newSeq[float32](M * N)
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("bf16MmaKernel", gpuC, (A, B))
    verifyBf16(atom, gpuC, A, B, 0.0'f32, "in-place trial " & $trial)

    var gpuD = newSeq[float32](M * N)
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("bf16MmaExplicitKernel", gpuD, (A, B))
    verifyBf16(atom, gpuD, A, B, 1.0'f32, "explicit trial " & $trial)

  echo "  OK: bf16 m8n8k8 microtile matches the exact f32 reference (", atom.name,
       ", 16 trials, in-place + explicit)"

when isMainModule:
  converterKat()
  runTest()
