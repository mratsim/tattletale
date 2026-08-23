## Manual GPU test: F16 Apple simdgroup microtile, bit-exact.
##
## Requires an Apple GPU. Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests/manual_apple_gpu_metal_f16.nim \
##     --nimcache:nimcache/tests/manual_apple_gpu_metal_f16.nim \
##     workspace/ceramic/tests/atoms_mma/manual_apple_gpu_metal_f16.nim

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
#  Host f16 ↔ f32 conversions (bit patterns)
# ═════════════════════════════════════════════════════════════════════════
#  Bit-pattern conversions: the kernel operand type is crucible's
#  builtin `float16` (distinct uint16). The host transports uint16 bits.

func f16ToF32(h: uint16): float32 =
  ## Exact binary16 → binary32: every f16 value is representable in f32.
  let sign = uint32(h and 0x8000'u16) shl 16
  let e = (h shr 10) and 0x1F'u16
  let m = h and 0x3FF'u16
  if e == 0:
    if m == 0: return cast[float32](sign)    # ±0
    # Subnormal: value m·2⁻²⁴, normalized to a normal f32.
    var m2 = uint32(m)
    var e2 = 113
    while (m2 and 0x400'u32) == 0:
      m2 = m2 shl 1
      dec e2
    return cast[float32](sign or (uint32(e2) shl 23) or ((m2 and 0x3FF'u32) shl 13))
  if e == 0x1F:                              # Inf / NaN
    return cast[float32](sign or 0x7F800000'u32 or (uint32(m) shl 13))
  return cast[float32](sign or ((uint32(e) + 127 - 15) shl 23) or (uint32(m) shl 13))

func f32ToF16(x: float32): uint16 =
  ## IEEE-754 binary32 → binary16 bit pattern, round-to-nearest-even.
  ## Covers normal, subnormal, zero, Inf and NaN.
  ## NaN is quieted to a 0x200 payload (the hardware default).
  let u = cast[uint32](x)
  let sign = uint16((u shr 16) and 0x8000'u32)
  let aexp = int32((u shr 23) and 0xFF)
  let mant = u and 0x7FFFFF'u32
  if aexp == 0xFF:                           # Inf / NaN
    if mant == 0: return sign or 0x7C00'u16
    return sign or 0x7C00'u16 or 0x0200'u16
  var bexp = aexp - 127 + 15
  if bexp >= 31:                             # overflow → Inf
    return sign or 0x7C00'u16
  if bexp <= 0:                              # subnormal or zero
    if bexp < -10:                           # underflow → zero
      return sign
    # Subnormal significand: RNE(m2·2^(aexp−126)), computed as a right shift of (126 − aexp)
    # with round-to-nearest-even on the remainder.
    let m2 = mant or 0x800000'u32
    let shift = uint32(126 - aexp)
    let half = 1'u32 shl (shift - 1)
    var r = m2 shr shift
    let rem = m2 and ((1'u32 shl shift) - 1)
    if rem > half or (rem == half and (r and 1) == 1):
      r += 1
    return sign or uint16(r)
  # Normal: drop 13 low mantissa bits with round-to-nearest-even.
  let half = 0x1000'u32                       # 2¹², the round bit
  var r = mant shr 13
  let rem = mant and 0x1FFF'u32
  if rem > half or (rem == half and (r and 1) == 1):
    r += 1
    if r == 0x400:                           # mantissa overflow → bump exponent
      bexp += 1
      r = 0
      if bexp >= 31:
        return sign or 0x7C00'u16
  return sign or uint16(bexp shl 10) or uint16(r)

proc converterKat() =
  ## Known-answer vectors: pins the RNE ties, subnormal boundaries, Inf/NaN behavior.
  doAssert f16ToF32(0x3C00'u16) == 1.0'f32
  doAssert f16ToF32(0xBC00'u16) == -1.0'f32
  doAssert f16ToF32(0x3C01'u16) == 1.0009765625'f32
  doAssert f16ToF32(0x0001'u16) == 5.9604644775390625e-8'f32   # 2⁻²⁴
  doAssert f16ToF32(0x0400'u16) == 6.103515625e-5'f32          # 2⁻¹⁴
  doAssert classify(f16ToF32(0x7C00'u16)) == fcInf
  doAssert classify(f16ToF32(0xFC00'u16)) == fcNegInf
  doAssert classify(f16ToF32(0x7E00'u16)) == fcNan
  doAssert f32ToF16(1.5'f32) == 0x3E00'u16
  doAssert f32ToF16(1.00048828125'f32) == 0x3C00'u16           # 1+2⁻¹¹, tie-to-even down
  doAssert f32ToF16(1.0009765625'f32) == 0x3C01'u16            # 1+2⁻¹⁰, the discriminator
  doAssert f32ToF16(5.9604644775390625e-8'f32) == 0x0001'u16   # 2⁻²⁴
  doAssert f32ToF16(6.103515625e-5'f32) == 0x0400'u16          # 2⁻¹⁴
  doAssert f32ToF16(6.097555160522461e-5'f32) == 0x03FF'u16    # 2⁻¹⁴ − 2⁻²⁴
  doAssert f32ToF16(65520.0'f32) == 0x7C00'u16                 # overflow tie → Inf
  doAssert f32ToF16(-0.0'f32) == 0x8000'u16
  doAssert f32ToF16(Inf.float32) == 0x7C00'u16
  doAssert classify(f16ToF32(f32ToF16(NaN.float32))) == fcNan
  for i in -15 .. 15:
    doAssert f16ToF32(f32ToF16(float32(i))) == float32(i)
  echo "  OK: f16↔f32 converters match known-answer vectors (RNE, subnormal, Inf/NaN)"

# ═════════════════════════════════════════════════════════════════════════
#  Microtile kernels (library path, one 8×8×8 atom per call)
# ═════════════════════════════════════════════════════════════════════════

const atom = APPLE_8x8x8_F16
const tiled = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

func f16MmaMicrotile(tma: static TiledMma; t: int;
                     C: ptr UncheckedArray[float32];
                     A, B: ptr UncheckedArray[float16]) {.inline.} =
  ## One 8×8×8 f16 simdgroup atom (C = A·B), in-place, via the library path.
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
  aFrag.copyFrom(Aview)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(Bview)
  # make_fragment_C yields the SimdgroupFragment type gemm_atom's simdgroup overload requires.
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(0.0'f32)

  gemm_atom(tma.atom, cFrag, aFrag, bFrag)   # one simdgroup_multiply_accumulate

  Cview.copyFrom(cFrag)

func f16MmaMicrotileExplicit(tma: static TiledMma; t: int;
                             C: ptr UncheckedArray[float32];
                             A, B: ptr UncheckedArray[float16]) {.inline.} =
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
  aFrag.copyFrom(Aview)
  var bFrag = make_fragment_B(tma.atom, tBv)
  bFrag.copyFrom(Bview)
  var cFrag = make_fragment_C(tma.atom, tCv)
  cFrag.fillWith(1.0'f32)                        # nonzero accumulator input
  var dFrag = make_fragment_C(tma.atom, tCv)

  dFrag.copyFrom(cFrag)
  gemm_atom(tma.atom, dFrag, aFrag, bFrag)   # dFrag = aFrag·bFrag + cFrag

  Cview.copyFrom(dFrag)

func f16HalfFill(tma: static TiledMma; t: int;
                 A: ptr UncheckedArray[float16]) {.inline.} =
  ## Pins the half make_filled spelling the f32 kernels never emit.
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
  proc f16MmaKernel(C: ptr UncheckedArray[float32],
                    A, B: ptr UncheckedArray[float16]) {.global.} =
    f16MmaMicrotile(tiled, int(thread_index_in_simdgroup), C, A, B)

  proc f16MmaExplicitKernel(C: ptr UncheckedArray[float32],
                            A, B: ptr UncheckedArray[float16]) {.global.} =
    f16MmaMicrotileExplicit(tiled, int(thread_index_in_simdgroup), C, A, B)

  proc f16HalfFillKernel(A: ptr UncheckedArray[float16]) {.global.} =
    f16HalfFill(tiled, int(thread_index_in_simdgroup), A)

  proc bf16CopyKernel(Dst, Src: ptr UncheckedArray[bfloat16]) {.global.} =
    let i = int(thread_position_in_grid.x)
    Dst[i] = Src[i]

# ═════════════════════════════════════════════════════════════════════════
#  Fixtures, reference and the run loop
# ═════════════════════════════════════════════════════════════════════════

proc f16Fixture(rng: var Rand; M, K: int; discriminator: bool): seq[uint16] =
  ## (M, K) col-major f16 fixture as uint16 bit patterns.
  ## Discriminator = 1+2⁻¹⁰ (0x3C01), else −15..15 integers.
  result = newSeq[uint16](M * K)
  if discriminator:
    let disc = f32ToF16(1.0009765625'f32)    # 1+2⁻¹⁰, exact in f32
    doAssert disc == 0x3C01'u16
    for i in 0 ..< result.len:
      result[i] = disc
  else:
    for i in 0 ..< result.len:
      result[i] = f32ToF16(float32(rng.rand(-15 .. 15)))

proc verifyF16(atom: static MmaAtom; gpuC: openArray[float32];
               A, B: openArray[uint16]; cInit: float32; context: string) =
  ## Reference: exact f32 GEMM over f16ToF32 values.
  ## Gate: maxAbsErr == 0.
  const M = atom.mnk.m
  const N = atom.mnk.n
  const K = atom.mnk.k
  var refC = newSeq[float32](M * N)
  for m in 0 ..< M:
    for n in 0 ..< N:
      var sum = cInit
      for k in 0 ..< K:
        sum += f16ToF32(A[m + k * M]) * f16ToF32(B[n + k * N])
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
  doAssert msl.contains("make_filled_simdgroup_matrix<half, 8>"),
    "half-fragment fill must emit the half make_filled spelling"
  const
    M = atom.mnk.m
    N = atom.mnk.n
    K = atom.mnk.k
  var rng = initRand(0xC0FFEE)
  for trial in 0 ..< 16:
    # Trial 0 is the discriminator: the −15..15 domain cannot tell f16 from f32 accumulation.
    let A = f16Fixture(rng, M, K, discriminator = trial == 0)
    let B = f16Fixture(rng, N, K, discriminator = trial == 0)

    var gpuC = newSeq[float32](M * N)
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("f16MmaKernel", gpuC, (A, B))
    verifyF16(atom, gpuC, A, B, 0.0'f32, "in-place trial " & $trial)

    # explicit-output (5-arg)
    var gpuD = newSeq[float32](M * N)
    engine.run<<(1, toIntVal(atom.threadCount(opA)))>>("f16MmaExplicitKernel", gpuD, (A, B))
    verifyF16(atom, gpuD, A, B, 1.0'f32, "explicit trial " & $trial)

  echo "  OK: f16 m8n8k8 microtile matches the exact f32 reference (", atom.name,
       ", 16 trials incl. the 1+2⁻¹⁰ discriminator trial, in-place + explicit)"

proc runBf16Copy() =
  ## bfloat16 buffer copy, bit-identical.
  var engine = bkMetal.init()
  engine.ingest(kernelCode)
  let msl = engine.getArtifact()
  doAssert msl.contains("bfloat"), "bfloat16 buffers must emit the MSL bfloat type"
  var src = newSeq[uint16](1024)
  var dst = newSeq[uint16](1024)
  var rng = initRand(0xB16)
  for i in 0 ..< src.len:
    src[i] = uint16(rng.rand(0xFFFF))
  engine.run<<(1, 1024)>>("bf16CopyKernel", dst, (src,))
  for i in 0 ..< src.len:
    doAssert dst[i] == src[i], &"bf16 copy mismatch at {i}"
  echo "  OK: bfloat16 buffer copy bit-identical (", src.len, " elements)"

proc discriminatorProbe() =
  ## Proves the 1+2⁻¹⁰ trial separates f16 from f32 accumulation.
  const v = 1.0009765625'f32                  # 1+2⁻¹⁰, exact in f32
  let exact = 8.0'f32 * v * v
  let f16Products = 8.0'f32 * f16ToF32(f32ToF16(v * v))
  doAssert exact != f16Products,
    "discriminator: the f16-rounded alternative must differ from the exact f32 result"
  echo "  OK: discriminator separates f16 from f32 accumulation " &
       &"(exact {exact}, f16-rounded products {f16Products})"

when isMainModule:
  converterKat()
  discriminatorProbe()
  runTest()
  runBf16Copy()
