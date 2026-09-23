# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/naive/t_naive_harness.nim
##
## The naive-tier harness self-checks, one flat main over the shared
## naive modules:
##
## - same-seed stream reproducibility, cross-seed divergence, checks
##   for the stream advance, golden stream values and an entropy floor
## - worstSeqDiff, bitExactCount and guarded-tail checks on known-good,
##   known-bad and edge shapes, plus the shape-budget guard raises
## - tensor element access, random-tensor dtype narrowing and widening
##
## Shapes stay inside the NaivePrefillMaxT / NaiveMaxDk ceilings.

import std/assertions
import naive_rng, naive_tensors, naive_metrics, naive_budget

proc main =
  runTimed "rng same-seed stream reproducibility":
    var a = initNaiveRng(0xC0FFEE'u64)
    var b = initNaiveRng(0xC0FFEE'u64)
    for i in 0 ..< 1024:
      doAssert a.nextU64() == b.nextU64(),
        "same-seed streams diverged at step " & $i
    var c = initNaiveRng(7'u64)
    var d = initNaiveRng(7'u64)
    for i in 0 ..< 256:
      let x = c.nextF32(-2.0'f32, 3.0'f32)
      doAssert x == d.nextF32(-2.0'f32, 3.0'f32),
        "same-seed float streams diverged at step " & $i
      doAssert c.nextInt(0, 97) == d.nextInt(0, 97),
        "same-seed int streams diverged at step " & $i

  runTimed "rng cross-seed divergence and ranges":
    # Distinct seeds start disjoint stream orbits, any seed corruption
    # surfaces as a stream mismatch at the first compared output.
    var a = initNaiveRng(0xC0FFEE'u64)
    var b = initNaiveRng(0xC0FFED'u64)
    doAssert a.nextU64() != b.nextU64(), "distinct seeds produced equal outputs"
    var r = initNaiveRng(42'u64)
    for i in 0 ..< 256:
      let x = r.nextF32(-1.5'f32, 2.5'f32)
      doAssert x >= -1.5'f32 and x < 2.5'f32, "out-of-range sample " & $x
      doAssert r.nextInt(0, 97) >= 0 and r.nextInt(0, 97) < 97,
        "out-of-range sample"

  runTimed "rng stream advance, golden values and entropy floor":
    var a = initNaiveRng(0xC0FFEE'u64)
    let first = a.nextU64()
    let second = a.nextU64()
    doAssert first != second, "stream did not advance between consecutive values"
    # Golden stream values of the xorshift64* sequence:
    # - the state update depends on the seed alone, never on compiler runtime
    # - re-measure these only on an intentional change of the generator
    doAssert first == 0xFEC579340D9C0AD5'u64,
      "stream value 1 diverged from the golden sequence"
    doAssert second == 0x2F333D299677249F'u64,
      "stream value 2 diverged from the golden sequence"
    var z = initNaiveRng(0'u64)
    doAssert z.nextU64() == 0x0D83B3E29A21487A'u64,
      "zero-seed golden state mapping wrong"
    # Entropy floor over 512 samples in [-1, 1), an 8-bit-quantized
    # stream yields at most 256 distinct values, the 2⁻²⁴ granularity
    # of nextF32 yields far more.
    var r = initNaiveRng(5'u64)
    var samples: array[512, float32]
    for i in 0 ..< 512:
      samples[i] = r.nextF32(-1.0'f32, 1.0'f32)
    var distinctCount = 0
    for i in 0 ..< 512:
      var seen = false
      for j in 0 ..< i:
        if samples[j] == samples[i]:
          seen = true
          break
      if not seen:
        inc distinctCount
    doAssert distinctCount > 300,
      "nextF32 entropy collapsed to " & $distinctCount & " distinct values over 512 samples"

  runTimed "worstSeqDiff known-good and known-bad pairs":
    let got = @[1.0'f32, 2.0, 4.0]
    let bad = @[1.0'f32, 7.0, 4.0]
    doAssert worstSeqDiff(got, got) == 0.0, "identical pairs differ"
    doAssert worstSeqDiff(got, bad) == 5.0, "worst difference mislocated"
    doAssert worstSeqDiff(bad, got) == 5.0, "worst difference not symmetric"
    doAssertRaises(ValueError):
      discard worstSeqDiff(@[0.0'f32, 1.0'f32], @[0.0'f32, NaN])

  runTimed "worstSeqDiff edge shapes":
    doAssertRaises(AssertionDefect):
      discard worstSeqDiff(@[0.0'f32, 1.0'f32], @[0.0'f32])
    # Empty-pair contract, two empty sequences compare as equal,
    # the worst difference is 0.0.
    doAssert worstSeqDiff(newSeq[float32](0), newSeq[float32](0)) == 0.0,
      "empty-pair contract drifted"

  runTimed "bitExactCount on mixed 16-bit patterns":
    let got: seq[uint16] = @[0x3C00'u16, 0x4000, 0x7FC0]
    let bad: seq[uint16] = @[0x3C00'u16, 0x4200, 0x7FC0]
    doAssert bitExactCount(got, got) == 3, "identical patterns counted low"
    doAssert bitExactCount(got, bad) == 2, "bit-exact count wrong on mixed pair"
    doAssert bitExactCount(got, @[0'u16, 0'u16, 0'u16]) == 0,
      "all-mismatch pair counted nonzero"

  runTimed "guarded-tail untouched-memory check, tiny case":
    var g = initGuardedTail(4, 3, 0xDEAD'u16)
    let p = g.data.asUnchecked()
    for i in 0 ..< g.used:
      p[i] = uint16(i + 1)
    assertTailUntouched(g)
    var buf = @[1'u16, 2, 3, 4, 5, 6]
    let snap = snapshotSeq(toOpenArray(buf, 2, 4))
    buf[3] = 99
    doAssert countChanged(toOpenArray(buf, 2, 4), snap) == 1,
      "detector missed a changed element"
    buf[3] = 4
    doAssert countChanged(toOpenArray(buf, 2, 4), snap) == 0,
      "detector reported a phantom change"

  runTimed "guarded tail raises on a corrupted tail element":
    var g = initGuardedTail(2, 2, 0xDEAD'u16)
    let p = g.data.asUnchecked()
    p[2] = 0x1234'u16
    doAssertRaises(AssertionDefect):
      assertTailUntouched(g)

  runTimed "guarded tail edge shapes":
    block:
      # Empty tail, nothing to protect, a used-region write passes.
      var g = initGuardedTail(3, 0, 0xDEAD'u16)
      g.data[2] = 99
      assertTailUntouched(g)
    block:
      # Zero used, the whole buffer is tail, any write raises.
      var g = initGuardedTail(0, 1, 0x0BB7'u16)
      doAssertRaises(AssertionDefect):
        g.data[0] = 0
        assertTailUntouched(g)
    block:
      # Single-element tail, the used region stays writable.
      var g = initGuardedTail(1, 1, 0xF00D'u16)
      g.data[0] = 7
      assertTailUntouched(g)

  runTimed "shape-budget ceilings":
    checkShapeBudget(t = NaivePrefillMaxT, dk = NaiveMaxDk)
    doAssert NaivePrefillMaxT == 256, "prefill ceiling is not the documented default"
    doAssert NaiveMaxDk == 128, "head-dimension ceiling is not the documented default"

  runTimed "shape-budget and range guards fire":
    doAssertRaises(AssertionDefect):
      checkShapeBudget(t = NaivePrefillMaxT + 1, dk = NaiveMaxDk)
    doAssertRaises(AssertionDefect):
      checkShapeBudget(t = NaivePrefillMaxT, dk = NaiveMaxDk + 1)
    doAssertRaises(AssertionDefect):
      discard bitExactCount(@[0x3C00'u16], @[0x3C00'u16, 0x4200'u16])
    var r = initNaiveRng(5'u64)
    doAssertRaises(AssertionDefect):
      discard r.nextF32(1.0'f32, 1.0'f32)
    doAssertRaises(AssertionDefect):
      discard r.nextInt(3, 3)
    # nextInt rejects a span above the 32-bit width of its sampled
    # high bits, the upper part of such a range is unreachable.
    doAssertRaises(AssertionDefect):
      discard r.nextInt(0, 4_294_967_297)

  runTimed "random tensor init and dtype roundtrip":
    var r1 = initNaiveRng(11'u64)
    var m = randomMat(r1, 8, 16, -1.5'f32, 2.5'f32)
    doAssert m.rows == 8 and m.cols == 16 and m.data.len == 128
    for v in m.data:
      doAssert v >= -1.5'f32 and v < 2.5'f32, "out-of-range sample " & $v
    var r2 = initNaiveRng(11'u64)
    doAssert randomMat(r2, 8, 16, -1.5'f32, 2.5'f32).data == m.data,
      "same seed produced different tensor inputs"
    var r3 = initNaiveRng(12'u64)
    doAssert randomMat(r3, 8, 16, -1.5'f32, 2.5'f32).data != m.data,
      "distinct seeds produced identical tensor inputs"
    var r4 = initNaiveRng(13'u64)
    let cube = randomCube(r4, 2, 3, 4, -1.0'f32, 1.0'f32)
    doAssert cube.planes == 2 and cube.rows == 3 and cube.cols == 4 and
      cube.data.len == 24
    doAssert narrowBf16(cube).data.len == 24

    # Hand-computed round-to-nearest-even truth, bf16 (ulp 2⁻⁷ on [1, 2)):
    # 1+2⁻⁹ sits below half, 1+2⁻⁸ sits at the exact tie, both round down
    # to even, 1+2⁻⁸+2⁻¹⁶ sits just above half and rounds up to 1+2⁻⁷.
    let bfIn = NaiveMat[float32](rows: 2, cols: 2, data: @[
      1.001953125'f32, 1.00390625'f32,
      1.0039215087890625'f32, 1.0078125'f32])
    let bfOut = widenBf16(narrowBf16(bfIn))
    doAssert bfOut.data[0] == 1.0'f32, "bf16 RNE down case wrong"
    doAssert bfOut.data[1] == 1.0'f32, "bf16 tie-to-even case wrong"
    doAssert bfOut.data[2] == 1.0078125'f32, "bf16 RNE up case wrong"
    doAssert bfOut.data[3] == 1.0078125'f32, "bf16 exact grid point wrong"

    # Hand-computed round-to-nearest-even truth, fp16 (ulp 2⁻¹⁰ on [1, 2)):
    # 1+2⁻¹¹ sits at the exact tie and rounds down to even, 1+2⁻¹¹+2⁻²³
    # sits one f32 ulp above the tie and rounds up to 1+2⁻¹⁰.
    let hIn = NaiveMat[float32](rows: 1, cols: 2, data: @[
      1.00048828125'f32, 1.00048840045928955078125'f32])
    let hOut = widenF16(narrowF16(hIn))
    doAssert hOut.data[0] == 1.0'f32, "fp16 tie-to-even case wrong"
    doAssert hOut.data[1] == 1.0009765625'f32, "fp16 RNE up case wrong"

  runTimed "tensor element access and cube dtype roundtrip":
    var r = initNaiveRng(21'u64)
    var m = randomMat(r, 3, 4, -1.0'f32, 1.0'f32)
    m.at(1, 2) = 0.5'f32
    doAssert m.at(1, 2) == 0.5'f32 and m.data[1 * 4 + 2] == 0.5'f32,
      "mat element access disagrees with row-major layout"
    var c = randomCube(r, 2, 2, 2, -1.0'f32, 1.0'f32)
    c.at(1, 0, 1) = 0.25'f32
    doAssert c.at(1, 0, 1) == 0.25'f32, "cube element access wrong"
    doAssert c.data[(1 * 2 + 0) * 2 + 1] == 0.25'f32,
      "cube element access disagrees with plane-major layout"
    # Hand-computed round-to-nearest-even truth on a cube, bf16
    # (the tie and up cases of the mat block above):
    var bfIn = NaiveCube[float32](planes: 1, rows: 1, cols: 2, data: @[1.00390625'f32, 1.0039215087890625'f32])
    let bfOut = widenBf16(narrowBf16(bfIn))
    doAssert bfOut.data[0] == 1.0'f32 and bfOut.data[1] == 1.0078125'f32,
      "cube bf16 roundtrip wrong"
    # Hand-computed round-to-nearest-even truth on a cube, fp16
    # (the tie and up cases of the mat block above):
    var hIn = NaiveCube[float32](planes: 1, rows: 1, cols: 2, data: @[1.00048828125'f32, 1.00048840045928955078125'f32])
    let hOut = widenF16(narrowF16(hIn))
    doAssert hOut.data[0] == 1.0'f32 and hOut.data[1] == 1.0009765625'f32,
      "cube fp16 roundtrip wrong"

  runTimed "f32ToBf16 non-finite passthrough":
    # NaN keeps a NaN pattern, payload not preserved, the round-to-nearest
    # increment must not carry across the Inf/NaN boundary.
    doAssert (f32ToBf16(cast[float32](0x7FC00001'u32)) and 0x7FC0'u16) == 0x7FC0'u16,
      "quiet NaN did not stay a NaN pattern"
    doAssert f32ToBf16(cast[float32](0x7F800001'u32)) == 0x7FC0'u16,
      "small-payload NaN rounded across the Inf/NaN boundary to Inf"
    doAssert f32ToBf16(cast[float32](0xFF800001'u32)) == 0xFFC0'u16,
      "negative small-payload NaN lost its sign and NaN pattern"
    doAssert f32ToBf16(cast[float32](0x7F800000'u32)) == 0x7F80'u16,
      "Inf did not pass through"

  runTimed "naive_metrics untouched region after a full write":
    # A reference writing every used element must leave the guarded
    # tail bit-identical end to end.
    var g = initGuardedTail(64, 16, 0xBEEF'u16)
    for i in 0 ..< g.used:
      g.data[i] = uint16(i * 3)
    assertTailUntouched(g)

main()
