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
## - same-seed stream reproducibility and cross-seed divergence
## - worstSeqDiff, bitExactCount on known-good and known-bad pairs
## - the guarded-tail untouched-memory check on a tiny case
##
## - the shape-budget ceilings and the random-tensor dtype
##   narrowing and widening against hand-computed round-to-nearest-even values
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

  runTimed "worstSeqDiff known-good and known-bad pairs":
    let got = @[1.0'f32, 2.0, 4.0]
    let bad = @[1.0'f32, 7.0, 4.0]
    doAssert worstSeqDiff(got, got) == 0.0, "identical pairs differ"
    doAssert worstSeqDiff(got, bad) == 5.0, "worst difference mislocated"
    doAssert worstSeqDiff(bad, got) == 5.0, "worst difference not symmetric"
    doAssertRaises(ValueError):
      discard worstSeqDiff(@[0.0'f32, 1.0'f32], @[0.0'f32, NaN])

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

  runTimed "shape-budget ceilings":
    checkShapeBudget(t = NaivePrefillMaxT, dk = NaiveMaxDk)
    doAssert NaivePrefillMaxT == 256, "prefill ceiling is not the documented default"
    doAssert NaiveMaxDk == 128, "head-dimension ceiling is not the documented default"

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

  runTimed "naive_metrics untouched region after a full write":
    # A reference writing every used element must leave the guarded
    # tail bit-identical end to end.
    var g = initGuardedTail(64, 16, 0xBEEF'u16)
    for i in 0 ..< g.used:
      g.data[i] = uint16(i * 3)
    assertTailUntouched(g)

main()
echo "t_naive_harness: all checks passed"
