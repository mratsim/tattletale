# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/naive/t_naive_grouped_mm.nim
##
## grouped_mm naive reference self-check, the fp32 accumulation and the one
## El store round judged against an independent fp64 reference
##
##   out[r, i] = El(sum_h a[r, h] · w[e, i, h])    group e owns row r
##
## | check            | contract                                                                                       |
## | ---------------- | ---------------------------------------------------------------------------------------------- |
## | offsets          | inclusive per-expert end offsets, non-decreasing, the last entry closes the row range          |
## | empty group      | a repeated offset contributes no rows, the groups either side keep their rows                  |
## | single-element   | a one-row group computes one dot product, closed-form checked on exact small values            |
## | fp64 reference   | exact widenings, fp64 sequential accumulation, the 16-bit products exact in fp64               |
## | row independence | a within-group row permutation leaves each row's output unchanged, the row follows its content |
##
## Band model, stated before measurement, u32 = 2⁻²⁴ fp32, u64 = 2⁻⁵³ fp64,
## u_fam = 2⁻⁸ bf16 / 2⁻¹¹ fp16, M = the row's largest |a_h·w_h| (input-derived):
##
## | term       | bar                 | covers                                                                         |
## | ---------- | ------------------- | ------------------------------------------------------------------------------ |
## | output     | u_fam·abs(y) + 2⁻²⁵ | the one El store round, the subnormal floor at tiny outputs                    |
## | fp32 sum   | 3·u32·abs(y)        | the fp32 store of the accumulator against the exact sum                        |
## | fp32 chain | H·(H + 1)·u32·M     | per-product rounding and the (H-1)-term forward error, each partial within H·M |
##
## | fp32 vs fp64 | bar                            | purpose                                              |
## | ------------ | ------------------------------ | ---------------------------------------------------- |
## | sum check    | 2·u32·abs(s64) + (H-1)·H·u32·M | the fp32 accumulation sits inside its rounding model |
##
## | shape | fam  | E | H  | I  | P  | group sizes       |
## | ----- | ---- | --- | --- | --- | --- | ----------------- |
## | main  | bf16 | 8 | 64 | 48 | 40 | 3 0 7 1 12 5 0 12 |
## | alt   | fp16 | 4 | 32 | 24 | 10 | 2 0 1 7           |
##
## - the measured divergence justifies the model, never sets the bar
## - adjudicated with fresh seeded xorshift64 inputs

import std/[strformat, math, assertions]
import std/sequtils
import naive_rng, naive_tensors, naive_budget, naive_grouped_mm

# ─── Independent fp64 reference, spelled in the test, not the ref module ──

proc gmmRefF64(fam: GmmFamily; a: NaiveMat[uint16]; w: NaiveCube[uint16];
    offs: seq[int32]): NaiveMat[float64] =
  ## Exact-value fp64 reference, fp64 widening, fp64 products exact at 22-bit mantissas
  ## fp64 sequential accumulation, unrounded.
  result = NaiveMat[float64](rows: a.rows, cols: w.rows)
  result.data = newSeq[float64](a.rows * w.rows)
  for e in 0 ..< w.planes:
    let lo = (if e == 0: 0 else: offs[e - 1].int)
    let hi = offs[e].int
    for r in lo ..< hi:
      for i in 0 ..< w.rows:
        var acc = 0.0'f64
        for h in 0 ..< a.cols:
          acc += gmmWiden(fam, a.data[r * a.cols + h]).float64 *
            gmmWiden(fam, w.data[(e * w.rows + i) * w.cols + h]).float64
        result.data[r * w.rows + i] = acc

proc rowMaxProduct(a: NaiveMat[uint16]; w: NaiveCube[uint16]; e, r: int): float64 =
  ## Returns the row's largest |a_h·w_h| over the widened exact products, the M
  ## of the band model, derived from the inputs before any output is seen.
  result = 0.0'f64
  for i in 0 ..< w.rows:
    for h in 0 ..< a.cols:
      let m = abs(gmmWiden(gmmBf16, a.data[r * a.cols + h]).float64 *
        gmmWiden(gmmBf16, w.data[(e * w.rows + i) * w.cols + h]).float64)
      result = max(result, m)

const
  U32 = 5.9604644775390625e-8'f64      # 2^-24, the fp32 unit roundoff
  FloorSub = 2.9802322387695312e-8'f64 # 2^-25, half the fp16 subnormal ulp

proc runBandCase(fam: GmmFamily; rng: var NaiveRng; a: NaiveMat[uint16];
    w: NaiveCube[uint16]; offs: seq[int32]; label: string) =
  ## One randomized (family, group layout) case, every output element
  ## judged under the stated bar, worst bar usage printed.
  let got = naiveGroupedMm(fam, a, w, offs)
  let ref64 = gmmRefF64(fam, a, w, offs)
  let uFam: float64 = (if fam == gmmBf16: 3.90625e-3 else: 4.8828125e-4)
  var worstUse = 0.0'f64
  var exact = 0
  for e in 0 ..< w.planes:
    let lo = (if e == 0: 0 else: offs[e - 1].int)
    let hi = offs[e].int
    for r in lo ..< hi:
      let m = rowMaxProduct(a, w, e, r)
      for i in 0 ..< w.rows:
        let y = gmmWiden(fam, got.data[r * w.rows + i]).float64
        let yExact = ref64.data[r * w.rows + i]
        let bar = uFam * abs(y) + 3.0 * U32 * abs(y) +
          float64(a.cols) * float64(a.cols + 1) * U32 * m + FloorSub
        let diff = abs(y - yExact)
        doAssert diff <= bar,
          &"output outside the bar at (r {r}, i {i}): {diff:.3e} > {bar:.3e}"
        worstUse = max(worstUse, diff / bar)
        if got.data[r * w.rows + i] == gmmRoundEl(fam, yExact.float32):
          inc exact
  echo &"[{label} {gmmName(fam)}] worst bar usage {worstUse:.3f}, " &
    &"bit-exact vs El(ref64) {exact}/{a.rows * w.rows}"

proc fillRandom(fam: GmmFamily; rng: var NaiveRng; a: var NaiveMat[uint16];
    w: var NaiveCube[uint16]) =
  ## Seeded family-dtype inputs over [-1, 1].
  for i in 0 ..< a.data.len:
    a.data[i] = gmmRoundEl(fam, rng.nextF32(-1.0'f32, 1.0'f32))
  for i in 0 ..< w.data.len:
    w.data[i] = gmmRoundEl(fam, rng.nextF32(-1.0'f32, 1.0'f32))

proc main =
  # Offsets contract, the inclusive end offsets partition the rows in order
  runTimed "offsets contract, groupOfRow walks the inclusive end offsets":
    let offs = [3'i32, 3, 10, 11, 23, 28, 28, 40].toSeq
    doAssert groupOfRow(offs, 0) == 0, "the first row owns group 0"
    doAssert groupOfRow(offs, 2) == 0, "row 2 still inside group 0"
    doAssert groupOfRow(offs, 3) == 2, "the empty group 1 owns no rows"
    doAssert groupOfRow(offs, 10) == 3, "row 10 opens the single-row group"
    doAssert groupOfRow(offs, 39) == 7, "the last row sits in the last group"

  # Closed-form small case, hand-computed, bf16-exact values.
  runTimed "closed-form small case, exact small values, bit-exact outputs":
    var a = NaiveMat[uint16](rows: 2, cols: 2)
    a.data = @[f32ToBf16(1.0'f32), f32ToBf16(2.0'f32),
               f32ToBf16(0.5'f32), f32ToBf16(-1.0'f32)]
    var w = NaiveCube[uint16](planes: 2, rows: 1, cols: 2)
    w.data = @[f32ToBf16(0.5'f32), f32ToBf16(0.25'f32),
               f32ToBf16(1.0'f32), f32ToBf16(0.5'f32)]
    let offs = @[1'i32, 2] # group 0 single row, group 1 single row
    let got = naiveGroupedMm(gmmBf16, a, w, offs)
    doAssert got.data[0] == f32ToBf16(1.0'f32),
      "1.0·0.5 + 2.0·0.25 must round to 1.0"
    doAssert got.data[1] == f32ToBf16(0.0'f32),
      "0.5·1.0 + (-1.0)·0.5 must round to 0.0"

  # Empty group and single-element group through the full call.
  runTimed "offsets boundaries, the empty group skips rows, the single-element group computes one dot":
    var a = NaiveMat[uint16](rows: 10, cols: 4)
    var w = NaiveCube[uint16](planes: 4, rows: 3, cols: 4)
    var rng = initNaiveRng(0xC04D0531'u64)
    a.data = newSeq[uint16](a.rows * a.cols)
    w.data = newSeq[uint16](w.planes * w.rows * w.cols)
    fillRandom(gmmBf16, rng, a, w)
    let offs = @[2'i32, 2, 3, 10] # empty group 1, single-element group 2
    let got = naiveGroupedMm(gmmBf16, a, w, offs)
    let solo = naiveGroupedMm(gmmBf16,
      NaiveMat[uint16](rows: 1, cols: 4, data: a.data[8 .. 11]),
      NaiveCube[uint16](planes: 1, rows: 3, cols: 4, data: w.data[24 ..< 36]),
      @[1'i32])
    for i in 0 ..< 3:
      doAssert got.data[2 * 3 + i] == solo.data[i],
        "the single-element group's row must match a solo one-row call"

  # Row independence, a within-group permutation moves each output row
  # with its input row
  runTimed "row independence, a within-group permutation follows the rows":
    var a = NaiveMat[uint16](rows: 6, cols: 8)
    var w = NaiveCube[uint16](planes: 2, rows: 5, cols: 8)
    var rng = initNaiveRng(0xC04D0532'u64)
    a.data = newSeq[uint16](a.rows * a.cols)
    w.data = newSeq[uint16](w.planes * w.rows * w.cols)
    fillRandom(gmmBf16, rng, a, w)
    let offs = @[4'i32, 6]
    let base = naiveGroupedMm(gmmBf16, a, w, offs)
    var perm = newSeq[uint16](a.data.len)
    # group 0 rows reversed in place, group 1 rows kept
    for local in 0 ..< 4:
      for h in 0 ..< 8:
        perm[(3 - local) * 8 + h] = a.data[local * 8 + h]
    for p in 4 ..< 6:
      for h in 0 ..< 8:
        perm[p * 8 + h] = a.data[p * 8 + h]
    let permuted = naiveGroupedMm(gmmBf16,
      NaiveMat[uint16](rows: 6, cols: 8, data: perm), w, offs)
    # group 0 rows were reversed, each output row must equal the base
    # output of the row whose content it now carries
    for local in 0 ..< 4:
      for i in 0 ..< 5:
        doAssert permuted.data[(3 - local) * 5 + i] == base.data[local * 5 + i],
          "a permuted row's output must follow the row's content"
    for p in 4 ..< 6:
      for i in 0 ..< 5:
        doAssert permuted.data[p * 5 + i] == base.data[p * 5 + i],
          "an untouched row's output must stay in place"

  # Randomized band cases against the fp64 reference.
  runTimed "randomized band cases, bf16 main layout and fp16 alt layout":
    block bf16Main:
      var rng = initNaiveRng(0xC04D0533'u64)
      for caseId in 0 ..< 3:
        var a = NaiveMat[uint16](rows: 40, cols: 64)
        var w = NaiveCube[uint16](planes: 8, rows: 48, cols: 64)
        a.data = newSeq[uint16](a.rows * a.cols)
        w.data = newSeq[uint16](w.planes * w.rows * w.cols)
        fillRandom(gmmBf16, rng, a, w)
        let offs = @[3'i32, 3, 10, 11, 23, 28, 28, 40].toSeq
        runBandCase(gmmBf16, rng, a, w, offs, &"band main case {caseId}")
    block fp16Alt:
      var rng = initNaiveRng(0xC04D0534'u64)
      for caseId in 0 ..< 3:
        var a = NaiveMat[uint16](rows: 10, cols: 32)
        var w = NaiveCube[uint16](planes: 4, rows: 24, cols: 32)
        a.data = newSeq[uint16](a.rows * a.cols)
        w.data = newSeq[uint16](w.planes * w.rows * w.cols)
        fillRandom(gmmF16, rng, a, w)
        let offs = @[2'i32, 2, 3, 10].toSeq
        runBandCase(gmmF16, rng, a, w, offs, &"band alt case {caseId}")

  # fp32-vs-fp64 accumulation check at the fp32 level, the unrounded sums.
  runTimed "fp32 accumulation check, the fp32 sums sit inside the rounding model":
    var rng = initNaiveRng(0xC04D0535'u64)
    var a = NaiveMat[uint16](rows: 40, cols: 64)
    var w = NaiveCube[uint16](planes: 8, rows: 48, cols: 64)
    a.data = newSeq[uint16](a.rows * a.cols)
    w.data = newSeq[uint16](w.planes * w.rows * w.cols)
    fillRandom(gmmBf16, rng, a, w)
    let offs = @[3'i32, 3, 10, 11, 23, 28, 28, 40].toSeq
    let s32 = naiveGroupedMmSums(gmmBf16, a, w, offs)
    let s64 = gmmRefF64(gmmBf16, a, w, offs)
    var worstUse = 0.0'f64
    for e in 0 ..< w.planes:
      let lo = (if e == 0: 0 else: offs[e - 1].int)
      let hi = offs[e].int
      for r in lo ..< hi:
        let m = rowMaxProduct(a, w, e, r)
        for i in 0 ..< w.rows:
          let s = s32.data[r * w.rows + i].float64
          let sExact = s64.data[r * w.rows + i]
          let bar = 2.0 * U32 * abs(sExact) +
            float64(a.cols - 1) * float64(a.cols) * U32 * m
          let diff = abs(s - sExact)
          doAssert diff <= bar,
            &"fp32 sum outside its bar at (r {r}, i {i}): {diff:.3e} > {bar:.3e}"
          worstUse = max(worstUse, diff / bar)
    echo &"[fp32 check] worst bar usage {worstUse:.3f}"

  echo "NAIVE GROUPED_MM VERDICT: offsets contract, boundary cases, band and check all inside the stated models"

main()
