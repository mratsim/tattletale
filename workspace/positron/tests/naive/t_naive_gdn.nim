# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run command, from the repo root:
## - nim test_positron_naive
## - nim c -r -d:release --warnings:off --outdir:build/tests --nimcache:nimcache/tests workspace/positron/tests/naive/t_naive_gdn.nim
##
## GDN naive reference suite, one flat main over naive_gdn:
##
## Checks, all closed-form or model-bar assertions:
## - the g = 0, g -> -inf and beta = 0 anchors, fp32 and fp64 runs, exact
## - the one-step equivalence at T = 1 (decode step vs walk), exact
## - the DIAGNOSTIC chunked-vs-per-token pair at the rounding-model bar,
##   plus the budget-ceiling pass at T = 256, Dk = 128
##
## Case contract:
## - every recurring (T > 1) case starts from a non-zero initial state
## - the final state and the outputs both take the comparison
##
##   anchors: g=0 → plain delta rule, g->-inf → replace rule, β=0 → pure decay
##
## Shapes stay inside the NaivePrefillMaxT / NaiveMaxDk ceilings.

import std/[assertions, math, strutils]
import naive_rng, naive_tensors, naive_metrics, naive_budget
import naive_gdn, naive_deltarule

# ─── Closed-form anchors ─────────────────────────────────────────────

proc g0AnchorCase[F: float32|float64](seed: uint64) =
  ## g = 0 anchor, the gated recurrence reduced to the plain delta rule:
  ##
  ## Contract:
  ## - the gated walk and an independently spelled decay-free walk must
  ##   agree exactly, both runs fed from the same widened inputs
  const B = 1
  const Hv = 4
  const Hk = 2
  const hkRatio = 2
  const Dk = 16
  const Dv = 16
  const T = 8
  checkShapeBudget(T, Dk)
  var rng = initNaiveRng(seed)
  let q32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let k32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let v32 = randomCube(rng, B * Hv, T, Dv, -0.5'f32, 0.5'f32)
  let beta32 = randomMat(rng, B * Hv, T, 0.1'f32, 1.0'f32)
  let s032 = randomCube(rng, B * Hv, Dv, Dk, -0.25'f32, 0.25'f32)  # non-zero initial state
  var g32 = randomMat(rng, B * Hv, T, -1.0'f32, -0.01'f32)
  g32.fillMat(0.0'f32)

  var sGated = castCube[F](s032)
  var yGated = zerosCube[F](B * Hv, T, Dv)
  gdnPrefillPerToken(sGated, yGated, castCube[F](q32), castCube[F](k32),
    castCube[F](v32), castMat[F](beta32), castMat[F](g32), Hv, Hk, hkRatio)

  var sPlain = castCube[F](s032)
  var yPlain = zerosCube[F](B * Hv, T, Dv)
  deltaRuleWalk(sPlain, yPlain, castCube[F](q32), castCube[F](k32),
    castCube[F](v32), castMat[F](beta32), Hv, Hk, hkRatio)

  when F is float32:
    let worstY = worstSeqDiff(yGated.data, yPlain.data)
    let worstS = worstSeqDiff(sGated.data, sPlain.data)
  else:
    let worstY = worstDiffF64(yGated.data, yPlain.data)
    let worstS = worstDiffF64(sGated.data, sPlain.data)
  # Exact bar, g = 0:
  # - the decay factor is exp(0) = 1, 1.0*x the identity in IEEE arithmetic
  # - the gated walk therefore performs the decay-free walk's operations
  #   bit for bit, any nonzero difference being a real spelling divergence
  doAssert worstY == 0.0,
    "g = 0 outputs diverged from the plain delta rule: worst " & $worstY &
    ", expected exactly 0.0"
  doAssert worstS == 0.0,
    "g = 0 final state diverged from the plain delta rule: worst " & $worstS &
    ", expected exactly 0.0"

proc replaceAnchorCase[F: float32|float64](seed: uint64) =
  ## g -> -inf anchor, every step reduced to the replace rule, exp
  ## underflowing to exact zero:
  ##
  ## Contract:
  ## - the final state is beta*(k (x) v) of the last token, whatever
  ##   state preceded it
  const B = 1
  const Hv = 4
  const Hk = 2
  const hkRatio = 2
  const Dk = 16
  const Dv = 16
  const T = 3
  checkShapeBudget(T, Dk)
  var rng = initNaiveRng(seed)
  let q32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let k32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let v32 = randomCube(rng, B * Hv, T, Dv, -0.5'f32, 0.5'f32)
  let beta32 = randomMat(rng, B * Hv, T, 0.1'f32, 1.0'f32)
  let s032 = randomCube(rng, B * Hv, Dv, Dk, -0.25'f32, 0.25'f32)  # non-zero initial state
  var g32 = randomMat(rng, B * Hv, T, -800.0'f32, -750.0'f32)
  doAssert exp(g32.data[0]) == 0.0'f32, "test g range does not underflow to zero"

  var sGated = castCube[F](s032)
  var yGated = zerosCube[F](B * Hv, T, Dv)
  gdnPrefillPerToken(sGated, yGated, castCube[F](q32), castCube[F](k32),
    castCube[F](v32), castMat[F](beta32), castMat[F](g32), Hv, Hk, hkRatio)

  # Independently spelled replace rule at the last token, the state entry
  # being (beta*v[row])*k[dk], formed with the same two products the gated
  # path forms, beta*(v - 0) = beta*v then k*delta, so the comparison is exact
  let bh = 0
  let hk = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
  let betaLast = castMat[F](beta32).data[bh * T + T - 1]
  var wantS = newSeq[F](Dv * Dk)
  for row in 0 ..< Dv:
    let bv = betaLast * castCube[F](v32).data[(bh * T + T - 1) * Dv + row]
    for dkc in 0 ..< Dk:
      wantS[row * Dk + dkc] = bv * castCube[F](k32).data[(hk * T + T - 1) * Dk + dkc]
  var plane = newSeq[F](Dv * Dk)
  for row in 0 ..< Dv:
    for dkc in 0 ..< Dk:
      plane[row * Dk + dkc] = sGated.data[(bh * Dv + row) * Dk + dkc]
  when F is float32:
    let worstS = worstSeqDiff(plane, wantS)
  else:
    let worstS = worstDiffF64(plane, wantS)
  doAssert worstS == 0.0,
    "g -> -inf final state diverged from the replace rule: worst " & $worstS &
    ", expected exactly 0.0"

  # Output cross-read bar, stated from the rounding model:
  # - the gated spelling sums the Dk products in ascending key-channel order,
  #   the replace-rule spelling factoring the beta*v[row] scale out,
  #   a pure reassociation of one Dk-length sum
  # - at the F unit roundoff that is 16*2^-24 ~ 1e-6 (fp32) plus
  #   16*2^-53 ~ 2e-15 (fp64) of the output magnitude, the bars below
  #   carrying three-plus orders of headroom
  var qkAcc: F = 0
  for dkc in 0 ..< Dk:
    qkAcc += castCube[F](k32).data[(hk * T + T - 1) * Dk + dkc] *
      (castCube[F](q32).data[(hk * T + T - 1) * Dk + dkc] / sqrt(F(Dk)))
  var wantY = newSeq[F](Dv)
  for row in 0 ..< Dv:
    let bv = betaLast * castCube[F](v32).data[(bh * T + T - 1) * Dv + row]
    wantY[row] = bv * qkAcc
  var yLast = newSeq[F](Dv)
  for row in 0 ..< Dv:
    yLast[row] = yGated.data[(bh * T + T - 1) * Dv + row]
  when F is float32:
    let worstY = worstSeqDiff(yLast, wantY)
    var yAbs: float64 = 0
    for value in yLast:
      yAbs = max(yAbs, abs(float64(value)))
    doAssert worstY <= 1e-4 * max(1.0, yAbs),
      "g -> -inf output reassociation exceeds the fp32 bar: worst " & $worstY
  else:
    let worstY = worstDiffF64(yLast, wantY)
    doAssert worstY <= 1e-12 * max(1.0, maxAbsF64(yLast)),
      "g -> -inf output reassociation exceeds the fp64 bar: worst " & $worstY

proc beta0AnchorCase[F: float32|float64](seed: uint64) =
  ## beta = 0 anchor, every rank-1 update removed:
  ##
  ## Contract:
  ## - the state is the pure decay chain S_t = gamma_t * S_(t-1), the outputs
  ##   reading the decayed initial state against each q
  const B = 1
  const Hv = 4
  const Hk = 2
  const hkRatio = 2
  const Dk = 16
  const Dv = 16
  const T = 8
  checkShapeBudget(T, Dk)
  var rng = initNaiveRng(seed)
  let q32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let k32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let v32 = randomCube(rng, B * Hv, T, Dv, -0.5'f32, 0.5'f32)
  var beta32 = randomMat(rng, B * Hv, T, 0.1'f32, 1.0'f32)
  beta32.fillMat(0.0'f32)
  let s032 = randomCube(rng, B * Hv, Dv, Dk, -0.25'f32, 0.25'f32)  # non-zero initial state
  let g32 = randomMat(rng, B * Hv, T, -1.0'f32, -0.01'f32)

  var sGated = castCube[F](s032)
  var yGated = zerosCube[F](B * Hv, T, Dv)
  gdnPrefillPerToken(sGated, yGated, castCube[F](q32), castCube[F](k32),
    castCube[F](v32), castMat[F](beta32), castMat[F](g32), Hv, Hk, hkRatio)

  # Independently spelled pure-decay walk, the anchor contract itself.
  var sDecay = castCube[F](s032)
  var yDecay = zerosCube[F](B * Hv, T, Dv)
  for bh in 0 ..< B * Hv:
    let hkHead = (bh mod Hv) div hkRatio + (bh div Hv) * Hk
    for t in 0 ..< T:
      let gamma = exp(castMat[F](g32).data[bh * T + t])
      for row in 0 ..< Dv:
        for dkc in 0 ..< Dk:
          sDecay.at(bh, row, dkc) = gamma * sDecay.at(bh, row, dkc)
        var acc: F = 0
        for dkc in 0 ..< Dk:
          acc += sDecay.at(bh, row, dkc) *
            (castCube[F](q32).data[(hkHead * T + t) * Dk + dkc] / sqrt(F(Dk)))
        yDecay.at(bh, t, row) = acc

  when F is float32:
    let worstY = worstSeqDiff(yGated.data, yDecay.data)
    let worstS = worstSeqDiff(sGated.data, sDecay.data)
  else:
    let worstY = worstDiffF64(yGated.data, yDecay.data)
    let worstS = worstDiffF64(sGated.data, sDecay.data)
  # Exact bar, beta = 0:
  # - the update delta is beta*(v - kv) = exact zero, state + k*0 = state
  #   in IEEE arithmetic
  # - the gated walk therefore performs the pure-decay walk's operations
  #   bit for bit
  doAssert worstY == 0.0,
    "beta = 0 outputs diverged from the pure decay chain: worst " & $worstY &
    ", expected exactly 0.0"
  doAssert worstS == 0.0,
    "beta = 0 final state diverged from the pure decay chain: worst " & $worstS &
    ", expected exactly 0.0"

# ─── One-step equivalence ────────────────────────────────────────────

proc oneStepEquivalenceCase[F: float32|float64](seed: uint64) =
  ## One decode step at T = 1 against the per-token walk's single step,
  ## both spellings from the same seeded inputs:
  ##
  ## Contract:
  ## - the step and the walk share no code, so a divergence in either
  ##   spelling is detected
  ##
  ## Exact bar, bitwise-identical outputs at T = 1:
  ##
  ## - both spellings apply the same operations in the same order
  ##   (read → decay → kv read → update → y), the walk restates the step's arithmetic
  ## - IEEE fp32/fp64 arithmetic is deterministic, so identical inputs
  ##   give identical bits
  const B = 1
  const Hv = 4
  const Hk = 2
  const hkRatio = 2
  const Dk = 16
  const Dv = 16
  const T = 1
  checkShapeBudget(T, Dk)
  var rng = initNaiveRng(seed)
  let q32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let k32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let v32 = randomCube(rng, B * Hv, T, Dv, -0.5'f32, 0.5'f32)
  let beta32 = randomMat(rng, B * Hv, T, 0.1'f32, 1.0'f32)
  let g32 = randomMat(rng, B * Hv, T, -1.0'f32, -0.01'f32)
  let s032 = randomCube(rng, B * Hv, Dv, Dk, -0.25'f32, 0.25'f32)  # non-zero initial state

  # The walk, one step from the seeded initial state.
  var sWalk = castCube[F](s032)
  var yWalk = zerosCube[F](B * Hv, T, Dv)
  gdnPrefillPerToken(sWalk, yWalk, castCube[F](q32), castCube[F](k32),
    castCube[F](v32), castMat[F](beta32), castMat[F](g32), Hv, Hk, hkRatio)

  # The step's inputs, token 0 of the same seeded tensors in the step
  # signature's per-token forms.
  var q1 = NaiveMat[F](rows: B * Hk, cols: Dk)
  q1.data = newSeq[F](B * Hk * Dk)
  var k1 = NaiveMat[F](rows: B * Hk, cols: Dk)
  k1.data = newSeq[F](B * Hk * Dk)
  var v1 = NaiveMat[F](rows: B * Hv, cols: Dv)
  v1.data = newSeq[F](B * Hv * Dv)
  var beta1 = newSeq[F](B * Hv)
  var g1 = newSeq[F](B * Hv)
  for hk in 0 ..< B * Hk:
    for dkc in 0 ..< Dk:
      q1.data[hk * Dk + dkc] = castCube[F](q32).data[(hk * T + 0) * Dk + dkc]
      k1.data[hk * Dk + dkc] = castCube[F](k32).data[(hk * T + 0) * Dk + dkc]
  for bh in 0 ..< B * Hv:
    for row in 0 ..< Dv:
      v1.data[bh * Dv + row] = castCube[F](v32).data[(bh * T + 0) * Dv + row]
    beta1[bh] = castMat[F](beta32).data[bh * T + 0]
    g1[bh] = castMat[F](g32).data[bh * T + 0]

  var sStep = castCube[F](s032)
  var yStep = NaiveMat[F](rows: B * Hv, cols: Dv)
  yStep.data = newSeq[F](B * Hv * Dv)
  gdnDecodeStep(sStep, yStep, q1, k1, v1, beta1, g1, Hv, Hk, hkRatio)

  when F is float32:
    let worstY = worstSeqDiff(yStep.data, yWalk.data)
    let worstS = worstSeqDiff(sStep.data, sWalk.data)
  else:
    let worstY = worstDiffF64(yStep.data, yWalk.data)
    let worstS = worstDiffF64(sStep.data, sWalk.data)
  doAssert worstY == 0.0,
    "decode step output diverged from the walk's step: worst " & $worstY &
    ", expected exactly 0.0"
  doAssert worstS == 0.0,
    "decode step state diverged from the walk's step: worst " & $worstS &
    ", expected exactly 0.0"

# ─── DIAGNOSTIC intra-family pair ────────────────────────────────────

proc gdnPairCase(seed: uint64; T, Hv, Hk, hkRatio, Dk, Dv, chunkLen: int;
    tag: string) =
  ## Compares the two independently spelled GDN forms on random inputs
  ## with a non-zero initial state:
  ##
  ## Contract:
  ## - the assert bar comes from the rounding model stated in the body
  ## - the printed worst differences and ratio are diagnostic output
  const B = 1
  checkShapeBudget(T, Dk)
  var rng = initNaiveRng(seed)
  let q32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let k32 = randomCube(rng, B * Hk, T, Dk, -0.5'f32, 0.5'f32)
  let v32 = randomCube(rng, B * Hv, T, Dv, -0.5'f32, 0.5'f32)
  let beta32 = randomMat(rng, B * Hv, T, 0.1'f32, 1.0'f32)
  let g32 = randomMat(rng, B * Hv, T, -1.0'f32, -0.01'f32)
  let s032 = randomCube(rng, B * Hv, Dv, Dk, -0.25'f32, 0.25'f32)  # non-zero initial state

  let qw = q32.widenF64()
  let kw = k32.widenF64()
  let vw = v32.widenF64()
  let betaw = beta32.widenF64()
  let gw = g32.widenF64()
  let s0w = s032.widenF64()

  var sPer = s0w.copyOf()
  var yPer = zerosCube[float64](B * Hv, T, Dv)
  gdnPrefillPerToken(sPer, yPer, qw, kw, vw, betaw, gw, Hv, Hk, hkRatio)
  let chunked = gdnPrefillChunked(s0w, qw, kw, vw, betaw, gw, Hv, Hk, hkRatio, chunkLen)

  let maxAbs = max(maxAbsF64(yPer.data), maxAbsF64(sPer.data))
  let worstY = worstDiffF64(yPer.data, chunked.y.data)
  let worstS = worstDiffF64(sPer.data, chunked.state.data)
  # Rounding-model bar, stated before any measurement:
  # - the spellings apply the same fp64 arithmetic reassociated differently
  # - one output or state entry accumulates at most (T + chunkLen)*Dk fp64
  #   roundings through either spelling, (256 + 64)*128 ~ 4e4 at the ceiling,
  #   drift model 4e4*2^-53 ~ 4e-12 relative to the entry magnitude
  # - the bar is 1e-8 relative, max(1, maxAbs) covering near-zero outputs,
  #   three orders of headroom above the model
  let bar = 1e-8 * max(1.0, maxAbs)
  let ratio = max(worstY, worstS) / max(1.0, maxAbs)
  echo "    [DIAGNOSTIC ", tag, "] worstY=",
    formatFloat(worstY, ffScientific, 3), " worstS=",
    formatFloat(worstS, ffScientific, 3), " maxAbs=",
    formatFloat(maxAbs, ffScientific, 3), " ratio=",
    formatFloat(ratio, ffScientific, 3)
  doAssert worstY <= bar,
    "chunked vs per-token outputs diverged beyond the rounding-model bar: " &
    "worst " & $worstY & ", bar " & $bar
  doAssert worstS <= bar,
    "chunked vs per-token final state diverged beyond the rounding-model " &
    "bar: worst " & $worstS & ", bar " & $bar

proc main =
  runTimed "harness fp64 additions: exact widening, independent copies, fp64 diffs":
    var rngA = initNaiveRng(0xFA4D'u64)
    var rngB = initNaiveRng(0xFA4D'u64)
    let mA = randomMat(rngA, 4, 5, -2.0'f32, 2.0'f32)
    let mB = randomMat(rngB, 4, 5, -2.0'f32, 2.0'f32)
    doAssert widenF64(mA).data == widenF64(mB).data,
      "same-seed fp64 widening produced different values"
    doAssert widenF64(mA).data[7] == float64(mA.data[7]),
      "fp64 widening is not the exact f32 value"
    var mCopy = mA.copyOf()
    mCopy.data[3] = 99.0'f32
    doAssert mA.data[3] != 99.0'f32, "copyOf is not independent of the original"
    doAssert worstDiffF64(@[1.0'f64, 2.0, 4.0], @[1.0'f64, 7.0, 4.0]) == 5.0,
      "worstDiffF64 hand value wrong, expected 5.0"
    doAssert worstDiffF64(@[1.0'f64, 2.0], @[1.0'f64, 2.0]) == 0.0,
      "worstDiffF64 on identical pairs is not zero"
    doAssert maxAbsF64(@[-3.5'f64, 2.0]) == 3.5, "maxAbsF64 hand value wrong"

  runTimed "anchor g=0 reduces to the plain delta rule (fp32 and fp64)":
    g0AnchorCase[float32](0x600D5EED'u64)
    g0AnchorCase[float64](0x600D5EED'u64)

  runTimed "anchor g->-inf reduces to the replace rule (fp32 and fp64)":
    replaceAnchorCase[float32](0x9E3779B9'u64)
    replaceAnchorCase[float64](0x9E3779B9'u64)

  runTimed "anchor beta=0 gives the pure decay chain (fp32 and fp64)":
    beta0AnchorCase[float32](0x5EEDC0DE'u64)
    beta0AnchorCase[float64](0x5EEDC0DE'u64)

  runTimed "one-step equivalence: decode step at T=1 vs the walk (fp32 and fp64)":
    oneStepEquivalenceCase[float32](0x7EA5EED'u64)
    oneStepEquivalenceCase[float64](0x7EA5EED'u64)

  runTimed "DIAGNOSTIC intra-family pair GDN chunked vs per-token (fp64)":
    # T = 100 with chunkLen 32 gives blocks 32, 32, 32, 4: the tail block
    # of the walk is exercised.
    gdnPairCase(0x6A11CE'u64, T = 100, Hv = 4, Hk = 2, hkRatio = 2,
      Dk = 64, Dv = 64, chunkLen = 32, tag = "gdn T=100 Dk=64")

  runTimed "DIAGNOSTIC intra-family pair GDN at the budget ceiling (fp64)":
    gdnPairCase(0x6A11CF'u64, T = 256, Hv = 8, Hk = 4, hkRatio = 2,
      Dk = 128, Dv = 128, chunkLen = 64, tag = "gdn T=256 Dk=128")

main()
