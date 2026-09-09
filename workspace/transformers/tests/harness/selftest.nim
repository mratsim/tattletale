
# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Selftest: two-sided verification of the test-suite checks themselves.
## A seeded fault corpus (v0) must be rejected, a known-good drift corpus must be accepted.
## Detection floors, one row per fault kind:
##
## | fault                  | size          | detector                     | floor
## |========================|===============|==============================|===================================
## | uniform scale          | x1.01         | element-wise f32 rtol 1e-6   | detected ≥ x1.000002
## | single-element flip    | +1.0          | element-wise f32 abstol 1e-6 | detected for |v| ≲ 1e6
## | permuted tensor        | full          | element-wise                 | detected
## | wrong sign             | full          | element-wise                 | detected
## | codebook entry         | +0.5          | element-wise                 | detected
## | dense drift            | +3 ulp all    | match-rate maxUlp 2/0        | detected > maxUlp
## | dense drift            | +3 ulp all    | stats quantiles maxUlp 2     | detected > maxUlp
## | sparse drift           | 0.5% +128 ulp | match-rate 4/1e-3            | detected above frac budget
## | sparse drift           | 0.5% +128 ulp | stats + histogram            | NOT detected (mass 0.5% < histL1)
## | tail-probability drift | x1.05         | softmax row sums             | NOT detected (sums stay 1)
## | tail-probability drift | x1.05         | tail-probability check       | detected ≥ x1.000002
##
## Chain-level corpus: synthetic checkpoint sequences and a synthetic greedy chain, built in memory.
## The mean-drift bound is stated per checkpoint: the bound derives from one longest
## f32 reduction per checkpoint, see SPEC.
## Honest signed means measure 1.0e-4 of the bulk or less, three orders of magnitude inside.
##
## | fault                        | size              | detector           | floor
## |==============================|===================|====================|====================================
## | coherent per-checkpoint bias | 1.5x bound        | mean-drift check   | detected past the chain drift bound
## | coherent per-checkpoint bias | 0.5x bound        | none, in bound     | accepted by derivation
## | zero-mean elementwise drift  | 2x bound each     | mean-drift check   | accepted, signed mean cancels
## | compounding per-block fault  | band width~sqrt(k) | drift-scaling     | rejected at the tail
## | bounded band width drift     | ratios <= 1.25    | drift-scaling      | accepted, the flat bound
## | margin-0 tie flip mid-chain  | full token        | tie clause         | accepted within cap
## | tie chain without recovery   | endless flips     | tie-flip cap       | rejected past the cap
## | wide-margin divergence       | full token        | greedy step check  | rejected
##
## Rejection rows: fault cases a check must reject. Acceptance rows: drift cases a
## check must accept.
##
## Descriptor rejection rows, the fixture contract: a synthetic
## 8192-element f32 tensor, all in memory:
##
## | fault                           | size                             | detector                  | floor
## |=================================|==================================|===========================|===============================
## | coherent relative scale         | x(1+2^-23) all                   | descriptor bands + probe  | accepted, honest rounding
## | alternating relative drift      | x(1+-2^-23) all                  | descriptor meanAbs        | accepted, inside the drift bound
## | coherent relative scale         | x(1+2^-20) all                   | max quantile + means      | rejected, systematic bias
## | absolute uniform shift          | +8 ulp at max                    | quantiles + means         | rejected by the absolute-scale bound
## | generic error from a check path | ValueError                       | none, must propagate      | the rejection rows never count it
## | single-element nudge            | +4 ulp at max, off probe         | none                      | documented one-element floor
## | single-element nudge            | +8 ulp at max on the max element | fingerprint max quantile  | rejected
## | single-element nudge            | +8 ulp at max, on probe          | descriptor probe          | rejected
## | value swap                      | two off-probe elements           | none (multiset invariant) | documented floor
## | value swap                      | one probed element               | descriptor probe          | rejected
##
## Known-good drift accepted:
## - 1-2 ulp f32 platform drift (bench_rmsnorm.nim measurement)
## - dense 1 ulp and dense 3 ulp at the attention budget (maxUlp 4)
## - honest bf16 rounding of an f32 tensor (1 bf16 ulp = 2^-8 relative)
##
## Evaluation-order rejection rows: the conv budget and the linear-in-decode-length
## state drift bound must keep rejection power at the derived bounds.
## Synthetic state tensors, built in memory. The conv state counts
## in bf16 ulp-units at the tensor max, the f32 state drift accumulates
## linearly in the decode length T (coherent rounding error, O(T), not
## a zero-mean random walk) at the calibrated decode length T = 70
## (abstol = 2^-3 * T * meanAbs, the chainCheckpointAbstol shape):
##
## | fault            | size                                  | detector                         | floor
## |==================|=======================================|==================================|================================
## | conv-state drift | +3 ulp-units at max                   | conv budget (assertConvEvalOrder) | rejected past the budget 2
## | conv-state drift | +1 ulp-unit at max                    | conv budget                      | accepted, inside the budget
## | state drift      | measured scale, 1027 fp32 ulps at max | linear drift bound (assertSsmEvalOrder) | accepted, inside the bound
## | state drift      | 0.9x bound                            | linear drift bound                     | accepted, just below the bound
## | state drift      | 2x bound                              | linear drift bound                     | rejected past the bound

import
  std/math,
  std/streams,
  std/os,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch_testutils

import ./tolerance
import workspace/zstd/zstd_highlevel

from workspace/libtorch/src/raw_libtorch import manual_seed

type FaultKind* = enum
  ## Seeded faults applied to a reference tensor or distribution.
  fkScale
    ## Uniform scale x1.01.
  fkSingleFlip
    ## One element shifted by +1.0.
  fkPermute
    ## Last two dims transposed, reshaped back to the original shape.
  fkSign
    ## Negated tensor.
  fkCodebookEntry
    ## One codebook entry shifted by +0.5.
  fkTailProbability
    ## Softmax tail beyond the top-k rescaled by x1.05, renormalized.
  fkDenseDrift
    ## Every element moved by k ulps on the f32 pattern.
  fkSparseDrift
    ## Every 199th element moved by 128 ulps (a 0.5% gross corruption).

proc applyFault*(t: Tensor, kind: FaultKind, k = 32): Tensor =
  ## Apply fault `kind` to a fresh copy of `t` (f32) and return it.
  result = t.to(F.kFloat32).clone()
  case kind
  of fkScale:
    copyFrom(result, result * 1.01'f32)
  of fkSingleFlip:
    let piece = result.view(-1).narrow(0, 0, 1)
    copyFrom(piece, piece + 1.0'f32)
  of fkPermute:
    doAssert result.dim == 2, "permute fault applies to 2D tensors"
    let rows = result.size(-2)
    let cols = result.size(-1)
    copyFrom(result, result.permute(1, 0).reshape(rows, cols))
  of fkSign:
    copyFrom(result, result.neg())
  of fkCodebookEntry:
    let piece = result.view(-1).narrow(0, k, 1)
    copyFrom(piece, piece + 0.5'f32)
  of fkTailProbability:
    let cols = result.size(-1)
    doAssert k < cols, "tail fault needs k below the last dim"
    let tail = result.narrow(-1, k, cols - k)
    copyFrom(tail, tail * 1.05'f32)
    copyFrom(result, result / result.sum(axis = -1, keepdim = true))
  of fkDenseDrift:
    # f32 bits read through an int32 view. data_ptr<T> refuses a T different from the tensor dtype.
    let iv = result.view(F.kInt32).contiguous()
    let raw = cast[ptr UncheckedArray[int32]](iv.data_ptr(int32))
    for i in 0 ..< result.numel():
      raw[i] = cast[int32](cast[uint32](raw[i]) + uint32(k))
  of fkSparseDrift:
    let iv = result.view(F.kInt32).contiguous()
    let raw = cast[ptr UncheckedArray[int32]](iv.data_ptr(int32))
    var i = 0
    while i < result.numel():
      raw[i] = cast[int32](cast[uint32](raw[i]) + 128'u32)
      i += 199

proc tailProbability*(probs: Tensor, k: int): float64 =
  ## Probability mass of everything beyond the top-k entries, in f32.
  let p32 = probs.to(F.kFloat32)
  let tail = p32.sort(axis = -1, descending = true).values
    .narrow(-1, k, p32.size(-1) - k)
  tail.sum().item(float64)

proc rejects*(body: proc(): bool): bool =
  ## True when `body` raises the check-layer rejection type, false when it completes. The rejection
  ## half of the two-sided selftest. Every other error propagates and the run fails.
  ## A crashing detector is an infrastructure bug, not a fault rejection.
  try:
    discard body()
    false
  except HarnessCheckError:
    true

proc assertClose*(actual, expected: Tensor, rtol, abstol: float64, msg = "") =
  ## Check one elementwise compare: the libtorch assertAllClose utility. The wrapper
  ## re-raises its AssertionDefect as the check-layer rejection type, so the element-wise
  ## corpus rows reject under the same tolerances as the harness checks.
  try:
    assertAllClose(actual, expected, rtol = rtol, abstol = abstol, msg = msg)
  except AssertionDefect:
    raise newException(HarnessCheckError,
      "[ttt] allClose assertion failed" & (if msg.len > 0: ": " & msg else: ""))

proc propagatesValueError*(body: proc(): bool): bool =
  ## True when a non-check-layer error of the body escapes `rejects` and surfaces here.
  ## Negative row proving a generic CatchableError from a check path propagates rather
  ## than counting as a fault rejection.
  try:
    discard rejects(body)
    false
  except ValueError:
    true

proc runSelftest*(): bool =
  ## Run both corpus halves. Prints one line per case. Returns true when every fault is rejected
  ## and every drift case is accepted.
  Torch.manual_seed(0x5EED'u64)
  var allGreen = true

  template check(label: string, ok: bool) =
    let verdict = ok
    echo (if verdict: "  ✅ " else: "  ❌ ") & label
    if not verdict:
      allGreen = false

  # reference tensors
  let x = F.randn(4, 8, F.kFloat32)
  let probs = F.softmax(F.randn(4, 64, F.kFloat32), dim = -1)
  let codebook = F.randn(8, 16, F.kFloat32)

  # fault corpus v0: every fault must be rejected
  for kind in [fkScale, fkSingleFlip, fkPermute, fkSign]:
    let faulty = applyFault(x, kind)
    let ok = rejects(proc(): bool =
      assertClose(faulty, x, rtol = 1e-6, abstol = 1e-6,
        msg = "selftest element-wise")
      true)
    check "fault " & $kind & " rejected", ok

  block:
    let faulty = applyFault(codebook, fkCodebookEntry)
    let ok = rejects(proc(): bool =
      assertClose(faulty, codebook, rtol = 1e-6, abstol = 1e-6,
        msg = "selftest codebook")
      true)
    check "fault fkCodebookEntry rejected", ok

  block:
    # Tail-probability drift keeps row sums at 1. The row-sum check alone cannot detect it, the tail-probability
    # check must catch the fault.
    let drifted = applyFault(probs, fkTailProbability)
    let tailBefore = tailProbability(probs, 8)
    let tailAfter = tailProbability(drifted, 8)
    let sumOk = rejects(proc(): bool =
      let worst = drifted.sum(axis = -1).add(-1.0'f32).abs().max().item(float64)
      doAssert worst <= 1e-5, "row sums moved"
      true)
    check "fault fkTailProbability keeps row sums (row sums alone cannot detect it)",
      not sumOk
    let tailOk = abs(tailAfter - tailBefore) > 1e-6
    check "fault fkTailProbability detected by the tail-probability check", tailOk

  block:
    # Dense drift beyond the match-rate cap: budget 2 ulp with zero mismatch allowance catches a +3
    # ulp shift of every element.
    let rmsBudget = defaultBudget(obRmsNorm, F.kCPU)
    let faulty = applyFault(x, fkDenseDrift, 3)
    let ok = rejects(proc(): bool =
      assertMatchRate(faulty, x, rmsBudget, msg = "selftest dense drift")
      true)
    check "fault fkDenseDrift (+3 ulp all) rejected by match-rate", ok

  block:
    # Stats quantiles drift with a dense shift: +3 ulp everywhere moves each order statistic by 3
    # ulp, past the norm budget of 2.
    let faulty = applyFault(x, fkDenseDrift, 3)
    let ok = rejects(proc(): bool =
      assertStats(faulty, tensorStats(x), defaultBudget(obRmsNorm, F.kCPU),
        msg = "selftest dense drift stats")
      true)
    check "fault fkDenseDrift (+3 ulp all) rejected by stats", ok

  block:
    # fp16 ulp unit of the exl3 families: the helpers count grid steps of the
    # fp16 pattern, one step at the binade boundary included, and the
    # match-rate budget rows reject at the same few-ulp magnitudes.
    let a = F.toTensor(@[1.0'f32, 1.9990234375'f32, 2.0'f32, -0.0'f32])
      .to(F.kFloat16)
    let b = F.toTensor(@[1.0009765625'f32, 2.0'f32, 2.001953125'f32, 0.0'f32])
      .to(F.kFloat16)
    check "fp16 ulp distances: one grid step, binade crossing included",
      ulpDistanceFp16(1.0'f32, 1.0009765625'f32) == 1 and
      ulpDistanceFp16(1.9990234375'f32, 2.0'f32) == 1 and
      ulpDistanceFp16(0.0'f32, -0.0'f32) == 0
    let f16Budget = ToleranceBudget(tier: ctDistribution,
      maxUlp: 2, maxMismatchFrac: 0.0, histL1: 0.01)
    let okSame = not rejects(proc(): bool =
      assertMatchRate(a, b, f16Budget, msg = "selftest fp16 one-ulp drift")
      true)
    check "fp16 one-step drift accepted at the 2-ulp budget", okSame
    let far = b.contiguous().clone()
    let farV = far.contiguous().view(F.kInt16).contiguous()
    let rv = cast[ptr UncheckedArray[int16]](farV.data_ptr(int16))
    rv[0] = int16(fp16BitsFromF32(1.0009765625'f32 * 4.0))
    let okFar = rejects(proc(): bool =
      assertMatchRate(a, far, f16Budget, msg = "selftest fp16 wide drift")
      true)
    check "fp16 multi-step drift rejected by match-rate", okFar

  block:
    # Sparse gross corruption: 0.5% of elements at +128 ulp passes quantiles and histogram, mass
    # below histL1). The match-rate frac budget catches it.
    let attnBudget = defaultBudget(obAttention, F.kCPU)
    let faulty = applyFault(x, fkSparseDrift)
    let statsOk = not rejects(proc(): bool =
      assertStats(faulty, tensorStats(x), attnBudget,
        msg = "selftest sparse drift stats")
      true)
    check "fault fkSparseDrift passes stats (documented floor)", statsOk
    let rateOk = rejects(proc(): bool =
      assertMatchRate(faulty, x, attnBudget, msg = "selftest sparse drift")
      true)
    check "fault fkSparseDrift rejected by match-rate frac", rateOk

  block:
    # Tie-flip class: a structural tie diverging to the tied runner-up is an expected flip
    # inside the cap. The tie band must hold pick's logit, the cap must stop an endless flip chain, and a wide
    # margin divergence must fail regardless of cap.
    var cfg = GreedyConfig(tieUlps: 1, epsBase: 0.05,
      tailBand: 0.3, klBand: 0.05, maxFlips: 2)
    # Reference step with an exact tie on the bf16 grid: top1 = top2. Top-32 order follows the recorded
    # first-max id, torch argmax semantics: the tie pair sits at slots 0 and 1, ids 3 and 7.
    let refLogits: seq[float32] = @[16.0'f32, 16.0, 15.0, 14.0, 13.0, 12.0,
      11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
      -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0,
      -11.0, -12.0, -13.0, -14.0]
    var refStep = GreedyStepRef(step: 0, chosenToken: 3,
      top32Ids: @[3, 7, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
      top32Logits: refLogits, argmaxMargin: 0.0, tailProbability: 0.0)
    # The recorded tail probability must match what the synthetic row carries, so the checksums
    # compare like against like exactly.
    block:
      var r = newSeq[float32](128)
      for i in 0 ..< 32:
        r[refStep.top32Ids[i]] = refLogits[i]
      refStep.tailProbability = observedTailProbability(F.toTensor(r), refStep.top32Ids)
    # Build a vocab row where the pick (id 99) and id 7 tie at 16.0.
    var rowSeq = newSeq[float32](128)
    for i in 0 ..< 32:
      rowSeq[refStep.top32Ids[i]] = refLogits[i]
    # The flip pick sits 2 bf16 ulp above the tie: inside the tie band, past the recorded first-max
    # id.
    rowSeq[99] = 16.125'f32
    let tieRow = F.toTensor(rowSeq)
    var st = GreedyState()
    let v1 = checkGreedyStep(st, cfg, refStep, tieRow)
    check "tie flip at structural tie returns gvTieFlip", v1 == gvTieFlip
    check "tie flip counted", st.flips == 1

    # Cap: two more flips exhaust maxFlips 2 and the third raises.
    let capOk = rejects(proc(): bool =
      discard checkGreedyStep(st, cfg, refStep, tieRow)
      discard checkGreedyStep(st, cfg, refStep, tieRow)
      true)
    check "tie-flip cap exceeded raises", capOk

    # Real divergence with a wide recorded margin fails even under cap.
    var wideStep = refStep
    wideStep.argmaxMargin = 2.0
    var wideLogits = refLogits
    wideLogits[0] = 18.0'f32
    wideStep.top32Logits = wideLogits
    var st2 = GreedyState()
    let wideOk = rejects(proc(): bool =
      discard checkGreedyStep(st2, cfg, wideStep, tieRow)
      true)
    check "real divergence with wide margin raises", wideOk

    # Post-flip re-convergence: after a tie flip the next step must match the recording again.
    var st3 = GreedyState()
    let agree = checkGreedyStep(st3, cfg, refStep, F.toTensor(
      block:
        var r = newSeq[float32](128)
        for i in 0 ..< 32: r[refStep.top32Ids[i]] = refLogits[i]
        r))
    check "agreement step returns gvAgree", agree == gvAgree
    check "agreement leaves flip count clean", st3.flips == 0

  # chain-level fault and drift corpus: synthetic checkpoint sequences and a synthetic greedy
  # chain, built in memory. The acceptance boundary must sit exactly where the derivation
  # says: the chain drift bound.
  block:
    let reductionLen = 3072
    let expected = F.randn(6144, F.tensorOptions(F.kBFloat16, F.kCPU))
    let bulk = meanAbsValue(expected)
    let mdBound = chainMeanDriftBound(bulk, reductionLen)
    let n = expected.numel()

    proc constDelta(c: float64): Tensor =
      var s = newSeq[float32](n)
      for i in 0 ..< n:
        s[i] = c.float32
      F.toTensor(s).to(F.kBFloat16)

    proc alternatingDelta(c: float64): Tensor =
      var s = newSeq[float32](n)
      for i in 0 ..< n:
        s[i] = (if i mod 2 == 0: c.float32 else: -c.float32)
      F.toTensor(s).to(F.kBFloat16)

    # A coherent bias above the chain drift bound passes the elementwise band, which absorbs
    # depth-linear offsets by design, and is rejected by the mean-drift check. The band compare
    # runs alone here, the mean-drift case below runs the full checkpoint assert.
    let faulty = expected + constDelta(1.5 * mdBound)
    let bandOnly = maxBandWidths(faulty, expected, ChainCheckpointRtol,
      chainCheckpointAbstol(8, bulk))
    check "chain fault: bias above the bound passes the band (by design)",
      bandOnly.violations == 0
    let mdReject = rejects(proc(): bool =
      discard assertChainCheckpoint(faulty, expected, depth = 8,
        reductionLen = reductionLen, msg = "corpus mean drift")
      true)
    check "chain fault: bias above the bound rejected by mean drift", mdReject

    # Acceptance row: the same bias shape below the bound is inside the chain drift bound and
    # must pass band and mean drift alike.
    let legal = expected + constDelta(0.5 * mdBound)
    let legalOk = not rejects(proc(): bool =
      discard assertChainCheckpoint(legal, expected, depth = 8,
        reductionLen = reductionLen, msg = "corpus legal bias")
      true)
    check "chain drift: bias below the bound accepted (matches the derivation)",
      legalOk

    # Zero-mean elementwise drift, double the bound per element: the signed mean cancels, the mean-drift
    # check accepts.
    let zeroMeanOk = not rejects(proc(): bool =
      assertChainMeanDrift(expected + alternatingDelta(2.0 * mdBound),
        expected, reductionLen, msg = "corpus zero-mean drift")
      true)
    check "chain drift: zero-mean drift at double the bound accepted",
      zeroMeanOk

    # Compounding per-block fault: the band width grows like sqrt(depth) and crosses the flat
    # bound exactly at the depth-28 tail.
    var compounding: seq[float64] = @[]
    for depth in [1, 2, 3, 4, 5, 6, 7, 8, 28]:
      compounding.add 0.31 * sqrt(depth.float64)
    let compoundingReject = rejects(proc(): bool =
      checkDriftScaling(compounding)
      true)
    check "chain fault: compounding growth rejected at the tail",
      compoundingReject

    # Bounded drift under the flat bound: accepted.
    let boundedOk = not rejects(proc(): bool =
      checkDriftScaling(@[0.31, 0.39, 0.35, 0.30, 0.28, 0.31, 0.25, 0.22])
      true)
    check "chain drift: bounded band widths accepted", boundedOk

  # Chain-level tie handling: a margin-0 tie flip mid-chain takes teacher-forced recovery,
  # the recorded token fed back in, and the chain re-converges. An endless flip chain trips the cap, and wide-margin
  # corruption fails inside the sequence.
  block:
    var cfg = GreedyConfig(tieUlps: 1, epsBase: 0.05,
      tailBand: 0.3, klBand: 0.05, maxFlips: 2)
    let refLogits: seq[float32] = @[16.0'f32, 16.0, 15.0, 14.0, 13.0, 12.0,
      11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
      -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0,
      -11.0, -12.0, -13.0, -14.0]
    let supportIds = @[3, 7, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    proc vocabRow(pickId: int, pickLogit: float32): Tensor =
      ## The recorded vocab row, with an optional diverging pick: id 99 one bf16 ulp above the tied
      ## top logit, inside the tie band.
      var r = newSeq[float32](128)
      for i in 0 ..< 32:
        r[supportIds[i]] = refLogits[i]
      if pickId >= 0:
        r[pickId] = pickLogit
      F.toTensor(r)

    proc recordedStep(step: int): GreedyStepRef =
      result = GreedyStepRef(step: step, chosenToken: supportIds[0],
        top32Ids: supportIds, top32Logits: refLogits,
        argmaxMargin: 0.0, tailProbability: 0.0)
      result.tailProbability = observedTailProbability(
        vocabRow(-1, 0.0'f32), supportIds)

    let step0 = recordedStep(0)
    let step1 = recordedStep(1)

    # Margin-0 tie sequence: step 0 flips to the tied runner-up inside the tie band, the recorded
    # token is teacher-forced back, and step 1 re-converges. The whole chain still matches the
    # recording with one flip.
    var st = GreedyState()
    let flip = checkGreedyStep(st, cfg, step0, vocabRow(99, 16.125'f32))
    let reAnchor = checkGreedyStep(st, cfg, step1, vocabRow(-1, 0.0'f32))
    check "chain ties: margin-0 flip recovers on the teacher-forced token",
      flip == gvTieFlip and reAnchor == gvAgree and st.flips == 1

    # Endless tie flips past the cap: the chain fails at the cap.
    let capReject = rejects(proc(): bool =
      var s2 = GreedyState()
      discard checkGreedyStep(s2, cfg, step0, vocabRow(99, 16.125'f32))
      discard checkGreedyStep(s2, cfg, step0, vocabRow(99, 16.125'f32))
      discard checkGreedyStep(s2, cfg, step0, vocabRow(99, 16.125'f32))
      true)
    check "chain ties: endless flip chain rejected at the cap", capReject

    # Wide-margin corruption inside the chain: the recorded step wins by a wide margin, the observed
    # row still ties: rejected.
    var wide = step0
    wide.argmaxMargin = 2.0
    var wideLogits = refLogits
    wideLogits[0] = 18.0'f32
    wide.top32Logits = wideLogits
    let wideReject = rejects(proc(): bool =
      var s3 = GreedyState()
      discard checkGreedyStep(s3, cfg, wide, vocabRow(99, 16.125'f32))
      true)
    check "chain fault: wide-margin corruption rejected", wideReject

  # known-good drift corpus: must be accepted
  block:
    # 1-2 ulp f32 platform drift.
    let drifted = x * (1.0'f32 + 2.4e-7'f32)
    let ok = not rejects(proc(): bool =
      assertClose(drifted, x, rtol = 1e-6, abstol = 1e-6,
        msg = "selftest ulp drift")
      true)
    check "drift 1-2 ulp f32 accepted", ok

  block:
    # Dense ulp shifts inside the attention budget: maxUlp 4, mismatch 1e-3. 1 ulp is bench-level
    # drift, 3 ulp shows headroom.
    for k in [1, 3]:
      let drifted = applyFault(x, fkDenseDrift, k)
      let ok = not rejects(proc(): bool =
        assertMatchRate(drifted, x, defaultBudget(obAttention, F.kCPU),
          msg = "selftest dense drift within budget")
        true)
      check "drift dense " & $k & " ulp accepted at attention budget", ok

  block:
    # Honest bf16 rounding of an f32 tensor.
    let rounded = x.to(F.kBFloat16).to(F.kFloat32)
    let ok = not rejects(proc(): bool =
      assertClose(rounded, x, rtol = 1e-2, abstol = 1e-2,
        msg = "selftest bf16 rounding")
      true)
    check "drift honest bf16 rounding accepted", ok

  block:
    # Tail-probability epsilon drift below the floor.
    let drifted = probs.clone()
    let tail = drifted.narrow(-1, 8, 64 - 8)
    copyFrom(tail, tail * 1.000001'f32)
    copyFrom(drifted, drifted / drifted.sum(axis = -1, keepdim = true))
    let ok = abs(tailProbability(drifted, 8) - tailProbability(probs, 8)) <= 1e-6
    check "drift tail-probability epsilon below floor accepted", ok

  block:
    # Cross-implementation corpus: fixture_stats.py already fingerprinted the committed tensor
    # bytes, Nim recomputes and requires bit-exact agreement on order statistics and histograms.
    # The histogram total is the summation statistic, integer math, also bit-exact.
    let dir = currentSourcePath().parentDir() / "stats-corpus"
    let st = Safetensor.open(dir / "crossimpl.safetensor")
    let statsFile = loadFingerprintStats(dir / "crossimpl.safetensor.stats")
    for ts in statsFile.tensors:
      let t = st.getTensorOwned(ts.name)
      let fp = tensorStats(t, allowMinusInf = ts.allowMinusInf,
        withHistogram = ts.hasHist, name = ts.name)
      var quantilesExact = fp.n == ts.n
      for i in 0 ..< QuantileCount:
        quantilesExact = quantilesExact and fp.quantiles[i] == ts.quantiles[i]
      let histExact = (not ts.hasHist) or
        (fp.histKeys == ts.histKeys and fp.histCounts == ts.histCounts and
         fp.histTotal == ts.histTotal)
      check "stats-corpus " & ts.name & " quantiles bit-exact", quantilesExact
      check "stats-corpus " & ts.name & " histogram bit-exact", histExact
      if ts.hasDescriptors:
        # The descriptor contract: the Nim recompute of the descriptor entry must agree bit-exactly
        # with the recorded fields, like the quantiles and histograms.
        let dp = tensorDescriptors(t, ts.name, ts.probeMode,
          allowMinusInf = ts.allowMinusInf, withHistogram = ts.hasHist)
        let descExact = dp.n == ts.n and dp.meanAbs == ts.meanAbs and
          dp.signedMean == ts.signedMean and
          dp.tailProbability == ts.tailProbability and
          dp.tailEdge == ts.tailEdge and dp.probeStride == ts.probeStride and
          dp.probeValues == ts.probeValues
        check "stats-corpus " & ts.name & " descriptors (" & $ts.probeMode &
          ") bit-exact", descExact
      # The assert path under one budget on the same recorded inputs.
      let budget = defaultBudget(obRmsNorm, F.kCPU)
      var ok = true
      try:
        assertStats(t, ts, budget, msg = "stats-corpus " & ts.name,
          allowMinusInf = ts.allowMinusInf)
      except HarnessCheckError:
        ok = false
      check "stats-corpus " & ts.name & " assertStats passed", ok

  block:
    # Format registry: the harness loaders accept exactly the three
    # ttt-tf registry ids; the h1-era shapes (no schema field, or the
    # numeric 1) are retired. The constants are the observable contract,
    # the selftest pins the literal values so a silent rename cannot pass.
    check "registry id ttt-tf-001-greedy-steps-h2",
      GreedyStepsSchema == "ttt-tf-001-greedy-steps-h2"
    check "registry id ttt-tf-002-logit-decisions-probe-h2",
      LogitDecisionsProbeSchema == "ttt-tf-002-logit-decisions-probe-h2"
    check "registry id ttt-tf-003-tensor-stats-h2",
      TensorStatsSchema == "ttt-tf-003-tensor-stats-h2"
    # Stats parse/write roundtrip under the string registry id: the writer
    # emits {"schema":"ttt-tf-003-tensor-stats-h2",...} and the reader
    # returns the same schema value and the same tensor entries.
    var roundtripStats: FingerprintStatsFile
    roundtripStats.schema = TensorStatsSchema
    roundtripStats.source = "selftest-roundtrip.safetensor"
    var roundtripEntry = tensorStats(probs, withHistogram = true)
    # tensorStats names the entry only in its error paths; the writer
    # emits ts.name as the JSON key, so set it like gen_stats does.
    roundtripEntry.name = "probs"
    roundtripStats.tensors.add roundtripEntry
    let roundtripPath = getTempDir() / "ttt-selftest-stats-roundtrip.json.zst"
    writeFingerprintStats(roundtripPath, roundtripStats)
    let readBack = loadFingerprintStats(roundtripPath)
    check "stats roundtrip schema is the registry string",
      readBack.schema == TensorStatsSchema
    check "stats roundtrip source round-trips",
      readBack.source == roundtripStats.source
    check "stats roundtrip tensor entry byte-exact",
      encodeTensorStatsBody(readBack.statsTensor("probs")) ==
        encodeTensorStatsBody(roundtripEntry)

  # Descriptor rejection rows: the recorded summary of the fixture
  # contract must reject the fault classes the full tensor caught.
  # Detected and undetected fault classes both carry their measured floor
  # in the module header tables. Synthetic 8192-element f32 tensor,
  # probe stride 16, built in memory.
  block:
    Torch.manual_seed(0x5EEDFA17'u64)
    let dref = F.randn(8192, F.tensorOptions(F.kFloat32, F.kCPU))
    let dts = tensorDescriptors(dref, "selftest-descriptor", dmDrift,
      allowMinusInf = false, withHistogram = true)
    let budget = descriptorStatsBudget(dts.probeMode)
    doAssert dts.probeStride == 16 and dts.probeValues.len == 512,
      "selftest descriptor geometry drifted"
    let maxAbs = dref.abs().max().item(float64)
    let ulpMax = ulpFp32At(maxAbs)

    proc denseShift(c: float64): Tensor =
      result = dref.to(F.kFloat32).clone()
      copyFrom(result, result + c.float32)

    proc nudge(idx: int, c: float64): Tensor =
      result = dref.to(F.kFloat32).clone()
      let piece = result.view(-1).narrow(0, idx, 1)
      copyFrom(piece, piece + c.float32)

    proc swapVals(a, b: int): Tensor =
      result = dref.to(F.kFloat32).clone()
      let va = result.view(-1).narrow(0, a, 1).item(float32)
      let vb = result.view(-1).narrow(0, b, 1).item(float32)
      copyFrom(result.view(-1).narrow(0, a, 1),
        F.toTensor(@[vb]))
      copyFrom(result.view(-1).narrow(0, b, 1),
        F.toTensor(@[va]))

    proc descOk(t: Tensor): bool =
      not rejects(proc(): bool =
        assertStats(t, dts, budget, msg = "selftest descriptor rejection row")
        assertDescriptors(t, dts, msg = "selftest descriptor rejection row")
        true)

    # Acceptance: honest rounding is relative (each element drifts ulps at its own binade), so the
    # following relative drifts all sit inside the derived absolute bound:
    # - a dense drift of one ulp per element
    # - a coherent relative scale of the same size, the class the signed mean tolerates
    # - the alternating relative drift
    proc relativeScale(c: float64): Tensor =
      result = dref.to(F.kFloat32).clone()
      copyFrom(result, result * (1.0 + c).float32)

    proc alternatingRelative(c: float64): Tensor =
      result = dref.to(F.kFloat32).clone()
      var signs = newSeq[float32](8192)
      for i in 0 ..< 8192:
        signs[i] = (if i mod 2 == 0: (1.0 + c).float32 else: (1.0 - c).float32)
      copyFrom(result, result * F.toTensor(signs))

    # The acceptance rows sit at x(1+2^-23), one half own ulp per element: the worst order
    # statistic drifts two ulps at the max, half the bound. The recorded max sits in the top
    # binade and the margins stay seed-independent under the unified absolute bound.
    check "descriptor: coherent relative scale x(1+2^-23) accepted",
      descOk(relativeScale(pow(2.0, -23.0)))
    check "descriptor: alternating relative drift accepted",
      descOk(alternatingRelative(pow(2.0, -23.0)))

    # Rejection: a coherent relative scale past the tolerance band is the systematic-bias class the signed
    # mean polices. A uniform absolute shift is a fault under the absolute-scale bound: it scatters
    # near-zero bulk into foreign histogram buckets. Rejection margins stay seed-independent:
    # the max quantile drifts eight to sixteen ulps of the max at x(1+2^-20) and +8 ulp,
    # twice the bound and up.
    check "descriptor: coherent relative scale x(1+2^-20) rejected",
      not descOk(relativeScale(pow(2.0, -20.0)))
    check "descriptor: absolute uniform shift +8 ulp at max rejected",
      not descOk(denseShift(8.0 * ulpMax))

    # Probe geometry: a bounded off-probe single-element nudge stays inside the quantile budget and below the detection
    # floor of the recorded summary, the documented one-element floor. A max-element nudge is policed
    # by the fingerprint max quantile wherever it sits, and a probed element is caught directly.
    check "descriptor: bounded off-probe nudge accepted (documented floor)",
      descOk(nudge(1, 4.0 * ulpMax))
    let argmaxIdx = dref.to(F.kFloat32).abs().argmax().item(int64).int
    check "descriptor: max-element nudge rejected by the max quantile",
      not descOk(nudge(argmaxIdx, 8.0 * ulpMax))
    check "descriptor: on-probe nudge rejected by the probe",
      not descOk(nudge(0, 8.0 * ulpMax))

    # Multiset invariant: a value swap leaves every order statistic, both means, the tail and the
    # histogram in place. The probe catches a swap only when a probed position moves, so the
    # off-probe swap is the documented detection floor of the recorded summary.
    check "descriptor: off-probe value swap accepted (documented floor)",
      descOk(swapVals(1, 3))
    check "descriptor: on-probe value swap rejected",
      not descOk(swapVals(0, 3))

  # Evaluation-order rejection rows: the conv budget and the linear-in-decode-length
  # state drift bound must keep rejection power at the derived bounds.
  # Synthetic state tensors, built in memory. The conv state counts
  # in bf16 ulp-units at the tensor max, the f32 state drift accumulates
  # linearly in the decode length T at the calibration point T = 70.
  block:
    Torch.manual_seed(0x5EED5EED'u64)
    let steps = 70

    proc nudgeMax(t: Tensor, c: float64): Tensor =
      result = t.to(F.kFloat32).clone()
      let idx = result.abs().argmax().item(int64).int
      let piece = result.view(-1).narrow(0, idx, 1)
      copyFrom(piece, piece + c.float32)

    # Conv budget: a synthetic bf16 conv-state tensor. The budget scales
    # with bf16UlpAt at the tensor max, so the seed moves the margins only.
    let convRef = (F.randn(8192, F.tensorOptions(F.kFloat32, F.kCPU)) *
      0.6'f32).to(F.kBFloat16)
    let convMax = convRef.to(F.kFloat32).abs().max().item(float64)
    let convUnit = bf16UlpAt(convMax)
    let convPast = nudgeMax(convRef, 3.0 * convUnit).to(F.kBFloat16)
    let convReject = rejects(proc(): bool =
      assertConvEvalOrder(convPast, convRef, msg = "corpus conv past budget")
      true)
    check "evaluation-order fault: conv drift 3 ulp-units rejected at the budget 2",
      convReject
    let convInside = nudgeMax(convRef, 1.0 * convUnit).to(F.kBFloat16)
    let convOk = not rejects(proc(): bool =
      assertConvEvalOrder(convInside, convRef, msg = "corpus conv sub-budget")
      true)
    check "evaluation-order drift: conv drift 1 ulp-unit accepted inside the budget",
      convOk

    # Linear drift bound: a synthetic f32 state tensor at the calibrated
    # decode length. The measured-scale drift must sit inside the bound: 1027
    # fp32 ulps at the max, the Metal T=70 calibration point. A 2x-bound
    # corruption rejects, a just-below-bound drift accepts.
    let stateRef = F.randn(4096, F.tensorOptions(F.kFloat32, F.kCPU)) * 0.01'f32
    let stateMax = stateRef.abs().max().item(float64)
    let bound = chainCheckpointAbstol(steps, meanAbsValue(stateRef))
    let measuredScale = 1027.4 * ulpFp32At(stateMax)
    doAssert measuredScale < bound,
      "the measured-scale drift must sit inside the bound for the corpus seed"
    let stateMeasured = nudgeMax(stateRef, measuredScale)
    let measuredOk = not rejects(proc(): bool =
      assertSsmEvalOrder(stateMeasured, stateRef, steps,
        msg = "corpus state measured-scale drift")
      true)
    check "evaluation-order drift: measured-scale state drift accepted (T=70 bound)",
      measuredOk
    let belowLaw = nudgeMax(stateRef, 0.9 * bound)
    let belowOk = not rejects(proc(): bool =
      assertSsmEvalOrder(belowLaw, stateRef, steps,
        msg = "corpus state just below the law")
      true)
    check "evaluation-order drift: just-below-bound state drift accepted", belowOk
    let pastLaw = nudgeMax(stateRef, 2.0 * bound)
    let pastOk = rejects(proc(): bool =
      assertSsmEvalOrder(pastLaw, stateRef, steps,
        msg = "corpus state 2x-law drift")
      true)
    check "evaluation-order fault: 2x-bound state drift rejected", pastOk

  # Negative row: a generic CatchableError from a check path must propagate out of `rejects`
  # instead of counting as a fault rejection. A fingerprint-only entry fed to assertDescriptors
  # raises ValueError, a caller misuse, never a fault verdict.
  block:
    Torch.manual_seed(0x5EEDFA17'u64)
    let dref = F.randn(64, F.tensorOptions(F.kFloat32, F.kCPU))
    let propagated = propagatesValueError(proc(): bool =
      assertDescriptors(dref, TensorStats(name: "fingerprint-only-entry"),
        msg = "propagation probe")
      true)
    check "generic CatchableError from a check path propagates (not counted as rejection)",
      propagated

  # Ulp-at-value correctness row. The projection and greedy tie
  # bands scale with bf16UlpAt, so the ulp width must stay locked:
  # one value per binade. bf16 stores 7 mantissa bits, so the width
  # halves at every binade boundary.
  block:
    check "ulp at value 5.0 (binade [4, 8)) is 2^-5", bf16UlpAt(5.0) == 0.03125
    check "ulp at value 10.0 (binade [8, 16)) is 2^-4", bf16UlpAt(10.0) == 0.0625
    check "ulp at value 17.125 (binade [16, 32)) is 2^-3", bf16UlpAt(17.125) == 0.125
    check "ulp at value 32.0 (binade [32, 64)) is 2^-2", bf16UlpAt(32.0) == 0.25
    check "bf16Ulp matches bf16UlpAt at the binade floor",
      bf16Ulp(2) == bf16UlpAt(4.0) and bf16Ulp(4) == bf16UlpAt(16.0)


  # frame contract rows: the fixture-frame layer (workspace/zstd/zstd_highlevel.nim)
  block:
    let payloadText = "{\"schema\":1,\"check\":\"frame contract\"}"
    var payload = newSeq[byte](payloadText.len)
    copyMem(payload[0].addr, payloadText[0].unsafeAddr, payloadText.len)
    let frame = zstdCompress(payload)

    func frameStr(b: seq[byte]): string =
      result = newString(b.len)
      if b.len > 0:
        copyMem(result[0].addr, b[0].unsafeAddr, b.len)

    check "frame round-trip byte-exact", zstdDecompress(frame) == payload

    let stream = zstdDecompressStream(newStringStream(frameStr(frame)))
    var evalCount = 0
    check "frame streaming round-trip byte-exact", stream.readAll() == payloadText
    check "frame stream reaches end", stream.atEnd()
    stream.close()

    var bodyCorrupted = frame
    bodyCorrupted[frame.len div 2] = bodyCorrupted[frame.len div 2] xor byte(0x5A)
    var raised = false
    try:
      discard zstdDecompress(bodyCorrupted)
    except ZstdError:
      raised = true
    check "corrupt frame body raises ZstdError", raised

    var magicCorrupted = frame
    magicCorrupted[0] = magicCorrupted[0] xor byte(0x5A)
    raised = false
    try:
      discard zstdDecompress(magicCorrupted)
    except ZstdError:
      raised = true
    check "corrupt frame magic raises ZstdError", raised

    raised = false
    try:
      discard zstdDecompressStream(
        newStringStream(frameStr(frame)[0 ..< frame.len div 3])).readAll()
    except IOError:
      raised = true
    check "truncated streaming frame raises IOError", raised

  result = allGreen

when isMainModule:
  let ok = runSelftest()
  if not ok:
    quit(1)
  echo "selftest passed"
