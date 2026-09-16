# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Selftest of tests/harness/harness.nim.
##
## Every case drives the real exported API and reads only outcomes:
## - accept, the call returns
## - reject, HarnessCheckError carries the expected message fragment
## - allowance math, deriveBands checked against hand-derived values
##
## Run through the test_tf_harness_selftest task in config.nims.

import
  std/importutils,
  std/os,
  std/sequtils,
  std/strutils,
  workspace/libtorch as F,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/tests/harness/harness {.all.}

privateAccess(StatsRecord)
privateAccess(ArgmaxRecord)

# ============================ Hand-derived allowances ============================

# deriveBands(2, 30.0) gives ulpBand = ceil(2 x sqrt(1) x 1) = 2, delta = 2 x 0.125 = 0.25,
# klBand = 0.5 x 0.0625 = 0.03125.
# deriveBands(4, 30.0) gives ulpBand = 4, delta = 0.5, klBand = 0.125.
# deriveBands(2, 30.0, depth = 2) gives ulpBand = ceil(2 x sqrt(2)) = 3, delta = 3 x 0.125 = 0.375,
# klBand = 0.5 x 0.140625 = 0.0703125.
# deriveBands(4, 30.0, depth = 2) gives ulpBand = ceil(4 x sqrt(2)) = 6, delta = 6 x 0.125 = 0.75,
# klBand = 0.5 x 0.5625 = 0.28125.
# The depth scaling is the root-sum-square (RSS) accumulation of independent
# per-stage reordering errors, ~ sqrt(depth) x the per-stage allowance, not
# the worst-case linear depth x it.

# ============================ Stimulus tensors ============================

# Base value pattern, 16.0 + 0.125 x (i mod 112), n = 4096.
# - every value is exactly bf16-representable (binade 4, step 0.125)
# - max magnitude 29.875, binade 4, one bf16 ulp 0.125
# - quantile ranks, p01 -> index 40, p50 -> 2047, p99 -> 4054

proc main() =
  proc demand(cond: bool, what: string) =
    if not cond:
      raise newException(AssertionDefect, what)

  proc expectReject(action: proc(), fragment: string, what: string) =
    var caught = false
    try:
      action()
    except HarnessCheckError as e:
      caught = true
      if fragment notin e.msg:
        raise newException(AssertionDefect,
          what & ": message '" & e.msg & "' lacks '" & fragment & "'")
    if not caught:
      raise newException(AssertionDefect, what & ": no HarnessCheckError raised")

  var baseSeq = newSeq[float32](4096)
  for i in 0 ..< 4096:
    baseSeq[i] = 16.0'f32 + 0.125'f32 * float32(i mod 112)

  proc perturb(offsets: seq[int], by: float32): seq[float32] =
    result = baseSeq
    for i in offsets:
      result[i] = result[i] + by

  proc constSeq(v: float32): seq[float32] =
    newSeq(result, 4096)
    for i in 0 ..< 4096:
      result[i] = v

  proc maxOffsets(): seq[int] =
    # Elements carrying the maximum value (i mod 112 == 111), 36 of them.
    result = toSeq(0 ..< 4096).filterIt(it mod 112 == 111)

  let baseF32 = F.toTensor(baseSeq)
  let baseBf16 = baseF32.to(F.kBfloat16)
  let constBf16 = F.toTensor(constSeq(30.0'f32)).to(F.kBfloat16)

  # ============================ Band math ============================

  block:
    let b = deriveBands(4, 30.0)
    demand(b.ulpBand == 4 and b.delta == 0.5 and b.klBand == 0.125,
      "deriveBands(4, 30.0) must be (4, 0.5, 0.125), got " & $b)
    let b2 = deriveBands(2, 30.0)
    demand(b2.ulpBand == 2 and b2.delta == 0.25 and b2.klBand == 0.03125,
      "deriveBands(2, 30.0) must be (2, 0.25, 0.03125), got " & $b2)
    let bd2 = deriveBands(4, 30.0, depth = 2)
    demand(bd2.ulpBand == 6 and bd2.delta == 0.75 and bd2.klBand == 0.28125,
      "deriveBands(4, 30.0, depth 2) must be (6, 0.75, 0.28125), got " & $bd2)
    var ampRaised = false
    try:
      discard deriveBands(2, 30.0, coarseAmplification = 0.5)
    except ValueError:
      ampRaised = true
    demand(ampRaised, "coarse amplification below 1.0 must raise")

  # ============================ Stats accept cases ============================

  let recBase = baseBf16.loadUniformStats("base")
  demand(recBase.ulpDatatype == ulpBf16,
    "record shape: ulp datatype " & ulpDatatypeName(recBase.ulpDatatype))

  block:
    # Round-trip case, the replay recomputes identical instruments, every
    # drift 0 <= ulpBand 2.
    harnessStats(baseBf16, recBase, kElementwise)
    harnessStats(baseBf16, recBase, kReduction)
    harnessStats(baseBf16, recBase, kReduction, depth = 2)

  block:
    # Constant tensor case, one histogram bucket, every element sits above
    # the tail threshold (recorded max / 16 = 1.875), so the tail
    # probability is 1.0 with tailEdge 0.
    let recConst = constBf16.loadUniformStats("const")
    demand(recConst.histKeys.len == 1 and recConst.tailProbability == 1.0 and
      recConst.tailEdge == 0,
      "constant record: buckets " & $recConst.histKeys.len & " tail " &
      $recConst.tailProbability)
    harnessStats(constBf16, recConst, kElementwise)

  block:
    # Depth-2 band edge case, the max quantile sits 3 steps up, 0.375 = delta.
    # The comparison is strict, so the edge itself passes.
    let up3 = F.toTensor(perturb(maxOffsets(), 0.375'f32)).to(F.kBfloat16)
    harnessStats(up3, recBase, kElementwise, depth = 2)

  # ============================ Stats reject cases ============================

  block:
    # Element count guard fires before any band math.
    let short = F.toTensor(baseSeq[0 ..< 2048]).to(F.kBfloat16)
    expectReject(proc() = harnessStats(short, recBase, kElementwise),
      "element count", "count mismatch")

  block:
    # Datatype guard case, same values recorded on a different ulp
    # datatype than the replay computes.
    expectReject(proc() = harnessStats(baseF32, recBase, kElementwise),
      "ulp datatype mismatch", "ulp datatype fault")

  block:
    # Depth-1 quantile instrument reads grid steps, max up 8 steps
    # against ulpBand 2. The harness reports the absolute drift
    # and the derived band. It does not report a grid-step count.
    let up8 = F.toTensor(perturb(maxOffsets(), 1.0'f32)).to(F.kBfloat16)
    expectReject(proc() = harnessStats(up8, recBase, kElementwise),
      "exceeds band", "quantile step fault")

  block:
    # Depth-2 quantile instrument reads absolute scale, max up 6 steps
    # = 0.75 against delta 0.5.
    let up6 = F.toTensor(perturb(maxOffsets(), 0.75'f32)).to(F.kBfloat16)
    expectReject(proc() = harnessStats(up6, recBase, kElementwise, depth = 2),
      "exceeds band", "quantile absolute fault")



  # Same-id record, id0 = 30.0, ids 1..31 = 20.0, off-set = 0.0,
  # V = 128. margin = 10, allowance 4: delta 0.5, klBand 0.125.
  let rowSeq = (block:
    var s = newSeq[float32](128)
    s[0] = 30.0'f32
    for i in 1 ..< 128:
      s[i] = if i <= 31: 20.0'f32 else: 0.0'f32
    s)
  let rowA = F.toTensor(rowSeq)
  let topKIds = toSeq(0 ..< 32)
  var recA = ArgmaxRecord(
    argmaxId: 0,
    topK: topKIds,
    topKLogits: topKIds.mapIt(rowSeq[it]),
    margin: 10.0,
    tailProbability: observedTailProbability(rowA, topKIds),
    kind: kReduction)

  block:
    var r = recA
    var flipCount = 0
    checkArgmaxRow(rowA, r, flipCount)
    demand(flipCount == 0, "same-id accept must not count a flip")

  block:
    # One recorded top-32 id drifts 1.5, twelve delta steps.
    # The per-id value band is retired, the top-32 distribution
    # instrument is the truncated KL, and a single mid-table id move
    # of 1.5 sits far under klBand 0.125, so the step accepts and no
    # flip is counted.
    var driftSeq = rowSeq
    driftSeq[5] = 21.5'f32
    var r = recA
    var flipCount = 0
    checkArgmaxRow(F.toTensor(driftSeq), r, flipCount)
    demand(flipCount == 0, "a per-id drift inside the KL band must not count a flip")

  block:
    # Distribution fault, one recorded top-32 id drifts 8.5, id 5
    # moves to 28.5, the truncated KL reaches 0.2 against klBand 0.125.
    var klSeq = rowSeq
    klSeq[5] = 28.5'f32
    var r = recA
    var flipCount = 0
    expectReject(proc() = checkArgmaxRow(F.toTensor(klSeq), r, flipCount),
      "truncated KL", "top-32 distribution fault")

  block:
    # Tail fault, 8 off-set logits 0 -> 21.5, observed tail moves
    # 1.6e-3 against limit 4e-4, top-32 and KL instruments clean.
    var tailSeq = rowSeq
    for i in 32 ..< 40:
      tailSeq[i] = 21.5'f32
    var r = recA
    var flipCount = 0
    expectReject(proc() = checkArgmaxRow(F.toTensor(tailSeq), r, flipCount),
      "tail probability", "tail fault")

  block:
    # KL allowance tightness:
    # - the delta^2 bound is not loose, the worst top-32 shape keeping
    #   every per-id drift at delta, the vertex split 13 up / 19 down
    #   by 0.5 over a flat record, gives KL 0.12323, under klBand 0.125
    #   by under 2 percent
    let flat = newSeqWith(32, 30.0'f32)
    var mixed = newSeq[float32](32)
    for i in 0 ..< 32:
      mixed[i] = if i < 13: 30.5'f32 else: 29.5'f32
    let kl = truncatedKl(flat, mixed)
    demand(kl <= 0.125 and kl > 0.1,
      "KL bound observation: got " & $kl)

  block:
    # Tie flip, recorded pick id3 = 30.0 against id7 = 29.9375,
    # margin 0.0625 <= one bf16 ulp at 30.0 (0.125). The replay swaps
    # the two logits, the flip counts and passes, and the fifth call
    # passes the cap of 4.
    var recRow = rowSeq
    recRow[3] = 30.0'f32
    recRow[7] = 29.9375'f32
    recRow[0] = 20.0'f32
    let tieIds = @[3, 7] & toSeq(0 ..< 32).filterIt(it != 3 and it != 7)
    var recTie = ArgmaxRecord(
      argmaxId: 3,
      topK: tieIds,
      topKLogits: tieIds.mapIt(recRow[it]),
      margin: 0.0625,
      tailProbability: observedTailProbability(F.toTensor(recRow), tieIds),
      kind: kReduction)
    var flipRow = recRow
    flipRow[3] = 29.9375'f32
    flipRow[7] = 30.0'f32
    var flipCount = 0
    for call in 1 ..< 5:
      checkArgmaxRow(F.toTensor(flipRow), recTie, flipCount)
    demand(flipCount == 4, "four flips must be counted, got " & $flipCount)
    expectReject(proc() = checkArgmaxRow(F.toTensor(flipRow), recTie, flipCount),
      "tie-flip cap", "tie cap overrun")

  block:
    # Real divergence, margin 2.0 is not tie-eligible, the replay picks
    # an off-set id at 29.5 while the recorded top-1 fell to 26.0.
    var divRow = rowSeq
    divRow[3] = 30.0'f32
    divRow[0] = 20.0'f32
    divRow[1] = 28.0'f32
    let divIds = @[3, 1] & toSeq(0 ..< 32).filterIt(it != 1 and it != 3)
    var recDiv = ArgmaxRecord(
      argmaxId: 3,
      topK: divIds,
      topKLogits: divIds.mapIt(divRow[it]),
      margin: 2.0,
      tailProbability: observedTailProbability(F.toTensor(divRow), divIds),
      kind: kReduction)
    # The record is built from the observed row, every instrument passes
    # and the pick alone diverges, recorded 3 against observed pick 1.
    var obsRow = divRow
    obsRow[3] = 26.0'f32
    recDiv = ArgmaxRecord(
      argmaxId: 3,
      topK: divIds,
      topKLogits: divIds.mapIt(obsRow[it]),
      margin: 2.0,
      tailProbability: observedTailProbability(F.toTensor(obsRow), divIds),
      kind: kReduction)
    var flipCount = 0
    expectReject(proc() = checkArgmaxRow(F.toTensor(obsRow), recDiv, flipCount),
      "argmax divergence", "real divergence")

  block:
    # The tie-flip cap through the public path form, the flip counts
    # survive between calls
    # - five tie flips breach the cap of four, the fifth call raises
    # - with a fresh record per call the cap never fires
    # - this case fails against that regression
    var flipSeq = newSeq[float32](128)
    flipSeq[5] = 20.0'f32
    flipSeq[9] = 19.875'f32
    let flipRow = F.toTensor(flipSeq)
    let flipIds = @[5, 9] & toSeq(0 ..< 32).filterIt(it != 5 and it != 9)
    let flipLogits = @[20.0'f32, 19.875'f32] & toSeq(0 ..< 30).mapIt(0.0'f32)
    let tailHex = "0x" & toHex(cast[uint64](
      observedTailProbability(flipRow, flipIds)), 16)
    let idsCsv = flipIds.mapIt($it).join(",")
    let bitsHex = flipLogits.mapIt("0x" & toHex(cast[uint32](it), 8)).join(" ")
    var steps = ""
    for si in 0 ..< 5:
      if si > 0: steps.add ","
      steps.add("{\"argmax_id\":5,\"margin\":\"0x3fc0000000000000\"," &
        "\"tail_probability\":\"" & tailHex & "\"," &
        "\"top_k\":[" & idsCsv & "],\"top_k_logits\":\"" &
        bitsHex & "\"}")
    let frameJson = "{\"schema\":\"ttt-tf-005-argmax-decisions\"," &
      "\"source\":\"selftest-flip-cap\",\"ulp_datatype\":\"bf16\",\"steps\":[" & steps & "]}"
    let framePath = getTempDir() / "selftest_flip_cap.json.zst"
    writeFile(framePath, zstdCompress(frameJson, string))
    # The observed row swaps the near-tie pair, the pick flips every step.
    var obsFlip = flipSeq
    obsFlip[5] = 19.875'f32
    obsFlip[9] = 20.0'f32
    let obsRow = F.toTensor(obsFlip)
    var flipCount = 0
    var capFired = false
    for step in 0 ..< 5:
      try:
        assertArgMax(obsRow, framePath, step, kReduction, flipCount)
      except HarnessCheckError as e:
        if step == 4 and "tie-flip cap" in e.msg:
          capFired = true
        else:
          raise newException(AssertionDefect,
            "flip chain: unexpected rejection at step " & $step & ": " & e.msg)
    demand(capFired, "the flip cap must fire on the fifth flip")


  block:
    # Serialized-allowance closure case, check-time allowance derivation:
    # - the frame serializes a wildly loose allowance (4096), honoring
    #   it would widen delta past 512 and the KL band past 131072
    # - the observed row drifts one top-32 id by 8.5 (id 5 -> 28.5),
    #   the derived delta (allowance 4, one bf16 ulp at 30.0 = 0.125)
    #   is 0.5, klBand 0.125
    # - the check must reject against the derived value
    var row = rowSeq
    var driftRow = rowSeq
    driftRow[5] = 28.5'f32
    let tailHex = "0x" & toHex(cast[uint64](
      observedTailProbability(F.toTensor(row), topKIds)), 16)
    let idsCsv = topKIds.mapIt($it).join(",")
    let bitsHex = topKIds.mapIt(
      "0x" & toHex(cast[uint32](rowSeq[it]), 8)).join(" ")
    let stepJson = "{\"argmax_id\":0,\"margin\":\"0x4024000000000000\"," &
      "\"tail_probability\":\"" & tailHex & "\",\"top_k\":[" &
      idsCsv & "],\"top_k_logits\":\"" & bitsHex &
      "\",\"ulp_drift_allowance\":4096}"
    let frameJson = "{\"schema\":\"ttt-tf-005-argmax-decisions\"," &
      "\"source\":\"selftest-allowance-closure\",\"ulp_datatype\":\"bf16\"," &
      "\"steps\":[" & stepJson & "]}"
    let framePath = getTempDir() / "selftest_allowance_closure.json.zst"
    writeFile(framePath, zstdCompress(frameJson, string))
    var flipCount = 0
    expectReject(proc() = assertArgMax(F.toTensor(driftRow), framePath, 0,
      kReduction, flipCount),
      "truncated KL", "a loose serialized allowance must not widen delta")

  block:
    # A NaN logit rejects before the instruments, ordered comparisons
    # would read NaN as no drift and accept silently.
    var nanSeq = rowSeq
    nanSeq[40] = NaN
    var r = recA
    var flipCount = 0
    expectReject(proc() = checkArgmaxRow(F.toTensor(nanSeq), r, flipCount),
      "non-finite", "NaN logits reject")

  block:
    echo "t_harness_selftest: all cases green"


when isMainModule:
  main()
