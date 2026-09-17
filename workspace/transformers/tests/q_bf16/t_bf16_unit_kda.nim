# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Model-free analytic suite for the KDA recurrence and its neighbors,
## running on seeded synthetic stimulus. Every value comparison reads
## a hand-spelled f64 reference or a closed form.
##
## Checks:
## - the sigmoid-gated output norm against the hand formula
## - the recurrent delta-rule kernel against hand f64 recurrences
## - the pure-decay closed form over the whole trajectory
## - the depthwise causal short conv decode window and packed forms
## - A_log checkpoint-form normalization and malformed-shape rejections
##
## Run:
##   nim cpp -d:release --stackTrace:on --debugger:native --passC:"-std=c++20" --verbosity:0 \
##     --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_unit_kda.nim

import
  std/math,
  std/strutils,
  workspace/libtorch as F,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/src/layers/norm

from workspace/libtorch/src/raw_libtorch import manual_seed

type KdaGeom = object
  ## Synthetic recurrence geometry, main binds every value and each helper
  ## reads the dimensions through the parameter.
  heads, keyDim, valueDim: int

type ConvGeom = object
  ## Synthetic conv geometry, main binds every value and the conv helper
  ## reads the dimensions through the parameter.
  channels, stateLen, kernelWidth: int

template orRaise(cond: bool; msg: string) =
  ## Suite enforcement form, cond false raises ValueError carrying msg.
  if not cond:
    raise newException(ValueError, msg)

func f32v(t: F.Tensor, i0: int): float64 =
  ## One f32 value of a rank-1 tensor as f64.
  t[i0].item(float32).float64

func f32v(t: F.Tensor, i0, i1: int): float64 =
  ## One f32 value of a rank-2 tensor as f64.
  t[i0, i1].item(float32).float64

func f32v(t: F.Tensor, i0, i1, i2: int): float64 =
  ## One f32 value of a rank-3 tensor as f64.
  t[i0, i1, i2].item(float32).float64

func f32v(t: F.Tensor, i0, i1, i2, i3: int): float64 =
  ## One f32 value of a rank-4 tensor as f64.
  t[i0, i1, i2, i3].item(float32).float64

func shapeOf(t: F.Tensor): seq[int] =
  ## Shape as a seq for equality comparisons, t.shape is an openArray view.
  for i in 0 ..< t.dim():
    result.add(t.size(i))

func f32UlpAt(v: float64): float64 =
  ## Width of one f32 ulp at magnitude v, the datatype the recurrence
  ## kernel rounds its state in and the state drift accumulates over.
  ## Precondition, v names a normal nonzero magnitude.
  pow(2.0, floor(log2(v)) - 23.0)

proc sigm(x: float64): float64 =
  ## Hand-spelled logistic sigmoid for the scalar references.
  1.0 / (1.0 + exp(-x))

proc bf16Opts(): F.TensorOptions =
  F.tensorOptions(F.kBFloat16, F.kCPU)

proc sigmoidGatedNorm(geom: KdaGeom, seed: uint64) =
  ## RmsNormGatedSigmoid against the hand formula, the norm reads x in f32
  ## through rsqrt of the mean square plus eps and the output multiplies
  ## the stored weight and the sigmoid of the gate values.
  ##
  ## Double bf16 rounding of the normed value sets the comparison band.
  Torch.manual_seed(seed)
  let weight = F.randn(geom.valueDim, bf16Opts()) * 0.2 + 1.0
  let norm = RmsNormGatedSigmoid.init(weight)
  let x = F.randn(1, 1, 1, geom.valueDim, bf16Opts()) * 0.5
  let gate = F.randn(1, 1, 1, geom.valueDim, bf16Opts()) * 0.5
  let gated = norm.forward(x, gate)
  orRaise(shapeOf(gated) == @[1, 1, 1, geom.valueDim],
    "gated norm output shape (" & $gated.size(0) & ", " & $gated.size(1) &
      ", " & $gated.size(2) & ", " & $gated.size(3) & ")")

  let x32 = x.to(F.kFloat32)
  let g32 = gate.to(F.kFloat32)
  for j in 0 ..< geom.valueDim:
    var sq = 0.0
    for c in 0 ..< geom.valueDim:
      sq += f32v(x32, 0, 0, 0, c) * f32v(x32, 0, 0, 0, c)
    let rstd = 1.0 / sqrt(sq / geom.valueDim.float64 + 1e-6)
    let normed = f32v(x32, 0, 0, 0, j) * rstd
    let expect = f32v(weight, j) * normed * sigm(f32v(g32, 0, 0, 0, j))
    let got = f32v(gated, 0, 0, 0, j)
    orRaise(abs(got - expect) <= 0.03 + 0.01 * abs(expect),
      "gated norm mismatch at channel " & $j & " norm output " & $got &
        " vs hand formula " & $expect)

proc handDeltaRule(geom: KdaGeom, q, k, v, beta: F.Tensor, steps: int,
    s0: F.Tensor): tuple[o: seq[float64], s: seq[float64], maxAbsS: float64] =
  ## Zero-decay delta rule spelled by hand in f64, one entry per output
  ## element (row-major over t, h, j) and one per state element (h, c, j),
  ## GDN math S <- S + k delta, the output read on the post-update state.
  ## maxAbsS carries the trajectory max |S|, the operating magnitude
  ## the state drift band references for its ulp.
  let q32 = q.to(F.kFloat32)
  let k32 = k.to(F.kFloat32)
  let v32 = v.to(F.kFloat32)
  var s: array[8, array[8, array[8, float64]]]
  for h in 0 ..< geom.heads:
    for c in 0 ..< geom.keyDim:
      for j in 0 ..< geom.valueDim:
        s[h][c][j] = f32v(s0, 0, h, c, j)
        result.maxAbsS = max(result.maxAbsS, abs(s[h][c][j]))
  for t in 0 ..< steps:
    for h in 0 ..< geom.heads:
      var qsq, ksq = 0.0
      for c in 0 ..< geom.keyDim:
        qsq += f32v(q32, 0, t, h, c) * f32v(q32, 0, t, h, c)
        ksq += f32v(k32, 0, t, h, c) * f32v(k32, 0, t, h, c)
      var kvMem: array[8, float64]
      for c in 0 ..< geom.keyDim:
        let kn = f32v(k32, 0, t, h, c) / sqrt(ksq + 1e-6)
        for j in 0 ..< geom.valueDim:
          kvMem[j] += s[h][c][j] * kn
      for j in 0 ..< geom.valueDim:
        let delta = (f32v(v32, 0, t, h, j) - kvMem[j]) * f32v(beta, 0, t, h)
        for c in 0 ..< geom.keyDim:
          s[h][c][j] += f32v(k32, 0, t, h, c) / sqrt(ksq + 1e-6) * delta
          result.maxAbsS = max(result.maxAbsS, abs(s[h][c][j]))
      for j in 0 ..< geom.valueDim:
        var oVal = 0.0
        for c in 0 ..< geom.keyDim:
          let qn = f32v(q32, 0, t, h, c) / sqrt(qsq + 1e-6) /
            sqrt(geom.keyDim.float64)
          oVal += qn * s[h][c][j]
        result.o.add(oVal)
  for h in 0 ..< geom.heads:
    for c in 0 ..< geom.keyDim:
      for j in 0 ..< geom.valueDim:
        result.s.add(s[h][c][j])

proc recurrentSingleStep(geom: KdaGeom, seed: uint64) =
  ## One decode step (T = 1, the batch-1 bmm form) against a hand-spelled
  ## scalar recurrence on a random initial state, decay applied before
  ## the memory read and the output read on the post-update state.
  Torch.manual_seed(seed)
  let q = F.randn(1, 1, geom.heads, geom.keyDim, bf16Opts()) * 0.5
  let k = F.randn(1, 1, geom.heads, geom.keyDim, bf16Opts()) * 0.5
  let v = F.randn(1, 1, geom.heads, geom.valueDim, bf16Opts()) * 0.5
  let g = -(F.randn(1, 1, geom.heads, geom.keyDim, F.kFloat32) * 2.0 + 0.01)
  let beta = F.sigmoid(F.randn(1, 1, geom.heads, F.kFloat32))
  let s0 = F.randn(1, geom.heads, geom.keyDim, geom.valueDim, F.kFloat32) * 0.5

  let (o, s) = gatedDeltaRuleRecurrence(perChannel, q, k, v, g, beta, s0)
  orRaise(shapeOf(o) == @[1, 1, geom.heads, geom.valueDim],
    "recurrent T=1 output shape (" & $o.size(0) & ", " & $o.size(1) & ", " &
      $o.size(2) & ", " & $o.size(3) & ")")
  orRaise(shapeOf(s) == @[1, geom.heads, geom.keyDim, geom.valueDim],
    "recurrent T=1 state shape (" & $s.size(0) & ", " & $s.size(1) & ", " &
      $s.size(2) & ", " & $s.size(3) & ")")

  # Hand recurrence in f64 from the bf16 and f32 input values.
  let q32 = q.to(F.kFloat32)
  let k32 = k.to(F.kFloat32)
  let v32 = v.to(F.kFloat32)
  var kvMem: array[8, array[8, float64]]
  var sNew: array[8, array[8, array[8, float64]]]
  for h in 0 ..< geom.heads:
    var qsq, ksq = 0.0
    for c in 0 ..< geom.keyDim:
      qsq += f32v(q32, 0, 0, h, c) * f32v(q32, 0, 0, h, c)
      ksq += f32v(k32, 0, 0, h, c) * f32v(k32, 0, 0, h, c)
    for c in 0 ..< geom.keyDim:
      let decay = exp(f32v(g, 0, 0, h, c))
      let kn = f32v(k32, 0, 0, h, c) / sqrt(ksq + 1e-6)
      for j in 0 ..< geom.valueDim:
        let sDecayed = f32v(s0, 0, h, c, j) * decay
        kvMem[h][j] += sDecayed * kn
        sNew[h][c][j] = sDecayed
    for j in 0 ..< geom.valueDim:
      let delta = (f32v(v32, 0, 0, h, j) - kvMem[h][j]) * f32v(beta, 0, 0, h)
      for c in 0 ..< geom.keyDim:
        let kn = f32v(k32, 0, 0, h, c) / sqrt(ksq + 1e-6)
        sNew[h][c][j] += kn * delta
    for j in 0 ..< geom.valueDim:
      var oExpect = 0.0
      for c in 0 ..< geom.keyDim:
        let qn = f32v(q32, 0, 0, h, c) / sqrt(qsq + 1e-6) /
          sqrt(geom.keyDim.float64)
        oExpect += qn * sNew[h][c][j]
      # o carries the bf16 output cast, the band covers one bf16 rounding.
      let got = f32v(o, 0, 0, h, j)
      orRaise(abs(got - oExpect) <= 0.02 + 0.01 * abs(oExpect),
        "recurrent o mismatch at head " & $h & " channel-out " & $j &
          " kernel output " & $got & " vs hand recurrence " & $oExpect)
      for c in 0 ..< geom.keyDim:
        orRaise(abs(f32v(s, 0, h, c, j) - sNew[h][c][j]) <= 1e-5,
          "recurrent state mismatch at (h=" & $h & ", c=" & $c & ", j=" &
            $j & ") kernel state " & $f32v(s, 0, h, c, j) &
            " vs hand recurrence " & $sNew[h][c][j])

proc noDecayDeltaRule(geom: KdaGeom, seed: uint64) =
  ## With zero decay the recurrent kernel reduces to the GDN delta rule
  ## and must match the hand-spelled f64 recurrence at T = 5 and T = 64
  ## on outputs and state.
  Torch.manual_seed(seed)
  for steps in [5, 64]:
    let q = F.randn(1, steps, geom.heads, geom.keyDim, bf16Opts()) * 0.5
    let k = F.randn(1, steps, geom.heads, geom.keyDim, bf16Opts()) * 0.5
    let v = F.randn(1, steps, geom.heads, geom.valueDim, bf16Opts()) * 0.5
    let g = F.zeros(1, steps, geom.heads, geom.keyDim, F.kFloat32)
    let beta = F.sigmoid(F.randn(1, steps, geom.heads, F.kFloat32))

    let hand = handDeltaRule(geom, q, k, v, beta, steps, F.zeros(
      1, geom.heads, geom.keyDim, geom.valueDim, F.kFloat32))
    let (oRec, sRec) = gatedDeltaRuleRecurrence(perChannel, q, k, v, g, beta, nil)

    var worstO = 0.0
    var idx = 0
    for t in 0 ..< steps:
      for h in 0 ..< geom.heads:
        for j in 0 ..< geom.valueDim:
          # bf16 output cast on the kernel output.
          let err = abs(f32v(oRec, 0, t, h, j) - hand.o[idx])
          worstO = max(worstO, err)
          orRaise(err <= 0.02 + 0.01 * abs(hand.o[idx]),
            "no-decay o mismatch at T=" & $steps & " t=" & $t & " head " &
              $h & " out " & $j & " kernel output " &
              $f32v(oRec, 0, t, h, j) & " vs hand recurrence " & $hand.o[idx])
          inc idx
    echo "    no-decay delta rule T=", steps,
      " worst hand-vs-kernel output drift ", worstO

    var worstS = 0.0
    idx = 0
    for h in 0 ..< geom.heads:
      for c in 0 ..< geom.keyDim:
        for j in 0 ..< geom.valueDim:
          worstS = max(worstS, abs(f32v(sRec, 0, h, c, j) - hand.s[idx]))
          inc idx
    echo "    no-decay delta rule T=", steps,
      " worst hand-vs-kernel state drift ", worstS
    # State drift band, derived at this binding, the kernel state
    # accumulates f32 rounding over the recurrence length while the hand
    # recurrence runs f64 over the same bf16 inputs.
    # - Operating magnitude, the hand trajectory max, one f32 ulp there
    #   comes from f32UlpAt(hand.maxAbsS) on the f32 state path.
    # - Per step, the state add rounding contributes one half-ulp unit
    #   at the operating magnitude, the kv_mem dot and the delta
    #   roundings stay under the same unit at their own magnitudes.
    # - Closed form, n steps x 1/2 ulp, rounded up to the power of two,
    #   T=5 derives 2^-22 and T=64 derives 2^-19.
    let stepBound = steps.float64 * 0.5 * f32UlpAt(hand.maxAbsS)
    let stateBand = pow(2.0, ceil(log2(stepBound)))
    orRaise(worstS <= stateBand,
      "no-decay state drift " & $worstS & " past the derived band " &
        $stateBand & " at T=" & $steps)

proc pureDecayClosedForm(geom: KdaGeom, seed: uint64) =
  ## beta = 0 zeroes the delta correction, state_T = state_0 * exp(cum_g)
  ## elementwise and o_t the scaled-query read of state_0 * exp(cum_t),
  ## one closed form over the whole trajectory at T = 5 and T = 65.
  Torch.manual_seed(seed)
  for steps in [5, 65]:
    let q = F.randn(1, steps, geom.heads, geom.keyDim, bf16Opts()) * 0.5
    let k = F.randn(1, steps, geom.heads, geom.keyDim, bf16Opts()) * 0.5
    let v = F.randn(1, steps, geom.heads, geom.valueDim, bf16Opts()) * 0.5
    # Small |g| keeps exp(cum) away from f32 underflow over 65 steps.
    let g = -(F.randn(1, steps, geom.heads, geom.keyDim, F.kFloat32) * 0.02)
    let beta = F.zeros(1, steps, geom.heads, F.kFloat32)
    let s0 = F.randn(1, geom.heads, geom.keyDim, geom.valueDim, F.kFloat32) * 0.5

    let (oRec, sRec) = gatedDeltaRuleRecurrence(perChannel, q, k, v, g, beta, s0)

    let q32 = q.to(F.kFloat32)
    for t in 0 ..< steps:
      for h in 0 ..< geom.heads:
        var qsq = 0.0
        for c in 0 ..< geom.keyDim:
          qsq += f32v(q32, 0, t, h, c) * f32v(q32, 0, t, h, c)
        for j in 0 ..< geom.valueDim:
          var oExpect = 0.0
          for c in 0 ..< geom.keyDim:
            var cum = 0.0
            for u in 0 .. t:
              cum += f32v(g, 0, u, h, c)
            let qn = f32v(q32, 0, t, h, c) / sqrt(qsq + 1e-6) /
              sqrt(geom.keyDim.float64)
            oExpect += qn * f32v(s0, 0, h, c, j) * exp(cum)
          let got = f32v(oRec, 0, t, h, j)
          orRaise(abs(got - oExpect) <= 0.02 + 0.01 * abs(oExpect),
            "pure-decay o mismatch at T=" & $steps & " t=" & $t & " head " &
              $h & " out " & $j & " kernel output " & $got &
              " vs closed form " & $oExpect)

    for h in 0 ..< geom.heads:
      for c in 0 ..< geom.keyDim:
        var cum = 0.0
        for u in 0 ..< steps:
          cum += f32v(g, 0, u, h, c)
        for j in 0 ..< geom.valueDim:
          let expect = f32v(s0, 0, h, c, j) * exp(cum)
          let got = f32v(sRec, 0, h, c, j)
          orRaise(abs(got - expect) <= 1e-5 + 1e-5 * abs(expect),
            "pure-decay state mismatch at T=" & $steps & " (h=" & $h &
              ", c=" & $c & ", j=" & $j & ") kernel state " & $got &
              " vs closed form " & $expect)
    echo "    pure-decay closed form checked at T=", steps

proc shortConvWindow(g: ConvGeom, seed: uint64) =
  ## Decode step of the depthwise causal conv spelled by hand, silu applied
  ## over the f32 dot of (history, x), the new state drops the oldest entry
  ## and appends x.
  Torch.manual_seed(seed)
  let w = F.randn(g.channels, 1, g.kernelWidth, bf16Opts()) * 0.4
  let state = F.randn(g.channels, g.stateLen, bf16Opts()) * 0.5
  let x = F.randn(1, g.channels, 1, bf16Opts()) * 0.5
  let (convOut, newState) = shortConvStep(x, w, state)
  orRaise(shapeOf(convOut) == @[1, g.channels, 1],
    "conv step output shape (" & $convOut.size(0) & ", " & $convOut.size(1) &
      ", " & $convOut.size(2) & ")")
  orRaise(shapeOf(newState) == @[g.channels, g.stateLen],
    "conv step state shape (" & $newState.size(0) & ", " &
      $newState.size(1) & ")")
  let w32 = w.to(F.kFloat32)
  let st32 = state.to(F.kFloat32)
  let x32 = x.to(F.kFloat32)
  for ch in 0 ..< g.channels:
    var acc = 0.0
    for i in 0 ..< g.stateLen:
      acc += f32v(st32, ch, i) * f32v(w32, ch, 0, i)
    acc += f32v(x32, 0, ch, 0) * f32v(w32, ch, 0, g.kernelWidth - 1)
    let expect = acc * sigm(acc)
    let got = f32v(convOut, 0, ch, 0)
    # Conv output rounds to bf16 before silu, the comparison band covers
    # the quantization (about 2 bf16 ulps at this magnitude) of the f64 dot.
    orRaise(abs(got - expect) <= 2e-3 + 1e-3 * abs(expect),
      "conv step mismatch at channel " & $ch & " conv output " & $got &
        " vs hand dot-silu " & $expect)
    # State window, drop oldest, append x.
    for i in 0 ..< g.stateLen - 1:
      orRaise(newState[ch, i].item(float32) == state[ch, i + 1].item(float32),
        "conv state shift mismatch at channel " & $ch & " slot " & $i)
    orRaise(
      newState[ch, g.stateLen - 1].item(float32) ==
        x[0, ch, 0].item(float32),
      "conv state append mismatch at channel " & $ch)

proc shortConvPerBranchPacked(seed: uint64) =
  ## Depthwise conv is channel-independent, so the packed fused weight
  ## moves the same data as the per-branch weights on outputs and states
  ## and across the decode continuation, over the concatenated qkv stimulus.
  Torch.manual_seed(seed)
  let dq = 6
  let dvv = 4
  let wq = F.randn(dq, 1, 4, bf16Opts()) * 0.4
  let wk = F.randn(dq, 1, 4, bf16Opts()) * 0.4
  let wv = F.randn(dvv, 1, 4, bf16Opts()) * 0.4
  let fused = F.cat([wq, wk, wv], axis = 0)
  let xq = F.randn(1, dq, 5, bf16Opts()) * 0.5
  let xk = F.randn(1, dq, 5, bf16Opts()) * 0.5
  let xv = F.randn(1, dvv, 5, bf16Opts()) * 0.5
  let xcat = F.cat([xq, xk, xv], axis = 1)

  let (oQ, sQ) = shortConvSequence(xq, wq, nil)
  let (oK, sK) = shortConvSequence(xk, wk, nil)
  let (oV, sV) = shortConvSequence(xv, wv, nil)
  let (oCat, sCat) = shortConvSequence(xcat, fused, nil)

  orRaise(oCat.narrow(1, 0, dq).equal(oQ),
    "packed q branch output diverged from the per-branch conv")
  orRaise(oCat.narrow(1, dq, dq).equal(oK),
    "packed k branch output diverged from the per-branch conv")
  orRaise(oCat.narrow(1, 2 * dq, dvv).equal(oV),
    "packed v branch output diverged from the per-branch conv")
  let sPack = F.cat([sQ, sK, sV], axis = 0)
  orRaise(sCat.equal(sPack),
    "packed conv state diverged from the per-branch states")

  # Decode continuation on the stored states, one token per form.
  let (stepQ, stepStateQ) = shortConvStep(xq.narrow(2, 4, 1), wq, sQ)
  let (stepCat, stepStateCat) = shortConvStep(xcat.narrow(2, 4, 1), fused, sCat)
  orRaise(stepCat.narrow(1, 0, dq).equal(stepQ),
    "packed decode continuation q branch diverged from the per-branch step")
  orRaise(stepStateCat.narrow(0, 0, dq).equal(stepStateQ),
    "packed decode continuation q state diverged from the per-branch state")

proc aLogForms(geom: KdaGeom, seed: uint64) =
  ## A_log arrives as flat (heads), (heads, 1) or the rank-4 form
  ## (1, 1, heads, 1), all three normalize to one (heads, 1) f32 tensor
  ## with identical values, malformed shapes raise naming the key path.
  Torch.manual_seed(seed)
  let aFlat = F.randn(geom.heads, F.kFloat32) * 0.2
  let aPair = aFlat.reshape([geom.heads, 1])
  let aRank4 = aFlat.reshape([1, 1, geom.heads, 1])

  let aFlatN = normalizeAGateLog("flat", aFlat, geom.heads)
  let aPairN = normalizeAGateLog("pair", aPair, geom.heads)
  let aRank4N = normalizeAGateLog("rank4", aRank4, geom.heads)
  orRaise(shapeOf(aFlatN) == @[geom.heads, 1],
    "flat A_log normalized to shape (" & $aFlatN.size(0) & ", " &
      $aFlatN.size(1) & ")")
  orRaise(aPairN.equal(aFlatN),
    "the (heads, 1) form normalized to different values than the flat form")
  orRaise(aRank4N.equal(aFlatN),
    "the rank-4 form normalized to different values than the flat form")

  # Rank-3 and a wrong rank-4 shape raise, naming the key path.
  let bad3 = aFlat.reshape([1, geom.heads, 1])
  var sawKey = false
  try:
    let sink = normalizeAGateLog("bad3", bad3, geom.heads)
  except ValueError as err:
    sawKey = "bad3" in err.msg
  orRaise(sawKey,
    "rank-3 A_log rejection must name the key path")

  let bad4 = aFlat.reshape([1, geom.heads, 1, 1])
  var refused4 = false
  try:
    let sink = normalizeAGateLog("bad4", bad4, geom.heads)
  except ValueError:
    refused4 = true
  orRaise(refused4, "malformed rank-4 A_log accepted")

proc main() =
  let geom = KdaGeom(heads: 2, keyDim: 4, valueDim: 4)

  sigmoidGatedNorm(geom, seed = 53'u64)
  recurrentSingleStep(geom, seed = 19'u64)
  noDecayDeltaRule(geom, seed = 23'u64)
  pureDecayClosedForm(geom, seed = 29'u64)
  let convGeom = ConvGeom(channels: 2, stateLen: 3, kernelWidth: 4)
  shortConvWindow(convGeom, seed = 37'u64)
  shortConvPerBranchPacked(seed = 39'u64)
  aLogForms(geom, seed = 17'u64)

  # Out of scope for this suite, the decay formulas and the mixer composition.
  # Their projections are checkpoint layers, only the deserialization loaders
  # build those over real weights, synthetic stimulus carries none.

when isMainModule:
  main()
