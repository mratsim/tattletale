# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Model-free analytic suite for the MoE router forms over synthetic weights.
## Seeded inputs and closed-form margins, no checkpoint and no recorded frame.
##
## Checks:
## - decision shape and dtype contract on both router forms
## - renormalized weight sum against the routed scaling factor
## - greedy weights against the scaled softmax gather
## - bias steering with unbiased weights and the degenerate-grouping identity
## - construction guards on both forms
##
## Run:
##   nim cpp -d:release --stackTrace:on --debugger:native --passC:"-std=c++20" --verbosity:0 \
##     --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_unit_router.nim

import
  std/importutils,
  workspace/libtorch as F,
  workspace/transformers/src/layers/moe_router {.all.}

from workspace/libtorch/src/raw_libtorch import manual_seed

{.experimental: "callOperator".}

privateAccess(NoauxTcRouter)

type RouterGeom = object
  ## Synthetic geometry and routing constants, main binds every value.
  ## Each helper reads the dimensions through the parameter.
  numExperts, hidden, numGroup, topkGroup, topK, numTokens: int
  scale: float64

template orRaise(cond: bool; msg: string) =
  ## Suite enforcement form, cond false raises ValueError carrying msg.
  if not cond:
    raise newException(ValueError, msg)

template mustRefuse(why: string; body: untyped) =
  ## Construction under test must reject its input, the body raises
  ## ValueError and any silent outcome raises with why.
  var fired = false
  try:
    body
  except ValueError:
    fired = true
  if not fired:
    raise newException(ValueError, "no rejection: " & why)

func bf16Opts(): F.TensorOptions =
  F.tensorOptions(F.kBFloat16, F.kCPU)

func f32Opts(): F.TensorOptions =
  F.tensorOptions(F.kFloat32, F.kCPU)

func maxDrift(a, b: F.Tensor): float64 =
  ## Largest absolute element difference of two tensors compared as f32,
  ## the drift quantity of every value comparison in this suite.
  let diff = (a.to(F.kFloat32) - b.to(F.kFloat32)).abs()
  result = diff.max().item(float32).float64

proc makeRouters(geo: RouterGeom): tuple[noaux, degenerate: NoauxTcRouter,
    greedy: GreedyRouter] =
  ## Router trio on one synthetic weight table, the grouped noaux_tc form,
  ## a degenerate noaux form sharing the weights and the greedy form.
  let weight = F.randn(geo.numExperts, geo.hidden, bf16Opts())
  let bias = F.randn(geo.numExperts, f32Opts()) * 0.05
  result.noaux = NoauxTcRouter.init(
    weight, bias, geo.topK, geo.numGroup, geo.topkGroup, geo.scale, true)
  result.degenerate = NoauxTcRouter.init(
    weight, bias, geo.topK, 1, 1, geo.scale, true)
  result.greedy = GreedyRouter.init(weight, geo.topK, geo.scale)

proc makeHidden(geo: RouterGeom, rows: int, seed: uint64): F.Tensor =
  ## Seeded stimulus rows over the hidden width, one table per call.
  Torch.manual_seed(seed)
  F.randn(rows, geo.hidden, bf16Opts())

proc choiceMargin(choice: F.Tensor, topK: int): float32 =
  ## Gap between the top-k boundary neighbors of the sorted choice values,
  ## the margin condition that keeps every selection comparison unambiguous.
  let sorted = choice.sort(-1, descending = true).values
  let above = sorted.narrow(-1, topK - 1, 1)
  let below = sorted.narrow(-1, topK, 1)
  result = (above - below).min().item(float32)

proc expectDecisionContract(geo: RouterGeom,
    router: NoauxTcRouter | GreedyRouter, form: string, seed: uint64) =
  ## One router form against the decision contract, f32 logits over
  ## (tokens, experts), f32 weights and int64 indices over (tokens, topk),
  ## and the decode form over one hidden row returns outputs sized (1, ...).
  let hidden = makeHidden(geo, geo.numTokens, seed)
  let decision = router.route(hidden)
  let logits = decision.logits
  let weights = decision.weights
  let indices = decision.indices
  orRaise(logits.dim() == 2 and logits.size(0) == geo.numTokens and
      logits.size(1) == geo.numExperts,
    form & " logits shape (" & $logits.size(0) & ", " & $logits.size(1) & ")")
  orRaise(weights.dim() == 2 and weights.size(0) == geo.numTokens and
      weights.size(1) == geo.topK,
    form & " weights shape (" & $weights.size(0) & ", " & $weights.size(1) & ")")
  orRaise(indices.dim() == 2 and indices.size(0) == geo.numTokens and
      indices.size(1) == geo.topK,
    form & " indices shape (" & $indices.size(0) & ", " & $indices.size(1) & ")")
  orRaise(logits.scalarType() == F.kFloat32, form & " logits dtype")
  orRaise(weights.scalarType() == F.kFloat32, form & " weights dtype")
  orRaise(indices.scalarType() == F.kInt64, form & " indices dtype")

  let decode = router.routeDecode(hidden.narrow(0, 0, 1))
  orRaise(decode.logits.size(0) == 1 and decode.logits.size(1) == geo.numExperts,
    form & " decode logits shape (" & $decode.logits.size(0) & ", " &
      $decode.logits.size(1) & ")")
  orRaise(decode.weights.size(0) == 1 and decode.weights.size(1) == geo.topK,
    form & " decode weights shape (" & $decode.weights.size(0) & ", " &
      $decode.weights.size(1) & ")")
  orRaise(decode.indices.size(0) == 1 and decode.indices.size(1) == geo.topK,
    form & " decode indices shape (" & $decode.indices.size(0) & ", " &
      $decode.indices.size(1) & ")")

proc decisionContract(geo: RouterGeom, seed: uint64) =
  ## Both router forms against the decision contract, one hidden stimulus
  ## table per form.
  let routers = makeRouters(geo)
  expectDecisionContract(geo, routers.noaux, "noaux_tc", seed)
  expectDecisionContract(geo, routers.greedy, "greedy", seed)

proc renormSumIdentity(geo: RouterGeom, seed: uint64) =
  ## Renormalized top-k weights sum to the routed scaling factor on every
  ## row under the gathered-sum identity. The slack covers the f32 divide
  ## rounding and the 1e-20 floor term inside the divisor.
  let routers = makeRouters(geo)
  let hidden = makeHidden(geo, geo.numTokens, seed)
  let decision = routers.noaux.route(hidden)
  let sums = decision.weights.sum(axis = -1)
  let worst = (sums - geo.scale).abs().max().item(float32)
  orRaise(worst < 1e-6'f32,
    "renormalized weight sum vs routed scaling factor drift " & $worst)

proc greedySoftmaxIdentity(geo: RouterGeom, seed: uint64) =
  ## Greedy route weights equal the softmax scores at the returned indices
  ## times the scale, recomputed from the returned logits over the same
  ## device and op sequence.
  let routers = makeRouters(geo)
  let hidden = makeHidden(geo, geo.numTokens, seed)
  let decision = routers.greedy.route(hidden)
  let scores = F.softmax(decision.logits, -1)
  let gathered = scores.gather(1, decision.indices) * geo.scale
  let drift = maxDrift(decision.weights, gathered)
  orRaise(drift == 0.0,
    "greedy route weights vs scaled softmax gather drift " & $drift)

proc biasSteersSelection(geo: RouterGeom, seed: uint64) =
  ## One positive mid-table bias value steers the selection of the boosted
  ## expert on every row, while the weights stay the unbiased sigmoid
  ## scores and the bias never reaches the gathered renormalized weights.
  Torch.manual_seed(seed)
  let weight = F.randn(geo.numExperts, geo.hidden, bf16Opts())
  let hidden = F.randn(geo.numTokens, geo.hidden, bf16Opts())
  let scores = F.sigmoid(F.matmul(hidden.to(F.kFloat32),
    weight.to(F.kFloat32).t()))

  # One expert-wide bias value sized past the boundary on every row,
  # the kth score minus the boosted expert score plus headroom.
  var boost = 0.0'f32
  for tok in 0 ..< geo.numTokens:
    let row = scores[tok]
    let sortedRow = row.sort(0, descending = true).values
    let kth = sortedRow[geo.topK - 1].item(float32)
    let target = row[geo.topK + 3].item(float32)
    boost = max(boost, kth - target)
  boost = boost + 0.01'f32

  var biasVals = newSeq[float32](geo.numExperts)
  biasVals[geo.topK + 3] = boost
  let bias = F.toTensor(biasVals).to(F.kFloat32)
  let steered = NoauxTcRouter.init(
    weight, bias, geo.topK, 1, 1, geo.scale, true)
  let decision = steered.route(hidden)

  let margin = choiceMargin(F.sigmoid(decision.logits) + bias, geo.topK)
  orRaise(margin > 0.0'f32,
    "boosted choice boundary margin " & $margin)

  let picked = scores.gather(1, decision.indices)
  let expected = picked /
    (picked.sum(axis = -1, keepdim = true) + 1e-20) * geo.scale
  let drift = maxDrift(decision.weights, expected)
  orRaise(drift == 0.0,
    "biased-selection route weights vs unbiased sigmoid renorm drift " & $drift)

  let pickedIds = decision.indices
  var found = 0
  for tok in 0 ..< geo.numTokens:
    for pos in 0 ..< geo.topK:
      if pickedIds[tok, pos].item(int64) == geo.topK + 3:
        inc found
  orRaise(found == geo.numTokens,
    "boosted expert selected in " & $found & " of " &
      $geo.numTokens & " rows")

proc degenerateIdentity(geo: RouterGeom, seed: uint64) =
  ## Degenerate grouping (one group, one winning group) selects the plain
  ## biased top-k exactly, the single group always wins, the mask stays
  ## all-ones and the selection sees unmasked scores.
  ##
  ## The seed loop skips any stimulus whose boundary ties would cloud
  ## the selection comparison.
  let routers = makeRouters(geo)
  let bias = routers.noaux.expertBias
  var runSeed = seed
  var hidden = makeHidden(geo, geo.numTokens, runSeed)
  var decision = routers.degenerate.route(hidden)
  var margin = choiceMargin(F.sigmoid(decision.logits) + bias, geo.topK)
  while margin <= 0.0'f32:
    runSeed = runSeed + 1'u64
    orRaise(runSeed < 200'u64, "no margin-clean seed found")
    hidden = makeHidden(geo, geo.numTokens, runSeed)
    decision = routers.degenerate.route(hidden)
    margin = choiceMargin(F.sigmoid(decision.logits) + bias, geo.topK)

  let choice = F.sigmoid(decision.logits) + bias
  let plainIndices = choice.topk(geo.topK, axis = -1, sorted = false).indices
  orRaise(decision.indices.equal(plainIndices),
    "degenerate grouping changed the selection")

proc initGuards(geo: RouterGeom, seed: uint64) =
  ## Construction guards on both router forms, every refused input must
  ## raise ValueError at init.
  ##
  ## Refused inputs:
  ## - a rank-3 weight
  ## - a bias buffer missing the expert count
  ## - an expert count indivisible by num_group
  ## - a topk_group past num_group
  ## - a grouped expert count below the two-expert floor
  ## - a top_k past the expert count
  let weight = F.randn(geo.numExperts, geo.hidden, bf16Opts())
  let bias = F.zeros(geo.numExperts, f32Opts())

  mustRefuse("rank-3 router weight accepted"):
    let sink = NoauxTcRouter.init(
      F.randn(geo.numExperts, geo.hidden, 4, bf16Opts()), bias,
      geo.topK, geo.numGroup, geo.topkGroup, geo.scale, true)
  mustRefuse("short bias buffer accepted"):
    let sink = NoauxTcRouter.init(weight,
      F.zeros(geo.numExperts - 1, f32Opts()),
      geo.topK, geo.numGroup, geo.topkGroup, geo.scale, true)
  mustRefuse("expert count indivisible by num_group accepted"):
    let sink = NoauxTcRouter.init(weight, bias, geo.topK, 3, 2,
      geo.scale, true)
  mustRefuse("topk_group past num_group accepted"):
    let sink = NoauxTcRouter.init(weight, bias, geo.topK, geo.numGroup, 5,
      geo.scale, true)
  mustRefuse("four experts in four groups accepted past the two-expert floor"):
    let sink = NoauxTcRouter.init(
      F.randn(4, geo.hidden, bf16Opts()),
      F.zeros(4, f32Opts()),
      2, 4, 1, geo.scale, true)
  mustRefuse("top_k past the expert count accepted"):
    let sink = NoauxTcRouter.init(weight, bias, geo.numExperts + 4,
      geo.numGroup, geo.topkGroup, geo.scale, true)
  mustRefuse("rank-3 greedy weight accepted"):
    let sink = GreedyRouter.init(
      F.randn(geo.numExperts, geo.hidden, 4, bf16Opts()), geo.topK, geo.scale)
  mustRefuse("greedy top_k past the expert count accepted"):
    let sink = GreedyRouter.init(weight, geo.numExperts + 4, geo.scale)

proc main() =
  # Routing constants are suite geometry, the stimulus is synthetic.
  # No checkpoint participates and no fixture reader is opened.
  let geo = RouterGeom(
    numExperts: 16, hidden: 32, numGroup: 4, topkGroup: 2, topK: 4,
    numTokens: 6, scale: 2.5)

  decisionContract(geo, seed = 7'u64)
  renormSumIdentity(geo, seed = 11'u64)
  greedySoftmaxIdentity(geo, seed = 13'u64)
  biasSteersSelection(geo, seed = 17'u64)
  degenerateIdentity(geo, seed = 19'u64)
  initGuards(geo, seed = 21'u64)

  # Out of scope for this suite, the routed FFN expert bodies, since layer
  # construction belongs to the deserialization loaders reading real
  # checkpoint weights, synthetic stimulus carries none.

when isMainModule:
  main()
