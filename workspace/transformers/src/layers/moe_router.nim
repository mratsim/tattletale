# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## MoE routers of the DeepSeek family, two typed forms plus one
## embedded-weight form:
## - NoauxTcRouter: the noaux_tc family. Sigmoid scoring over f32 logits,
##   a bias buffer that steers selection only, group limiting, weights
##   gathered from the unbiased scores, renormalized per config and scaled
##   by the routed scaling factor. One algorithm serves every noaux_tc
##   checkpoint, degenerate grouping included: n_group 1 leaves the mask
##   at all-ones inside the same op sequence, no runtime branch.
## - GreedyRouter: the legacy greedy form. Softmax scores, straight
##   top-k, scaled, no renorm and no bias.
## - `routeToExperts` is the embedded-weight form of the Qwen family:
##   the router weight lives on the FFN object, softmax scores,
##   renormalized top-k weights at the hidden dtype.
##
## Routing constants arrive from config: expert count, top-k, group counts,
## scaling factor and the bias values are all init arguments. Bias-buffer
## key naming is a model-load concern, out of this module.

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/instrumentation

{.experimental: "callOperator".}

type
  RouteDecision* = tuple[logits: Tensor, weights: Tensor, indices: Tensor]
    ## Router output surface: f32 logits [T, E], f32 weights [T, K],
    ## int64 expert ids [T, K] in selection order. F32 weights carry
    ## the decision projection, the hidden-dtype cast happens at the FFN
    ## boundary, one recorded rounding.

  NoauxTcRouter* = ref object
    ## noaux_tc-family router, parameterized by config:
    ## scoring sigmoid, selection under the bias buffer, group limiting,
    ## optional renormalization, routed scaling factor last.
    routerWeight: Tensor ## [E, H], checkpoint dtype, upcast to f32 per route
    expertBias: Tensor   ## [E] f32, selection bias, weights gather around it
    topK: int
    numGroup: int
    topkGroup: int
    routedScalingFactor: float64
    normTopkProb: bool

  GreedyRouter* = ref object
    ## Legacy greedy router (DeepSeek-V2-Lite family): softmax scores cover
    ## all experts, straight top-k, no renorm, scaled by the routed factor.
    routerWeight: Tensor ## [E, H]
    topK: int
    routedScalingFactor: float64

# ─── Construction ──────────────────────────────────────────────────────────

func init*(
    _: type NoauxTcRouter,
    routerWeight, expertBias: Tensor,
    topK, numGroup, topkGroup: int,
    routedScalingFactor: float64,
    normTopkProb: bool
  ): NoauxTcRouter =
  ## Builds the router from the checkpoint gate weight, bias buffer,
  ## config routing constants.
  ##
  ## Raises ValueError:
  ## - routerWeight is not rank 2
  ## - expertBias misses the expert count of the weight
  ## - group counts degenerate past their bounds or miss the expert count
  let e = routerWeight.size(0)
  checkValue(routerWeight.dim() == 2,
    "[ttt] NoauxTcRouter.init: router weight must be rank 2, found rank " &
    $routerWeight.dim())
  checkValue(expertBias.numel() == e,
    "[ttt] NoauxTcRouter.init: bias buffer holds " & $expertBias.numel() &
    " entries, expected one per expert (" & $e & ")")
  checkValue(topK >= 1,
    "[ttt] NoauxTcRouter.init: top_k must be positive, found " & $topK)
  checkValue(topK <= e,
    "[ttt] NoauxTcRouter.init: top_k " & $topK &
    " exceeds the expert count " & $e)
  checkValue(topkGroup >= 1 and topkGroup <= numGroup,
    "[ttt] NoauxTcRouter.init: topk_group " & $topkGroup &
    " outside 1..num_group " & $numGroup)
  checkValue(e mod numGroup == 0,
    "[ttt] NoauxTcRouter.init: " & $e & " experts do not split into " &
    $numGroup & " equal groups")
  checkValue(e >= 2 * numGroup,
    "[ttt] NoauxTcRouter.init: group scores sum the top 2 biased scores" &
    " per group, " & $numGroup & " groups of " & $(e div numGroup) &
    " experts is below the 2-expert floor")
  NoauxTcRouter(
    routerWeight: routerWeight,
    expertBias: expertBias.to(F.kFloat32),
    topK: topK,
    numGroup: numGroup,
    topkGroup: topkGroup,
    routedScalingFactor: routedScalingFactor,
    normTopkProb: normTopkProb
  )

func init*(
    _: type GreedyRouter,
    routerWeight: Tensor,
    topK: int,
    routedScalingFactor: float64
  ): GreedyRouter =
  ## Builds the legacy greedy router from the checkpoint gate weight
  ## and the config routing constants.
  let e = routerWeight.size(0)
  checkValue(routerWeight.dim() == 2,
    "[ttt] GreedyRouter.init: router weight must be rank 2, found rank " &
    $routerWeight.dim())
  checkValue(topK >= 1,
    "[ttt] GreedyRouter.init: top_k must be positive, found " & $topK)
  checkValue(topK <= e,
    "[ttt] GreedyRouter.init: top_k " & $topK &
    " exceeds the expert count " & $e)
  GreedyRouter(
    routerWeight: routerWeight,
    topK: topK,
    routedScalingFactor: routedScalingFactor
  )

# ─── Routing forms ─────────────────────────────────────────────────────────

proc routerLogits(routerWeight: Tensor, hidden: Tensor): Tensor =
  ## Router scoring GEMM of the hidden rows against the router weight
  ## transpose: the scoring input of both router forms. Routing math runs
  ## in f32, weights and rows upcast first, matching the HF reference
  ## module's F.linear on f32 views.
  F.matmul(hidden.to(F.kFloat32), routerWeight.to(F.kFloat32).t())

proc route*(self: NoauxTcRouter, hidden: Tensor): RouteDecision =
  ## noaux_tc routing over rank-2 [T, H] hidden rows.
  ##
  ##   logits [T, E]  = f32 GEMM
  ##   scores         = sigmoid(logits)
  ##   choice         = scores + bias, selection only
  ##   group scores   = sum of the top-2 biased scores per group
  ##   group mask     = topk_group winning groups, all-ones at n_group 1
  ##   indices        = topk(choice masked to -inf outside the winners)
  ##   weights        = gather(scores, indices), renormed per config, scaled
  ##
  ## Ties at a top-k or group boundary take the kernel order
  ## of the sorted = false topk, fixture margins record the disambiguation.
  let numTokens = hidden.size(0)
  let numExperts = self.routerWeight.size(0)
  let expertsPerGroup = numExperts div self.numGroup

  let logits = routerLogits(self.routerWeight, hidden)
  let scores = F.sigmoid(logits)
  var choice = scores + self.expertBias

  # Group limiting over the biased scores. At numGroup 1 the single group
  # always wins and the mask stays all-ones, so the degenerate config runs
  # the same ops with an identity mask.
  let groupScores = choice.view(numTokens, self.numGroup, expertsPerGroup)
    .topk(2, axis = -1).values.sum(axis = -1)
  let groupIdx = groupScores.topk(self.topkGroup, axis = -1, sorted = false).indices
  let device = choice.deviceType()
  let groupMask = F.zeros(numTokens, self.numGroup,
      F.tensorOptions(F.kFloat32, device))
    .scatter(1, groupIdx, F.ones(groupIdx.size(0), groupIdx.size(1),
      F.tensorOptions(F.kFloat32, device)))
  let scoreMask = groupMask.unsqueeze(-1)
    .expand(numTokens, self.numGroup, expertsPerGroup, implicit = false)
    .reshape(numTokens, numExperts)
  choice.masked_fill_mut(scoreMask.eq(F.zeros(1,
      F.tensorOptions(F.kFloat32, device))), Scalar(NegInf))

  let indices = choice.topk(self.topK, axis = -1, sorted = false).indices
  var weights = scores.gather(1, indices)
  if self.normTopkProb:
    weights = weights / (weights.sum(axis = -1, keepdim = true) + Scalar(1e-20))
  weights = weights * Scalar(self.routedScalingFactor)
  result = (logits: logits, weights: weights, indices: indices)

proc routeDecode*(self: NoauxTcRouter, hidden: Tensor): RouteDecision =
  ## Batch-1 routing contract: one hidden row [1, H], the same f32 GEMM
  ## scoring as route, outputs sized [1, K] straight into the decode
  ## expert gather path. No reshape staging, no flag: the batch-1 shape
  ## is a contract, not a distinct kernel.
  checkValue(hidden.dim() == 2 and hidden.size(0) == 1,
    "[ttt] NoauxTcRouter.routeDecode: hidden_states must be one row [1, H]," &
    " found shape (" & $hidden.size(0) & ", " & $hidden.size(1) & ")")
  route(self, hidden)

proc route*(self: GreedyRouter, hidden: Tensor): RouteDecision =
  ## Legacy greedy routing over rank-2 [T, H] hidden rows.
  ##
  ##   logits [T, E]  = f32 GEMM
  ##   scores         = softmax(logits)
  ##   indices        = topk(scores, sorted = false)
  ##   weights        = topk values, scaled by the routed factor, no renorm
  let logits = routerLogits(self.routerWeight, hidden)
  let scores = F.softmax(logits, -1)
  let (topValues, topIndices) = scores.topk(self.topK, axis = -1, sorted = false)
  let weights = topValues * Scalar(self.routedScalingFactor)
  result = (logits: logits, weights: weights, indices: topIndices)

proc routeDecode*(self: GreedyRouter, hidden: Tensor): RouteDecision =
  ## Batch-1 greedy routing, one hidden row [1, H], one f32 GEMV, outputs
  ## sized [1, K].
  checkValue(hidden.dim() == 2 and hidden.size(0) == 1,
    "[ttt] GreedyRouter.routeDecode: hidden_states must be one row [1, H]," &
    " found shape (" & $hidden.size(0) & ", " & $hidden.size(1) & ")")
  route(self, hidden)

# ─── Embedded-weight form (Qwen family) ───────────────────────────────────

proc routeToExperts*(
    hidden: Tensor,
    routerWeight: Tensor,
    numExpertsPerTok: int
  ): tuple[indices: Tensor, weights: Tensor] =
  ## Router selection over rank-2 hidden states [T, H], the embedded-weight
  ## form: the router weight lives on the FFN object instead of a typed
  ## router object. Softmax scores over the hidden-dtype logits,
  ## renormalized top-k weights cast to the hidden dtype:
  ##
  ##   logits = matmul(hidden, routerWeight.T) → [T, E] at the hidden dtype
  ##   probs  = softmax(logits.to(fp32), -1)
  ##   values [T, K], indices [T, K] = topk(probs, numExpertsPerTok, -1)
  ##   weights = (values / values.sum(-1, keepdim)).to(hidden dtype)
  ##
  ## Ties at the top-k boundary take the selection's own order, with no re-sort.
  let logits = F.matmul(hidden, routerWeight.t())
  let probs = F.softmax(logits.to(kFloat32), -1)
  let topk = F.topk(probs, numExpertsPerTok, -1)
  let topValuesFp32 = topk.values
  let topIndices = topk.indices
  let renormFp32 = topValuesFp32 / topValuesFp32.sum(-1, keepdim = true)
  let routingWeights = renormFp32.to(hidden.scalarType())
  (topIndices, routingWeights)

template `()`*(router: NoauxTcRouter | GreedyRouter, hidden: Tensor): untyped =
  ## Route-call sugar, both forms.
  route(router, hidden)
