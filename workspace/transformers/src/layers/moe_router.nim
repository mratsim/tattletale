# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## MoE routers of the DeepSeek family, three typed forms plus one embedded-weight form.
##
## Typed forms:
## - NoAuxTopCorr (noaux_tc family) sigmoid-scores, the bias steers only
##   the pick, weights gather from the unbiased scores, then renormalize
##   per config. Lineage spellings live in the `scalesWeights`/`scoreBf16` docs.
## - GreedyRouter softmax-scores, straight top-k, scaled, no renorm and no bias.
## - SoftmaxTopkRouter softmax-scores over a shaped input, renormalizes
##   the picked probabilities and applies the per-expert scale (gemma-4).
##
## Embedded-weight form:
## - `routeToExperts` (Qwen family) softmax-scores, renormalizes top-k weights
##   at the hidden dtype, router weight on the FFN object.
##
## One algorithm serves every noaux_tc checkpoint, degenerate grouping
## included. n_group 1 leaves the mask at all-ones inside the same
## op sequence, no runtime branch.
##
## Routing constants arrive from config:
##   expert count, top-k, group counts, scaling factor and the bias
## values are all init arguments.
##
## Bias-buffer key naming is a model-load concern, out of this module.

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/instrumentation,
  ./norm

{.experimental: "callOperator".}

type
  RouteDecision* = tuple[logits: Tensor, weights: Tensor, indices: Tensor]
    ## Router output surface:
    ## - logits [T, E] f32
    ## - weights [T, K], f32 or hidden dtype
    ## - indices [T, K] int64 expert ids in selection order
    ##
    ## Contract:
    ## - f32 weights carry the decision projection
    ## - the hidden-dtype cast lands at the FFN boundary
    ##   (weightsHiddenDtype casts them at the router instead)

  NoAuxTopCorr* = ref object
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
    scalesWeights: bool
      ## Placement of the routed scaling factor:
      ##   - true carries the factor in the returned weights, the Moonlight
      ##     reference router scales them itself
      ##   - false applies the factor at the FFN routed output, the returned
      ##     weights stay the unscaled renormalized scores (Laguna)
    scoreBf16: bool
      ## Scoring GEMM dtype:
      ##   - true runs the GEMM at the checkpoint dtype, the logits upcast
      ##     to f32 after (Laguna, F.linear on bf16 views)
      ##   - false runs the GEMM on f32 upcasts (DeepSeek/glm lineage)
    sortedTopk: bool
      ## Selection order of the returned expert ids:
      ##   - true returns the ids sorted by biased score, the torch.topk
      ##     default the Laguna reference selection carries, a pair-sum
      ##     accumulation consumes the ids in this order
      ##   - false returns the kernel selection order (sorted = false)
    weightsHiddenDtype: bool
      ## Returned-weights dtype contract:
      ##   - true rounds the renormalized f32 weights to the hidden
      ##     dtype at the router output (Laguna, bf16 weights)
      ##   - false returns f32 weights of the DeepSeek/glm lineage,
      ##     the FFN boundary then carries the hidden-dtype cast

  GreedyRouter* = ref object
    ## Greedy router of the DeepSeek-V2-Lite family:
    ##   softmax scores cover
    ## all experts, straight top-k, no renorm, scaled by the routed factor.
    routerWeight: Tensor ## [E, H]
    topK: int
    routedScalingFactor: float64

  SoftmaxTopkRouter* = ref object
    ## Softmax top-k router, the gemma-4 routed spelling.
    ##
    ## - the input shaping mirrors the reference module, an unscaled RMS
    ##   norm (a ones-weight FusedRmsNorm, with_scale=False) plus the scale
    ##   row and the hidden-size root
    ## - scoring runs a hidden-dtype GEMM, softmax in f32, straight
    ##   top-k in probability order
    ## - the picked probabilities renormalize, the per-expert scale applies
    ##
    ## The decision weights stay f32 from the softmax through the FFN boundary, the expert
    ## multiply consumes them without a hidden-dtype rounding.
    inputNorm: FusedRmsNorm ## ones-weight unscaled norm over the hidden rows
    scaleRow: Tensor        ## [H], checkpoint dtype
    projWeight: Tensor      ## [E, H], checkpoint dtype
    perExpertScale: Tensor  ## [E], checkpoint dtype
    topK: int
    hiddenSize: int

# ─── Construction ──────────────────────────────────────────────────────────

func init*(
    _: type NoAuxTopCorr,
    routerWeight, expertBias: Tensor,
    topK, numGroup, topkGroup: int,
    routedScalingFactor: float64,
    normTopkProb: bool,
    scalesWeights: bool = true,
    scoreBf16: bool = false,
    weightsHiddenDtype: bool = false,
    sortedTopk: bool = false
  ): NoAuxTopCorr =
  ## Builds the router from the checkpoint gate weight, bias buffer,
  ## config routing constants.
  ##
  ## Wiring contracts:
  ## - scalesWeights = false pairs with an FFN applying the routed
  ##   scaling factor to the routed output
  ## - scoreBf16 = true pairs with a router reference scoring at hidden dtype
  ## - weightsHiddenDtype = true pairs with a reference router returning
  ##   hidden-dtype routing weights, sortedTopk = true with a reference
  ##   selection keeping the score-sorted id order
  ##
  ## Raises ValueError:
  ## - routerWeight is not rank 2
  ## - expertBias misses the expert count of the weight
  ## - group counts degenerate past their bounds or miss the expert count
  let e = routerWeight.size(0)
  checkValue(routerWeight.dim() == 2,
    "[ttt] NoAuxTopCorr.init: router weight must be rank 2, found rank " &
    $routerWeight.dim())
  checkValue(expertBias.numel() == e,
    "[ttt] NoAuxTopCorr.init: bias buffer holds " & $expertBias.numel() &
    " entries, expected one per expert (" & $e & ")")
  checkValue(topK >= 1,
    "[ttt] NoAuxTopCorr.init: top_k must be positive, found " & $topK)
  checkValue(topK <= e,
    "[ttt] NoAuxTopCorr.init: top_k " & $topK &
    " exceeds the expert count " & $e)
  checkValue(topkGroup >= 1 and topkGroup <= numGroup,
    "[ttt] NoAuxTopCorr.init: topk_group " & $topkGroup &
    " outside 1..num_group " & $numGroup)
  checkValue(e mod numGroup == 0,
    "[ttt] NoAuxTopCorr.init: " & $e & " experts do not split into " &
    $numGroup & " equal groups")
  checkValue(e >= 2 * numGroup,
    "[ttt] NoAuxTopCorr.init: group scores sum the top 2 biased scores" &
    " per group, " & $numGroup & " groups of " & $(e div numGroup) &
    " experts is below the 2-expert floor")
  NoAuxTopCorr(
    routerWeight: routerWeight,
    expertBias: expertBias.to(F.kFloat32),
    topK: topK,
    numGroup: numGroup,
    topkGroup: topkGroup,
    routedScalingFactor: routedScalingFactor,
    normTopkProb: normTopkProb,
    scalesWeights: scalesWeights,
    scoreBf16: scoreBf16,
    weightsHiddenDtype: weightsHiddenDtype,
    sortedTopk: sortedTopk
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

func init*(
    _: type SoftmaxTopkRouter,
    projWeight, scaleRow, perExpertScale: Tensor,
    topK: int,
    eps: float64,
    device: F.DeviceKind
  ): SoftmaxTopkRouter =
  ## Builds the router from the checkpoint projection weight [E, H], the scale row [H]
  ## and the per-expert scale [E], all at the checkpoint dtype, plus
  ## the routing constants.
  ##
  ## Raises ValueError:
  ## - projWeight is not rank 2
  ## - scaleRow misses the hidden width of the projection weight
  ## - perExpertScale misses the expert count of the projection weight
  let e = projWeight.size(0)
  let h = projWeight.size(1)
  checkValue(projWeight.dim() == 2,
    "[ttt] SoftmaxTopkRouter.init: router weight must be rank 2, found rank " &
    $projWeight.dim())
  checkValue(scaleRow.numel() == h,
    "[ttt] SoftmaxTopkRouter.init: scale row holds " & $scaleRow.numel() &
    " entries, expected the hidden width " & $h)
  checkValue(perExpertScale.numel() == e,
    "[ttt] SoftmaxTopkRouter.init: per-expert scale holds " &
    $perExpertScale.numel() & " entries, expected one per expert (" & $e & ")")
  checkValue(topK >= 1 and topK <= e,
    "[ttt] SoftmaxTopkRouter.init: top_k " & $topK &
    " outside 1.." & $e)
  let inputNorm = FusedRmsNorm.init(
    F.ones(h, F.tensorOptions(F.kBFloat16, device)), eps = eps)
  SoftmaxTopkRouter(
    inputNorm: inputNorm,
    scaleRow: scaleRow,
    projWeight: projWeight,
    perExpertScale: perExpertScale,
    topK: topK,
    hiddenSize: h
  )

# ─── Routing forms ─────────────────────────────────────────────────────────

proc routerLogits(routerWeight, hidden: Tensor, scoreBf16: bool): Tensor =
  ## Router scoring GEMM of the hidden rows against the transposed
  ## router weight, the scoring input of both router forms.
  ##
  ## - f32 spelling, rows and weights upcast first, the HF reference
  ##   module's F.linear on f32 views (DeepSeek/glm lineage)
  ## - hidden-dtype spelling, the GEMM runs at the checkpoint dtype,
  ##   the logits upcast after (Laguna), the bf16 rounding the recorded
  ##   decision rows carry
  if scoreBf16:
    F.matmul(hidden, routerWeight.t()).to(F.kFloat32)
  else:
    F.matmul(hidden.to(F.kFloat32), routerWeight.to(F.kFloat32).t())

proc route*(self: NoAuxTopCorr, hidden: Tensor): RouteDecision =
  ## noaux_tc routing over rank-2 [T, H] hidden rows.
  ##
  ##   logits [T, E]  = f32 GEMM
  ##   scores         = sigmoid(logits)
  ##   choice         = scores + bias, selection only
  ##   group scores   = sum of the top-2 biased scores per group
  ##   group mask     = topk_group winning groups, all-ones at n_group 1
  ##   indices        = topk(choice masked to -inf outside the winners)
  ##   weights        = gather(scores, indices), renormed per config,
  ##                    scaled when scalesWeights
  ##
  ## Ties at a top-k or group boundary follow the `sortedTopk` order
  ## with fixture margins recording the resolution.
  ##
  ## Returns:
  ##   - the RouteDecision surface, logits [T, E] f32, weights [T, K],
  ##     indices [T, K]
  let numTokens = hidden.size(0)
  let numExperts = self.routerWeight.size(0)
  let expertsPerGroup = numExperts div self.numGroup

  let logits = routerLogits(self.routerWeight, hidden, self.scoreBf16)
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

  let indices = choice.topk(self.topK, axis = -1, sorted = self.sortedTopk).indices
  var weights = scores.gather(1, indices)
  if self.normTopkProb:
    weights = weights / (weights.sum(axis = -1, keepdim = true) + Scalar(1e-20))
  if self.scalesWeights:
    weights = weights * Scalar(self.routedScalingFactor)
  if self.weightsHiddenDtype:
    weights = weights.to(hidden.scalarType())
  result = (logits: logits, weights: weights, indices: indices)

proc routeDecode*(self: NoAuxTopCorr, hidden: Tensor): RouteDecision =
  ## Batch-1 routing contract:
  ##   one hidden row [1, H], the same f32 GEMM scoring as route,
  ##   outputs sized [1, K] straight into the decode expert gather path.
  ##
  ## No reshape staging and no flag, the batch-1 shape is a contract,
  ## not a distinct kernel.
  checkValue(hidden.dim() == 2 and hidden.size(0) == 1,
    "[ttt] NoAuxTopCorr.routeDecode: hidden_states must be one row [1, H]," &
    " found shape (" & $hidden.size(0) & ", " & $hidden.size(1) & ")")
  route(self, hidden)

proc route*(self: GreedyRouter, hidden: Tensor): RouteDecision =
  ## Legacy greedy routing over rank-2 [T, H] hidden rows.
  ##
  ##   logits [T, E]  = f32 GEMM
  ##   scores         = softmax(logits)
  ##   indices        = topk(scores, sorted = false)
  ##   weights        = topk values, scaled by the routed factor, no renorm
  let logits = routerLogits(self.routerWeight, hidden, false)
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

proc route*(self: SoftmaxTopkRouter, hidden: Tensor): RouteDecision =
  ## Softmax top-k routing over rank-2 [T, H] hidden rows, the gemma-4
  ## routed spelling:
  ##
  ##   shaped   = rmsnorm(hidden) * scaleRow * hiddenSize^-0.5
  ##   scores   = shaped @ projWeight.T, the hidden dtype
  ##
  ##   probs    = softmax(scores f32)
  ##   indices  = topk(probs, topK) in probability order
  ##   weights  = probs gather / sum * perExpertScale gather, f32
  ##
  ## Returns:
  ##   - the RouteDecision surface, `logits` carries the softmax
  ##     probabilities [T, E] f32, `weights` [T, K] f32, `indices` [T, K]
  let shaped = self.inputNorm.forward(hidden) *
    self.scaleRow * Scalar(pow(self.hiddenSize.float64, -0.5))
  let scores = F.matmul(shaped, self.projWeight.t())
  let probs = F.softmax(scores.to(F.kFloat32), -1)
  let picked = probs.topk(self.topK, axis = -1, sorted = true)
  var weights = picked.values /
    picked.values.sum(axis = -1, keepdim = true)
  # The per-expert scale is a rank-1 [E] buffer, the gather rows travel
  # through an expanded [T, E] view.
  weights = weights *
    self.perExpertScale.unsqueeze(0)
      .expand(picked.indices.size(0), self.perExpertScale.numel(),
        implicit = false)
      .gather(1, picked.indices).to(F.kFloat32)
  result = (logits: probs, weights: weights, indices: picked.indices)

proc routeDecode*(self: SoftmaxTopkRouter, hidden: Tensor): RouteDecision =
  ## Batch-1 routing contract:
  ##   one hidden row [1, H], the same softmax scoring as route, outputs
  ##   sized [1, K] straight into the decode expert gather path.
  checkValue(hidden.dim() == 2 and hidden.size(0) == 1,
    "[ttt] SoftmaxTopkRouter.routeDecode: hidden_states must be one row [1, H]," &
    " found shape (" & $hidden.size(0) & ", " & $hidden.size(1) & ")")
  route(self, hidden)

# ─── Embedded-weight form (Qwen family) ───────────────────────────────────

proc routeToExperts*(
    hidden: Tensor,
    routerWeight: Tensor,
    numExpertsPerTok: int
  ): tuple[indices: Tensor, weights: Tensor] =
  ## Router selection over rank-2 hidden states [T, H], the embedded-weight
  ## form:
  ##   the router weight lives on the FFN object, the call site passes
  ## it in. Softmax scores over the hidden-dtype logits,
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

template `()`*(router: NoAuxTopCorr | GreedyRouter | SoftmaxTopkRouter,
              hidden: Tensor): untyped =
  ## Route-call sugar, all three typed forms.
  route(router, hidden)
