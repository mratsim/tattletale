# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  workspace/positron,
  workspace/transformers/src/layers/mlp

################################################################################
#                Routed-expert body over rank-3 fused weights                   #
################################################################################

type
  MixtureOfExperts* = ref object
    ## Expert MLP body of a routed Mixture-of-Experts block: one shared module
    ## over rank-3 fused expert tensors, visited expert by expert. No router
    ## and no packing step here. Routing arrives as preselected expert ids
    ## and weights.
    ##
    ## Layout of the fused gate/up projection [E, 2I, H]: gate rows 0:I,
    ## up rows I:2I. Each expert carries one fused gate/up weight of that shape
    ## and one down projection [E, H, I].
    ##
    ## Numerical contract, bitwise fidelity with the reference expert body
    ## (transformers modeling, CPU):
    ## - one fused GEMM per hit expert covers the full [2I, H] gate/up
    ##   weight. Narrow GEMMs per half would round differently.
    ## - experts visited in ascending index order. Within one expert,
    ##   every token appears at most once, so each accumulator row gets
    ##   exactly one addition per scatter_add call. Accumulation order
    ##   matches the reference index_add loop on CPU.
    ## - topKWeights arrive at the multiply dtype and the module changes
    ##   no weight values: the fp32 spelling round-trips bf16 exactly,
    ##   the router owns the renormalization and its cast.
    gateUpProj*: Tensor   ## [E, 2I, H] fused: gate rows 0:I, up rows I:2I
    downProj*: Tensor     ## [E, H, I]
    numExperts*: int      ## E, derived from gateUpProj at init
    hiddenDim*: int       ## H, derived from gateUpProj at init
    activation*: ActivationKind

func init*(
    _: type MixtureOfExperts,
    gateUpProj: Tensor,
    downProj: Tensor,
    activation: ActivationKind = kSilu
  ): MixtureOfExperts =
  ## Creates the expert-body collection from rank-3 fused weights.
  ## numExperts and hiddenDim derive from gateUpProj:
  ## E is size(0), I is size(1) div 2, H is size(2).
  ##
  ## Raises ValueError when gateUpProj or downProj is not rank 3, or when downProj
  ## is not [E, H, I] against gateUpProj.
  if gateUpProj.dim() != 3:
    raise newException(ValueError,
      "[ttt] MixtureOfExperts.init: gate_up_proj must be rank 3, found rank " &
      $gateUpProj.dim())
  if downProj.dim() != 3:
    raise newException(ValueError,
      "[ttt] MixtureOfExperts.init: down_proj must be rank 3, found rank " &
      $downProj.dim())

  let e = gateUpProj.size(0)
  let inter = gateUpProj.size(1) div 2
  let h = gateUpProj.size(2)
  if downProj.size(0) != e or downProj.size(1) != h or downProj.size(2) != inter:
    raise newException(ValueError,
      "[ttt] MixtureOfExperts.init: down_proj shape is [" & $downProj.size(0) & ", " &
      $downProj.size(1) & ", " & $downProj.size(2) & "], expected [" & $e & ", " &
      $h & ", " & $inter & "]")

  MixtureOfExperts(
    gateUpProj: gateUpProj,
    downProj: downProj,
    numExperts: e,
    hiddenDim: h,
    activation: activation
  )

proc forward*(
    self: MixtureOfExperts,
    hiddenStates: Tensor,
    topKIndex: Tensor,
    topKWeights: Tensor
  ): Tensor =
  ## Routed-expert forward: per-expert SwiGLU over each hit expert's tokens,
  ## projected down, weighted and scattered into a zeroed [T, H]
  ## accumulator.
  ##
  ## Expected input:
  ## - hiddenStates [T, H] at the multiply dtype, one row per token
  ## - topKIndex [T, K] int64, expert ids per token. Values within a row
  ##   must be distinct: a duplicate would put two additions on one accumulator
  ##   row inside one scatter_add call, an order scatter_add does not define
  ## - topKWeights [T, K] at the multiply dtype, the routing weights
  ##
  ## Output:
  ## - [T, H]: per token the sum over selected experts, each contribution
  ##   down_proj(silu(gate) * up) * routing weight, zero rows for unselected
  ##   tokens
  let t = hiddenStates.size(0)
  let topK = topKIndex.size(1)
  let device = hiddenStates.deviceType()

  # Expert id per (token, position), extracted once for the grouping scan
  var expertIds = newSeq[int64](t * topK)
  for tok in 0 ..< t:
    for pos in 0 ..< topK:
      let cell = topKIndex[tok, pos]
      expertIds[tok * topK + pos] = cell.item(int64)

  # Guard: the scatter_add accumulation relies on disjoint token groups
  for tok in 0 ..< t:
    for i in 0 ..< topK:
      for j in (i + 1) ..< topK:
        if expertIds[tok * topK + i] == expertIds[tok * topK + j]:
          raise newException(ValueError,
            "[ttt] MixtureOfExperts.forward: topKIndex row " & $tok & " repeats expert id " &
            $expertIds[tok * topK + i] & ", the accumulation order is undefined")

  var finalHiddenStates =
    F.zeros(t, self.hiddenDim, F.tensorOptions(hiddenStates.scalarType(), device))

  for e in 0 ..< self.numExperts:
    # Token group of expert e, pairs ordered by position then token:
    # the row-major scan of the reference one-hot mask row
    var tokenIdx = newSeq[int]()
    var topKPos = newSeq[int]()
    for pos in 0 ..< topK:
      for tok in 0 ..< t:
        if expertIds[tok * topK + pos] == e.int64:
          topKPos.add pos
          tokenIdx.add tok
    if tokenIdx.len == 0:
      continue

    let n = tokenIdx.len
    let tokenIdxTensor = tokenIdx.toTensor().to(device)
    let currentStates = F.index_select(hiddenStates, 0, tokenIdxTensor)

    # One fused GEMM over the full [2I, H] gate/up weight of expert e
    let gateUpWeight = self.gateUpProj[e]
    let gateUpOut = F.matmul(currentStates, gateUpWeight.t())

    let chunks = F.chunk(gateUpOut, 2, -1)
    let gateChunk = chunks[0]
    let upChunk = chunks[1]
    let act =
      case self.activation
      of kSilu: F.silu(gateChunk) * upChunk

    let downWeight = self.downProj[e]
    let currentHiddenStates = F.matmul(act, downWeight.t())

    # Routing weight per (token, position) pair, read through an fp32
    # spelling of topKWeights so bf16 values round-trip exactly
    let weights32 = topKWeights.to(kFloat32)
    var weightVals = newSeq[float32](n)
    for i in 0 ..< n:
      let tokI = tokenIdx[i]
      let posI = topKPos[i]
      let cell = weights32[tokI, posI]
      weightVals[i] = cell.item(float32)
    let weightCol = weightVals.toTensor().reshape(n, 1)
      .to(device, hiddenStates.scalarType())

    let weighted = currentHiddenStates * weightCol

    # One column per hidden feature: scatter_add requires index ndims equal
    # to the accumulator ndims. Token groups are disjoint within one expert,
    # so each row receives at most one addition per call.
    let idx2d = tokenIdxTensor.unsqueeze(1).expand(n, self.hiddenDim, implicit = false)
    finalHiddenStates = F.scatter_add(finalHiddenStates, 0, idx2d, weighted)

  result = finalHiddenStates

################################################################################
#                 Router, shared expert and the routed block                   #
################################################################################

type
  SparseMoeResult* = ref object
    ## Outputs of one routed-block forward: the MoE output and intermediates
    ## a caller inspects.
    routerLogits*: Tensor   ## [T, E] at the hidden-state dtype, softmax input
    topkIndices*: Tensor    ## [T, K] int64 expert ids per token, descending probability
    renormValues*: Tensor   ## [T, K] fp32 renormalized values, before the dtype cast
    routingWeights*: Tensor ## [T, K] routing weights, the same values after the cast
    sharedGate*: Tensor     ## [T, 1] post-sigmoid shared-expert gate
    output*: Tensor         ## [T, H], routed sum plus gated shared expert

proc sparseMoeForward*(
    numExpertsPerTok: int,
    hiddenStates: Tensor,
    routerWeight: Tensor,
    experts: MixtureOfExperts,
    sharedExpert: GatedMLP,
    sharedGateWeight: Tensor
  ): SparseMoeResult =
  ## One routed-block forward on rank-2 hidden states [T, H], the flattened
  ## form the reference block reaches through `view(-1, hidden_dim)`.
  ##
  ## Router: `F.matmul(hiddenStates, routerWeight.t())` [T, E], the reference
  ## `F.linear`, softmax over the fp32 spelling of all E experts, top-k
  ## selection of `numExpertsPerTok` experts with `F.topk`, naming
  ## the reference `torch.topk`. Ties at the top-k selection boundary
  ## order by the selection's own tie-break, and no sort reproduces
  ## such a rule. Renormalized in fp32.
  ## The cast back to the hidden-state dtype is last, the reference
  ## `router_top_value.to(router_logits.dtype)`.
  ##
  ## Routed branch: `experts.forward` over the selected ids and weights.
  ##
  ## Shared branch: `F.sigmoid(hiddenStates @ sharedGateWeight.t())` [T, 1]
  ## times the shared-expert output, added to the routed sum.
  ##
  ## The expert activation lives in the expert bodies, not here. Raises
  ## ValueError when hiddenStates is not rank 2, or when routerWeight
  ## or sharedGateWeight disagrees with hiddenStates on the hidden
  ## dimension.
  if hiddenStates.dim() != 2:
    raise newException(ValueError,
      "[ttt] sparseMoeForward: hidden_states must be rank 2 [T, H], found rank " &
      $hiddenStates.dim())
  if routerWeight.size(1) != hiddenStates.size(1):
    raise newException(ValueError,
      "[ttt] sparseMoeForward: router weight hidden width is " &
      $routerWeight.size(1) & ", expected " & $hiddenStates.size(1))
  if sharedGateWeight.size(1) != hiddenStates.size(1):
    raise newException(ValueError,
      "[ttt] sparseMoeForward: shared gate weight hidden width is " &
      $sharedGateWeight.size(1) & ", expected " & $hiddenStates.size(1))

  let logits = F.matmul(hiddenStates, routerWeight.t())
  let probs = F.softmax(logits.to(kFloat32), -1)
  let topk = F.topk(probs, numExpertsPerTok, -1)
  let topValuesFp32 = topk.values
  let topIndices = topk.indices
  let renormFp32 = topValuesFp32 / topValuesFp32.sum(-1, keepdim = true)
  let routingWeights = renormFp32.to(hiddenStates.scalarType())

  let sharedGate = F.sigmoid(F.matmul(hiddenStates, sharedGateWeight.t()))
  let sharedGated = sharedGate * sharedExpert.forward(hiddenStates)

  let routed = experts.forward(hiddenStates, topIndices, routingWeights)

  SparseMoeResult(
    routerLogits: logits,
    topkIndices: topIndices,
    renormValues: renormFp32,
    routingWeights: routingWeights,
    sharedGate: sharedGate,
    output: routed + sharedGated
  )

