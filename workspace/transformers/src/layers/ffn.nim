# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Transformer feed-forward layers:
## - GatedDenseFFN: dense SwiGLU FFN with separate gate and up projections
## - GatedBlockSparseFFN: routed mixture-of-experts FFN with a shared expert

import
  std/options,
  workspace/libtorch as F,
  workspace/positron,
  workspace/transformers/src/instrumentation,
  ./linear,
  ./moe_router

{.experimental: "callOperator".}

type
  GatedDenseFFN* = ref object
    ## Dense SwiGLU feed-forward network (FFN) with separate gate and up projections.
    ##
    ## Forward:
    ##   gate = gate_proj(x)        # (..., intermediate_size)
    ##   up   = up_proj(x)          # (..., intermediate_size)
    ##   activation = silu(gate) * up
    ##   output = down_proj(activation)
    ##
    ## Input:
    ##   - An externally provided `x` of shape (..., hidden_size)
    ##
    ## Return:
    ##   - Output tensor of shape (..., hidden_size)
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear
    activation: ActivationKind

func init*(
    _: type GatedDenseFFN,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: ActivationKind = kSilu
  ): GatedDenseFFN =
  ## Creates a GatedDenseFFN layer from pre-constructed Linear projections.
  GatedDenseFFN(
    gate_proj: gate_proj,
    up_proj: up_proj,
    down_proj: down_proj,
    activation: activation
  )

func init*(
    _: type GatedDenseFFN,
    gate_weight, up_weight, down_weight: Tensor,
    activation: ActivationKind = kSilu
  ): GatedDenseFFN =
  ## Creates a GatedDenseFFN layer from separate gate and up weight tensors.
  ## The weights are still stored as separate projections (no fusion).
  GatedDenseFFN(
    gate_proj: Linear.init(gate_weight),
    up_proj: Linear.init(up_weight),
    down_proj: Linear.init(down_weight),
    activation: activation
  )

proc forward*(self: GatedDenseFFN, x: Tensor): Tensor =
  ## Forward pass for inference.
  ##
  ## Args:
  ##   x: Input tensor of shape (..., hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (..., hidden_size)
  ##
  ## Computes:
  ##   gate_out = self.gate_proj(x)     # (..., intermediate_size)
  ##   up_out   = self.up_proj(x)       # (..., intermediate_size)
  ##   act      = silu(gate_out) * up_out
  ##   return self.down_proj(act)
  let gate_out = self.gate_proj(x)
  let up_out = self.up_proj(x)
  let act_out =
    case self.activation
    of kSilu: F.silu(gate_out) * up_out # TODO silu_and_mul fusion
    of kGeluTanh: gelu_tanh(gate_out) * up_out
  result = self.down_proj(act_out)

template `()`*(layer: GatedDenseFFN, x: Tensor): untyped =
  forward(layer, x)

################################################################################
#            Routed expert FFN: router, expert body, shared expert             #
################################################################################

type
  GatedBlockSparseFFN* = ref object
    ## Block-sparse mixture-of-experts FFN: a router picks the top-K experts
    ## per token and every hit expert runs its own gate/up/SwiGLU/down body.
    ## A sigmoid-gated shared expert joins the routed contribution.
    ##
    ## One object owns:
    ## - the router weight [E, H]
    ## - the rank-3 fused expert bodies, visited expert by expert
    ## - the shared expert and its sigmoid gate
    ## - the flatten/reshape around the token rows
    ##
    ## `forward(hidden)` returns the FFN contribution for the token rows.
    ## Dispatch is by token count:
    ## - the shared expertForwardDecode at T = 1, per-expert loop on CPU,
    ##   batched gather off CPU, no host syncs off CPU
    ## - expertForwardPrefill at T > 1, per-expert loop
    gateUpProj: Tensor   ## [E, 2I, H] fused: gate rows 0:I, up rows I:2I
    downProj: Tensor     ## [E, H, I]
    numExperts: int
    hiddenSize: int
    activation: ActivationKind
    routerWeight: Tensor
    sharedExpert: GatedDenseFFN
    sharedGateWeight: Tensor
    numExpertsPerTok: int

func init*(
    _: type GatedBlockSparseFFN,
    gateUpProj: Tensor,
    downProj: Tensor,
    routerWeight: Tensor,
    sharedExpert: GatedDenseFFN,
    sharedGateWeight: Tensor,
    numExpertsPerTok: int,
    activation: ActivationKind = kSilu
  ): GatedBlockSparseFFN =
  ## Create the routed FFN from the rank-3 fused expert weights, the router
  ## weight [E, H], the shared expert and its gate weight [1, H].
  ##
  ## Raises ValueError:
  ## - gateUpProj or downProj is not rank 3
  ## - downProj is not [E, H, I] against gateUpProj
  ## - routerWeight or sharedGateWeight misses the hidden width
  checkValue(gateUpProj.dim() == 3,
    "[ttt] GatedBlockSparseFFN.init: gate_up_proj must be rank 3, found rank " &
    $gateUpProj.dim())
  checkValue(downProj.dim() == 3,
    "[ttt] GatedBlockSparseFFN.init: down_proj must be rank 3, found rank " &
    $downProj.dim())

  let e = gateUpProj.size(0)
  let inter = gateUpProj.size(1) div 2
  let h = gateUpProj.size(2)
  checkValue(downProj.size(0) == e and downProj.size(1) == h and
    downProj.size(2) == inter,
    "[ttt] GatedBlockSparseFFN.init: down_proj shape is [" & $downProj.size(0) & ", " &
    $downProj.size(1) & ", " & $downProj.size(2) & "], expected [" & $e & ", " &
    $h & ", " & $inter & "]")
  checkValue(routerWeight.size(1) == h,
    "[ttt] GatedBlockSparseFFN.init: router weight hidden width is " &
    $routerWeight.size(1) & ", expected " & $h)
  checkValue(sharedGateWeight.size(1) == h,
    "[ttt] GatedBlockSparseFFN.init: shared gate weight hidden width is " &
    $sharedGateWeight.size(1) & ", expected " & $h)

  GatedBlockSparseFFN(
    gateUpProj: gateUpProj,
    downProj: downProj,
    numExperts: e,
    hiddenSize: h,
    activation: activation,
    routerWeight: routerWeight,
    sharedExpert: sharedExpert,
    sharedGateWeight: sharedGateWeight,
    numExpertsPerTok: numExpertsPerTok
  )

################################################################################
#                 Routed-expert compute: decode and prefill                    #
################################################################################
# Two paths per shared tail, dispatched by token count in forward.
# "Prefill" covers every multi-token forward. The decode path is one
# shared proc for every router type, the routing-weight rounding
# selected at compile time by a policy type.
#
# Maintainer note, numerical contract for the prefill path: bitwise
# fidelity with the HF reference expert body on CPU.
# - one fused GEMM per hit expert covers the full [2I, H] gate/up
#   weight, narrow per-half GEMMs round differently
# - experts visited in ascending index order, token groups disjoint,
#   so each accumulator row takes exactly one addition per scatter_add
#   call, matching the HF reference index_add loop
# - weight values pass through unchanged, renormalization and dtype
#   cast belong to the router
#
# Decode path: routed expert weights are gathered per top-k position,
# each projection stage runs as one batched matmul, contributions sum
# over the top-k positions. Routing data stays on the device throughout.
# Gathered traffic grows as T*K, pricing this path out of large T.
# On CPU the decode path swaps the gather for a per-expert narrow +
# GEMV loop, the routed weights are read in place and the host reads
# are free, rationale in expertForwardDecode.
#
# Prefill path: token groups are scanned on the host, every hit expert
# runs its own fused GEMMs, and scatter_add writes the weighted results
# in a zeroed accumulator in ascending expert order.
# Cost scales with hit experts, not with T*K.
#
# Roundoff is the only difference: decode accumulates in routing order,
# prefill in ascending expert order, at most one bf16 ulp on the layer
# output. Comparison: workspace/libtorch/bench/metal/bench_moe.nim,
# weight samples and devices.

# CPU form: per-expert narrow + GEMV loop instead of the batched
# gather, measured in workspace/libtorch/bench/cpu/bench_moe_decode.nim
# (both decode shapes, measured dispatch records in its header). On CPU
# the loop reads each routed expert in place, the host reads are free,
# while the batched gather copies the K expert bodies first, so the copy
# traffic prices the gather out once the routed weight bytes outgrow the
# caches. MPS keeps the batched gather form.

proc expertBody(gateUpWeight, downWeight, currentStates: Tensor,
    activation: ActivationKind): Tensor =
  ## One routed expert body on the gathered token rows: the fused gate/up
  ## GEMM covers the full [2I, H] weight, narrow per-half GEMMs round
  ## differently. SiLU gate times up, down GEMM.
  ## Output at the states dtype, weight untouched.
  let gateUpOut = F.matmul(currentStates, gateUpWeight.t())
  let chunks = F.chunk(gateUpOut, 2, -1)
  let act =
    case activation
    of kSilu: F.silu(chunks[0]) * chunks[1]
    of kGeluTanh: gelu_tanh(chunks[0]) * chunks[1]
  F.matmul(act, downWeight.t())

type
  RoundAfterSum* = object
    ## Routing-weight rounding policy of the ungated shared-expert form
    ## (noaux_tc): the routing weight stays f32, the f32 weight multiplies
    ## the f32 expert output and the weighted sum rounds to the hidden
    ## dtype once at the accumulated sum, the HF reference index_add_
    ## rounding location.
  RoundBeforeMultiply* = object
    ## Routing-weight rounding policy of the gated shared-expert form
    ## (Qwen): the routing weight rounds to the hidden dtype before the
    ## multiply, the HF gated router computes the routing weights at the
    ## hidden dtype.

proc weightedTerms(policy: typedesc[RoundAfterSum],
    downOut, topKWeights: Tensor): Tensor =
  ## Weighted expert outputs per top-k position as [K, 1, H] f32, the
  ## product stays f32 and rounds to the hidden dtype once at the
  ## accumulated sum.
  let weightCol = topKWeights.transpose(0, 1).unsqueeze(2).to(F.kFloat32)
  downOut.to(F.kFloat32) * weightCol

proc weightedTerms(policy: typedesc[RoundBeforeMultiply],
    downOut, topKWeights: Tensor): Tensor =
  ## Weighted expert outputs per top-k position as [K, 1, H] at the
  ## expert output dtype, one rounding per weighted term.
  let weightCol = topKWeights.transpose(0, 1).unsqueeze(2)
    .to(downOut.scalarType())
  downOut * weightCol

proc weightedTerm(policy: typedesc[RoundAfterSum],
    downOut, topKWeights: Tensor, pos: int): Tensor =
  ## One weighted expert output inside the per-expert loop form, the
  ## f32 product joins the f32 accumulator unchanged.
  downOut.to(F.kFloat32) * topKWeights[0, pos].item(float32)

proc weightedTerm(policy: typedesc[RoundBeforeMultiply],
    downOut, topKWeights: Tensor, pos: int): Tensor =
  ## One weighted expert output inside the per-expert loop form, one
  ## rounding per weighted term, the product joins the f32 accumulator.
  (downOut * topKWeights[0, pos]).to(F.kFloat32)

proc expertForwardDecode*(policy: typedesc,
    gateUpProj, downProj: Tensor,
    hiddenStates: Tensor,
    topKIndex: Tensor,
    topKWeights: Tensor,
    activation: ActivationKind
  ): Tensor =
  ## Routed expert compute for a single token (T = 1), one body for every
  ## router type, the routing-weight rounding selected at compile time
  ## by the policy type. On CPU the expert weights read in place through
  ## per-expert narrow plus GEMV, on other devices the gather form
  ## runs one batched GEMM per projection stage. The f32 accumulator
  ## carries the evaluation-order difference against the HF eager
  ## per-expert bf16 additions, recorded and budgeted.
  ##
  ## Data flow, gather form:
  ##
  ##   gather routed expert weights by top-k id, device-side index_select
  ##     ├→ matmul(hiddenStates, gathered gate_up weight.T)   [K, 2I]
  ##     ├→ chunk(2) → silu(gate) * up                        [K, I]
  ##     ├→ matmul(act, gathered down weight.T)               [K, H]
  ##     └→ weightedTerms per policy, sum over K
  ##
  ## On CPU the loop swaps the weight gather for per-expert narrow, each
  ## projection stage a GEMV, one weightedTerm per position into an f32
  ## accumulator, the measured variants live in
  ## workspace/libtorch/bench/cpu/bench_moe_decode.nim.
  ##
  ## Input:
  ##   - gateUpProj, downProj: rank-3 fused expert weights [E, 2I, H]
  ##     and [E, H, I]
  ##   - hiddenStates: [1, H] at the hidden dtype
  ##   - topKIndex: [1, K] int64 expert ids
  ##   - topKWeights: [1, K] decision weights, f32 pre-cast never for
  ##     RoundAfterSum, at the hidden dtype for RoundBeforeMultiply
  ##
  ## Output:
  ##   - [1, H] at the hidden dtype, the sum over selected experts, each
  ##     term down_proj(silu(gate) * up) weighted by its routing weight
  ##
  ## Routing indices never leave the device on the gather form, so
  ## there are no host syncs off CPU. topk never repeats an id within a
  ## row, so no duplicate check here. Accumulation runs in routing order.
  let topK = topKIndex.size(1)
  if hiddenStates.deviceType() == F.kCPU:
    var acc = F.zeros(1, gateUpProj.size(2), F.kFloat32)
    for pos in 0 ..< topK:
      let e = topKIndex[0, pos].item(int64).int
      let downOut = expertBody(gateUpProj[e], downProj[e], hiddenStates, activation)
      acc = acc + weightedTerm(policy, downOut, topKWeights, pos)
    return acc.to(hiddenStates.scalarType())
  let flatIdx = topKIndex.reshape(topK)
  let gatheredGateUp = F.index_select(gateUpProj, 0, flatIdx)
  let gatheredDown = F.index_select(downProj, 0, flatIdx)
  let xs = hiddenStates.unsqueeze(0)
    .expand(topK, 1, gateUpProj.size(2), implicit = false)
    .contiguous()
  let gateUpOut = F.matmul(xs, gatheredGateUp.transpose(1, 2))
  let chunks = F.chunk(gateUpOut, 2, -1)
  let act =
    case activation
    of kSilu: F.silu(chunks[0]) * chunks[1]
    of kGeluTanh: gelu_tanh(chunks[0]) * chunks[1]
  let downOut = F.matmul(act, gatheredDown.transpose(1, 2))
  result = weightedTerms(policy, downOut, topKWeights).sum(0)
    .to(hiddenStates.scalarType())

proc expertForwardPrefill(
    self: GatedBlockSparseFFN,
    hiddenStates: Tensor,
    topKIndex: Tensor,
    topKWeights: Tensor
  ): Tensor =
  ## Routed expert compute for multi-token inputs (T > 1).
  ##
  ## Data flow, per hit expert e in ascending index order:
  ##
  ##   host scan of topKIndex → token group of e
  ##     ├→ index_select(hiddenStates, tokenIdx)      [n, H]
  ##     ├→ matmul(·, gateUpProj[e].T)                [n, 2I]
  ##     ├→ chunk(2) → silu(gate) * up                [n, I]
  ##     ├→ matmul(·, downProj[e].T)                  [n, H]
  ##     ├→ multiply the routing weight per (token, position) pair
  ##     └→ scatter_add into the zeroed [T, H] accumulator
  ##
  ## Input:
  ##   - hiddenStates: [T, H] at the multiply dtype
  ##   - topKIndex: [T, K] int64 expert ids from routeToExperts
  ##   - topKWeights: [T, K] routing weights at the multiply dtype
  ##
  ## Output:
  ##   - [T, H], zero rows for unselected tokens
  ##
  ## Preconditions:
  ##   - ids within one topKIndex row stay distinct, a duplicate puts
  ##     two additions on one accumulator row inside one scatter_add
  ##     call, an order scatter_add leaves undefined
  ##
  ## The single-token case runs expertForwardDecode instead, a batched
  ## gather path with no host syncs.
  let t = hiddenStates.size(0)
  let topK = topKIndex.size(1)
  let device = hiddenStates.deviceType()

  # TODO: the grouping scan still syncs per routing cell via .item(),
  # reading expert ids and weights on the host. A sync drains the whole
  # device on hardware without unified memory (CUDA). A segment
  # GEMM or grouped GEMM with device-side offsets removes them: gather
  # cost scales as T*K and prices this design out of large T.

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
        checkValue(expertIds[tok * topK + i] != expertIds[tok * topK + j],
          "[ttt] GatedBlockSparseFFN.expertForwardPrefill: topKIndex row " & $tok & " repeats expert id " &
          $expertIds[tok * topK + i] & ", the accumulation order is undefined")

  var finalHiddenStates =
    F.zeros(t, self.hiddenSize, F.tensorOptions(hiddenStates.scalarType(), device))

  let weights32 = topKWeights.to(F.kFloat32)
  for e in 0 ..< self.numExperts:
    # Token group of expert e, pairs ordered by position then token
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

    let gateUpWeight = self.gateUpProj[e]
    let gateUpOut = F.matmul(currentStates, gateUpWeight.t())

    let chunks = F.chunk(gateUpOut, 2, -1)
    let gateChunk = chunks[0]
    let upChunk = chunks[1]
    let act =
      case self.activation
      of kSilu: F.silu(gateChunk) * upChunk
      of kGeluTanh: gelu_tanh(gateChunk) * upChunk

    let downWeight = self.downProj[e]
    let currentHiddenStates = F.matmul(act, downWeight.t())

    # Routing weight per (token, position) pair, read through an fp32
    # form of topKWeights so bf16 values round-trip exactly
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
    let idx2d = tokenIdxTensor.unsqueeze(1).expand(n, self.hiddenSize, implicit = false)
    finalHiddenStates = F.scatter_add(finalHiddenStates, 0, idx2d, weighted)

  result = finalHiddenStates

proc forward*(self: GatedBlockSparseFFN, hidden: Tensor): Tensor =
  ## Routed FFN forward on rank-2 [T, H] or rank-3 [B, T, H] hidden
  ## states, the FFN contribution at the input shape: routing from the
  ## embedded routerWeight, the sigmoid shared-expert gate, the routed
  ## contribution dispatched on the token count (T = 1 -> decode,
  ## T > 1 -> prefill), one reshape back on output.
  checkValue(hidden.dim() == 2 or hidden.dim() == 3,
    "[ttt] GatedBlockSparseFFN.forward: hidden_states must be rank 2 [T, H] or rank 3 [B, T, H], found rank " &
    $hidden.dim())
  checkValue(hidden.size(hidden.dim() - 1) == self.hiddenSize,
    "[ttt] GatedBlockSparseFFN.forward: hidden_states width is " &
    $hidden.size(hidden.dim() - 1) & ", expected hidden size " & $self.hiddenSize)
  let batchTokens = hidden.numel() div self.hiddenSize
  let hiddenStates = hidden.reshape(batchTokens, self.hiddenSize)

  let (topkIndices, routingWeights) =
    routeToExperts(hiddenStates, self.routerWeight, self.numExpertsPerTok)
  let sharedGate = F.sigmoid(F.matmul(hiddenStates, self.sharedGateWeight.t()))
  let sharedGated = sharedGate * self.sharedExpert.forward(hiddenStates)
  let routed =
    if batchTokens == 1:
      expertForwardDecode(RoundBeforeMultiply, self.gateUpProj, self.downProj,
        hiddenStates, topkIndices, routingWeights, self.activation)
    else:
      self.expertForwardPrefill(hiddenStates, topkIndices, routingWeights)
  let output = routed + sharedGated

  if hidden.dim() == 2:
    output
  else:
    output.reshape(hidden.size(0), hidden.size(1), self.hiddenSize)

################################################################################
#   Block-sparse FFN with an ungated shared expert                            #
################################################################################

type
  BlockSparseFFN* = ref object
    ## Block-sparse mixture-of-experts FFN with an optional ungated shared expert
    ## (the DeepSeek-V2/V3 form, Moonlight instantiates it, the cohere lineage ships none)
    ##
    ## - the shared expert output adds unchanged to the routed contribution,
    ##   no multiplicative weight scales it
    ## - the checkpoint carries no shared-expert scaling weight, this form
    ##   carries no GatedBlockSparseFFN-style scale row, a distinct
    ##   typed shape, never a flag nor a dummy weight
    ##
    ## Routed-expert bodies mirror the HF eager expert loop where
    ## the token count allows:
    ## - prefill (T > 1): fused gate/up GEMM per hit expert, SiLU gating,
    ##   down GEMM, then the f32 routing weight multiplies the bf16 expert
    ##   output in f32 and the product rounds to the hidden dtype when it
    ##   joins the accumulator. The gated form rounds the routing
    ##   weight to the hidden dtype BEFORE the multiply, the HF gated
    ##   router computes the routing weights at the hidden dtype.
    ## - decode (T = 1): one routed expert run per top-k position, f32
    ##   accumulator, routing-order sum. The HF eager loop adds one
    ##   contribution in ascending index order on a bf16 accumulator,
    ##   one recorded evaluation-order drift, budgeted.
    gateUpProj: Tensor   ## [E, 2I, H] fused: gate rows 0:I, up rows I:2I
    downProj: Tensor     ## [E, H, I]
    numExperts: int
    hiddenSize: int
    activation: ActivationKind
    router: NoAuxTopCorr ## the typed routing decision source
    sharedExpert: Option[GatedDenseFFN]
      ## Ungated shared-expert tail, present when the checkpoint config routes to one
      ## (n_shared_experts / num_shared_experts > 0), absent otherwise.

func init*(
    _: type BlockSparseFFN,
    gateUpProj: Tensor,
    downProj: Tensor,
    sharedExpert: Option[GatedDenseFFN],
    router: NoAuxTopCorr,
    activation: ActivationKind = kSilu
  ): BlockSparseFFN =
  ## Create the routed FFN from the rank-3 fused expert weights, the optional shared expert and the typed noaux_tc router.
  ##
  ## - the router is a composed typed object, no router weight embedding
  ## - a present shared tail adds unchanged, an absent one routes the whole
  ##   output through the experts
  ##
  ## Raises ValueError:
  ## - gateUpProj or downProj is not rank 3
  ## - downProj is not [E, H, I] against gateUpProj
  checkValue(gateUpProj.dim() == 3,
    "[ttt] BlockSparseFFN.init: gate_up_proj must be rank 3, found rank " &
    $gateUpProj.dim())
  checkValue(downProj.dim() == 3,
    "[ttt] BlockSparseFFN.init: down_proj must be rank 3, found rank " &
    $downProj.dim())

  let e = gateUpProj.size(0)
  let inter = gateUpProj.size(1) div 2
  let h = gateUpProj.size(2)
  checkValue(downProj.size(0) == e and downProj.size(1) == h and
    downProj.size(2) == inter,
    "[ttt] BlockSparseFFN.init: down_proj shape is [" & $downProj.size(0) & ", " &
    $downProj.size(1) & ", " & $downProj.size(2) & "], expected [" & $e & ", " &
    $h & ", " & $inter & "]")

  BlockSparseFFN(
    gateUpProj: gateUpProj,
    downProj: downProj,
    numExperts: e,
    hiddenSize: h,
    activation: activation,
    router: router,
    sharedExpert: sharedExpert
  )

proc expertForwardPrefillPlain(
    self: BlockSparseFFN,
    hiddenStates: Tensor,
    topKIndex: Tensor,
    topKWeights: Tensor
  ): Tensor =
  ## Routed expert compute for multi-token inputs (T > 1), the HF
  ## eager loop: per hit expert e in ascending index order, one expert body
  ## on the token group of e, the f32 routing weight multiplies the expert
  ## output in f32 and the product rounds to the hidden dtype when it joins
  ## the bf16 accumulator.
  ##
  ## Input:
  ##   - hiddenStates: [T, H] at the hidden dtype
  ##   - topKIndex: [T, K] int64 expert ids
  ##   - topKWeights: [T, K] f32 decision weights, pre-cast never
  ##
  ## Output:
  ##   - [T, H], zero rows for unselected tokens
  ##
  ## Preconditions:
  ##   - ids within one topKIndex row stay distinct, a duplicate puts
  ##     two additions on one accumulator row inside one scatter_add
  ##     call, an order scatter_add leaves undefined
  let t = hiddenStates.size(0)
  let topK = topKIndex.size(1)
  let device = hiddenStates.deviceType()

  var expertIds = newSeq[int64](t * topK)
  for tok in 0 ..< t:
    for pos in 0 ..< topK:
      expertIds[tok * topK + pos] = topKIndex[tok, pos].item(int64)

  # Guard: the scatter_add accumulation relies on disjoint token groups
  for tok in 0 ..< t:
    for i in 0 ..< topK:
      for j in (i + 1) ..< topK:
        checkValue(expertIds[tok * topK + i] != expertIds[tok * topK + j],
          "[ttt] BlockSparseFFN.expertForwardPrefillPlain: topKIndex row " & $tok &
          " repeats expert id " & $expertIds[tok * topK + i] &
          ", the accumulation order is undefined")

  var finalHiddenStates =
    F.zeros(t, self.hiddenSize, F.tensorOptions(hiddenStates.scalarType(), device))

  let weights32 = topKWeights.to(F.kFloat32)
  for e in 0 ..< self.numExperts:
    # Token group of expert e, pairs ordered by position then token,
    # the HF reference torch.where(expert_mask[e]) order
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
    let currentHiddenStates = expertBody(
      self.gateUpProj[e], self.downProj[e], currentStates, self.activation)

    # f32 routing weight per (token, position) pair, the product stays f32
    # and rounds once at the accumulated sum on the bf16 accumulator, the
    # HF reference index_add_ rounding location
    var weightVals = newSeq[float32](n)
    for i in 0 ..< n:
      weightVals[i] = weights32[tokenIdx[i], topKPos[i]].item(float32)
    let weightCol = weightVals.toTensor().reshape(n, 1).to(device)
    let weighted = (currentHiddenStates.to(F.kFloat32) * weightCol)
      .to(hiddenStates.scalarType())

    let idx2d = tokenIdxTensor.unsqueeze(1).expand(n, self.hiddenSize, implicit = false)
    finalHiddenStates = F.scatter_add(finalHiddenStates, 0, idx2d, weighted)

  result = finalHiddenStates

proc forward*(self: BlockSparseFFN, hidden: Tensor): Tensor =
  ## Routed FFN forward on the embedded noaux_tc router, the ungated
  ## shared-expert tail. Token rows flatten to [T, H], route, run the
  ## expert bodies and the shared expert, reshape on output; the routed
  ## contribution dispatches on the token count (T = 1 -> decode
  ## RoundAfterSum, T > 1 -> prefill plain).
  checkValue(hidden.dim() == 2 or hidden.dim() == 3,
    "[ttt] BlockSparseFFN.forward: hidden_states must be rank 2 [T, H] or rank 3 [B, T, H], found rank " &
    $hidden.dim())
  checkValue(hidden.size(hidden.dim() - 1) == self.hiddenSize,
    "[ttt] BlockSparseFFN.forward: hidden_states width is " &
    $hidden.size(hidden.dim() - 1) & ", expected hidden size " & $self.hiddenSize)
  let batchTokens = hidden.numel() div self.hiddenSize
  let hiddenStates = hidden.reshape(batchTokens, self.hiddenSize)

  let (_, weights32, topkIndices) =
    if batchTokens == 1:
      routeDecode(self.router, hiddenStates)
    else:
      route(self.router, hiddenStates)

  let routed =
    if batchTokens == 1:
      expertForwardDecode(RoundAfterSum, self.gateUpProj, self.downProj,
        hiddenStates, topkIndices, weights32, self.activation)
    else:
      expertForwardPrefillPlain(self, hiddenStates, topkIndices, weights32)
  let output =
    if self.sharedExpert.isSome():
      routed + self.sharedExpert.unsafeGet().forward(hiddenStates)
    else:
      routed

  if hidden.dim() == 2:
    output
  else:
    output.reshape(hidden.size(0), hidden.size(1), self.hiddenSize)

template `()`*(layer: BlockSparseFFN, hidden: Tensor): untyped =
  forward(layer, hidden)
