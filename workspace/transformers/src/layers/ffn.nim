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
  workspace/libtorch as F,
  workspace/positron,
  workspace/transformers/src/instrumentation,
  ./linear

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
    ## - expertForwardDecode at T = 1, batched gather, no host syncs
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

func routeToExperts*(
    hidden: Tensor,
    routerWeight: Tensor,
    numExpertsPerTok: int
  ): tuple[indices: Tensor, weights: Tensor] =
  ## Router selection over rank-2 hidden states [T, H].
  ##
  ## Returns the per-token top-K expert ids and routing weights:
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

################################################################################
#                 Routed-expert compute: decode and prefill                    #
################################################################################
# Two implementations, dispatched by token count in forward. "Prefill"
# covers every multi-token forward.
#
# Maintainer note, numerical contract for the prefill path: bitwise
# fidelity with the HF reference expert body on CPU.
# - one fused GEMM per hit expert covers the full [2I, H] gate/up
#   weight, narrow per-half GEMMs round differently
# - experts visited in ascending index order, token groups disjoint,
#   so each accumulator row takes exactly one addition per scatter_add
#   call, matching the reference index_add loop
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
# runs its own fused GEMMs, and scatter_add lands the weighted results
# in a zeroed accumulator in ascending expert order.
# Cost scales with hit experts, not with T*K.
#
# Roundoff is the only difference: decode accumulates in routing order,
# prefill in ascending expert order, at most one bf16 ulp on the layer
# output. Comparison: workspace/libtorch/bench/metal/bench_moe.nim,
# weight samples and devices.

# CPU spelling: per-expert narrow + GEMV loop instead of the batched
# gather, measured in workspace/libtorch/bench/cpu/bench_moe_decode.nim
# at the Qwen3.6-35B decode shape: 1.496 to 0.807 ms per call, output
# bit-identical on the measured sample. The batched gather copies the K
# routed expert bodies (50 MB at the bench shape) before reading them
# again, while the loop reads each routed expert in place and .item()
# is a free host read on CPU. The f32 accumulator reproduces the sum(0)
# rounding of the gather spelling, one rounding of the weighted terms.
# MPS keeps the gather spelling in the batched form.

proc expertForwardDecode(
    self: GatedBlockSparseFFN,
    hiddenStates: Tensor,
    topKIndex: Tensor,
    topKWeights: Tensor
  ): Tensor =
  ## Routed expert compute for a single token (T = 1).
  ##
  ## Data flow:
  ##
  ##   gather routed expert weights by top-k id, device-side index_select
  ##     ├→ matmul(hiddenStates, gathered gate_up weight.T)   [K, 2I]
  ##     ├→ chunk(2) → silu(gate) * up                        [K, I]
  ##     ├→ matmul(act, gathered down weight.T)               [K, H]
  ##     └→ multiply the routing weight per position, sum over K
  ##
  ## Input:
  ##   - hiddenStates: [1, H] at the multiply dtype
  ##   - topKIndex: [1, K] int64 expert ids from routeToExperts
  ##   - topKWeights: [1, K] routing weights at the multiply dtype
  ##
  ## Output:
  ##   - [1, H], the sum over selected experts, each term
  ##     down_proj(silu(gate) * up) weighted by its routing weight
  ##
  ## Routing indices never leave the device, so there are no host syncs.
  ## topk never repeats an id within a row, so no duplicate check here.
  ## Accumulation runs in routing order.
  let topK = topKIndex.size(1)
  if hiddenStates.deviceType() == F.kCPU:
    var acc = F.zeros(1, self.hiddenSize, F.kFloat32)
    for pos in 0 ..< topK:
      let e = topKIndex[0, pos].item(int64).int
      let gateUpWeight = self.gateUpProj[e]      # (2I, H) view, no copy
      let gateUpOut = F.matmul(hiddenStates, gateUpWeight.t())
      let chunks = F.chunk(gateUpOut, 2, -1)
      let act =
        case self.activation
        of kSilu: F.silu(chunks[0]) * chunks[1]
      let downOut = F.matmul(act, self.downProj[e].t())
      acc = acc + (downOut * topKWeights[0, pos]).to(F.kFloat32)
    return acc.to(hiddenStates.scalarType())
  let flatIdx = topKIndex.reshape(topK)
  let gatheredGateUp = F.index_select(self.gateUpProj, 0, flatIdx)
  let gatheredDown = F.index_select(self.downProj, 0, flatIdx)
  let xs = hiddenStates.unsqueeze(0)
    .expand(topK, 1, self.hiddenSize, implicit = false)
    .contiguous()
  let gateUpOut = F.matmul(xs, gatheredGateUp.transpose(1, 2))
  let chunks = F.chunk(gateUpOut, 2, -1)
  let act =
    case self.activation
    of kSilu: F.silu(chunks[0]) * chunks[1]
  let downOut = F.matmul(act, gatheredDown.transpose(1, 2))
  let weightCol = topKWeights.transpose(0, 1).unsqueeze(2)
    .to(hiddenStates.scalarType())
  result = (downOut * weightCol).sum(0)

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
    let idx2d = tokenIdxTensor.unsqueeze(1).expand(n, self.hiddenSize, implicit = false)
    finalHiddenStates = F.scatter_add(finalHiddenStates, 0, idx2d, weighted)

  result = finalHiddenStates

proc forward*(self: GatedBlockSparseFFN, hidden: Tensor): Tensor =
  ## Routed FFN forward on rank-2 [T, H] or rank-3 [B, T, H] hidden
  ## states, returning the FFN contribution at the input shape.
  ##
  ## The token rows flatten to the [batchTokens, H] view for the router,
  ## the expert bodies and the shared expert, and reshape back on output.
  ##
  ##   hidden [T, H]
  ##     ├→ routeToExperts(hidden, routerWeight, numExpertsPerTok)
  ##     │    → topkIndices [T, K], routingWeights [T, K]
  ##     ├→ routed contribution, dispatched on the token count:
  ##     │    T = 1 → expertForwardDecode
  ##     │    T > 1 → expertForwardPrefill
  ##     ├→ sharedGate [T, 1] = sigmoid(hidden · sharedGateWeight.T)
  ##     └→ output = routed + sharedGate * sharedExpert.forward(hidden)
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
      self.expertForwardDecode(hiddenStates, topkIndices, routingWeights)
    else:
      self.expertForwardPrefill(hiddenStates, topkIndices, routingWeights)
  let output = routed + sharedGated

  if hidden.dim() == 2:
    output
  else:
    output.reshape(hidden.size(0), hidden.size(1), self.hiddenSize)
