# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/instrumentation,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/stateful/inference_context

{.experimental: "callOperator".}

type
  DecayAxis* = enum
    ## Layout of the log decay g fed the recurrent kernel: one scalar
    ## per head (batch, seq, heads) or one channel per key dim
    ## (batch, seq, heads, dk). The consumer checkpoints split on more
    ## axes than the decay shape. Per-head families use the fused qkv
    ## projection, the fused conv weight and the silu-gated norm
    ## (Qwen3.5, Qwen3.6-MoE). Per-channel families use the separate
    ## q/k/v projections, the three branch conv weights and the
    ## sigmoid-gated output norm (Kimi-Linear, Ling-3.0).
    perHead
    perChannel

  GateForm* = enum
    ## Decay-gate formula applied to a + dt_bias:
    ## softplus: g = -exp(A_log) * softplus(a + dt_bias)
    ##   (Gated DeltaNet, Yang et al., 2024, arXiv:2412.06464).
    ## lowerBoundSigmoid: g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
    ##   (Kimi Delta Attention, Kimi Team, 2025, arXiv:2510.26692).
    softplus
    lowerBoundSigmoid

  FullRankGateIn* = ref object
    ## Full-rank gate-input projection, one Linear.
    proj: Linear

  LowRankGateIn* = ref object
    ## Rank-reduced gate-input pair, the proj_a output feeds proj_b.
    proj_a: Linear
    proj_b: Linear

func init*(_: type FullRankGateIn, proj: Linear): FullRankGateIn =
  FullRankGateIn(proj: proj)

func init*(_: type LowRankGateIn, proj_a, proj_b: Linear): LowRankGateIn =
  LowRankGateIn(proj_a: proj_a, proj_b: proj_b)

template gateInput*(self: FullRankGateIn, x: Tensor): Tensor =
  ## Gate-input projection, (batch, seq, heads) or (batch, seq, heads * dk)
  ## per the projection width, checkpoint dtype.
  self.proj.forward(x)

template gateInput*(self: LowRankGateIn, x: Tensor): Tensor =
  ## Rank-reduced gate input, proj_b(proj_a(x)), checkpoint dtype.
  self.proj_b.forward(self.proj_a.forward(x))

proc computeG*(Decay: static DecayAxis, Form: static GateForm,
    gateIn: FullRankGateIn | LowRankGateIn,
    aLog, dtBias, x: Tensor, lowerBound: float64): Tensor =
  ## Log-space decay from the gate-input projection, A_log and dt_bias.
  ##
  ## softplus: g = -exp(A_log) * softplus(a + dt_bias), the Gated
  ## DeltaNet formula. lowerBoundSigmoid: g = lower_bound *
  ## sigmoid(exp(A_log) * (a + dt_bias)), the Kimi Delta Attention
  ## formula, lower_bound the checkpoint config's kda_lower_bound. a is
  ## the gate-input projection output in the checkpoint dtype, the gate
  ## math runs in f32 with dt_bias promoting, output (batch, seq, heads)
  ## f32 perHead or (batch, seq, heads, dk) f32 perChannel.
  ##
  ## Example: `computeG(perChannel, lowerBoundSigmoid, decayGate, aLog,
  ## dtBias, x, -5.0)` with aLog (heads, 1) f32, dtBias (heads, dk) f32.
  let aLogExp = aLog.to(kFloat32).exp()
  when Decay == perHead:
    let a = gateIn.gateInput(x).to(kFloat32)
  else:
    let heads = aLogExp.size(0)
    let dk = dtBias.numel() div heads
    let a = gateIn.gateInput(x).reshape(
      [x.size(0), x.size(1), heads, dk]).to(kFloat32)
  when Form == lowerBoundSigmoid:
    result = Scalar(lowerBound) * F.sigmoid(aLogExp * (a + dtBias))
  else:
    result = aLogExp.neg() * F.softplus(a + dtBias, 1.0, 20.0)

proc computeG*(Decay: static DecayAxis, Form: static GateForm,
    gateIn: FullRankGateIn | LowRankGateIn,
    aLog, dtBias, x: Tensor): Tensor =
  ## The bound-free call form, the gate formulas that carry no lower
  ## bound (softplus, both decay axes). The lowerBoundSigmoid form must
  ## route its checkpoint's config kda_lower_bound through the
  ## bound-taking overload.
  when Form == lowerBoundSigmoid:
    {.error: "the lowerBoundSigmoid gate takes its checkpoint config" &
      " kda_lower_bound: add the bound argument".}
  # The placeholder bound is never read: the bound formula is statically
  # excluded on this overload.
  computeG(Decay, Form, gateIn, aLog, dtBias, x, 0.0)

type
  GatedDeltaNet*[Decay: static DecayAxis, GateIn: FullRankGateIn | LowRankGateIn, Form: static GateForm] = ref object
    ## Gated delta-rule linear-attention block over a static decay
    ## axis, gate-input variant and gate formula: causal short conv
    ## over the q/k/v projections, then the f32 delta-rule recurrence
    ## with an l2-normalized query/key and a gated RMSNorm ahead
    ## of out_proj. The layer is stateless: per-sequence conv history
    ## and SSM state live in InferenceContext (indexed by layer_idx). One
    ## recurrent kernel serves every sequence length, the decay axis
    ## selects the per-token g reshape, no length dispatch exists.
    ##
    ## Papers:
    ## - DeltaNet, arXiv:2406.06484 (Yang et al., 2024).
    ## - Gated DeltaNet, arXiv:2412.06464 (Yang et al., ICLR 2025).
    layer_idx*: int             # Layer index, indexes ctx.gdn*/kda* state
    name*: string               # Safetensor key prefix
    when Decay == perHead:
      in_proj_qkv: Linear       # [conv_dim, hidden] fused q/k/v projection
      conv1d_weight: Tensor     # [conv_dim, 1, conv_kernel_size] bf16 depthwise conv
      norm: RmsNormGated        # F32 [head_v_dim] silu-gated RMSNorm over value head dim
      conv_dim: int
    else:
      q_proj: Linear            # (heads * dk, hidden), bias-free
      k_proj: Linear
      v_proj: Linear            # (heads * dv, hidden), bias-free
      conv_q: Tensor            # (heads * dk, 1, kernel) bf16 depthwise weight
      conv_k: Tensor
      conv_v: Tensor            # (heads * dv, 1, kernel)
      when Form == lowerBoundSigmoid:
        o_norm: FusedRmsNormGatedSigmoid  # single rounding at the fused sigmoid-gated norm (flash-linear-attention kernel)
      else:
        o_norm: RmsNormGatedSigmoid       # two-rounding form: the sigmoid gate multiplies after the norm rounds
      kda_lower_bound: float64  # config kda_lower_bound: sigmoid decay floor, read by the lowerBoundSigmoid gate, carried unread by softplus
    in_proj_b: Linear           # (heads, hidden) beta projection
    decay_gate: GateIn          # gate-input projections of the log decay
    norm_gate: GateIn           # gate-input projection of the output norm
    a_log: Tensor               # (heads,) perHead or (heads, 1) perChannel, f32 on use
    dt_bias: Tensor             # (heads,) perHead or (heads, dk) perChannel
    out_proj: Linear            # [hidden, num_v_heads * head_v_dim]
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int

# =============================================================================
# Data flow through GatedDeltaNet
# =============================================================================
#
#   x (batch, seq, hidden)
#   │
#   ├─→ (perHead) in_proj_qkv → transpose(1,2) → (b, conv_dim, T)
#   │        └─→ causal conv1d (K=4, groups=conv_dim, silu)
#   │              ├─ prefill: left-pad 3 → conv → silu, take first T
#   │              └─ decode:  cat([state(3), x]) → conv → silu, take last 1
#   │        └─→ transpose(1,2) → split q/k/v [k_dim | k_dim | v_dim]
#   ├─→ (perChannel) q/k/v projections → per-branch causal conv (K=4, silu)
#   │        └─→ head views q, k (b, T, k_heads, k_dim), v (b, T, v_heads, v_dim)
#   ├─→ norm_gate → (b, T, v_heads, v_dim)             (norm gate)
#   ├─→ in_proj_b → beta logits, (b, T, heads)
#   ├─→ decay_gate + a_log + dt_bias → g via computeG
#   │
#   beta = sigmoid(b) (bf16 perHead, f32 perChannel)
#   recurrence (f32, per step): S = S*exp(g_t)
#                               S += k_t^T (x) beta_t(v_t - S.k_t)
#                               o_t = (S * q_t^T).sum(-2)
#   core_attn_out (b, T, heads, dv) → gated RMSNorm(x, norm gate) → out_proj
#
# GDN layers never touch ctx.pages: the conv + SSM state is the cache.
# =============================================================================

func l2norm(x: Tensor): Tensor =
  ## L2-normalize over the last dim in the input dtype, eps 1e-6.
  ## No dtype conversion: callers cast to f32 after normalize.
  let invNorm = F.rsqrt((x * x).sum(axis = -1, keepdim = true) + Scalar(1e-6))
  x * invNorm

# ──────────────────────────────────────────────────────────────────────
# Short conv (k=4, silu, per-branch states):
# ──────────────────────────────────────────────────────────────────────

func shortConvValid(input, convWeight: Tensor): Tensor =
  ## Depthwise causal conv over an already-history-prefixed input:
  ## CPU runs the dot form (K shifted multiplies over the sequence
  ## slices, fp32 accumulation, one rounding to the storage dtype
  ## before the activation); other devices run the grouped conv1d
  ## kernel. Measured comparisons: bench_decode_stage.nim (dot vs
  ## grouped conv on CPU), bench_short_conv_bmm.nim (dot vs bmm).
  let kernel = convWeight.size(2)
  let seqLen = input.size(2) - kernel + 1
  let dim = input.size(1)
  if input.deviceType() == F.kCPU:
    let wFlat32 = convWeight.reshape([dim, kernel]).to(kFloat32)
    let input32 = input.to(kFloat32)
    var conv32 = input32.narrow(2, 0, seqLen) *
      wFlat32.narrow(1, 0, 1).reshape([1, dim, 1])
    for k in 1 ..< kernel:
      conv32 = conv32 + input32.narrow(2, k, seqLen) *
        wFlat32.narrow(1, k, 1).reshape([1, dim, 1])
    F.silu(conv32.to(input.scalarType()))
  else:
    let conv = F.conv1d(input, convWeight, padding = [0], groups = dim)
    F.silu(conv.narrow(2, conv.size(2) - seqLen, seqLen))

proc shortConvStep*(x, convWeight: Tensor, state: Tensor): (Tensor, Tensor) =
  ## One decode step of the depthwise causal short conv.
  ##
  ## Args:
  ##   x: (batch, dim, 1) pre-conv branch input
  ##   convWeight: (dim, 1, kernel) bf16 depthwise weight
  ##   state: (dim, kernel - 1) bf16 branch history
  ##
  ## Returns:
  ##   (conv output (batch, dim, 1) after silu, new state (dim, kernel-1))
  let catInput = F.cat([state.unsqueeze(0), x], -1)
  let convOut = shortConvValid(catInput, convWeight)
  let tail = convWeight.size(2) - 1
  let newState = catInput.narrow(2, catInput.size(2) - tail, tail)[0]
    .contiguous()
  (convOut, newState)

proc shortConvSequence*(x, convWeight: Tensor, state: Tensor): (Tensor, Tensor) =
  ## Multi-token pass of the depthwise causal short conv, the stored
  ## history prepended so positions before the pass stay in the window.
  ##
  ## Args:
  ##   x: (batch, dim, seq) pre-conv branch input
  ##   convWeight: (dim, 1, kernel) bf16 depthwise weight
  ##   state: (dim, kernel - 1) bf16 branch history, nil prepends zeros
  ##
  ## Returns:
  ##   (conv outputs (batch, dim, seq) after silu, new state (dim, k-1))
  let kernel = convWeight.size(2)
  let tail = kernel - 1
  let pre =
    if state.isNil:
      F.zeros(1, x.size(1), tail, F.tensorOptions(F.kBFloat16, x.deviceType()))
    else:
      state.unsqueeze(0)
  let padded = F.cat([pre, x], -1)
  let convOut = shortConvValid(padded, convWeight)
  let newState = padded.narrow(2, padded.size(2) - tail, tail)[0].contiguous()
  (convOut, newState)

# ──────────────────────────────────────────────────────────────────────
# Recurrent kernel, one variant per decay axis
# ──────────────────────────────────────────────────────────────────────

proc decodeStepViews(k32, v32, beta32, qScaled: Tensor):
    tuple[batchHeads: int, betaT, kT, vT, qRow: Tensor] =
  ## T = 1 decode slices shared by both decay axes of the recurrence:
  ## the time-0 views of beta, k, v and the scaled q, each narrowed to
  ## the single token and reshaped to the contraction shapes the
  ## batched matmuls consume (narrow + squeeze forms the integer time
  ## index of the sugar, one stride-identical view).
  result.batchHeads = k32.size(0) * k32.size(1)
  result.betaT = beta32.narrow(2, 0, 1).squeeze(2).unsqueeze(-1)  # (b, h, 1)
  result.kT = k32.narrow(2, 0, 1).squeeze(2).unsqueeze(-1)        # (b, h, Dk, 1)
  result.vT = v32.narrow(2, 0, 1).squeeze(2)                      # (b, h, Dv)
  result.qRow = qScaled.narrow(2, 0, 1).squeeze(2).unsqueeze(-1)  # (b, h, Dk, 1)

proc gatedDeltaRuleRecurrence*(Decay: static DecayAxis,
    q, k, v, g, beta: Tensor, initialS: Tensor): (Tensor, Tensor) =
  ## Sequential delta-rule recurrence. Every sequence length, prefill
  ## included, runs through this one proc.
  ##
  ## Args:
  ##   q, k, v: (batch, seq, heads, dim) checkpoint dtype, l2-normalized here
  ##   g: perHead takes (batch, seq, heads) f32 log-space decay,
  ##      perChannel takes (batch, seq, heads, dk) per-channel decay
  ##   beta: (batch, seq, heads) gate
  ##   initialS: (batch, heads, Dk, Dv) f32 SSM state, nil starts zeros
  ##
  ## Returns:
  ##   (output (batch, seq, heads, Dv) in the q dtype, finalS (batch, heads, Dk, Dv) f32)
  ##
  ## Per step t (all f32): S = S * exp(g_t) FIRST,
  ## then kv_mem = (S * k_t^T).sum(-2), delta = (v_t - kv_mem) * beta_t,
  ##   S = S + k_t^T (x) delta_t, o_t = (S * q_t^T).sum(-2).
  let initialDtype = q.scalarType()
  when Decay == perHead:
    # Qwen3.5 native reference order: the l2norm runs in the input dtype,
    # the f32 staging follows.
    let q32 = l2norm(q).transpose(1, 2).contiguous().to(kFloat32)
    let k32 = l2norm(k).transpose(1, 2).contiguous().to(kFloat32)
  else:
    # Kimi-linear native reference order: the f32 staging runs first,
    # the l2norm follows in f32.
    let q32 = l2norm(q.transpose(1, 2).contiguous().to(kFloat32))
    let k32 = l2norm(k.transpose(1, 2).contiguous().to(kFloat32))
  # All five inputs share one layout, dtype, and operation order.
  let v32 = v.transpose(1, 2).contiguous().to(kFloat32)
  let beta32 = beta.transpose(1, 2).contiguous().to(kFloat32)
  let g32 = g.transpose(1, 2).contiguous().to(kFloat32)

  let batch = q32.size(0)
  let numHeads = q32.size(1)
  let seqLen = q32.size(2)
  let dkDim = k32.size(3)
  let dvDim = v32.size(3)
  # Query scaled by the head dim: the per-channel port divides, the
  # per-head port multiplies by the reciprocal, each keeping one
  # reference order per family.
  var qScaled: Tensor
  when Decay == perChannel:
    qScaled = q32 / sqrt(dkDim.float64)
  else:
    let scale = 1.0 / sqrt(dkDim.float64)
    qScaled = q32 * Scalar(scale)

  var s =
    if initialS.isNil:
      F.zeros(batch, numHeads, dkDim, dvDim,
        F.tensorOptions(kFloat32, q32.deviceType()))
    else:
      initialS.to(kFloat32)
  var coreOut = F.zeros(batch, numHeads, seqLen, dvDim,
    F.tensorOptions(kFloat32, q32.deviceType()))

  if seqLen == 1 and q32.deviceType() == F.kCPU:
    # T = 1 decode slices shared by both decay axes: the per-token
    # views of the decay, beta, k, v and q, each squeezed to the
    # recurrence's contraction shapes.
    when Decay == perChannel:
      let gTrow = g32.narrow(2, 0, 1).squeeze(2).exp() # (batch, heads, dk)
    else:
      let gTrow = g32.narrow(2, 0, 1).squeeze(2).exp().unsqueeze(-1)  # (b, h, 1)
    let gT4 = gTrow.unsqueeze(-1)
    let (batchHeads, betaT, kT, vT, qRow) =
      decodeStepViews(k32, v32, beta32, qScaled)
    let s3 = s.view(batchHeads, dkDim, dvDim)
    let kT3 = kT.view(batchHeads, dkDim, 1)
    let qRow3 = qRow.view(batchHeads, 1, dkDim)
    when Decay == perChannel:
      # CPU decode path (T = 1) of the per-channel decay: the decay is
      # (batch, heads, dk) channel-wise on the key axis and cannot
      # factor out of the k^T contraction, so it decays the state
      # before the memory read.
      let sDecayed = (s * gT4).view(batchHeads, dkDim, dvDim)
      let kvMem = F.bmm(kT3.transpose(1, 2), sDecayed)
        .view(batch, numHeads, dvDim)
      let delta = (vT - kvMem) * betaT
      let outer = F.bmm(kT3, delta.view(batchHeads, 1, dvDim))
        .view(batch, numHeads, dkDim, dvDim)
      s = sDecayed.view(batch, numHeads, dkDim, dvDim) + outer
    else:
      # CPU decode path (T = 1) of the per-head decay: the decay is one
      # scalar per head, it commutes with the k^T contraction and
      # factors out of it. The kv and output reductions run as batched
      # matmuls over the flattened head batch, measured in
      # workspace/libtorch/bench/cpu/bench_gdn_recurrence.nim.
      let kvMem = F.bmm(kT3.transpose(1, 2), s3)
        .view(batch, numHeads, dvDim) * gTrow
      let delta = (vT - kvMem) * betaT
      let outer = F.bmm(kT3, delta.view(batchHeads, 1, dvDim))
        .view(batch, numHeads, dkDim, dvDim)
      s = s * gT4 + outer
    coreOut.narrow(2, 0, 1).copyFrom(
      F.bmm(qRow3, s.view(batchHeads, dkDim, dvDim))
        .view(batch, numHeads, dvDim).unsqueeze(2))
  else:
    for t in 0 ..< seqLen:
      # narrow + squeeze forms the integer time index of the sugar, one
      # stride-identical view, the decay axis selects only the g reshape.
      var gT: Tensor
      when Decay == perChannel:
        gT = g32.narrow(2, t, 1).squeeze(2).exp().unsqueeze(-1)  # (b, h, dk, 1)
      else:
        gT = g32.narrow(2, t, 1).squeeze(2).exp().unsqueeze(-1).unsqueeze(-1)  # (b,h,1,1)
      let betaT = beta32.narrow(2, t, 1).squeeze(2).unsqueeze(-1)
      let kT = k32.narrow(2, t, 1).squeeze(2).unsqueeze(-1)
      let vT = v32.narrow(2, t, 1).squeeze(2)
      let qT = qScaled.narrow(2, t, 1).squeeze(2).unsqueeze(-1)
      s = s * gT
      let kvMem = (s * kT).sum(axis = -2)
      let delta = (vT - kvMem) * betaT
      s = s + kT * delta.unsqueeze(-2)
      coreOut.narrow(2, t, 1).copyFrom(((s * qT).sum(axis = -2)).unsqueeze(2))
  let output = coreOut.transpose(1, 2).contiguous().to(initialDtype)
  (output, s)

# ──────────────────────────────────────────────────────────────────────
# Init:
# ──────────────────────────────────────────────────────────────────────

proc init*[GateIn](_: type GatedDeltaNet[perHead, GateIn, GateForm.softplus],
    layer_idx: int,
    name: string,
    in_proj_qkv, in_proj_z, in_proj_a, in_proj_b: Linear,
    conv1d_weight, a_log, dt_bias: Tensor,
    norm: RmsNormGated,
    out_proj: Linear,
    num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size: int
  ): GatedDeltaNet[perHead, GateIn, GateForm.softplus] =
  ## Initialize a per-head Gated DeltaNet layer.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1), indexes the per-sequence
  ##     conv/SSM state in InferenceContext
  ##   name: Safetensor key prefix
  ##   in_proj_qkv, in_proj_z, in_proj_a, in_proj_b: Preinitialized projections
  ##   conv1d_weight: (conv_dim, 1, conv_kernel_size) bf16 depthwise conv weight
  ##   a_log: (num_v_heads,) log decay, dtype per checkpoint, cast to f32 on use
  ##   dt_bias: (num_v_heads,) bf16 discretization bias
  ##   norm: RmsNormGated over head_v_dim
  ##   out_proj: (hidden, num_v_heads * head_v_dim) projection
  ##   num_k_heads, num_v_heads, head_k_dim, head_v_dim: GDN head dims
  ##   conv_kernel_size: causal conv kernel width (4)
  ##
  ## Raises ValueError naming the layer key path:
  ## - a non-positive head count
  ## - a value-head count that is not a multiple of the key-head count
  #
  # Head-count checks run before the conv-dim arithmetic: a zero count
  # satisfies the alignment check vacuously.
  checkValue(num_k_heads > 0,
    "[ttt] " & name & ": linear_num_key_heads is " & $num_k_heads &
    ", expected a positive count")
  checkValue(num_v_heads > 0,
    "[ttt] " & name & ": linear_num_value_heads is " & $num_v_heads &
    ", expected a positive count")
  checkValue(num_v_heads mod num_k_heads == 0,
    "[ttt] " & name & ": linear_num_value_heads (" & $num_v_heads &
    ") is not a multiple of linear_num_key_heads (" & $num_k_heads & ")")
  let convDim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
  # conv_dim sizes the conv state and the q/k/v split, so the fused qkv
  # projection and the conv weight must agree on it.
  checkValue(in_proj_qkv.out_features == convDim,
    "[ttt] GDN in_proj_qkv out_features is " & $in_proj_qkv.out_features &
    ", expected conv_dim " & $convDim)
  checkValue(conv1d_weight.size(0) == convDim,
    "[ttt] GDN conv1d weight channels is " & $conv1d_weight.size(0) &
    ", expected conv_dim " & $convDim)
  GatedDeltaNet[perHead, GateIn, GateForm.softplus](
    layer_idx: layer_idx,
    name: name,
    in_proj_qkv: in_proj_qkv,
    in_proj_b: in_proj_b,
    decay_gate: FullRankGateIn.init(in_proj_a),
    norm_gate: FullRankGateIn.init(in_proj_z),
    conv1d_weight: conv1d_weight,
    a_log: a_log,
    dt_bias: dt_bias,
    norm: norm,
    out_proj: out_proj,
    num_k_heads: num_k_heads,
    num_v_heads: num_v_heads,
    head_k_dim: head_k_dim,
    head_v_dim: head_v_dim,
    conv_kernel_size: conv_kernel_size,
    conv_dim: convDim
  )

proc init*[GateIn: FullRankGateIn | LowRankGateIn, Form: static GateForm](
    _: type GatedDeltaNet[perChannel, GateIn, Form],
    layer_idx: int,
    name: string,
    decay_gate, norm_gate: GateIn,
    q_proj, k_proj, v_proj, b_proj, o_proj: Linear,
    o_norm: RmsNormGatedSigmoid | FusedRmsNormGatedSigmoid,
    conv_q, conv_k, conv_v, a_log, dt_bias: Tensor,
    num_heads, head_k_dim, head_v_dim, conv_kernel_size: int,
    kda_lower_bound: float64 = 0.0
  ): GatedDeltaNet[perChannel, GateIn, Form] =
  ## Initialize a per-channel Kimi Delta Attention layer. The static
  ## Form selects the gate formula and its o_norm class: softplus pairs
  ## with RmsNormGatedSigmoid, the two-rounding output norm;
  ## lowerBoundSigmoid pairs with FusedRmsNormGatedSigmoid, the
  ## single-rounding form of the flash-linear-attention kernel.
  ##
  ## Args:
  ##   decay_gate, norm_gate: Preinitialized gate-input projections
  ##   a_log: (heads, 1) f32 log decay, normalized by the load convention
  ##   dt_bias: (heads, dk) f32 discretization bias
  ##   o_norm: the Form's output-norm class over head_v_dim
  ##   kda_lower_bound: the checkpoint config's kda_lower_bound; read by
  ##     the lowerBoundSigmoid gate, carried unread by softplus
  ##
  ## Raises ValueError naming the layer key path:
  ## - a non-positive head count or a kernel width not above 1
  ## - projection widths that disagree with the head dims
  ## - conv weights whose channel width disagrees with the projections
  ## - a lowerBoundSigmoid layer whose bound is not negative
  when Form == GateForm.softplus:
    when o_norm is not RmsNormGatedSigmoid:
      {.error: "the softplus form takes RmsNormGatedSigmoid, the two-rounding output norm".}
  else:
    when o_norm is not FusedRmsNormGatedSigmoid:
      {.error: "the lowerBoundSigmoid form takes FusedRmsNormGatedSigmoid, the single-rounding output norm".}
    checkValue(kda_lower_bound < 0.0,
      "[ttt] " & name & ": kda_lower_bound is " & $kda_lower_bound &
      ", expected a negative sigmoid decay floor: the gate is the bound" &
      " times a sigmoid, a non-negative bound compounds the decay past 1")
  let keyDim = num_heads * head_k_dim
  let valueDim = num_heads * head_v_dim
  checkValue(num_heads > 0 and head_k_dim > 0 and head_v_dim > 0 and
    conv_kernel_size > 1,
    "[ttt] " & name & ": KDA head counts must be positive and the kernel" &
    " width above 1")
  checkValue(q_proj.out_features == keyDim and k_proj.out_features == keyDim,
    "[ttt] " & name & ": q_proj/k_proj out_features " & $q_proj.out_features &
    "/" & $k_proj.out_features & ", expected the key span " & $keyDim)
  checkValue(v_proj.out_features == valueDim and o_proj.in_features == valueDim,
    "[ttt] " & name & ": v_proj out_features " & $v_proj.out_features &
    " and o_proj in_features " & $o_proj.in_features &
    ", expected the value span " & $valueDim)
  checkValue(b_proj.out_features == num_heads,
    "[ttt] " & name & ": b_proj out_features " & $b_proj.out_features &
    ", expected the head count " & $num_heads)
  checkValue(conv_q.size(0) == keyDim and conv_k.size(0) == keyDim and
    conv_v.size(0) == valueDim and
    conv_q.size(2) == conv_kernel_size and
    conv_k.size(2) == conv_kernel_size and
    conv_v.size(2) == conv_kernel_size,
    "[ttt] " & name & ": conv weights (" & $conv_q.size(0) & ", " &
    $conv_k.size(0) & ", " & $conv_v.size(0) & ") channels and kernel " &
    $conv_q.size(2) & " disagree with the head dims")
  GatedDeltaNet[perChannel, GateIn, Form](
    layer_idx: layer_idx,
    name: name,
    q_proj: q_proj,
    k_proj: k_proj,
    v_proj: v_proj,
    in_proj_b: b_proj,
    decay_gate: decay_gate,
    norm_gate: norm_gate,
    o_norm: o_norm,
    conv_q: conv_q,
    conv_k: conv_k,
    conv_v: conv_v,
    a_log: a_log,
    dt_bias: dt_bias,
    out_proj: o_proj,
    num_k_heads: num_heads,
    num_v_heads: num_heads,
    head_k_dim: head_k_dim,
    head_v_dim: head_v_dim,
    conv_kernel_size: conv_kernel_size,
    kda_lower_bound: kda_lower_bound
  )

# ──────────────────────────────────────────────────────────────────────
# Forward
# ──────────────────────────────────────────────────────────────────────

proc convBranches[Decay: static DecayAxis, GateIn: FullRankGateIn | LowRankGateIn, Form: static GateForm](
    self: GatedDeltaNet[Decay, GateIn, Form], ctx: var InferenceContext,
    qFlat, kFlat, vFlat: Tensor, seqLen: int): (Tensor, Tensor, Tensor) =
  ## Per-branch causal short conv with the stored branch histories
  ## and silu activation, then the state write-back of the last
  ## kernel-1 positions of each history-prefixed branch input.
  var states = ctx.kdaConvState[self.layer_idx]
  let (outQ, stateQ) =
    if seqLen == 1:
      shortConvStep(qFlat, self.conv_q, states[0])
    else:
      shortConvSequence(qFlat, self.conv_q, states[0])
  let (outK, stateK) =
    if seqLen == 1:
      shortConvStep(kFlat, self.conv_k, states[1])
    else:
      shortConvSequence(kFlat, self.conv_k, states[1])
  let (outV, stateV) =
    if seqLen == 1:
      shortConvStep(vFlat, self.conv_v, states[2])
    else:
      shortConvSequence(vFlat, self.conv_v, states[2])
  states[0] = stateQ
  states[1] = stateK
  states[2] = stateV
  ctx.kdaConvState[self.layer_idx] = states
  (outQ, outK, outV)

proc forward[Decay: static DecayAxis, GateIn: FullRankGateIn | LowRankGateIn, Form: static GateForm](
    self: GatedDeltaNet[Decay, GateIn, Form],
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass with per-sequence state.
  ##
  ## Args:
  ##   ctx: InferenceContext holding this layer's conv/SSM state
  ##   x: Input tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (batch, seq, hidden_size)
  ##
  ## Decode (seq_len 1) reads the stored conv/SSM state and writes
  ## updated state back. Prefill (seq_len > 1) starts from the stored
  ## state (zeros on a fresh sequence) and overwrites it. Both cases
  ## run the sequential recurrence, so decode after prefill
  ## is bit-identical to a one-shot forward over the same tokens.
  let batch = x.size(0)
  checkValue(batch == 1,
    "[ttt] Gated delta-rule mixers currently support batch_size == 1 only, got " &
    $batch)
  when Decay == perHead:
    let seqLen = x.size(1)
    let device = x.deviceType()
    ctx.ensureGdnStates(
      self.layer_idx, self.conv_dim,
      self.num_v_heads, self.head_k_dim, self.head_v_dim, device)

    # ── Projections ──
    let mixedQkv = self.in_proj_qkv.forward(x).transpose(1, 2)  # (b, conv_dim, T)
    let z = self.norm_gate.gateInput(x).reshape(
      [batch, seqLen, self.num_v_heads, self.head_v_dim])
    let bProj = self.in_proj_b.forward(x)  # (b, T, num_v_heads)

    # ── Causal conv1d (K=4, groups=conv_dim, no bias, silu) ──
    # One fused depthwise conv over the packed qkv, the stored conv
    # history prepended; the helpers carry the device split (dot form
    # on CPU, grouped conv1d elsewhere) and the state write-back.
    var convOut: Tensor
    if seqLen == 1:
      let (stepOut, newState) = shortConvStep(
        mixedQkv, self.conv1d_weight, ctx.gdnConvState[self.layer_idx])
      convOut = stepOut
      ctx.gdnConvState[self.layer_idx] = newState
    else:
      let (seqOut, newState) = shortConvSequence(
        mixedQkv, self.conv1d_weight, ctx.gdnConvState[self.layer_idx])
      convOut = seqOut
      ctx.gdnConvState[self.layer_idx] = newState

    # ── Split conv output into q/k/v and reshape to heads ──
    # The conv output is [key | key | value], three narrows on the last dim
    # are bitwise-equivalent to torch.split.
    let convT = convOut.transpose(1, 2)
    let keyDim = self.num_k_heads * self.head_k_dim
    let valueDim = self.num_v_heads * self.head_v_dim
    let queryFlat = convT.narrow(2, 0, keyDim)
    let keyFlat = convT.narrow(2, keyDim, keyDim)
    let valueFlat = convT.narrow(2, 2 * keyDim, valueDim)
    let query = queryFlat.reshape([batch, seqLen, self.num_k_heads, self.head_k_dim])
    let key = keyFlat.reshape([batch, seqLen, self.num_k_heads, self.head_k_dim])
    let value = valueFlat.reshape([batch, seqLen, self.num_v_heads, self.head_v_dim])

    # ── Gates ──
    let beta = F.sigmoid(bProj)  # bf16
    let g = computeG(Decay, Form, self.decay_gate, self.a_log, self.dt_bias, x)

    # Value heads per key head: one shared recurrent-state slot per group.
    # The init refuses a non-multiple pair, the division is exact here.
    let headGroupRatio = self.num_v_heads div self.num_k_heads
    let qFinal =
      if headGroupRatio > 1:
        query.repeat_interleave(headGroupRatio, 2)
      else:
        query
    let kFinal =
      if headGroupRatio > 1:
        key.repeat_interleave(headGroupRatio, 2)
      else:
        key

    # ── Sequential delta-rule recurrence with stored SSM state ──
    let ssmState = ctx.gdnSsmState[self.layer_idx].unsqueeze(0)  # (1, H, Dk, Dv) f32
    let (coreAttnOut, finalS) = gatedDeltaRuleRecurrence(Decay,
      qFinal, kFinal, value, g, beta, ssmState)
    ctx.gdnSsmState[self.layer_idx] = finalS[0].contiguous()

    # ── Gated RMSNorm over value head dim, then out_proj ──
    let normed = self.norm.forward(coreAttnOut, z)
    let reshaped = normed.reshape([batch, seqLen, self.num_v_heads * self.head_v_dim])
    result = self.out_proj.forward(reshaped)
  else:
    let seqLen = x.size(1)
    let device = x.deviceType()
    let keyDim = self.num_k_heads * self.head_k_dim
    let valueDim = self.num_v_heads * self.head_v_dim
    ctx.ensureKdaStates(
      self.layer_idx, keyDim, keyDim, valueDim,
      self.num_v_heads, self.head_k_dim, self.head_v_dim,
      self.conv_kernel_size, device)

    # Projections (batch, seq, span) each, bias-free. The transpose
    # yields the (batch, span, seq) conv layout.
    let qFlat = self.q_proj.forward(x).transpose(1, 2)
    let kFlat = self.k_proj.forward(x).transpose(1, 2)
    let vFlat = self.v_proj.forward(x).transpose(1, 2)

    # Per-branch causal short conv with silu.
    let (convQ, convK, convV) =
      self.convBranches(ctx, qFlat, kFlat, vFlat, seqLen)

    # Head views: q and k use the key dims, v the value dims.
    let query = convQ.transpose(1, 2).reshape(
      [batch, seqLen, self.num_k_heads, self.head_k_dim])
    let key = convK.transpose(1, 2).reshape(
      [batch, seqLen, self.num_k_heads, self.head_k_dim])
    let value = convV.transpose(1, 2).reshape(
      [batch, seqLen, self.num_v_heads, self.head_v_dim])

    # Gates: beta from the sigmoid of the f32-cast beta logits, the log
    # decay from computeG over the config lower bound, both before the
    # kernel.
    let beta = F.sigmoid(self.in_proj_b.forward(x).to(kFloat32))
    let g = computeG(Decay, Form, self.decay_gate, self.a_log,
      self.dt_bias, x, self.kda_lower_bound)

    # Delta rule over the stored state, the recurrent kernel serves
    # every sequence length.
    let ssmState = ctx.kdaSsmState[self.layer_idx].unsqueeze(0)
    let (coreAttnOut, finalS) = gatedDeltaRuleRecurrence(Decay,
      query, key, value, g, beta, ssmState)
    ctx.kdaSsmState[self.layer_idx] = finalS[0].contiguous()

    # Sigmoid-gated RMSNorm over the value head dim, then out_proj.
    let normGate = self.norm_gate.gateInput(x).reshape(
      [batch, seqLen, self.num_v_heads, self.head_v_dim])
    let normed = self.o_norm.forward(coreAttnOut, normGate)
    let reshaped = normed.reshape(
      [batch, seqLen, self.num_v_heads * self.head_v_dim])
    result = self.out_proj.forward(reshaped)

template `()`*[Decay: static DecayAxis, GateIn: FullRankGateIn | LowRankGateIn, Form: static GateForm](
    layer: GatedDeltaNet[Decay, GateIn, Form],
    ctx: var InferenceContext,
    x: Tensor): untyped =
  layer.forward(ctx, x)
