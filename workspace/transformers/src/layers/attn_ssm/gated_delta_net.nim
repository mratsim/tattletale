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
  GatedDeltaNet* = ref object
    ## Gated DeltaNet linear-attention block:
    ## causal conv1d over the fused qkv projection, then a delta-rule
    ## recurrence in f32 with an l2-normalized query/key, and a SiLU-gated
    ## RMSNorm before out_proj. The layer is stateless: per-sequence conv
    ## history and SSM state live in InferenceContext (indexed by layer_idx).
    ##
    ## Papers:
    ## - DeltaNet, arXiv:2406.06484 (Yang et al., 2024).
    ## - Gated DeltaNet, arXiv:2412.06464 (Yang et al., ICLR 2025).
    layer_idx*: int             # Layer index, indexes ctx.gdnConvState/gdnSsmState
    name*: string               # Safetensor key prefix (e.g. "model.language_model.layers.0.linear_attn")
    in_proj_qkv: Linear         # [conv_dim, hidden] fused q/k/v projection
    in_proj_z: Linear           # [num_v_heads * head_v_dim, hidden] norm gate
    in_proj_a: Linear           # [num_v_heads, hidden] decay projection
    in_proj_b: Linear           # [num_v_heads, hidden] beta projection
    conv1d_weight: Tensor       # [conv_dim, 1, conv_kernel_size] bf16 depthwise conv
    a_log: Tensor               # [num_v_heads] log decay, dtype per checkpoint, cast to f32 on use
    dt_bias: Tensor             # [num_v_heads] bf16 discretization bias
    norm: RmsNormGated          # F32 [head_v_dim] gated RMSNorm over value head dim
    out_proj: Linear            # [hidden, num_v_heads * head_v_dim]
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    conv_dim: int

# =============================================================================
# Data flow through GatedDeltaNet
# =============================================================================
#
#   x (batch, seq, hidden)
#   │
#   ├─→ in_proj_qkv → transpose(1,2) → (b, conv_dim, T)
#   │        └─→ causal conv1d (K=4, groups=conv_dim, silu)
#   │              ├─ prefill: left-pad 3 → conv → silu, take first T
#   │              └─ decode:  cat([state(3), x]) → conv → silu, take last 1
#   │        └─→ transpose(1,2) → split q/k/v [k_dim | k_dim | v_dim]
#   │              └─→ q, k (b, T, k_heads, k_dim), v (b, T, v_heads, v_dim)
#   ├─→ in_proj_z → (b, T, v_heads, v_dim)             (norm gate)
#   ├─→ in_proj_a → a, in_proj_b → b                   ((b, T, v_heads))
#   │
#   beta = sigmoid(b) (bf16)
#   g = -exp(A_log) * softplus(a + dt_bias) (f32)
#   (q, k) = l2norm(q, k) (bf16, eps 1e-6)
#   recurrence (f32, per step): S = S*exp(g_t)
#                               S += k_t ⊗ beta_t(v_t − S·k_t)
#                               o_t = S · (q_t * Dk^-0.5)
#   core_attn_out (b, T, v_heads, v_dim) → RmsNormGated(·, silu(z)) → out_proj → (b, T, hidden)
#
# GDN layers never touch ctx.pages: the conv + SSM state is the cache.
# =============================================================================

func init*(
    _: type GatedDeltaNet,
    layer_idx: int,
    name: string,
    in_proj_qkv, in_proj_z, in_proj_a, in_proj_b: Linear,
    conv1d_weight, a_log, dt_bias: Tensor,
    norm: RmsNormGated,
    out_proj: Linear,
    num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size: int
  ): GatedDeltaNet =
  ## Initialize a Gated DeltaNet layer.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1), indexes the per-sequence
  ##     conv/SSM state in InferenceContext
  ##   name: Safetensor key prefix (e.g. "model.language_model.layers.0.linear_attn")
  ##   in_proj_qkv, in_proj_z, in_proj_a, in_proj_b: Preinitialized projections
  ##   conv1d_weight: (conv_dim, 1, conv_kernel_size) bf16 depthwise conv weight
  ##   a_log: (num_v_heads,) log decay, dtype per checkpoint, cast to f32 on use
  ##   dt_bias: (num_v_heads,) bf16 discretization bias
  ##   norm: RmsNormGated over head_v_dim
  ##   out_proj: (hidden, num_v_heads * head_v_dim) projection
  ##   num_k_heads, num_v_heads, head_k_dim, head_v_dim: GDN head geometry
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
  GatedDeltaNet(
    layer_idx: layer_idx,
    name: name,
    in_proj_qkv: in_proj_qkv,
    in_proj_z: in_proj_z,
    in_proj_a: in_proj_a,
    in_proj_b: in_proj_b,
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

func l2norm(x: Tensor): Tensor =
  ## L2-normalize over the last dim in the input dtype (bf16), eps 1e-6.
  ## No dtype conversion: callers cast to f32 after normalize.
  let invNorm = F.rsqrt((x * x).sum(axis = -1, keepdim = true) + Scalar(1e-6))
  x * invNorm

proc gatedDeltaRuleRecurrence(
    q, k, v, g, beta: Tensor,
    initialS: Tensor): (Tensor, Tensor) =
  ## Sequential delta-rule recurrence.
  ##
  ## Args:
  ##   q, k, v: (batch, seq, heads, dim) bf16, l2norm applied to q/k here
  ##   g: (batch, seq, heads) f32 log-space decay
  ##   beta: (batch, seq, heads) bf16 gate
  ##   initialS: (batch, heads, Dk, Dv) f32 SSM state
  ##
  ## Returns:
  ##   (output (batch, seq, heads, Dv) bf16, finalS (batch, heads, Dk, Dv) f32)
  ##
  ## Per step t (all f32):
  ##   S = S * exp(g_t)                       g_t (batch, heads, 1, 1)
  ##   kv_mem = (S * k_t^T).sum(-2)           k_t (batch, heads, Dk)
  ##   delta = (v_t − kv_mem) * beta_t        beta_t (batch, heads, 1)
  ##   S = S + k_t^T * delta^T
  ##   o_t = (S * q_t^T).sum(-2)              q_t scaled by Dk^-0.5
  let initialDtype = q.scalarType()
  # The parallel-form reductions of arXiv:2412.06464 are layout-sensitive:
  # all five inputs must share one layout, dtype, and operation order.
  let q32 = l2norm(q).transpose(1, 2).contiguous().to(kFloat32)
  let k32 = l2norm(k).transpose(1, 2).contiguous().to(kFloat32)
  let v32 = v.transpose(1, 2).contiguous().to(kFloat32)
  let beta32 = beta.transpose(1, 2).contiguous().to(kFloat32)
  let g32 = g.transpose(1, 2).contiguous().to(kFloat32)

  let batch = q32.size(0)
  let numHeads = q32.size(1)
  let seqLen = q32.size(2)
  let vDim = v32.size(3)
  let scale = 1.0 / sqrt(k32.size(3).float64)
  let qScaled = q32 * Scalar(scale)

  var s = initialS
  var coreOut = F.zeros(batch, numHeads, seqLen, vDim,
    F.tensorOptions(kFloat32, q32.deviceType()))
  if seqLen == 1 and q32.deviceType() == F.kCPU:
    # CPU decode spelling (T = 1), measured in
    # workspace/libtorch/bench/cpu/bench_gdn_recurrence.nim:
    # - kv and output reductions run as batched matmuls over the flattened
    #   head batch, and the state update outer product is one bmm,
    #   replacing five full-size elementwise passes and two broadcast
    #   reduces with three full-size ops and three batched reduces
    # - 0.291 to 0.221 ms per step
    # - state drift 2.2 fp32 ulps against the mul + sum spelling
    #   (fixture budget: 4 fp32 ulps)
    # - the math and the f32 core stay unchanged, MPS keeps the loop spelling
    let batchHeads = batch * numHeads
    let dkDim = k32.size(3)
    let gTrow = g32[_, _, 0].exp().unsqueeze(-1)     # (batch, heads, 1)
    let gT4 = gTrow.unsqueeze(-1)                    # (batch, heads, 1, 1)
    let betaT = beta32[_, _, 0].unsqueeze(-1)        # (batch, heads, 1)
    let kT = k32[_, _, 0, _].unsqueeze(-1)           # (batch, heads, Dk, 1)
    let vT = v32[_, _, 0, _]                         # (batch, heads, Dv)
    let qRow = qScaled[_, _, 0, _].unsqueeze(-1)     # (batch, heads, Dk, 1)
    let s3 = s.view(batchHeads, dkDim, vDim)
    let kT3 = kT.view(batchHeads, dkDim, 1)
    let qRow3 = qRow.view(batchHeads, 1, dkDim)
    let kvMem = F.bmm(kT3.transpose(1, 2), s3)
      .view(batch, numHeads, vDim) * gTrow
    let delta = (vT - kvMem) * betaT
    let outer = F.bmm(kT3, delta.view(batchHeads, 1, vDim))
      .view(batch, numHeads, dkDim, vDim)
    s = s * gT4 + outer
    coreOut[_, _, 0, _] = F.bmm(qRow3, s.view(batchHeads, dkDim, vDim))
      .view(batch, numHeads, vDim)
  else:
    for t in 0 ..< seqLen:
      let gT = g32[_, _, t].exp().unsqueeze(-1).unsqueeze(-1)
      let betaT = beta32[_, _, t].unsqueeze(-1)
      let kT = k32[_, _, t, _].unsqueeze(-1)
      let vT = v32[_, _, t, _]
      let qT = qScaled[_, _, t, _].unsqueeze(-1)
      s = s * gT
      let kvMem = (s * kT).sum(axis = -2)
      let delta = (vT - kvMem) * betaT
      s = s + kT * delta.unsqueeze(-2)
      coreOut[_, _, t, _] = (s * qT).sum(axis = -2)
  let output = coreOut.transpose(1, 2).contiguous().to(initialDtype)
  (output, s)

proc forward(
    self: GatedDeltaNet,
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass for the Gated DeltaNet block with per-sequence state.
  ##
  ## Args:
  ##   ctx: InferenceContext holding this layer's conv/SSM state
  ##     (ctx.gdnConvState[layer_idx], ctx.gdnSsmState[layer_idx])
  ##   x: Input tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (batch, seq, hidden_size)
  ##
  ## Decode (seq_len 1) reads the stored conv/SSM state and writes the updated state back.
  ## Prefill (seq_len > 1) starts from the stored state (zeros on a fresh sequence) and overwrites it.
  ## The recurrence is sequential in both cases, so decode after prefill is bit-identical
  ## to a one-shot forward over the same tokens.
  let batch = x.size(0)
  checkValue(batch == 1,
    "[ttt] GDN currently supports batch_size == 1 only, got " & $batch)

  let seqLen = x.size(1)
  let device = x.deviceType()
  ctx.ensureGdnStates(
    self.layer_idx, self.conv_dim,
    self.num_v_heads, self.head_k_dim, self.head_v_dim, device)

  # ── Projections ──
  let mixedQkv = self.in_proj_qkv.forward(x).transpose(1, 2)  # (b, conv_dim, T)
  let z = self.in_proj_z.forward(x).reshape(
    [batch, seqLen, self.num_v_heads, self.head_v_dim])
  let bProj = self.in_proj_b.forward(x)  # (b, T, num_v_heads)
  let aProj = self.in_proj_a.forward(x)  # (b, T, num_v_heads)

  # ── Causal conv1d (K=4, groups=conv_dim, no bias, silu) ──
  var convOut: Tensor
  if seqLen == 1:
    # Decode step: prepend the stored conv context, valid conv, take last 1.
    let state = ctx.gdnConvState[self.layer_idx]  # (conv_dim, 3) bf16
    let catInput = F.cat([state.unsqueeze(0), mixedQkv], -1)  # (b, conv_dim, 4)
    # Depthwise dot spelling of the grouped conv on CPU:
    # - torch's CPU grouped-conv kernel thrashes its thread pool on 8192
    #   tiny per-channel convs: 270 ms vs 6.5 ms single-thread, 0.072 ms
    #   with groups = 1, drill mode,
    #   workspace/libtorch/bench/cpu/bench_decode_stage.nim
    # - elementwise multiply plus a sum over the kernel window avoids
    #   that kernel
    # - products and accumulation run in fp32, mirroring the conv kernel
    #   internally, one rounding to the storage dtype at the output
    # - the fixture recording path is thereby reproduced within the fp32
    #   ulp budget of the fixtures
    # - the MPS grouped-conv kernel is healthy and stays on conv1d
    if device == F.kCPU:
      let kernel = self.conv1d_weight.size(2)
      let wWindow32 = self.conv1d_weight.reshape([1, self.conv_dim, kernel])
        .to(kFloat32)
      let conv32 = (catInput.to(kFloat32) * wWindow32)
        .sum(-1, keepdim = true)  # (b, conv_dim, 1), fp32 accumulation
      convOut = F.silu(conv32.to(x.scalarType()))
    else:
      let conv = F.conv1d(
        catInput, self.conv1d_weight,
        padding = [0], groups = self.conv_dim)
      convOut = F.silu(conv.narrow(2, conv.size(2) - seqLen, seqLen))
    # New state = last conv_kernel_size - 1 positions of the concatenated
    # input.
    ctx.gdnConvState[self.layer_idx] =
      catInput.narrow(2, catInput.size(2) - 3, 3)[0].contiguous()
  else:
    # Prefill: prepend the stored conv history (zeros on a fresh sequence)
    # and take the last seqLen valid outputs, so a continuation after an
    # earlier multi-token forward keeps the preceding positions in the
    # conv window instead of re-zeroing them.
    let state = ctx.gdnConvState[self.layer_idx]  # (conv_dim, 3) bf16, or nil
    let pre =
      if state.isNil:
        F.zeros(batch, self.conv_dim, 3, F.tensorOptions(F.kBFloat16, device))
      else:
        state.unsqueeze(0)
    let padded = F.cat([pre, mixedQkv], -1)  # (b, conv_dim, 3 + T)
    # Depthwise dot spelling on CPU, same rationale as the decode branch:
    # the grouped-conv kernel is pathological there, K shifted multiplies
    # and adds over the sequence slices avoid it entirely. The products
    # and accumulation run in fp32, mirroring the conv kernel internals.
    # One rounding to the storage dtype at the output. The fixture
    # recording path is thereby reproduced inside the fp32 ulp budget.
    if device == F.kCPU:
      let kernel = self.conv1d_weight.size(2)
      let wFlat32 = self.conv1d_weight.reshape([self.conv_dim, kernel])
        .to(kFloat32)
      let padded32 = padded.to(kFloat32)
      var conv32 = padded32.narrow(2, 0, seqLen) *
        wFlat32.narrow(1, 0, 1).reshape([1, self.conv_dim, 1])
      for k in 1 ..< kernel:
        conv32 = conv32 + padded32.narrow(2, k, seqLen) *
          wFlat32.narrow(1, k, 1).reshape([1, self.conv_dim, 1])
      convOut = F.silu(conv32.to(x.scalarType()))
    else:
      let conv = F.conv1d(
        padded, self.conv1d_weight,
        padding = [0], groups = self.conv_dim)
      convOut = F.silu(conv.narrow(2, conv.size(2) - seqLen, seqLen))
    # New state = last 3 positions of the pre-conv input, so a prefill
    # shorter than the kernel still yields a full 3-wide context.
    ctx.gdnConvState[self.layer_idx] =
      padded.narrow(2, padded.size(2) - 3, 3)[0].contiguous()

  # ── Split conv output into q/k/v and reshape to heads ──
  # torch.split(mixed_qkv, [key_dim, key_dim, value_dim], -1): the conv
  # output is [key | key | value], so the split widths follow the head
  # geometry. Three narrows on the last dim are bitwise-equivalent.
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
  # g = -exp(A_log) * softplus(a + dt_bias), all f32 (dt_bias bf16 promotes)
  let aLogExp = self.a_log.to(kFloat32).exp()
  let aPlusBias = aProj.to(kFloat32) + self.dt_bias
  let g = aLogExp.neg() * F.softplus(aPlusBias, 1.0, 20.0)

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
  let (coreAttnOut, finalS) = gatedDeltaRuleRecurrence(
    qFinal, kFinal, value, g, beta, ssmState)
  ctx.gdnSsmState[self.layer_idx] = finalS[0].contiguous()

  # ── Gated RMSNorm over value head dim, then out_proj ──
  let normed = self.norm.forward(coreAttnOut, z)
  let reshaped = normed.reshape([batch, seqLen, self.num_v_heads * self.head_v_dim])
  result = self.out_proj.forward(reshaped)

template `()`*(layer: GatedDeltaNet,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)
