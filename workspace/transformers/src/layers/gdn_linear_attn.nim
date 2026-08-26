# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/stateful/inference_context

{.experimental: "callOperator".}

type
  GatedDeltaNet* = ref object
    ## Gated DeltaNet linear-attention block (Qwen3.5 hybrid layers):
    ## causal conv1d over the fused qkv projection, then a delta-rule
    ## recurrence in f32 with an l2-normalized query/key, and a SiLU-gated
    ## RMSNorm before out_proj. The layer is stateless: per-sequence conv
    ## history and SSM state live in InferenceContext (indexed by layer_idx).
    layer_idx*: int             # Layer index, indexes ctx.gdnConvState/gdnSsmState
    name*: string               # Safetensor key prefix (e.g. "model.language_model.layers.0.linear_attn")
    in_proj_qkv: Linear         # [6144, 1024] fused q/k/v projection
    in_proj_z: Linear           # [2048, 1024] norm gate projection
    in_proj_a: Linear           # [16, 1024] decay projection
    in_proj_b: Linear           # [16, 1024] beta projection
    conv1d_weight: Tensor       # [conv_dim, 1, 4] bf16, depthwise causal conv, no bias
    a_log: Tensor               # [16] f32 log decay
    dt_bias: Tensor             # [16] bf16 discretization bias
    norm: RmsNormGated          # F32 [128] gated RMSNorm over value head dim
    out_proj: Linear            # [1024, 2048]
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    conv_dim: int

# =============================================================================
# Data flow through GatedDeltaNet (vendored Qwen3_5GatedDeltaNet.forward)
# =============================================================================
#
#   x (batch, seq, hidden)
#   │
#   ├─→ in_proj_qkv → transpose(1,2) → (b, conv_dim, T)
#   │        └─→ causal conv1d (K=4, groups=conv_dim, silu)
#   │              ├─ prefill: left-pad 3 → conv → silu, take first T
#   │              └─ decode:  cat([state(3), x]) → conv → silu, take last 1
#   │        └─→ transpose(1,2) → split q/k/v (2048/2048/2048) → (b, T, 16, 128)
#   ├─→ in_proj_z → (b, T, 16, 128)                    (norm gate)
#   ├─→ in_proj_a → a, in_proj_b → b                   ((b, T, 16))
#   │
#   beta = sigmoid(b) (bf16)
#   g = -exp(A_log) * softplus(a + dt_bias) (f32)
#   (q, k) = l2norm(q, k) (bf16, eps 1e-6)
#   recurrence (f32, per step): S = S*exp(g_t)
#                               S += k_t ⊗ beta_t(v_t − S·k_t)
#                               o_t = S · (q_t * Dk^-0.5)
#   core_attn_out (b, T, 16, 128) → RmsNormGated(·, silu(z)) → out_proj → (b, T, 1024)
#
# GDN layers never touch ctx.pages: the conv + SSM state IS the cache.
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
  ##   a_log: (num_v_heads,) f32 log decay
  ##   dt_bias: (num_v_heads,) bf16 discretization bias
  ##   norm: RmsNormGated over head_v_dim
  ##   out_proj: (hidden, num_v_heads * head_v_dim) projection
  ##   num_k_heads, num_v_heads, head_k_dim, head_v_dim: GDN head geometry
  ##   conv_kernel_size: causal conv kernel width (4)
  let convDim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
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
  ## Matches the vendored l2norm (FLA alignment): all arithmetic stays in
  ## the input dtype and the cast to f32 happens after, so the rounding is
  ## identical to the fixture.
  let invNorm = F.rsqrt((x * x).sum(axis = -1, keepdim = true) + Scalar(1e-6))
  x * invNorm

proc gatedDeltaRuleRecurrence(
    q, k, v, g, beta: Tensor,
    initialS: Tensor): (Tensor, Tensor) =
  ## Sequential delta-rule recurrence, the vendored
  ## `torch_recurrent_gated_delta_rule` with use_qk_l2norm_in_kernel=True.
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
  # Vendored staging: transpose(1,2) → contiguous → to(f32), same order for
  # all five inputs so the f32 sums are bit-identical.
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
  ## Decode (seq_len 1) reads the stored conv/SSM state and writes the
  ## updated state back. Prefill (seq_len > 1) starts from the stored state
  ## (zeros on a fresh sequence) and overwrites it. The recurrence is
  ## sequential in both cases, so decode after prefill is bit-identical to a
  ## one-shot forward over the same tokens.
  let batch = x.size(0)
  if batch != 1:
    raise newException(ValueError,
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
    let conv = F.conv1d(
      catInput, self.conv1d_weight,
      padding = [0], groups = self.conv_dim)
    convOut = F.silu(conv.narrow(2, conv.size(2) - seqLen, seqLen))
    # New state = last conv_kernel_size - 1 positions of the concatenated input.
    ctx.gdnConvState[self.layer_idx] =
      catInput.narrow(2, catInput.size(2) - 3, 3)[0].contiguous()
  else:
    # Prefill: causal conv with 3-position left padding, take first T.
    let conv = F.conv1d(
      mixedQkv, self.conv1d_weight,
      padding = [3], groups = self.conv_dim)
    convOut = F.silu(conv.narrow(2, 0, seqLen))
    # New state = last 3 positions of the zero-padded pre-conv input, so a
    # prefill shorter than the kernel still yields a full 3-wide context.
    let padded = F.cat([
      F.zeros(batch, self.conv_dim, 3, F.tensorOptions(F.kBFloat16, device)),
      mixedQkv], -1)
    ctx.gdnConvState[self.layer_idx] =
      padded.narrow(2, padded.size(2) - 3, 3)[0].contiguous()

  # ── Split conv output into q/k/v and reshape to heads ──
  let split = F.chunk(convOut.transpose(1, 2), 3, -1)  # (b, T, 2048) each
  let query = split[0].reshape([batch, seqLen, self.num_k_heads, self.head_k_dim])
  let key = split[1].reshape([batch, seqLen, self.num_k_heads, self.head_k_dim])
  let value = split[2].reshape([batch, seqLen, self.num_v_heads, self.head_v_dim])

  # ── Gates ──
  let beta = F.sigmoid(bProj)  # bf16
  # g = -exp(A_log) * softplus(a + dt_bias), all f32 (dt_bias bf16 promotes)
  let aLogExp = self.a_log.to(kFloat32).exp()
  let aPlusBias = aProj.to(kFloat32) + self.dt_bias
  let g = aLogExp.neg() * F.softplus(aPlusBias, 1.0, 20.0)

  # Guard for num_v_heads // num_k_heads > 1 (not the case here: 16 // 16).
  let qFinal =
    if self.num_v_heads div self.num_k_heads > 1:
      query.repeat_interleave(self.num_v_heads div self.num_k_heads, 2)
    else:
      query
  let kFinal =
    if self.num_v_heads div self.num_k_heads > 1:
      key.repeat_interleave(self.num_v_heads div self.num_k_heads, 2)
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
