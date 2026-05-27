# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/options,
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/quantizations/datatypes {.all.},
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  ./rope

{.experimental: "callOperator".}

type
  GroupedQueryAttention = object
    head_dim: int
    num_qo_head: int
    num_kv_head: int
    num_kv_groups: int
    qo_attn_dim: int
    kv_attn_dim: int
    softmax_scale: float64

  RopeGQAttention* = ref object
    ## Rope + Grouped Query Attention.
    ##
    ## State is external (InferenceContext), not owned by this layer.
    layer_idx*: int             # Layer index for self-indexing KV cache
    name*: string               # Safetensor key prefix (e.g., "model.layers.23.self_attn")
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    gqa_attn: GroupedQueryAttention
    rotary: RotaryPositionEmbeddingRef
    q_norm: Option[RmsNorm]
    k_norm: Option[RmsNorm]

# =============================================================================
# Data flow through RopeGQAttention
# =============================================================================
#
#   x (batch, seq, hidden)
#   │
#   ├─→ q_proj → reshape → q_norm → applyRope → q_rot ────────────────────┐
#   │                                                                     │
#   ├─→ k_proj → reshape → k_norm → applyRope → k_rot ─→ cache.write ◄────┤
#   │                                         offset from ctx.position_ids│
#   │                                            cache.read ─→ k_full     │
#   │                                                                     │
#   └─→ v_proj → reshape ─────────────────→ cache.write ◄─────────────────┘
#                                                cache.read ─→ v_full
#
#   q_rot, k_full, v_full → SDPA(is_causal, enable_gqa) → attn_out
#   attn_out → o_proj → output
#
# Note: V is NOT rotated. RoPE (Su et al., 2021) only rotates Q and K
# because the attention mechanism computes similarity via q·k. V simply
# carries content to be aggregated by attention weights — it has no
# positional role in the similarity computation.
# =============================================================================

func init(_: type GroupedQueryAttention, num_qo_head, num_kv_head, head_dim: int): GroupedQueryAttention =
  let num_kv_groups = num_qo_head div num_kv_head
  GroupedQueryAttention(
    head_dim: head_dim,
    num_qo_head: num_qo_head,
    num_kv_head: num_kv_head,
    num_kv_groups: num_kv_groups,
    qo_attn_dim: num_qo_head * head_dim,
    kv_attn_dim: num_kv_head * head_dim,
    softmax_scale: 1.0'f64 / sqrt(head_dim.float64)
  )

func forward(
      self: GroupedQueryAttention,
      q: Tensor,
      k: Tensor,
      v: Tensor,
      is_causal: bool = true,
      attn_mask = none(Tensor),
      dropout_p = 0.0'f64): Tensor =
  ## Scaled dot-product attention with GQA support.
  ##
  ## Args:
  ##   q: Query tensor of shape (batch, seq, num_qo_head, head_dim)
  ##   k: Key tensor of shape (batch, seq, num_kv_head, head_dim)
  ##   v: Value tensor of shape (batch, seq, num_kv_head, head_dim)
  ##   is_causal: If true, apply causal mask
  ##
  ## Returns:
  ##   Attention output of shape (batch, seq, num_qo_head * head_dim)

  # Backend: permute to (batch, head, seq, head_dim), ensure dtype, SDPA, reshape
  let batch = q.size(0)
  let seq_len = q.size(1)

  var q_attn = q.permute([0, 2, 1, 3])
  var k_attn = k.permute([0, 2, 1, 3])
  let v_attn = v.permute([0, 2, 1, 3])

  let target_dtype = v_attn.scalarType()
  let q_final = q_attn.to(target_dtype)
  let k_final = k_attn.to(target_dtype)

  let attn_out = F.scaled_dot_product_attention(
    q_final, k_final, v_attn,
    attn_mask = attn_mask,
    dropout_p = dropout_p,
    is_causal = is_causal,
    scale = some(self.softmax_scale),
    enable_gqa = self.num_kv_groups > 1
  )

  let attn_perm = attn_out.permute([0, 2, 1, 3])
  result = attn_perm.reshape([batch, seq_len, self.qo_attn_dim])

template `()`*(layer: GroupedQueryAttention,
            q, k, v: Tensor,
            is_causal: bool = true,
            attn_mask = none(Tensor),
            dropout_p = 0.0'f64): untyped =
  layer.forward(q, k, v, is_causal, attn_mask, dropout_p)


func init*(
    _: type RopeGQAttention,
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    q_norm, k_norm: RmsNorm,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbeddingRef): RopeGQAttention =
  ## Initialize RopeGQAttention.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1)
  ##   name: Safetensor key prefix (e.g., "model.layers.23.self_attn")
  ##   q_proj, k_proj, v_proj, o_proj: Preinitialized projections
  ##   q_norm, k_norm: Q/K normalization
  ##   num_qo_head: Number of query/output heads
  ##   num_kv_head: Number of KV heads (GQA)
  ##   head_dim: Dimension per head
  ##   rotary: RoPE module (shared across layers)
  ##   quant_format: Quantization format of this layer (determines RMSNorm impl)
  ##   rms_norm_eps: Epsilon for Q/K norm
  let has_qk_norm = rotary.head_dim == head_dim
  RopeGQAttention(
    layer_idx: layer_idx,
    name: name,
    q_proj: q_proj,
    k_proj: k_proj,
    v_proj: v_proj,
    o_proj: o_proj,
    gqa_attn: GroupedQueryAttention.init(num_qo_head, num_kv_head, head_dim),
    rotary: rotary,
    q_norm: if has_qk_norm: some(q_norm) else: none(RmsNorm),
    k_norm: if has_qk_norm: some(k_norm) else: none(RmsNorm)
  )

proc forward(
    self: RopeGQAttention,
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass for attention with paged KV cache.
  ##
  ## Args:
  ##   ctx: InferenceContext with page refs (ctx.pages, ctx.cos, ctx.sin)
  ##   x: Input tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (batch, seq, num_qo_head * head_dim)
  ##
  ## Computes:
  ##   q = self.q_proj(x)
  ##   k = self.k_proj(x)
  ##   v = self.v_proj(x)
  ##   (q_rot, k_rot) = self.rotary.applyRope(q, k, ctx.cos, ctx.sin)
  ##   Write k_rot, v_reshaped into ctx.pages page slots
  ##   Gather pages into contiguous k_full, v_full
  ##   attn_out = self.gqa_attn(q_rot, k_full, v_full)
  ##   return self.o_proj(attn_out)

  # Use separate Q, K, V projections (matching HF/Qwen3)
  let q = self.q_proj(x)
  let k = self.k_proj(x)
  let v = self.v_proj(x)

  let batch = x.size(0)
  let seq_len = x.size(1)

  # Reshape to (batch, seq, heads, head_dim)
  let q_reshaped = q.reshape([batch, seq_len, self.gqa_attn.num_qo_head, self.gqa_attn.head_dim])
  let k_reshaped = k.reshape([batch, seq_len, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim])
  let v_reshaped = v.reshape([batch, seq_len, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim])

  # Apply q/k norm (on reshaped tensor before RoPE)
  var q_norm_input = q_reshaped
  var k_norm_input = k_reshaped
  if self.q_norm.isSome:
    q_norm_input = self.q_norm.get()(q_reshaped)
  if self.k_norm.isSome:
    k_norm_input = self.k_norm.get()(k_reshaped)

  # Apply RoPE using precomputed cos/sin
  let (q_rot, k_rot) = self.rotary.applyRope(q_norm_input, k_norm_input, ctx.cos, ctx.sin)

  # ── Write new KV into page slots ──
  # Each page covers TokensPerPage token positions.
  # page.k_view[layer_idx] is (PAGE_SIZE, kv_heads, head_dim)
  let offset = ctx.position_ids.min().item(int)
  # Skip writing cached prefix positions (already in trie from COW)
  # TODO: how to test usage of cache?
  let writeStart = max(0, ctx.kv_position - offset)
  for t in writeStart ..< seq_len:
    let globalPos = offset + t
    let pageIdx = globalPos div TokensPerPage
    let withinPage = globalPos mod TokensPerPage
    let page = ctx.pages[pageIdx]
    page.k_view[self.layer_idx, withinPage] = k_rot[0, t]
    page.v_view[self.layer_idx, withinPage] = v_reshaped[0, t]

  # ── Gather pages into contiguous K/V for SDPA ──
  let totalSeqLen = offset + seq_len
  let numPages = ceilDiv(totalSeqLen, TokensPerPage)
  # k_full/v_full: (1, totalSeqLen, kv_heads, head_dim)
  let kvDtype = v_reshaped.scalarType()
  # Determine device (page views may be on GPU)
  let kvDevice: F.DeviceKind = v_reshaped.deviceType()
  let kvOpts = F.tensorOptions(kvDtype, kvDevice)
  var k_full = F.empty(
    1, totalSeqLen, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim, kvOpts)
  var v_full = F.empty(
    1, totalSeqLen, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim, kvOpts)
  for p in 0 ..< numPages:
    let pageStart = p * TokensPerPage
    let pageEnd = min(pageStart + TokensPerPage, totalSeqLen)
    let pageValidLen = pageEnd - pageStart
    let page = ctx.pages[p]
    k_full[0, pageStart ..< pageEnd] = page.k_view[self.layer_idx, 0 ..< pageValidLen]
    v_full[0, pageStart ..< pageEnd] = page.v_view[self.layer_idx, 0 ..< pageValidLen]

  # k_full/v_full are already (batch, seq, kv_heads, head_dim) — the format GQA expects.
  # GQA's forward permutes internally to (batch, kv_heads, seq, head_dim) for SDPA.
  # is_causal only makes sense when Q and K seq_lens are equal (prefill).
  # In decode mode (Q=1, K=N), causal mask would block K[1..N-1].
  let doCausal = q_rot.size(1) == k_full.size(1)
  let attn_out_reshaped = self.gqa_attn(q_rot, k_full, v_full, is_causal = doCausal)

  result = self.o_proj(attn_out_reshaped)

template `()`*(layer: RopeGQAttention,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)
