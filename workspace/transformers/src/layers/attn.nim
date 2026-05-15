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
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  ./rope

type
  GroupedQueryAttention* = object
    head_dim*: int
    num_qo_head*: int
    num_kv_head*: int
    num_kv_groups*: int
    qo_attn_dim*: int
    kv_attn_dim*: int
    softmax_scale*: float64

  RopeGQAttention* = object
    ## Rope + Grouped Query Attention.
    ##
    ## State is external (InferenceContext), not owned by this layer.
    layer_idx*: int             # Layer index for self-indexing KV cache
    name*: string               # Safetensor key prefix (e.g., "model.layers.23.self_attn")
    q_proj*: Linear
    k_proj*: Linear
    v_proj*: Linear
    o_proj*: Linear
    attn*: GroupedQueryAttention
    rotary*: RotaryPositionEmbeddingRef
    q_norm*: Option[RmsNorm]
    k_norm*: Option[RmsNorm]


  # =====================================================================
  # Data flow through RopeGQAttention
  # =====================================================================
  #
  #   x (batch, seq, hidden)
  #   │
  #   ├─→ q_proj → reshape → q_norm → applyRope → q_rot ───────────────────┐
  #   │                                                                    │
  #   ├─→ k_proj → reshape → k_norm → applyRope → k_rot ─→ cache.write ◄───┤
  #   │                                             ↑                      │
  #   │                                    cache.getKV ─→ k_full           │
  #   │                                                                    │
  #   └─→ v_proj → reshape ─────────────────────→ cache.write ◄────────────┘
  #                                                 ↑
  #                                        cache.getKV ─→ v_full
  #
  #   q_rot, k_full, v_full → SDPA(is_causal, enable_gqa) → attn_out
  #   attn_out → o_proj → output
  #
  # Note: V is NOT rotated. RoPE (Su et al., 2021) only rotates Q and K
  # because the attention mechanism computes similarity via q·k. V simply
  # carries content to be aggregated by attention weights — it has no
  # positional role in the similarity computation.
  # =====================================================================

func init*(_: type GroupedQueryAttention, num_qo_head, num_kv_head, head_dim: int): GroupedQueryAttention =
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

func forward*(
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

func init*(
    _: type RopeGQAttention,
    layer_idx: int,
    name: string,
    q_weight, k_weight, v_weight, o_weight, q_norm_weight, k_norm_weight: Tensor,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbeddingRef,
    rms_norm_eps = 1e-6'f64): RopeGQAttention =
  ## Initialize RopeGQAttention.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1)
  ##   name: Safetensor key prefix (e.g., "model.layers.23.self_attn")
  ##   q_weight, k_weight, v_weight, o_weight: Projection weights
  ##   q_norm_weight, k_norm_weight: Q/K normalization weights
  ##   num_qo_head: Number of query/output heads
  ##   num_kv_head: Number of KV heads (GQA)
  ##   head_dim: Dimension per head
  ##   rotary: RoPE module (shared across layers)
  ##   rms_norm_eps: Epsilon for Q/K norm

  let q_proj = Linear.init(q_weight)
  let k_proj = Linear.init(k_weight)
  let v_proj = Linear.init(v_weight)
  let o_proj = Linear.init(o_weight)

  let has_qk_norm = rotary.head_dim == head_dim
  let q_norm =
    if has_qk_norm: some(RmsNorm.init(weight = q_norm_weight, eps = rms_norm_eps))
    else: none(RmsNorm)
  let k_norm =
    if has_qk_norm: some(RmsNorm.init(weight = k_norm_weight, eps = rms_norm_eps))
    else: none(RmsNorm)

  let attn = GroupedQueryAttention.init(
    num_qo_head = num_qo_head,
    num_kv_head = num_kv_head,
    head_dim = head_dim
  )

  RopeGQAttention(
    layer_idx: layer_idx,
    name: name,
    q_proj: q_proj,
    k_proj: k_proj,
    v_proj: v_proj,
    o_proj: o_proj,
    attn: attn,
    rotary: rotary,
    q_norm: q_norm,
    k_norm: k_norm
  )

proc forward*(
    self: RopeGQAttention,
    ctx: var InferenceContext,
    cos, sin: Tensor,
    x: Tensor): Tensor =
  ## Forward pass for attention.
  ##
  ## Args:
  ##   ctx: InferenceContext with KV caches (ctx.kv_caches[self.layer_idx])
  ##   cos, sin: Precomputed RoPE of shape (seq_len, head_dim)
  ##   x: Input tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (batch, seq, num_qo_head * head_dim)
  ##
  ## Computes:
  ##   q = self.q_proj(x)
  ##   k = self.k_proj(x)
  ##   v = self.v_proj(x)
  ##   cache = ctx.kv_caches[self.layer_idx]
  ##   cache.write(k, v, ctx.position_ids)
  ##   (q_rot, k_rot) = self.rotary.applyRope(q, k, cos, sin)
  ##   attn_out = self.attn(q_rot, k_rot, cache.values)
  ##   return self.o_proj(attn_out)

  # Use separate Q, K, V projections (matching HF/Qwen3)
  let q = self.q_proj.forward(x)
  var k = self.k_proj.forward(x)
  var v = self.v_proj.forward(x)

  let batch = x.size(0)
  let seq_len = x.size(1)

  # Reshape to (batch, seq, heads, head_dim)
  let q_reshaped = q.reshape([batch, seq_len, self.attn.num_qo_head, self.attn.head_dim])
  let k_reshaped = k.reshape([batch, seq_len, self.attn.num_kv_head, self.attn.head_dim])
  let v_reshaped = v.reshape([batch, seq_len, self.attn.num_kv_head, self.attn.head_dim])

  # Apply q/k norm (on reshaped tensor before RoPE)
  var q_norm_input = q_reshaped
  var k_norm_input = k_reshaped
  if self.q_norm.isSome:
    q_norm_input = self.q_norm.get().forward(q_reshaped)
  if self.k_norm.isSome:
    k_norm_input = self.k_norm.get().forward(k_reshaped)

  # Apply RoPE using precomputed cos/sin
  let (q_rot, k_rot) = self.rotary.applyRope(q_norm_input, k_norm_input, cos, sin)

  # Get this layer's KV cache and write new KV
  var cache = ctx.kv_caches[self.layer_idx]
  let offset = ctx.position_ids.min().item(int) # Get current position offset

  # Append to KV cache (K rotated, V)
  cache.store(k_rot, v_reshaped, offset)
  # Get full KV (cached + appended)
  let (k_full, v_full) = cache.getKV(cache.offset)

  # Transpose k_full, v_full from (batch, kv_heads, seq, head_dim) to (batch, seq, kv_heads, head_dim)
  let k_attn = k_full.permute([0, 2, 1, 3])
  let v_attn = v_full.permute([0, 2, 1, 3])

  # Pass to backend (GroupedQueryAttention) which handles permute/dtype/SDPA/reshape
  let attn_out_reshaped = self.attn.forward(q_rot, k_attn, v_attn, is_causal = true)

  result = self.o_proj.forward(attn_out_reshaped)
