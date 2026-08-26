# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  ./rope,
  ./attn

{.experimental: "callOperator".}

type
  GatedAttention* = ref object
    ## Gated full attention (Qwen3.5 hybrid layers): q_proj emits `[q | gate]`
    ## per head, GemmaRMSNorm qk-norm over head_dim, partial RoPE, causal
    ## SDPA, and the attention output is scaled by `sigmoid(gate)` before
    ## o_proj. State is external (InferenceContext), not owned by this layer.
    layer_idx*: int             # Layer index for self-indexing KV cache
    name*: string               # Safetensor key prefix (e.g. "model.language_model.layers.3.self_attn")
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: GemmaRmsNorm
    k_norm: GemmaRmsNorm
    gqa_attn: GroupedQueryAttention
    rotary: RotaryPositionEmbeddingRef
    rotary_dim: int

# =============================================================================
# Data flow through GatedAttention
# =============================================================================
#
#   x (batch, seq, hidden)
#   │
#   ├─→ q_proj → [q | gate] per head → q_norm → applyRope → q_rot ──────────┐
#   │                                             (partial, rotary_dim)     │
#   ├─→ k_proj → k_norm → applyRope → k_rot ─→ cache.write ◄───────────────┤
#   │                                   offset from ctx.kv_position         │
#   │                                      cache.read ─→ k_full             │
#   │                                                                       │
#   └─→ v_proj → reshape ─────────→ cache.write ◄───────────────────────────┘
#                                       cache.read ─→ v_full
#
#   q_rot, k_full, v_full → repeat_interleave ×kv_groups → SDPA(is_causal)
#   attn_out × sigmoid(gate) → o_proj → output
#
# K/V heads are pre-expanded (repeat_interleave) before SDPA so the call runs
# without enable_gqa, matching the HF reference convention bit for bit
# (FIXTURE_GENERATION.md section 6).
# =============================================================================

func init*(
    _: type GatedAttention,
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    q_norm, k_norm: GemmaRmsNorm,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbeddingRef): GatedAttention =
  ## Initialize the gated full-attention layer.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1), indexes the KV pages
  ##   name: Safetensor key prefix (e.g. "model.language_model.layers.3.self_attn")
  ##   q_proj, k_proj, v_proj, o_proj: Preinitialized projections
  ##   q_norm, k_norm: GemmaRMSNorm (weight applied as 1 + w) over head_dim
  ##   num_qo_head: Number of query/output heads
  ##   num_kv_head: Number of KV heads (GQA)
  ##   head_dim: Dimension per head
  ##   rotary: RoPE module (shared across layers), partial rotary_dim baked in
  GatedAttention(
    layer_idx: layer_idx,
    name: name,
    q_proj: q_proj,
    k_proj: k_proj,
    v_proj: v_proj,
    o_proj: o_proj,
    q_norm: q_norm,
    k_norm: k_norm,
    gqa_attn: GroupedQueryAttention.init(num_qo_head, num_kv_head, head_dim),
    rotary: rotary,
    rotary_dim: rotary.rotary_dim
  )

proc forward(
    self: GatedAttention,
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass for gated full attention with paged KV cache.
  ##
  ## Args:
  ##   ctx: InferenceContext with page refs (ctx.pages, ctx.cos, ctx.sin)
  ##   x: Input tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (batch, seq, hidden_size)
  ##
  ## Computes (matching the vendored Qwen3_5Attention.forward):
  ##   q, gate = chunk(q_proj(x) as (batch, seq, heads, 2*head_dim), 2)
  ##   q = q_norm(q)
  ##   k = k_norm(k_proj(x))
  ##   v = v_proj(x)
  ##   (q, k) = rotary.applyRope(q, k, ctx.cos, ctx.sin)
  ##   Write k_rot, v_reshaped into ctx.pages page slots
  ##   Gather pages into contiguous k_full, v_full
  ##   attn_out = gqa(q_rot, k_full, v_full)   (causal when q_len == k_len)
  ##   return o_proj(attn_out * sigmoid(gate))
  let batch = x.size(0)

  # Guard against batch_size > 1
  # The paged KV cache write/gather path indexes with [0, ...] throughout
  # (k_rot[0, ...], v_reshaped[0, ...], ctx.k_gather_buf[0, ...]).
  # Multi-batch support requires per-sequence page allocation and gather.
  if batch != 1:
    raise newException(ValueError,
      "[ttt] Paged KV attention currently supports batch_size == 1 only, got " & $batch)

  let seq_len = x.size(1)
  let head_dim = self.gqa_attn.head_dim
  let num_qo = self.gqa_attn.num_qo_head
  let num_kv = self.gqa_attn.num_kv_head
  let qo_attn_dim = self.gqa_attn.qo_attn_dim

  # q_proj emits [q | gate] per head: (batch, seq, num_qo * 2 * head_dim)
  let qg = self.q_proj.forward(x)
  let qg_reshaped = qg.reshape([batch, seq_len, num_qo, 2 * head_dim])
  let query_reshaped = qg_reshaped.narrow(3, 0, head_dim)
  let gate_reshaped = qg_reshaped.narrow(3, head_dim, head_dim)
  let gate = gate_reshaped.reshape([batch, seq_len, qo_attn_dim])

  # Separate K, V projections, reshaped to (batch, seq, kv_heads, head_dim)
  let k_reshaped = self.k_proj.forward(x).reshape([batch, seq_len, num_kv, head_dim])
  let v_reshaped = self.v_proj.forward(x).reshape([batch, seq_len, num_kv, head_dim])

  # GemmaRMSNorm qk-norm over head_dim (weight applied as 1 + w)
  let q_normed = self.q_norm.forward(query_reshaped)
  let k_normed = self.k_norm.forward(k_reshaped)

  # Partial RoPE using precomputed cos/sin (rotary_dim columns rotate)
  let (q_rot, k_rot) = self.rotary.applyRope(q_normed, k_normed, ctx.cos, ctx.sin)

  # ── Write new KV into page slots ──
  # Each page covers TokensPerPage token positions.
  # page.k_view[layer_idx] is (PAGE_SIZE, kv_heads, head_dim)
  #
  # offset = ctx.kv_position (instead of position_ids.min().item(int))
  # to avoid a GPU->CPU synchronous read every forward pass.
  #
  # Lifecycle overview:
  #   1. Prefill:  startSequence sets kv_position=0
  #                forward writes at offset=0
  #                generate() calls setKvPosition(ids.len)
  #   2. Decode:   decodeStep sets position_ids without incrementing
  #                forward writes at offset=kv_position (matches pos_ids.min())
  #                generate() increments kv_position after forward
  #   => Invariant: kv_position == position_ids.min() during forward.
  let offset = ctx.kv_position
  # Skip writing cached prefix positions (already in trie from COW)
  let writeStart = max(0, ctx.cached_tokens - offset)
  block pageWrite:
    var t = writeStart
    while t < seq_len:
      let globalPos = offset + t
      let pageIdx = globalPos div TokensPerPage
      let withinPage = globalPos mod TokensPerPage
      let page = ctx.pages[pageIdx]
      # Chunk size = min(remaining in this page, remaining to write)
      let chunkRemaining = TokensPerPage - withinPage
      let seqRemaining = seq_len - t
      let chunkLen = min(chunkRemaining, seqRemaining)
      let chunkEnd = t + chunkLen
      # Single copyFrom per page instead of one kernel per token
      page.k_view[self.layer_idx, withinPage ..< withinPage + chunkLen].copyFrom(
        k_rot[0, t ..< chunkEnd, _, _])
      page.v_view[self.layer_idx, withinPage ..< withinPage + chunkLen].copyFrom(
        v_reshaped[0, t ..< chunkEnd, _, _])
      t = chunkEnd

  # ── Gather pages into contiguous K/V for SDPA ──
  let totalSeqLen = offset + seq_len
  let numPages = ceilDiv(totalSeqLen, TokensPerPage)

  # Reuse pre-allocated buffers to avoid F.empty allocation per forward pass.
  # Allocate once at max_seq size, narrow to actual totalSeqLen each call.
  let kvDtype = v_reshaped.scalarType()
  let kvDevice: F.DeviceKind = v_reshaped.deviceType()
  if ctx.k_gather_buf.isNil or ctx.k_gather_buf.size(1) < totalSeqLen:
    let allocSize = max(totalSeqLen, ctx.max_seq)
    let kvOpts = F.tensorOptions(kvDtype, kvDevice)
    ctx.k_gather_buf = F.zeros(
      1, allocSize, num_kv, head_dim, kvOpts)
    ctx.v_gather_buf = F.zeros(
      1, allocSize, num_kv, head_dim, kvOpts)

  for p in 0 ..< numPages:
    let pageStart = p * TokensPerPage
    let pageEnd = min(pageStart + TokensPerPage, totalSeqLen)
    let pageValidLen = pageEnd - pageStart
    let page = ctx.pages[p]
    ctx.k_gather_buf[0, pageStart ..< pageEnd, _, _] = page.k_view[self.layer_idx, 0 ..< pageValidLen]
    ctx.v_gather_buf[0, pageStart ..< pageEnd, _, _] = page.v_view[self.layer_idx, 0 ..< pageValidLen]

  # Narrow pre-allocated buffers to actual sequence length for SDPA
  let k_full = ctx.k_gather_buf.narrow(1, 0, totalSeqLen)
  let v_full = ctx.v_gather_buf.narrow(1, 0, totalSeqLen)

  # Pre-expand K/V heads (repeat_interleave) and run standard SDPA without
  # enable_gqa, matching the HF reference (FIXTURE_GENERATION.md section 6).
  let k_expanded = k_full.repeat_interleave(self.gqa_attn.num_kv_groups, 2)
  let v_expanded = v_full.repeat_interleave(self.gqa_attn.num_kv_groups, 2)

  # is_causal only makes sense when Q and K seq_lens are equal (prefill).
  # In decode mode (Q=1, K=N), causal mask would block K[1..N-1].
  let doCausal = q_rot.size(1) == k_full.size(1)
  let attn_out = self.gqa_attn.forward(
    q_rot, k_expanded, v_expanded, is_causal = doCausal, enable_gqa = false)

  # Gate: raw pre-sigmoid per-head vector, broadcast over head_dim
  let attn_gated = attn_out * F.sigmoid(gate)
  result = self.o_proj.forward(attn_gated)

template `()`*(layer: GatedAttention,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)
