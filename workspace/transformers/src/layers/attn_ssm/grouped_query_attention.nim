# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Grouped-query attention: the core math and the rope-carrying layer
## of the Qwen3 and Qwen3.5 hybrid stacks.
##
## Two tiers:
## - `GroupedQueryAttention`, the core tier: head counts, KV expansion
##   and the softmax scale around the attention call, ropeless
## - `RopeGQAttention`, the layer tier: projections, qk-norm options,
##   fused gate, rope policy from the rotary ref, paged KV cache
##   plumbing
##
## Flags parameterizing the two served-(and future) shapes of the layer:
## - `fused_gate`: `q_proj` emits `[q | gate]` per head and the attention
##   output is scaled by `sigmoid(gate)` (Qwen3.5), or a plain `q_proj`
##   emits q only (Qwen3)
## - qk-norm presence: `q_norm`/`k_norm` as options, set by the model
##   wiring. Norm internals are uniform `RmsNorm`: the 1-centered
##   `1 + w` spelling is its `constant_bias` value, not a variant here
## - RoPE policy: the layer rotates whatever the `rotary` ref spells,
##   full head or partial columns. Nothing here re-reads config
##
## GQA head grouping runs one way only. The layer pre-expands K/V
## (repeat_interleave) before calling the core, so the core never needs
## `enable_gqa`: this is the bit-for-bit HF reference convention.
## The `enable_gqa` true path is bitwise identical on both served
## geometries, so the choice costs numbers nothing.

import
  std/math,
  std/options,
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  ../rope

{.experimental: "callOperator".}

type
  GroupedQueryAttention* = object
    ## Core tier, ropeless: head counts, KV group expansion, softmax scale
    ## and the attention call. No rope, no projections.
    ##
    ## Shapes are (batch, seq, heads, head_dim) throughout. GQA is handled
    ## either by the `enable_gqa` flag or by pre-expanding K/V heads
    ## before the call. The `enable_gqa` false path is the bit-for-bit
    ## HF reference convention (FIXTURE_GENERATION.md section 6).
    head_dim*: int
    num_qo_head*: int
    num_kv_head*: int
    num_kv_groups*: int
    qo_attn_dim*: int
    kv_attn_dim*: int
    softmax_scale*: float64

  RopeGQAttention* = ref object
    ## Layer tier: rope + grouped-query attention. Decoder layer over a paged KV cache.
    ## The attention math runs in the `gqa_attn` core.
    ## State is external (InferenceContext), not owned by this layer.
    layer_idx*: int             # Layer index for self-indexing KV cache
    name*: string               # Safetensor key prefix (e.g. "model.layers.23.self_attn")
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: Option[RmsNorm]     ## Head geometry (GQA grouping) and rope policy
    k_norm: Option[RmsNorm]     ## arrive from config, not from the layer kind
    gqa_attn: GroupedQueryAttention
    rotary: RotaryPositionEmbeddingRef
    fused_gate: bool            ## q_proj emits `[q | gate]` per head

# =============================================================================
# Data flow through RopeGQAttention
# =============================================================================
#
#   x (batch, seq, hidden)
#   │
#   ├─→ q_proj → reshape → qk-norm ─→ applyRope → q_rot ──────────────────┐
#   │     fused: [q | gate] per head, gate kept for the output scale       │
#   │                                                                     │
#   ├─→ k_proj → reshape → qk-norm ─→ applyRope → k_rot ─→ cache.write ◄──┤
#   │                                                       offset from   │
#   │                                                       ctx.kv_position
#   │                                                       cache.read ─→ k_full
#   │                                                                     │
#   └─→ v_proj → reshape ──────────────────────────→ cache.write ◄────────┘
#                                                   cache.read ─→ v_full
#
#   q_rot, k_full, v_full → repeat_interleave ×kv_groups → GQA(is_causal)
#   fused: attn_out × sigmoid(gate) → o_proj → output
#   plain:                                     o_proj → output
#
# K/V heads are pre-expanded (repeat_interleave) before the core call,
# which runs without enable_gqa, matching the HF reference convention,
# bit for bit (FIXTURE_GENERATION.md section 6).
#
# V is NOT rotated. RoPE (Su et al., 2021) only rotates Q and K
# because the attention mechanism computes similarity via q·k. V simply
# carries content to be aggregated by the attention weights and holds
# no positional role in the similarity computation.
# =============================================================================

func init*(_: type GroupedQueryAttention, num_qo_head, num_kv_head, head_dim: int): GroupedQueryAttention =
  ## Configure GQA over `num_qo_head` query heads and `num_kv_head` KV heads,
  ## each of width `head_dim`. The softmax scale is `head_dim^-0.5`.
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
      dropout_p = 0.0'f64,
      enable_gqa: bool = true): Tensor =
  ## Grouped-query attention forward, core tier.
  ##
  ## Expected input:
  ## - q: (batch, seq, num_qo_head, head_dim)
  ## - k, v: (batch, seq, num_kv_head, head_dim)
  ##   with `enable_gqa` false, K/V must already carry num_qo_head heads
  ##   (pre-expanded)
  ##
  ## Output:
  ## - (batch, seq, num_qo_head * head_dim)
  ##
  # Backend: permute to (batch, head, seq, head_dim), cast, run
  # F.scaled_dot_product_attention, reshape to (batch, seq, qo_attn_dim)
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
    enable_gqa = enable_gqa and self.num_kv_groups > 1
  )

  let attn_perm = attn_out.permute([0, 2, 1, 3])
  result = attn_perm.reshape([batch, seq_len, self.qo_attn_dim])

template `()`*(layer: GroupedQueryAttention,
            q, k, v: Tensor,
            is_causal: bool = true,
            attn_mask = none(Tensor),
            dropout_p = 0.0'f64,
            enable_gqa: bool = true): untyped =
  layer.forward(q, k, v, is_causal, attn_mask, dropout_p, enable_gqa)

func init*(
    _: type RopeGQAttention,
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbeddingRef,
    q_norm: Option[RmsNorm] = none(RmsNorm),
    k_norm: Option[RmsNorm] = none(RmsNorm),
    fused_gate: bool = false): RopeGQAttention =
  ## Assemble one attention layer over preinitialized projections.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1), indexes the KV pages
  ##   name: Safetensor key prefix (e.g. "model.layers.23.self_attn")
  ##   q_proj, k_proj, v_proj, o_proj: Preinitialized projections
  ##   num_qo_head: Number of query/output heads
  ##   num_kv_head: Number of KV heads (GQA)
  ##   head_dim: Dimension per head
  ##   rotary: RoPE module shared across layers, its `rotary_dim` carrying
  ##     the partial-vs-full rope policy
  ##   q_norm, k_norm: Q/K RMSNorms when the stack applies qk-norm
  ##   fused_gate: `q_proj` doubles per head as `[q | gate]`, the attention output scaled by `sigmoid(gate)`
  ##
  ## Raises ValueError naming the layer key path for a non-positive head
  ## count, for a query-head count outside the KV-head divisibility law,
  ## both before the GQA group division reads them.
  #
  # Head-count gates run before the GQA group division: a non-divisible
  # pair would truncate there silently.
  if num_qo_head <= 0:
    raise newException(ValueError,
      "[ttt] " & name & ": num_attention_heads is " & $num_qo_head &
      ", expected a positive count")
  if num_kv_head <= 0:
    raise newException(ValueError,
      "[ttt] " & name & ": num_key_value_heads is " & $num_kv_head &
      ", expected a positive count")
  if num_qo_head mod num_kv_head != 0:
    raise newException(ValueError,
      "[ttt] " & name & ": num_attention_heads (" & $num_qo_head &
      ") leaves a remainder under num_key_value_heads (" & $num_kv_head & ")")
  RopeGQAttention(
    layer_idx: layer_idx,
    name: name,
    q_proj: q_proj,
    k_proj: k_proj,
    v_proj: v_proj,
    o_proj: o_proj,
    gqa_attn: GroupedQueryAttention.init(num_qo_head, num_kv_head, head_dim),
    rotary: rotary,
    q_norm: q_norm,
    k_norm: k_norm,
    fused_gate: fused_gate
  )

proc forward(
    self: RopeGQAttention,
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass with paged KV cache.
  ##
  ## Expected input:
  ## - ctx: InferenceContext with page refs (ctx.pages, ctx.cos, ctx.sin)
  ## - x: (batch, seq, hidden_size)
  ##
  ## Output:
  ## - (batch, seq, hidden_size)
  ##
  ## Computes:
  ##   q = q_proj(x), chunked as [q | gate] when `fused_gate`
  ##   q, k: qk-norm when their norms are set
  ##   (q_rot, k_rot) = rotary.applyRope(q, k, ctx.cos, ctx.sin)
  ##   k_rot, v → ctx.pages page slots → gather into contiguous k_full, v_full
  ##   attn_out = gqa_attn(q_rot, expand(k_full), expand(v_full), causal when q_len == k_len)
  ##   return o_proj(attn_out) or o_proj(attn_out * sigmoid(gate))
  #
  # Guard against batch_size > 1
  # The paged KV cache write/gather path indexes with [0, ...] throughout
  # (k_rot[0, ...], v_reshaped[0, ...], ctx.k_gather_buf[0, ...]).
  # Multi-batch support requires per-sequence page allocation and gather.
  let batch = x.size(0)
  if batch != 1:
    raise newException(ValueError,
      "[ttt] Paged KV attention currently supports batch_size == 1 only, got " & $batch)

  let seq_len = x.size(1)
  let head_dim = self.gqa_attn.head_dim
  let num_qo = self.gqa_attn.num_qo_head
  let num_kv = self.gqa_attn.num_kv_head

  # q_proj emits q, plus the raw pre-sigmoid per-head gate when fused
  var gate: Tensor
  var q_reshaped: Tensor
  if self.fused_gate:
    let qg = self.q_proj.forward(x)
    let qg_reshaped = qg.reshape([batch, seq_len, num_qo, 2 * head_dim])
    q_reshaped = qg_reshaped.narrow(3, 0, head_dim)
    gate = qg_reshaped.narrow(3, head_dim, head_dim)
      .reshape([batch, seq_len, self.gqa_attn.qo_attn_dim])
  else:
    q_reshaped = self.q_proj.forward(x)
      .reshape([batch, seq_len, num_qo, head_dim])

  # Separate K, V projections, reshaped to (batch, seq, heads, head_dim)
  let k_reshaped = self.k_proj.forward(x).reshape([batch, seq_len, num_kv, head_dim])
  let v_reshaped = self.v_proj.forward(x).reshape([batch, seq_len, num_kv, head_dim])

  # RMSNorm qk-norm over head_dim when the stack ships the norms
  let q_norm_input =
    if self.q_norm.isSome: self.q_norm.get().forward(q_reshaped)
    else: q_reshaped
  let k_norm_input =
    if self.k_norm.isSome: self.k_norm.get().forward(k_reshaped)
    else: k_reshaped

  # RoPE on precomputed cos/sin, partial-vs-full policy from the rotary ref's rotary_dim
  # (applyRope rotates that many columns only)
  let (q_rot, k_rot) = self.rotary.applyRope(q_norm_input, k_norm_input, ctx.cos, ctx.sin)

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

  # ── Gather pages into contiguous K/V ──
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

  # Narrow pre-allocated buffers to actual sequence length for attention
  let k_full = ctx.k_gather_buf.narrow(1, 0, totalSeqLen)
  let v_full = ctx.v_gather_buf.narrow(1, 0, totalSeqLen)

  # Pre-expand K/V heads (repeat_interleave), then run standard attention
  # without enable_gqa, matching the HF reference convention
  # (FIXTURE_GENERATION.md section 6).
  let k_expanded = k_full.repeat_interleave(self.gqa_attn.num_kv_groups, 2)
  let v_expanded = v_full.repeat_interleave(self.gqa_attn.num_kv_groups, 2)

  # is_causal only makes sense when Q and K seq_lens are equal (prefill).
  # In decode mode (Q=1, K=N), causal mask would block K[1..N-1].
  let doCausal = q_rot.size(1) == k_full.size(1)
  let attn_out = self.gqa_attn.forward(
    q_rot, k_expanded, v_expanded, is_causal = doCausal, enable_gqa = false)

  if self.fused_gate:
    # Gate: raw pre-sigmoid per-head vector, broadcast over head_dim
    let attn_gated = attn_out * F.sigmoid(gate)
    result = self.o_proj.forward(attn_gated)
  else:
    result = self.o_proj.forward(attn_out)

template `()`*(layer: RopeGQAttention,
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)
