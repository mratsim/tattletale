# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/options,
  workspace/tensorcery/src/tensorcery,
  ./linear,
  ./norm,
  ./rope,
  ../stateful/page_pool,
  ../stateful/inference_context

from ../stateful/kvcache import TokensPerPage

{.experimental: "callOperator".}

type
  GroupedQueryAttention* = object
    ## Scaled dot-product attention over (1, seq, heads, head_dim) tensors.
    head_dim*: int
    num_qo_head*: int
    num_kv_head*: int
    num_kv_groups: int
    qo_attn_dim*: int
    softmax_scale*: float64

  RopeGQAttention*[QKNorm] = ref object
    ## Rope + Grouped Query Attention.
    ## Created at model load. Weights and rotary are immutable after init.
    ##
    ##  - `QKNorm` is `RmsNorm`, `RmsNormOne`, or `void`.
    ##    `void` is the variant without qk-norm weights.
    ##  - The layer owns projections, norms and rotary only.
    ##    The KV cache travels through the InferenceContext of forward.
    layer_idx: int             # Layer index for self-indexing KV cache
    name: string               # Safetensor key prefix (e.g., "model.layers.23.self_attn")
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    gqa_attn: GroupedQueryAttention
    rotary: RotaryPositionEmbedding
    window: int
      ## Visibility band of the layer kind.
      ## A query attends to itself and the previous `window - 1` cached keys.
      ## `FullVisibilityWindow` removes the band entirely.
    when QKNorm isnot void:
      q_norm: QKNorm
      k_norm: QKNorm

# -----------------------------------------------------------------------------
# Data flow through RopeGQAttention
# -----------------------------------------------------------------------------
#
#   x (1, seq, hidden)
#   │
#   ├─→ q_proj → reshape → q_norm → applyRope → q_rot ────────────────────┐
#   │                                                                     │
#   ├─→ k_proj → reshape → k_norm → applyRope → k_rot ─→ pool_write ◄─────┤
#   │                                         offset from ctx.position_ids│
#   │                                                                     │
#   └─→ v_proj → reshape ─────────────────→ pool_write ◄──────────────────┘
#
#   q_rot, pool slabs → jagged attention (native GQA over the page pool)
#   → attn_out → o_proj → output
#
#   pool_write offset from ctx.kv_position
#
# Note: V is NOT rotated. RoPE (Su et al., 2021) rotates Q and K only,
# since attention computes similarity via q·k.
# V supplies the content that the attention weights aggregate. V plays no positional role in the similarity computation.
# =============================================================================
#
# Jagged-routed form: the reference stack's K/V expand/repeat to the query head
# count and the masked SDPA call are REPLACED by the jagged attention
# kernel's native GQA (kv_head = q_head div (H div Nkv)).
# Its write/gather pair becomes the in-graph pool write over the page pool.
# Reference-side gatherKv staging buffers never existed on this face: the kernel walks the pool pages directly.

const FullVisibilityWindow* = int.high
  ## Sentinel of the windowed attention form.
  ## The visibility band never binds: every key at or before the query
  ## stays visible under the causal rule.

func init*(_: type GroupedQueryAttention, num_qo_head, num_kv_head, head_dim: int, softmaxScale = 0.0'f64): GroupedQueryAttention =
  ## Configures GQA over `num_qo_head` query heads and `num_kv_head` KV heads,
  ## each of width `head_dim`.
  ##
  ## Contract:
  ## - softmax scale defaults to `head_dim^-0.5`
  ## - `softmaxScale` overrides the default when positive. A checkpoint that scales attention by a pre-attention scalar passes it here.
  ## - Gemma-3 passes `query_pre_attn_scalar^-0.5` here
  let num_kv_groups = num_qo_head div num_kv_head
  let scale =
    if softmaxScale > 0.0'f64: softmaxScale
    else: 1.0'f64 / sqrt(head_dim.float64)
  GroupedQueryAttention(
    head_dim: head_dim,
    num_qo_head: num_qo_head,
    num_kv_head: num_kv_head,
    num_kv_groups: num_kv_groups,
    qo_attn_dim: num_qo_head * head_dim,
    softmax_scale: scale
  )

proc checkLayerValue(cond: bool, msg: string) =
  ## Raises ValueError with msg when cond is false, for the layer-content
  ## defect class of the load path.
  if not cond:
    raise newException(ValueError, msg)

func initBase[QKNorm](
    _: type RopeGQAttention[QKNorm],
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbedding,
    window: int, softmaxScale: float64): RopeGQAttention[QKNorm] =
  checkLayerValue(num_qo_head > 0,
    "[ttt] " & name & ": num_attention_heads is " & $num_qo_head &
    ", expected a positive count")
  checkLayerValue(num_kv_head > 0,
    "[ttt] " & name & ": num_key_value_heads is " & $num_kv_head &
    ", expected a positive count")
  checkLayerValue(num_qo_head mod num_kv_head == 0,
    "[ttt] " & name & ": num_attention_heads (" & $num_qo_head &
    ") leaves a remainder under num_key_value_heads (" & $num_kv_head & ")")
  checkLayerValue(window > 0,
    "[ttt] " & name & ": the visibility band is " & $window &
    ", expected a positive count or FullVisibilityWindow")
  RopeGQAttention[QKNorm](
    layer_idx: layer_idx,
    name: name,
    q_proj: q_proj,
    k_proj: k_proj,
    v_proj: v_proj,
    o_proj: o_proj,
    gqa_attn: GroupedQueryAttention.init(num_qo_head, num_kv_head, head_dim,
      softmaxScale),
    rotary: rotary,
    window: window
  )

func init*[QKNorm](
    _: type RopeGQAttention[QKNorm],
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbedding,
    window: int = FullVisibilityWindow,
    softmaxScale = 0.0'f64): RopeGQAttention[QKNorm] =
  ## Builds the attention block with no qk-norms.
  ##
  ## Contract:
  ## - `window` defaults to `FullVisibilityWindow` (causal attention)
  ## - a sliding-window layer kind passes its window width
  ## - `softmaxScale` overrides the head-width scale when positive
  initBase(RopeGQAttention[QKNorm], layer_idx, name,
    q_proj, k_proj, v_proj, o_proj,
    num_qo_head, num_kv_head, head_dim, rotary, window, softmaxScale)

func init*[QKNorm](
    _: type RopeGQAttention[QKNorm],
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbedding,
    q_norm, k_norm: QKNorm,
    window: int = FullVisibilityWindow,
    softmaxScale = 0.0'f64): RopeGQAttention[QKNorm] =
  ## Initialize RopeGQAttention.
  ##
  ## Args:
  ## - layer_idx: layer index (0..num_layers-1)
  ## - name: safetensor key prefix (example: "model.layers.23.self_attn")
  ## - q_proj, k_proj, v_proj, o_proj: preinitialized projections
  ## - q_norm, k_norm: the Q/K normalization, per head over `head_dim`
  ##
  ## Shape arguments:
  ## - num_qo_head: number of query/output heads
  ## - num_kv_head: number of KV heads (GQA)
  ## - head_dim: dimension per head
  ## - rotary: RoPE module (shared across layers)
  ##
  ## `window` is the visibility band of the layer kind. A query attends
  ## to itself and the previous `window - 1` cached keys.
  ## `FullVisibilityWindow` (the default) removes the band.
  ##
  ## `softmaxScale` overrides the attention scale when positive.
  ## Without it the scale is the head-width `head_dim^-0.5`.
  ##
  ## Raises ValueError naming the layer key path for a non-positive head count
  ## or for a query-head count not divisible by the KV-head count,
  ## checked before the GQA group division truncates it.
  result = initBase(RopeGQAttention[QKNorm], layer_idx, name,
    q_proj, k_proj, v_proj, o_proj,
    num_qo_head, num_kv_head, head_dim, rotary, window, softmaxScale)
  when QKNorm isnot void:
    result.q_norm = q_norm
    result.k_norm = k_norm

proc forward*[QKNorm](self: RopeGQAttention[QKNorm], ctx: var InferenceContext[uint32, Page], x: Tensor): Tensor =
  ## Forward pass for attention with the paged KV cache.
  ##
  ## Args:
  ## - ctx: InferenceContext with page refs (ctx.pages, ctx.cos, ctx.sin)
  ## - x: input tensor of shape (1, seq, hidden_size)
  ##
  ## Returns:
  ##   Output tensor of shape (1, seq, num_qo_head * head_dim)
  ##
  ## Computes:
  ## - q = q_proj.forward(x), k = k_proj.forward(x), v = v_proj.forward(x)
  ## - (q_rot, k_rot) = rotary.applyRope(q_norm_input, k_norm_input, ctx.cos, ctx.sin)
  ## - write k_rot, v_reshaped into the ctx.pages page slots (pool write node, in graph)
  ## - o_proj.forward(attn_out), attn_out from the jagged attention (native GQA) over the pool slabs
  ##
  ## The write reaches the attention as the attention node's poolK child,
  ## one eval orders write then attention through the child walk.
  ##
  ## A windowed layer kind raises: the op runs non-windowed (Qwen3 has no sliding window).
  ## A windowed dispatch needs its own op instantiation.
  if self.window != FullVisibilityWindow:
    raise ValueError.newException(
      "[ttt] " & self.name & ": the visibility band " & $self.window &
      " needs a windowed attention op instantiation, Qwen3 runs full visibility")

  # Guard against multi-sequence input: the paged KV write/gather path
  # indexes one sequence throughout (k_rot[0, ...], v_reshaped[0, ...]).
  # Multi-sequence support requires per-sequence page allocation and gather.
  let numSeq = x.shape()[0]
  if numSeq != 1:
    raise newException(ValueError,
      "[ttt] Paged KV attention runs one sequence at a time, got " & $numSeq)

  # Use separate Q, K, V projections (matching HF/Qwen3)
  let q = self.q_proj.forward(x)

  let seq_len = x.shape()[1]
  # Reshape to (1, seq, heads, head_dim)
  let q_reshaped = q.reshape([numSeq, seq_len, self.gqa_attn.num_qo_head, self.gqa_attn.head_dim])

  # Apply q norm (on reshaped tensor before RoPE)
  let q_norm_input =
    when QKNorm is void:
      q_reshaped
    else:
      self.q_norm.forward(q_reshaped)

  let offset = ctx.kv_position

  # Use separate Q, K, V projections (matching HF/Qwen3)
  let k = self.k_proj.forward(x)
  let k_reshaped = k.reshape([numSeq, seq_len, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim])
  let v_reshaped = self.v_proj.forward(x).reshape(
    [numSeq, seq_len, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim])

  # Apply k norm (on reshaped tensor before RoPE)
  let k_norm_input =
    when QKNorm is void:
      k_reshaped
    else:
      self.k_norm.forward(k_reshaped)

  # Apply RoPE using precomputed cos/sin.
  # Partial RoPE rotates the leading rotary_dim columns only.
  let (qRotated, k_rot) = self.rotary.applyRope(q_norm_input, k_norm_input, ctx.cos, ctx.sin)

  # ── Write new KV into page slots, jagged-routed ──
  # Reference-side writeKvPages/gatherKv becomes an in-graph pool write
  # over the pool slabs. One launch writes both pools, the jagged attention
  # kernel walks the pages directly, and the gathered contiguous K/V
  # staging never exists on this face.
  #
  # offset = ctx.kv_position, not position_ids.min().item(int).
  # A direct read costs a synchronous GPU->CPU read every forward pass.
  #
  # Lifecycle:
  #   1. Prefill:  startSequence sets kv_position=0
  #                forward writes at offset=0
  #                generate() calls setKvPosition(ids.len)
  #   2. Decode:   appendToken sets position_ids WITHOUT incrementing
  #                forward writes at the kv_position offset (the pos_ids minimum)
  #                generate() increments kv_position after forward
  #   => Invariant: kv_position == position_ids.min() during forward.
  # Skip writing cached prefix positions (already in trie from COW)
  # TODO: how to test usage of cache?
  let slots = ctx.pageSlots()
  let writeStart = max(0, ctx.cached_tokens - offset)
  var poolK, poolV: Tensor
  if writeStart < seq_len:
    # Pool write's result aliases the K slab. The attention below consumes it,
    # and the eval walk orders write then attention.
    poolK = pool_write(ctx.pool.kBuffer(), ctx.pool.vBuffer(),
      k_rot, v_reshaped, self.layer_idx, offset, seq_len,
      ctx.cached_tokens, slots)
    poolV = ctx.pool.vBuffer()
  else:
    # Fully cached write range (a replayed prompt): no write, the raw slabs
    # feed the attention.
    poolK = ctx.pool.kBuffer()
    poolV = ctx.pool.vBuffer()

  # ── Jagged attention over the page pool, jagged-routed ──
  # Reference-side K/V expand/repeat to the query head count and the masked SDPA call
  # are REPLACED by the kernel's native GQA (kv_head = q_head div (H div Nkv) inside the kernel).
  #
  # is_causal only makes sense when Q and K seq_lens are equal (prefill).
  # In decode mode (Q=1, K=N), causal mask would block K[1..N-1].
  # Kernel-side rule: query token j of a multi-token segment attends keys [0, kvLen - qLen + j],
  # a single-query segment attends [0, kvLen).
  let kvLen = offset + seq_len
  let q_flat = qRotated.reshape([numSeq * seq_len, self.gqa_attn.num_qo_head, self.gqa_attn.head_dim])
  var kvMeta = JaggedKvMeta(
    offsets: @[0], indices: @[], seqlens: @[kvLen])
  let numKvPages = (kvLen + TokensPerPage - 1) div TokensPerPage
  kvMeta.offsets.add numKvPages
  for i in 0 ..< numKvPages:
    kvMeta.indices.add slots[i]
  let qOffsets = @[0, numSeq * seq_len]
  let attn_out = jagged_attention(q_flat, poolK, poolV,
    qOffsets, kvMeta, float32(self.gqa_attn.softmax_scale),
    self.layer_idx, self.gqa_attn.num_qo_head, self.gqa_attn.num_kv_head)

  result = self.o_proj.forward(
    attn_out.reshape([numSeq, seq_len, self.gqa_attn.qo_attn_dim]))

template `()`*[QKNorm](layer: RopeGQAttention[QKNorm], ctx: var InferenceContext[uint32, Page], x: Tensor): untyped =
  layer.forward(ctx, x)
