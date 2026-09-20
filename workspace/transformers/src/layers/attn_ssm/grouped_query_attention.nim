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
  workspace/transformers/src/instrumentation,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  ../rope

{.experimental: "callOperator".}

type
  GroupedQueryAttention* = object
    ## Scaled dot-product attention over (batch, seq, heads, head_dim) tensors.
    head_dim*: int
    num_qo_head*: int
    num_kv_head*: int
    num_kv_groups*: int
    qo_attn_dim*: int
    kv_attn_dim*: int
    softmax_scale*: float64

  RopeGQAttention*[QKNorm] = ref object
    ## Rope + Grouped Query Attention.
    ## Created at model load. Weights and rotary are immutable after init.
    ##
    ##  - `QKNorm` is `RmsNorm`, `RmsNormOne`, or `void`.
    ##    `void` is the variant without qk-norm weights.
    ##  - The layer owns projections, norms and rotary only.
    ##    The KV cache arrives through the InferenceContext of forward.
    layer_idx*: int             # Layer index for self-indexing KV cache
    name*: string               # Safetensor key prefix (e.g., "model.layers.23.self_attn")
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    gqa_attn: GroupedQueryAttention
    rotary: RotaryPositionEmbedding
    window*: int
      ## Visibility band of the layer kind.
      ## A query attends to itself and the previous `window - 1` cached keys.
      ## `FullVisibilityWindow` removes the band entirely.
    when QKNorm isnot void:
      q_norm: QKNorm
      k_norm: QKNorm
    gProj: Option[Linear]
      ## Per-head output gate for the per-head-gated layer kinds (Laguna),
      ## none on the ungated kinds.
      ##
      ## Checkpoint spelling:
      ## a `[hidden, heads]` projection of the attention input.
      ##   - softplus computes in f32, one rounding back to the dtype
      ##   - one scalar per head multiplies the attention output
      ##     right before o_proj
    kvSourceLayer: int
      ## Layer index supplying this layer's K/V pages, the gemma-4
      ## shared-kv spelling. A shared layer sits at or past
      ## num_hidden_layers - num_kv_shared_layers, it projects no k/v.
      ##
      ## The shared layer gathers the source layer's post-rope pages
      ## instead. `-1` keeps the self-cached default, the layer then
      ## writes and reads its own pages.
    vNorm: FusedRmsNorm
      ## Value-path single-rounding norm applied to the reshaped v before
      ## the cache write, the gemma-4 v_norm with_scale=False spelling
      ## (a ones weight, no checkpoint tensor). Nil on layers without it.
# -----------------------------------------------------------------------------
# Data flow through RopeGQAttention
# -----------------------------------------------------------------------------
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
#   cache.write offset from ctx.kv_position
#
# Note:
#   V is NOT rotated. RoPE (Su et al., 2021) only rotates Q and K
#   because the attention mechanism computes similarity via q·k.
#   V simply carries content to be aggregated by attention weights,
#   and it has no positional role in the similarity computation.
# =============================================================================

const FullVisibilityWindow* = int.high
  ## Sentinel of the windowed attention spelling.
  ## The visibility band never binds, every key at or before the query stays
  ## visible under the plain causal rule.

proc windowedCausalMask*(qLen, kvLen, offset, window: int, device: F.DeviceKind): Tensor =
  ## Visibility-band causal mask of the windowed attention spelling.
  ##
  ## Expected input:
  ##
  ## - `qLen`, the query count of one forward pass
  ## - `kvLen`, the gathered key count of one forward pass
  ## - `offset`, the absolute position of the first query (kv_position)
  ##
  ## - `window`, the visibility band of the layer kind
  ## - `device`, the query tensor's device
  ##
  ## Output:
  ##
  ## - a `(1, 1, qLen, kvLen)` bool mask
  ## - `true` keeps a key visible, `false` masks it
  ## - the mask broadcasts over batch and heads
  ##
  ## Visibility rule, matching the reference sliding-window causal rule:
  ##
  ## - query row `i` covers absolute position `offset + i`
  ## - key column `j` covers absolute position `j`
  ## - a key stays visible iff `j <= offset + i` and `j > offset + i - window`
  doAssert window > 0, "windowedCausalMask: the visibility band must be positive"
  let opts = F.tensorOptions(F.kInt64, device)
  let qPos = F.arange(offset, offset + qLen, opts).unsqueeze(1)
  let kPos = F.arange(0, kvLen, opts).unsqueeze(0)
  let distance = qPos - kPos   # (qLen, kvLen) query position minus key position
  # The two band inequalities as a 0/1 int64 product, cast to bool because
  # the SDPA reference spelling consumes a bool mask.
  let visible = (distance >=. Scalar(0.0'f64)) * (distance <. Scalar(window.float64))
  visible.to(F.kBool).unsqueeze(0).unsqueeze(0)

func init*(_: type GroupedQueryAttention, num_qo_head, num_kv_head, head_dim: int, softmaxScale = 0.0'f64): GroupedQueryAttention =
  ## Configure GQA over `num_qo_head` query heads and `num_kv_head` KV heads,
  ## each of width `head_dim`.
  ##
  ## Softmax scale is `head_dim^-0.5` by default.
  ##
  ## `softmaxScale` overrides it when positive, for checkpoints that scale
  ## attention by a pre-attention scalar.
  ## Gemma-3 passes `query_pre_attn_scalar^-0.5` here.
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
    kv_attn_dim: num_kv_head * head_dim,
    softmax_scale: scale
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
  ## Scaled dot-product attention with GQA support.
  ##
  ## Args:
  ##   q: Query tensor of shape (batch, seq, num_qo_head, head_dim)
  ##   k: Key tensor of shape (batch, seq, num_kv_head, head_dim)
  ##   v: Value tensor of shape (batch, seq, num_kv_head, head_dim)
  ##   is_causal: If true, apply causal mask
  ##   enable_gqa: selects the head layout of K/V.
  ##     GQA, grouped-query (true): K/V have num_kv_head heads, fewer
  ##     than queries. Each K/V head serves g = num_qo_head / num_kv_head
  ##     query heads:
  ##
  ##       q    q0 … q(g-1) | qg … q(2g-1) | ...    num_qo_head heads
  ##       k,v        k0    |      k1      | ...    num_kv_head heads
  ##
  ##     MHA, multi-head (false): num_kv_head == num_qo_head, g = 1,
  ##     one K/V head per query head:
  ##
  ##       q    q0    | q1    | ...               num_qo_head heads
  ##       k,v  k0    | k1    | ...               num_qo_head heads
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

  # Masked SDPA spelling of the reference stack. K/V expand to the query
  # head count with each KV head repeated for its query-head group while
  # enable_gqa drops. The GQA path stays reserved for the mask-free kernel.
  let useGqa = attn_mask.isNone and enable_gqa and self.num_kv_groups > 1
  # Expansion requires KV-head-count keys (the repeat_kv precondition).
  # Callers replaying already-expanded keys skip it.
  let keysGrouped = k_final.size(1) == self.num_kv_head and
    self.num_kv_groups > 1
  let (k_sdpa, v_sdpa) =
    if useGqa or not keysGrouped:
      (k_final, v_attn)
    else:
      # Every expansion dim sizes from the KEY sequence length.
      # The masked decode shape runs one query row against a gathered
      # history longer than the window, and a sizing from the query
      # length would collapse the key axis down to 1.
      let kv_len = k_final.size(2)
      let expanded = k_final.unsqueeze(2)
        .expand(batch, self.num_kv_head, self.num_kv_groups, kv_len,
          self.head_dim, implicit = false)
        .reshape([batch, self.num_qo_head, kv_len, self.head_dim])
      let vExpanded = v_attn.unsqueeze(2)
        .expand(batch, self.num_kv_head, self.num_kv_groups, kv_len,
          self.head_dim, implicit = false)
        .reshape([batch, self.num_qo_head, kv_len, self.head_dim])
      (expanded, vExpanded)

  let attn_out = F.scaled_dot_product_attention(
    q_final, k_sdpa, v_sdpa,
    attn_mask = attn_mask,
    dropout_p = dropout_p,
    is_causal = is_causal,
    scale = some(self.softmax_scale),
    enable_gqa = useGqa
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


func initBase[QKNorm](
    _: type RopeGQAttention[QKNorm],
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbedding,
    window: int, softmaxScale: float64,
    gProj: Option[Linear] = none(Linear),
    kvSourceLayer = -1,
    vNorm: FusedRmsNorm = nil): RopeGQAttention[QKNorm] =
  checkValue(num_qo_head > 0,
    "[ttt] " & name & ": num_attention_heads is " & $num_qo_head &
    ", expected a positive count")
  checkValue(num_kv_head > 0,
    "[ttt] " & name & ": num_key_value_heads is " & $num_kv_head &
    ", expected a positive count")
  checkValue(num_qo_head mod num_kv_head == 0,
    "[ttt] " & name & ": num_attention_heads (" & $num_qo_head &
    ") leaves a remainder under num_key_value_heads (" & $num_kv_head & ")")
  checkValue(window > 0,
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
    window: window,
    gProj: gProj,
    kvSourceLayer: kvSourceLayer,
    vNorm: vNorm
  )

func init*[QKNorm](
    _: type RopeGQAttention[QKNorm],
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbedding,
    window: int = FullVisibilityWindow,
    softmaxScale = 0.0'f64,
    gProj: Option[Linear] = none(Linear),
    kvSourceLayer = -1,
    vNorm: FusedRmsNorm = nil): RopeGQAttention[QKNorm] =
  ## Build the attention block with no qk-norms.
  ##
  ## `window` defaults to `FullVisibilityWindow`, plain causal attention.
  ## A sliding-window layer kind passes its band width.
  ## `softmaxScale` overrides the head-width scale when positive.
  initBase(RopeGQAttention[QKNorm], layer_idx, name,
    q_proj, k_proj, v_proj, o_proj,
    num_qo_head, num_kv_head, head_dim, rotary, window, softmaxScale,
    gProj, kvSourceLayer, vNorm)

func init*[QKNorm](
    _: type RopeGQAttention[QKNorm],
    layer_idx: int,
    name: string,
    q_proj, k_proj, v_proj, o_proj: Linear,
    num_qo_head, num_kv_head, head_dim: int,
    rotary: RotaryPositionEmbedding,
    q_norm, k_norm: QKNorm,
    window: int = FullVisibilityWindow,
    softmaxScale = 0.0'f64,
    gProj: Option[Linear] = none(Linear),
    kvSourceLayer = -1,
    vNorm: FusedRmsNorm = nil): RopeGQAttention[QKNorm] =
  ## Initialize RopeGQAttention.
  ##
  ## Args:
  ##   layer_idx: Layer index (0..num_layers-1)
  ##   name: Safetensor key prefix (e.g., "model.layers.23.self_attn")
  ##   q_proj, k_proj, v_proj, o_proj: Preinitialized projections
  ##   q_norm, k_norm: Q/K normalization, the per-head norm over `head_dim`
  ##   num_qo_head: Number of query/output heads
  ##   num_kv_head: Number of KV heads (GQA)
  ##   head_dim: Dimension per head
  ##   rotary: RoPE module (shared across layers)
  ##
  ## `window` is the visibility band of the layer kind.
  ## A query attends to itself and the previous `window - 1` cached keys.
  ## `FullVisibilityWindow`, the default, removes the band.
  ##
  ## `softmaxScale` overrides the attention scale when positive.
  ## Without it the scale is the head-width `head_dim^-0.5`.
  ##
  ## Raises ValueError naming the layer key path for a non-positive head
  ## count, or for a query-head count not divisible by the KV-head count,
  ## before the GQA group division truncates it.
  result = initBase(RopeGQAttention[QKNorm], layer_idx, name,
    q_proj, k_proj, v_proj, o_proj,
    num_qo_head, num_kv_head, head_dim, rotary, window, softmaxScale,
    gProj, kvSourceLayer, vNorm)
  when QKNorm isnot void:
    result.q_norm = q_norm
    result.k_norm = k_norm

proc writeKvPages(
    ctx: var InferenceContext, layer_idx: int,
    k_rot, v_reshaped: Tensor, offset, seq_len: int) =
  # ── Write new KV into page slots ──
  # Each page covers TokensPerPage token positions.
  # page.k_view[layer_idx] is (PAGE_SIZE, kv_heads, head_dim)
  #
  # offset = ctx.kv_position  (instead of position_ids.min().item(int))
  # to avoid a GPU->CPU synchronous read every forward pass.
  #
  # Lifecycle overview:
  #   1. Prefill:  startSequence sets kv_position=0
  #                forward writes at offset=0
  #                generate() calls setKvPosition(ids.len)
  #   2. Decode:   appendToken sets position_ids WITHOUT incrementing
  #                forward writes at offset=kv_position (matches pos_ids.min())
  #                generate() increments kv_position after forward
  #   => Invariant: kv_position == position_ids.min() during forward.
  # Skip writing cached prefix positions (already in trie from COW)
  # TODO: how to test usage of cache?
  let writeStart = max(0, ctx.cached_tokens - offset)
  # Write in page-sized chunks instead of per-token indexed writes.
  # Reduces GPU kernel launches from O(seq_len) to O(num_pages).
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
      # Single copyFrom per page instead of one kernel per token.
      # Dual-width pools carry the widest kv width in the slot, a narrower
      # layer writes the leading head_dim channels through a narrowed view.
      let kView = page.k_view[layer_idx, withinPage ..< withinPage + chunkLen]
      let vView = page.v_view[layer_idx, withinPage ..< withinPage + chunkLen]
      let kSrc = k_rot[0, t ..< chunkEnd, _, _]
      let vSrc = v_reshaped[0, t ..< chunkEnd, _, _]
      if kView.size(2) != kSrc.size(2):
        kView.narrow(2, 0, kSrc.size(2)).copyFrom(kSrc)
        vView.narrow(2, 0, vSrc.size(2)).copyFrom(vSrc)
      else:
        kView.copyFrom(kSrc)
        vView.copyFrom(vSrc)
      t = chunkEnd

proc gatherKv(
    ctx: var InferenceContext, layer_idx, num_kv_head, head_dim: int,
    offset, seq_len: int, kvDtype: F.ScalarKind,
    kvDevice: F.DeviceKind): (Tensor, Tensor) =
  # ── Gather pages into contiguous K/V for SDPA ──
  let totalSeqLen = offset + seq_len
  let numPages = ceilDiv(totalSeqLen, TokensPerPage)

  # Reuse pre-allocated buffers to avoid F.empty allocation per forward pass.
  # Allocate once at max_seq size, narrow to actual totalSeqLen each call.
  # Dual-width layers share one context, a realloc is due whenever
  # the stashed buffer width disagrees with this layer's head_dim too.
  if ctx.k_gather_buf.isNil or ctx.k_gather_buf.size(1) < totalSeqLen or
      ctx.k_gather_buf.size(3) != head_dim:
    let allocSize = max(totalSeqLen, ctx.max_seq)
    let kvOpts = F.tensorOptions(kvDtype, kvDevice)
    ctx.k_gather_buf = F.zeros(
      1, allocSize, num_kv_head, head_dim, kvOpts)
    ctx.v_gather_buf = F.zeros(
      1, allocSize, num_kv_head, head_dim, kvOpts)

  for p in 0 ..< numPages:
    let pageStart = p * TokensPerPage
    let pageEnd = min(pageStart + TokensPerPage, totalSeqLen)
    let pageValidLen = pageEnd - pageStart
    let page = ctx.pages[p]
    # A narrower layer reads the leading head_dim channels of its wide
    # slot through a narrowed view.
    let kSlot = page.k_view[layer_idx, 0 ..< pageValidLen]
    let vSlot = page.v_view[layer_idx, 0 ..< pageValidLen]
    let kvWidth = ctx.k_gather_buf.size(3)
    ctx.k_gather_buf[0, pageStart ..< pageEnd, _, _] =
      (if kSlot.size(2) != kvWidth: kSlot.narrow(2, 0, kvWidth) else: kSlot)
    ctx.v_gather_buf[0, pageStart ..< pageEnd, _, _] =
      (if vSlot.size(2) != kvWidth: vSlot.narrow(2, 0, kvWidth) else: vSlot)

  # Narrow pre-allocated buffers to actual sequence length for SDPA
  let k_full = ctx.k_gather_buf.narrow(1, 0, totalSeqLen)
  let v_full = ctx.v_gather_buf.narrow(1, 0, totalSeqLen)
  (k_full, v_full)

proc forward[QKNorm](
    self: RopeGQAttention[QKNorm],
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
  ##   q = self.q_proj(x);  (k, v = self.k_proj(x), self.v_proj(x))
  ##   (q_rot, k_rot) = self.rotary.applyRope(q, k, ctx.cos, ctx.sin)
  ##   Write k_rot, v_reshaped into ctx.pages page slots
  ##   Gather pages into contiguous k_full, v_full
  ##   attn_out = self.gqa_attn(q_rot, k_full, v_full)
  ##   [gProj present] attn_out = attn_out * softplus(g_proj(x)) per head
  ##   return self.o_proj(attn_out)
  ##
  ## A kvSourceLayer >= 0 skips the k/v projection, norm, rope and write:
  ## k_full/v_full gather straight from the source layer's pages, q ropes
  ## on top of them.
  let batch = x.size(0)

  # Guard against batch_size > 1
  # The paged KV cache write/gather path indexes with [0, ...] throughout
  # (k_rot[0, ...], v_reshaped[0, ...], ctx.k_gather_buf[0, ...]).
  # Multi-batch support requires per-sequence page allocation and gather.
  if batch != 1:
    raise newException(ValueError,
      "[ttt] Paged KV attention currently supports batch_size == 1 only, got " & $batch)

  # Use separate Q, K, V projections (matching HF/Qwen3)
  let q = self.q_proj.forward(x)

  let seq_len = x.size(1)
  # Reshape to (batch, seq, heads, head_dim)
  let q_reshaped = q.reshape([batch, seq_len, self.gqa_attn.num_qo_head, self.gqa_attn.head_dim])

  # Apply q norm (on reshaped tensor before RoPE)
  let q_norm_input =
    when QKNorm is void:
      q_reshaped
    else:
      forward(self.q_norm, q_reshaped)

  let offset = ctx.kv_position
  let kvDtype = q_reshaped.scalarType()
  let kvDevice: F.DeviceKind = q_reshaped.deviceType()

  var q_rot: Tensor
  var k_full, v_full: Tensor
  if self.kvSourceLayer < 0:
    let k = self.k_proj.forward(x)
    let v = self.v_proj.forward(x)
    let k_reshaped = k.reshape([batch, seq_len, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim])
    var v_reshaped = v.reshape([batch, seq_len, self.gqa_attn.num_kv_head, self.gqa_attn.head_dim])

    # Apply k norm (on reshaped tensor before RoPE)
    let k_norm_input =
      when QKNorm is void:
        k_reshaped
      else:
        forward(self.k_norm, k_reshaped)

    # Apply RoPE using precomputed cos/sin.
    # Partial RoPE rotates the leading rotary_dim columns only.
    let (qRotated, k_rot) = self.rotary.applyRope(q_norm_input, k_norm_input, ctx.cos, ctx.sin)
    q_rot = qRotated

    # The value-path norm applies before the cache write
    let vWrite =
      if self.vNorm != nil:
        self.vNorm.forward(v_reshaped)
      else:
        v_reshaped

    writeKvPages(ctx, self.layer_idx, k_rot, vWrite, offset, seq_len)
    (k_full, v_full) = gatherKv(ctx, self.layer_idx,
      self.gqa_attn.num_kv_head, self.gqa_attn.head_dim,
      offset, seq_len, kvDtype, kvDevice)
  else:
    # A shared-kv layer (gemma-4 spelling) runs no k/v projection, norm,
    # rope or cache write of its own. The k/v come from the source layer's
    # post-rope full-length pages, the same rows the reference stack
    # carries in its shared_kv_states dict.
    let (qRotated, _) = self.rotary.applyRope(q_norm_input, q_norm_input, ctx.cos, ctx.sin)
    q_rot = qRotated
    (k_full, v_full) = gatherKv(ctx, self.kvSourceLayer,
      self.gqa_attn.num_kv_head, self.gqa_attn.head_dim,
      offset, seq_len, kvDtype, kvDevice)

  # k_full/v_full are already (batch, seq, kv_heads, head_dim) — the format GQA expects.
  # GQA's forward permutes internally to (batch, kv_heads, seq, head_dim) for SDPA.
  # is_causal only makes sense when Q and K seq_lens are equal (prefill).
  # In decode mode (Q=1, K=N), causal mask would block K[1..N-1].
  #
  # Mask dispatch of the visibility band:
  # - band unbound, window >= gathered key count
  #   prefill takes the causal path, decode sees the whole cached history
  # - band bound
  #   the windowed causal mask, query rows at absolute positions
  #   offset .. offset + seqQ - 1, one -Inf per key outside the band
  let kvSeqLen = k_full.size(1)
  var attnMask = none(Tensor)
  var doCausal = false
  if self.window >= kvSeqLen:
    doCausal = q_rot.size(1) == kvSeqLen
  elif q_rot.size(1) == 1:
    # Single-query decode past the window. The visibility band keeps
    # exactly the newest `window` keys and every one of them is visible,
    # each key j in the slice satisfies j > query_pos - window.
    # The reference stack's sliding cache serves that same slice, so
    # the gathered history truncates to it while the mask reduces
    # to all-visible. Running the full history through the band mask
    # prices a different SDPA kernel shape whose rounding drifts away
    # from the reference.
    let kvStart = kvSeqLen - self.window
    k_full = k_full.narrow(1, kvStart, self.window)
    v_full = v_full.narrow(1, kvStart, self.window)
    attnMask = some(F.ones(1, 1, 1, self.window,
      F.tensorOptions(F.kBool, q_rot.deviceType())))
  else:
    attnMask = some(windowedCausalMask(q_rot.size(1), kvSeqLen, offset,
      self.window, q_rot.deviceType()))
  var attn_out = self.gqa_attn.forward(q_rot, k_full, v_full,
    is_causal = doCausal, attn_mask = attnMask)

  if self.gProj.isSome():
    # Per-head output scaling, the Laguna checkpoint spelling.
    # The g_proj projection runs on the attention input, softplus
    # computes in f32, one rounding back to the attention dtype, one
    # scalar per head multiplies the attention output before o_proj.
    let gate = F.softplus(
      self.gProj.unsafeGet().forward(x).to(kFloat32)).to(attn_out.scalarType())
    attn_out = (attn_out.reshape(
        [batch, seq_len, self.gqa_attn.num_qo_head, self.gqa_attn.head_dim]) *
      gate.unsqueeze(-1)).reshape([batch, seq_len, self.gqa_attn.qo_attn_dim])

  result = self.o_proj.forward(attn_out)

template `()`*[QKNorm](layer: RopeGQAttention[QKNorm],
            ctx: var InferenceContext,
            x: Tensor): untyped =
  layer.forward(ctx, x)
