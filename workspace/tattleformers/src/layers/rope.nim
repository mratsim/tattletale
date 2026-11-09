# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  workspace/tensorcery/src/tensor_datatypes,
  ../ops_Ni1o/embed_gather,
  ../ops_Ni1o/rope

type
  RotaryPositionEmbedding* = ref object
    ## Rotary Position Embedding (RoPE), a per-model precomputed
    ## cosine/sine lookup table, immutable after init
    ##
    ## new(head_dim, max_seq_len, rope_theta, engine, rotary_dim) ─►
    ##   cos_cache, sin_cache, shape (max_seq_len, rotary_dim div 2)
    ## ropeByPositions(position_ids) ─► cos, sin, shape (seq_len, rotary_dim div 2)
    ##
    ## Ownership split:
    ## | Kind                | Lifetime    | Owner                     |
    ## | ------------------- | ----------- | ------------------------- |
    ## | Layer weights       | Per model   | Layer objects             |
    ## | Precomputed tables  | Per model   | Config objects (this one) |
    ## | Sliced cos/sin      | Per request | InferenceContext          |
    ## | Rope variant config | Per model   | Model (not orchestrator)  |
    ##
    ## Invariants:
    ## - `cos_cache`/`sin_cache` shape `(max_seq_len, rotary_dim div 2)`,
    ##   `rotary_dim <= head_dim`
    ## - `cos_cache[p, d] == cos(p * rope_theta^(-2*d/rotary_dim))`, the NEOX
    ##   half-repeat, the column pair (d, d + rotary_dim/2) shares one frequency
    ## - `applyRope` rotates the first `rotary_dim` columns, the remaining
    ##   columns pass through unchanged, the rotation is pure
    ##
    head_dim: int
    rotary_dim: int
    max_seq_len: int
    rope_theta: float64
    cos_cache*: Tensor  ## Precomputed (max_seq_len, rotary_dim div 2). Immutable after init.
    sin_cache*: Tensor  ## Precomputed (max_seq_len, rotary_dim div 2). Immutable after init.

proc new*(_: type RotaryPositionEmbedding,
      engine: AnyEngine,
      head_dim, max_seq_len: int,
      rope_theta: float64,
      rotary_dim = -1): RotaryPositionEmbedding =
  ## Build RoPE lookup table for all positions `0..max_seq_len-1`.
  ##
  ## `rotary_dim` defaults to `head_dim` (full rotation). A smaller value
  ## (e.g. 64 for a partial factor 0.5 over head_dim 128) rotates only
  ## the first `rotary_dim` columns.
  ##
  ## The cache is sized `(max_seq_len, rotary_dim div 2)`, a partial rotation
  ## never allocates the full head_dim table.
  ##
  ## **Algorithm (NEOX-style)**:
  ##
  ##  1. Compute inverse frequencies for even dimensions only:
  ##     `inv_freq[d] = theta^(-d/rotary_dim)` for `d in {0, 2, 4, ..., rotary_dim-2}`
  ##     This gives `rotary_dim/2` unique frequencies.
  ##     Odd dimensions reuse the same frequency (pairwise repetition).
  ##
  ##  2. For each position `p in {0, ..., max_seq_len-1}` and each
  ##     unique dimension `d`, compute `cos(p * inv_freq[d])` and `sin(p * inv_freq[d])`
  ##     in FP64 for precision.
  ##
  ##  3. The reference duplicates the half table to cover all `rotary_dim`
  ##     positions `[f0 .. f_{m-1}, f0 .. f_{m-1}]` (m = rotary_dim/2) by
  ##     self-concatenation: this port keeps the half table, the kernel's
  ##     pair indexing reproduces the duplication.
  ##
  ##  4. Cast to f32 and upload. The tables reach the device once per model load.
  ##
  ## **Complexity**:
  ##   O(max_seq_len * rotary_dim), done once per model load.
  ##
  ## Unported from the reference new(): the yarn blend, the activePairs mask, the f32Cache table construction and attentionFactor
  ## (other-model features, the Qwen3 path uses none of them).
  let dim = if rotary_dim < 0: head_dim else: rotary_dim
  doAssert dim <= head_dim, "rotary_dim " & $dim & " exceeds head_dim " & $head_dim
  doAssert (dim mod 2) == 0, "rotary_dim must be even"
  let half_dim = dim div 2

  # f64 host math: reference inv_freq construction with the tensor calls
  # (F.pow/F.arange/F.full) inlined to the same values,
  # inv_freq[d] = theta^(-d/rotary_dim) for the even dimensions d,
  # one entry per pair
  var inv_freq = newSeq[float64](half_dim)
  for j in 0 ..< half_dim:
    inv_freq[j] = pow(rope_theta, -(2.0 * float64(j)) / float64(dim))

  # angles (max_seq_len, rotary_dim/2): the positions × inv_freq outer product, one cos/sin half table per angle grid
  var cos_half = newSeq[float32](max_seq_len * half_dim)
  var sin_half = newSeq[float32](max_seq_len * half_dim)
  for p in 0 ..< max_seq_len:
    for j in 0 ..< half_dim:
      let angle = float64(p) * inv_freq[j]
      cos_half[p * half_dim + j] = float32(cos(angle))
      sin_half[p * half_dim + j] = float32(sin(angle))

  result = RotaryPositionEmbedding(
    head_dim: head_dim,
    rotary_dim: dim,
    max_seq_len: max_seq_len,
    rope_theta: rope_theta,
    cos_cache: newTensor(engine, @[max_seq_len, half_dim], kFp32),
    sin_cache: newTensor(engine, @[max_seq_len, half_dim], kFp32))
  result.cos_cache.copyToDeviceFromHost(cos_half)
  result.sin_cache.copyToDeviceFromHost(sin_half)

proc ropeByPositions*(self: RotaryPositionEmbedding, position_ids: openArray[int]): (Tensor, Tensor) =
  ## Slice cos/sin cache using position_ids.
  ##
  ## **input_ids vs position_ids: they are NOT the same**:
  ##
  ##   - `input_ids`:
  ##     Token IDs. *What* to compute (e.g., `[9707, 11, 1246]` = "Hello, how")
  ##   - `position_ids`:
  ##     Absolute positions in the sequence. *Where* each token goes
  ##
  ##   For the common case (prefill from 0, decode sequentially):
  ##     input_ids = `[9707, 11, 1246]`  →  position_ids = `[0, 1, 2]`
  ##     input_ids = `[498]`             →  position_ids = `[3]`  (next token at offset 3)
  ##
  ##   They diverge for continuous batching (different sequences at different positions),
  ##   prefix caching (skip cached tokens), and speculative decoding (non-contiguous).
  ##
  ## Args:
  ##   position_ids:
  ##     Flat positions in [0, max_seq_len), one per token. The reference accepts (seq_len,) or (batch, seq_len) tensors
  ##     and takes the first batch item, flattening to one position per token happens caller-side here.
  ##
  ## Returns:
  ##   (cos, sin) of shape (seq_len, rotary_dim div 2), gathered from the cache
  ##
  ## Note:
  ##   Called once per forward pass at model level.
  ##   Result is stored in InferenceContext via `ctx.setRopeForPositions(rotary)`.

  # Slice cache using position_ids (advanced indexing)
  # cos_cache[position_ids, :] → (seq_len, rotary_dim div 2)
  result = (
    Tensor(engine: self.cos_cache.engine,
      node: tn_embed_gather(self.cos_cache.engine, self.cos_cache.node, position_ids)),
    Tensor(engine: self.sin_cache.engine,
      node: tn_embed_gather(self.sin_cache.engine, self.sin_cache.node, position_ids)))

proc applyRopeImpl(
      q: Tensor,
      k: Tensor,
      cos: Tensor,
      sin: Tensor): (Tensor, Tensor) =
  ## Freestanding RoPE implementation.
  ##
  ## **Contract:** cos and sin MUST be 2D `(seq, rotary_dim div 2)` half tables.
  ## The rotation width is derived from twice cos's last axis. When it equals
  ## `head_dim` this is the full-head_dim rotation.
  ##
  ## Input q,k: (batch, seq, head, head_dim)
  ## Input cos, sin: (seq, rotary_dim div 2)
  ## Output: (batch, seq, head, head_dim)
  ##
  ## Only the first `rotary_dim` columns of head_dim rotate
  ## (`q_rot * cos + rotateHalf(q_rot) * sin`, NEOX pairwise repetition).
  ## Columns `rotary_dim ..< head_dim` pass through unchanged.
  let
    sq = q.node.getImpl().shape()
    sc = cos.node.getImpl().shape()
  doAssert sc.len == 2, "applyRopeImpl: cos must be 2D (seq, rotary_dim div 2), got " & $sc.len & "D"
  doAssert sin.node.getImpl().shape().len == 2,
    "applyRopeImpl: sin must be 2D (seq, rotary_dim div 2), got " &
    $sin.node.getImpl().shape().len & "D"

  let
    rotary_dim = 2 * sc[1]
    head_dim = sq[3]
  doAssert rotary_dim <= head_dim,
    "applyRopeImpl: rotary_dim " & $rotary_dim & " exceeds head_dim " & $head_dim
  doAssert (rotary_dim mod 2) == 0, "applyRopeImpl: rotary_dim must be even"
  doAssert (head_dim mod 2) == 0, "applyRopeImpl: head_dim must be even"
  # GQA: k has fewer heads than q, the batch/seq/head_dim axes must match
  let sk = k.node.getImpl().shape()
  doAssert sk.len == 4, "applyRopeImpl: k must be rank 4 (batch, seq, head, head_dim)"
  doAssert sk[0] == sq[0] and sk[1] == sq[1] and sk[3] == sq[3],
    "applyRopeImpl: q and k must share batch, seq and head_dim, got " &
    $sq & " and " & $sk

  # Broadcast: each (seq, rotary_dim/2) half table pairs with the (i, i + rotary_dim/2) columns of every head.
  # Reference flow: transpose (batch, seq, head, head_dim) to (batch, head, seq, head_dim) first.
  # This port keeps q's layout, the broadcast over the head axis happens
  # inside the kernel through stride-0 views.
  if sq[0] != 1:
    raise ValueError.newException(
      "applyRopeImpl: batch " & $sq[0] & " unported, the paged KV path " &
      "runs batch_size == 1 only (reference guard)")
  if rotary_dim != head_dim:
    raise ValueError.newException(
      "applyRopeImpl: partial rotation unported (rotary_dim " & $rotary_dim &
      " < head_dim " & $head_dim & "), the Qwen3 path runs partial_rotary_factor 1.0")

  # Full-head_dim rotation:
  #   qwen3 path, an identity with the split when the pass-through
  #   slice is empty.
  let
    tokens = sq[0] * sq[1]
    qFlat = q.reshape([tokens, sq[2], head_dim])
    kFlat = k.reshape([tokens, sk[2], head_dim])
    qRot = Tensor(engine: q.engine,
      node: tn_rope(q.engine, qFlat.node, cos.node, sin.node))
    kRot = Tensor(engine: k.engine,
      node: tn_rope(k.engine, kFlat.node, cos.node, sin.node))
  result = (qRot.reshape([sq[0], sq[1], sq[2], head_dim]),
            kRot.reshape([sk[0], sk[1], sk[2], head_dim]))

proc applyRope*(
    self: RotaryPositionEmbedding,
    q: Tensor,
    k: Tensor,
    cos, sin: Tensor): (Tensor, Tensor) =
  ## Apply RoPE using precomputed cos/sin.
  ##
  ## Args:
  ##   - q, k:
  ##     input tensors of shape (batch, seq, head, head_dim)
  ##   - cos, sin:
  ##     precomputed RoPE half tables of shape (seq, rotary_dim div 2)
  ##
  ## Returns:
  ##   (q_rot, k_rot) of shape (batch, seq, head, head_dim)
  ##
  ## Note:
  ##   Pure function, self stays unchanged.
  ##   cos/sin must match seq_len of q/k.
  applyRopeImpl(q, k, cos, sin)
