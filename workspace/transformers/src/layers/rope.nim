# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch as F

type
  RotaryPositionEmbeddingRef* = ref object
    ## Rotary Position Embedding (RoPE): precomputed cosine/sine lookup table.
    ##
    ## **LIFETIME**: Per-model. Created once by `new()` at model initialization;
    ##   destroyed when the model (and its shared `rotary` ref) is dropped.
    ##   `cos_cache` and `sin_cache` are **immutable after init** — they are
    ##   **precomputed tables**, not mutable runtime state.
    ##
    ## **DATA FLOW**:
    ##
    ## ```
    ##  ┌─────────────────────────────────────────────────────────┐
    ##  │  Model Init (once, per model)                           │
    ##  │  new(head_dim, max_seq_len, rope_theta, dtype, device,  │
    ##  │      rotary_dim = head_dim)                             │
    ##  │  └─► fills cos_cache, sin_cache  (max_seq_len, rotary_dim)│
    ##  └─────────────────────────────────────────────────────────┘
    ##
    ##  ┌─────────────────────────────────────────────────────────┐
    ##  │  Each Forward Pass                                      │
    ##  │  ropeByPositions(position_ids)                                  │
    ##  │    └─► index_select on cache  (seq_len, rotary_dim)       │
    ##  │        └─► ctx.setRopeForPositions(rotary)                     │
    ##  │              └─► attention layers read ctx.cos, ctx.sin │
    ##  └─────────────────────────────────────────────────────────┘
    ## ```
    ##
    ## **WHY THE CACHE LIVES HERE (not the sliced cos/sin)**:
    ##
    ## The **cache** is **model config** (immutable after init). The **sliced cos/sin**
    ## for each forward pass are **request state** and live in InferenceContext:
    ##
    ##  | Kind                  | Lifetime     | Owner                      |
    ##  |-----------------------|--------------|------------------------------|
    ##  | Layer weights         | Per model    | Layer objects              |
    ##  | Precomputed tables    | Per model    | Config objects (this one)    |
    ##  | Sliced cos/sin        | Per request  | InferenceContext             |
    ##  | Rope variant config   | Per model    | Model (not orchestrator)     |
    ##
    ## `cos_cache`/`sin_cache` fall in the second category. They are derived
    ## from `rope_theta` (model config) and reused across every forward pass
    ## and every request on this model. The sliced cos/sin for each forward pass
    ## are stored in InferenceContext via `ctx.setRopeForPositions(rotary)`.
    ##
    ## **INVARIANTS**:
    ##
    ##  - `cos_cache.shape == sin_cache.shape == (max_seq_len, rotary_dim)`,
    ##    with `rotary_dim <= head_dim` and `max_seq_len == self.max_seq_len`
    ##  - Each value `cos_cache[p, d]` equals `cos(p * rope_theta^(-d/rotary_dim))`
    ##    (with NEOX-style dim repetition: odd dims copy even dims)
    ##  - `ropeByPositions(position_ids)` returns tensors of shape `(seq_len, rotary_dim)`
    ##    where `seq_len == position_ids.numel()`
    ##  - `applyRope` rotates only the first `rotary_dim` columns of head_dim;
    ##    the remaining columns pass through unchanged
    ##  - `applyRope` is pure: same inputs always produce same outputs
    ##
    ## **USAGE**:
    ##
    ##  ```nim
    ##  # Model init (once)
    ##  let rotary = RotaryPositionEmbeddingRef.new(128, 8192, 1e6, kBFloat16, kCPU)
    ##
    ##  # Partial rotary: only the first 64 of 256 dims rotate (Qwen3.5)
    ##  let rotary = RotaryPositionEmbeddingRef.new(256, 262144, 1e7, kBFloat16, kCPU, rotary_dim = 64)
    ##
    ##  # Each forward pass: model calls ctx.setRopeForPositions(rotary)
    ##  ctx.setRopeForPositions(rotary)
    ##  # ... attention layers read ctx.cos, ctx.sin internally
    ##  ```
    ##
    head_dim*: int
    rotary_dim*: int
    max_seq_len*: int
    rope_theta*: float64
    cos_cache*: Tensor     ## Precomputed (max_seq_len, rotary_dim). Immutable after init.
    sin_cache*: Tensor     ## Precomputed (max_seq_len, rotary_dim). Immutable after init.

func rotateHalf(x: Tensor): Tensor =
  # Input/Output: (batch, head, seq, dim) with dim even.
  # Pairwise split: [-x2, x1] over the last dimension.
  let dim = x.size(3)
  let half_dim = dim div 2
  let x1 = x[_, _, _, 0..<half_dim]
  let x2 = x[_, _, _, half_dim..<dim]
  F.cat([x2.neg(), x1], -1)

func applyRopeImpl(
      q: Tensor,
      k: Tensor,
      cos: Tensor,
      sin: Tensor): (Tensor, Tensor) =
  ## Freestanding RoPE implementation.
  ##
  ## **Contract:** cos and sin MUST be 2D `(seq, rotary_dim)`. The rotation
  ## width is derived from `cos.size(-1)`; when it equals `head_dim` this is
  ## the plain full-head_dim rotation.
  ##
  ## Input q,k: (batch, seq, head, head_dim)
  ## Input cos, sin: (seq, rotary_dim)
  ## Output: (batch, seq, head, head_dim)
  ##
  ## Only the first `rotary_dim` columns of head_dim rotate (`q_rot * cos +
  ## rotateHalf(q_rot) * sin`, NEOX pairwise repetition); columns
  ## `rotary_dim ..< head_dim` pass through unchanged. This matches the
  ## vendored `apply_rotary_pos_emb` (rotary_dim = cos.shape[-1], split
  ## q_rot/q_pass, rotate q_rot, concatenate).
  doAssert cos.dim == 2, "applyRopeImpl: cos must be 2D (seq, rotary_dim), got " & $cos.dim & "D"
  doAssert sin.dim == 2, "applyRopeImpl: sin must be 2D (seq, rotary_dim), got " & $sin.dim & "D"

  let rotary_dim = cos.size(1)
  let head_dim = q.size(3)
  doAssert rotary_dim <= head_dim,
    "applyRopeImpl: rotary_dim " & $rotary_dim & " exceeds head_dim " & $head_dim
  doAssert (rotary_dim mod 2) == 0, "applyRopeImpl: rotary_dim must be even"
  doAssert (head_dim mod 2) == 0, "applyRopeImpl: head_dim must be even"

  # Transpose to (batch, head, seq, head_dim) for rotation
  var q_t = q.transpose(1, 2)
  var k_t = k.transpose(1, 2)

  # Broadcast: (seq, rotary_dim) -> (1, 1, seq, rotary_dim) -> matches (batch, head, seq, rotary_dim)
  let cos = cos.unsqueeze(0).unsqueeze(0)
  let sin = sin.unsqueeze(0).unsqueeze(0)

  if rotary_dim == head_dim:
    # Full-head_dim rotation: qwen3 path (identity with the split when the
    # pass-through slice is empty).
    let q_rot_t = q_t * cos + rotateHalf(q_t) * sin
    let k_rot_t = k_t * cos + rotateHalf(k_t) * sin
    result = (q_rot_t.transpose(1, 2), k_rot_t.transpose(1, 2))
  else:
    # Partial rotation: rotate the first rotary_dim columns, keep the rest.
    let qRot = q_t[_, _, _, 0..<rotary_dim]
    let qPass = q_t[_, _, _, rotary_dim..<head_dim]
    let qRotated = qRot * cos + rotateHalf(qRot) * sin
    let kRot = k_t[_, _, _, 0..<rotary_dim]
    let kPass = k_t[_, _, _, rotary_dim..<head_dim]
    let kRotated = kRot * cos + rotateHalf(kRot) * sin
    result = (F.cat([qRotated, qPass], -1).transpose(1, 2),
              F.cat([kRotated, kPass], -1).transpose(1, 2))

func new*(_: type RotaryPositionEmbeddingRef,
      head_dim, max_seq_len: int,
      rope_theta: float64,
      dtype: ScalarKind,
      device: DeviceKind,
      rotary_dim = -1): RotaryPositionEmbeddingRef =
  ## Build RoPE lookup table for all positions `0..max_seq_len-1`.
  ##
  ## `rotary_dim` defaults to `head_dim` (full rotation); pass a smaller value
  ## (e.g. 64 for Qwen3.5) to rotate only the first `rotary_dim` columns.
  ## The cache is sized `(max_seq_len, rotary_dim)` so a partial rotation does
  ## not allocate the full head_dim table.
  ##
  ## **Algorithm (NEOX-style)**:
  ##
  ##  1. Compute inverse frequencies for even dimensions only:
  ##     `inv_freq[d] = theta^(-d/rotary_dim)` for `d in {0, 2, 4, ..., rotary_dim-2}`
  ##     This gives `rotary_dim/2` unique frequencies.
  ##     Odd dimensions reuse the same frequency (pairwise repetition).
  ##
  ##  2. For each position `p in {0, ..., max_seq_len-1}` and each
  ##     unique dimension `d`, compute `cos(p * inv_freq[d])` and
  ##     `sin(p * inv_freq[d])` in FP64 for precision.
  ##
  ##  3. Duplicate even-dimension values to cover all `rotary_dim` positions:
  ##     `[freq_0, freq_0, freq_1, freq_1, ...]` by concatenating the
  ##     even-dim table with itself along the dimension axis.
  ##
  ##  4. Cast result to `dtype` (e.g., BF16) and move to `device`.
  ##
  ## **Complexity**: O(max_seq_len * rotary_dim) — done once per model load.
  ##
  let dim = if rotary_dim < 0: head_dim else: rotary_dim
  doAssert dim <= head_dim, "rotary_dim " & $dim & " exceeds head_dim " & $head_dim
  doAssert (dim mod 2) == 0, "rotary_dim must be even"
  let half_dim = dim div 2
  let inv_freq = F.arange(0, dim, 2).to(kFloat64) / dim.float64
  let inv_freq_final = F.pow(F.full([1], rope_theta, kFloat64), -inv_freq)  # theta^(-inv_freq), (rotary_dim/2,)
  let angles = F.arange(0, max_seq_len, kFloat64).unsqueeze(1) * inv_freq_final.unsqueeze(0)
  let cos_half = angles.cos()   # (max_seq_len, rotary_dim/2)
  let sin_half = angles.sin()   # (max_seq_len, rotary_dim/2)
  new(result)
  result.head_dim = head_dim
  result.rotary_dim = dim
  result.max_seq_len = max_seq_len
  result.rope_theta = rope_theta
  # NEOX-style: [c0, c0, c1, c1, ...] to cover rotary_dim columns
  result.cos_cache = F.cat([cos_half, cos_half], -1).to(dtype).to(device)
  result.sin_cache = F.cat([sin_half, sin_half], -1).to(dtype).to(device)

proc ropeByPositions*(self: RotaryPositionEmbeddingRef, position_ids: Tensor): (Tensor, Tensor) =
  ## Slice cos/sin cache using position_ids.
  ##
  ## **input_ids vs position_ids — they are NOT the same**:
  ##
  ##   - `input_ids`: Token IDs. *What* to compute (e.g., `[9707, 11, 1246]` = "Hello, how")
  ##   - `position_ids`: Absolute positions in the sequence. *Where* each token sits
  ##
  ##   For the common case (prefill from 0, decode sequentially):
  ##     input_ids = `[9707, 11, 1246]`  →  position_ids = `[0, 1, 2]`
  ##     input_ids = `[498]`             →  position_ids = `[3]`  (next token at offset 3)
  ##
  ##   They diverge for continuous batching (different sequences at different positions),
  ##   prefix caching (skip cached tokens), and speculative decoding (non-contiguous).
  ##
  ## Args:
  ##   position_ids: Tensor of shape (seq_len,) or (batch, seq_len)
  ##
  ## Returns:
  ##   (cos, sin) of shape (seq_len, rotary_dim) — sliced from cache
  ##
  ## Note:
  ##   Called once per forward pass at model level.
  ##   Result is stored in InferenceContext via `ctx.setRopeForPositions(rotary)`.

  # Handle 1D or 2D position_ids
  var pos_ids = position_ids.to(self.cos_cache.deviceType())
  if pos_ids.dim == 2:
    # Take first batch item (positions same for all batch items)
    pos_ids = pos_ids[0, _]

  # Slice cache using position_ids (advanced indexing)
  # cos_cache[position_ids, :] → (seq_len, rotary_dim)
  result = (self.cos_cache.index_select(0, pos_ids), self.sin_cache.index_select(0, pos_ids))

proc applyRope*(
    self: RotaryPositionEmbeddingRef,
    q: Tensor,
    k: Tensor,
    cos, sin: Tensor): (Tensor, Tensor) =
  ## Apply RoPE using precomputed cos/sin.
  ##
  ## Args:
  ##   q, k: Input tensors of shape (batch, seq, head, head_dim)
  ##   cos, sin: Precomputed RoPE of shape (seq, rotary_dim)
  ##
  ## Returns:
  ##   (q_rot, k_rot) of shape (batch, seq, head, head_dim)
  ##
  ## Note:
  ##   Pure function — no mutation of self.
  ##   cos/sin must match seq_len of q/k.
  applyRopeImpl(q, k, cos, sin)
