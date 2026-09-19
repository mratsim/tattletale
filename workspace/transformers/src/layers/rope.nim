# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/instrumentation

type
  RotaryPositionEmbedding* = ref object
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
    ##    (NEOX half-repeat: dim d and dim d + rotary_dim/2 share a frequency)
    ##  - `ropeByPositions(position_ids)` returns tensors of shape `(seq_len, rotary_dim)`
    ##    where `seq_len == position_ids.numel()`
    ##  - `applyRope` rotates only the first `rotary_dim` columns of head_dim.
    ##    The remaining columns pass through unchanged.
    ##  - `applyRope` is pure: same inputs always produce same outputs
    ##
    ## **USAGE**:
    ##
    ##  ```nim
    ##  # Model init (once)
    ##  let rotary = RotaryPositionEmbedding.new(128, 8192, 1e6, kBFloat16, kCPU)
    ##
    ##  # Partial rotary: only the first 64 of 256 dims rotate (Qwen3.5)
    ##  let rotary = RotaryPositionEmbedding.new(256, 262144, 1e7, kBFloat16, kCPU, rotary_dim = 64)
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
  ## width is derived from `cos.size(-1)`. When it equals `head_dim` this is
  ## the plain full-head_dim rotation.
  ##
  ## Input q,k: (batch, seq, head, head_dim)
  ## Input cos, sin: (seq, rotary_dim)
  ## Output: (batch, seq, head, head_dim)
  ##
  ## Only the first `rotary_dim` columns of head_dim rotate (`q_rot * cos +
  ## rotateHalf(q_rot) * sin`, NEOX pairwise repetition). Columns
  ## `rotary_dim ..< head_dim` pass through unchanged.
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

proc yarnInvFreq(dim: int, theta, factor, betaFast, betaSlow: float64, originalMaxPos: int): Tensor =
  ## Yarn-blended inverse frequencies over `dim div 2` pair entries
  ## (the DeepSeek-V2-Lite lineage, HF `_compute_yarn_parameters` construction).
  ##
  ## The blend extrapolates below the correction range, ramps linearly inside
  ## and interpolates (factor division) above.
  ##
  ## The correction range runs in f64 over the full `dim` width, floored,
  ## ceiled and clamped to `0 ..< dim - 1` per the truncated range.
  ##
  ## Computation runs on CPU, the same home as the plain inv_freq path.
  ## MPS carries no float64, only the cast cos/sin table reaches the device.
  let half = dim div 2
  let plain = F.pow(F.full(1, theta, kFloat64),
    (F.arange(0, dim, 2).to(kFloat64) / dim.float64).neg())
  let interpolated = plain / Scalar(factor)

  let lowF = dim.float64 * ln(originalMaxPos.float64 / (betaFast * 2.0 * PI)) /
    (2.0 * ln(theta))
  let highF = dim.float64 * ln(originalMaxPos.float64 / (betaSlow * 2.0 * PI)) /
    (2.0 * ln(theta))
  let low = max(floor(lowF), 0.0)
  let high = min(ceil(highF), dim.float64 - 1.0)

  let rampIdx = F.arange(0, half, kFloat64)
  let ramp =
    if low == high:
      F.full(half, 0.0, kFloat64)
    else:
      ((rampIdx - low) / (high - low)).clamp(0.0, 1.0)
  let extrapWeight = 1.0 - ramp
  plain * extrapWeight + interpolated * ramp

func new*(_: type RotaryPositionEmbedding,
      head_dim, max_seq_len: int,
      rope_theta: float64,
      dtype: ScalarKind,
      device: DeviceKind,
      rotary_dim = -1,
      activePairs = -1,
      yarnFactor = 0.0'f64,
      yarnBetaFast = 32.0'f64,
      yarnBetaSlow = 1.0'f64,
      yarnOriginalMaxPos = 0,
      attentionFactor = 1.0'f64): RotaryPositionEmbedding =
  ## Build RoPE lookup table for all positions `0..max_seq_len-1`.
  ##
  ## `rotary_dim` defaults to `head_dim` (full rotation). A smaller value
  ## (e.g. 64 for a partial factor 0.5 over head_dim 128) rotates only
  ## the first `rotary_dim` columns.
  ##
  ## The cache is sized `(max_seq_len, rotary_dim)`, a partial rotation
  ## never allocates the full head_dim table.
  ##
  ## `activePairs` bounds the rotating pair count of a proportional
  ## checkpoint (gemma-4 lineage):
  ##
  ## The first `activePairs` pairs carry their theta angles, every
  ## remaining pair stays at zero angle, an exact pass-through rotation.
  ## The default -1 keeps every pair active.
  ##
  ## `yarnFactor > 1` yarn-blends the inverse frequencies over the rotary
  ## table width, `yarnBetaFast`/`yarnBetaSlow`/`yarnOriginalMaxPos`
  ## carry the checkpoint's correction-range parameters.
  ##
  ## `attentionFactor` scales the cos/sin rows after the trigonometry,
  ## the checkpoint's attention_scaling (the yarn mscale by default).
  ##
  ## **Algorithm (NEOX-style)**:
  ##
  ##  1. Compute inverse frequencies for even dimensions only:
  ##     `inv_freq[d] = theta^(-d/rotary_dim)` for `d in {0, 2, 4, ..., rotary_dim-2}`
  ##     This gives `rotary_dim/2` unique frequencies.
  ##     Odd dimensions reuse the same frequency (pairwise repetition).
  ##     Yarn checkpoints blend interpolation and extrapolation per pair,
  ##     proportional checkpoints zero the pairs past `activePairs`.
  ##
  ##  2. For each position `p in {0, ..., max_seq_len-1}` and each
  ##     unique dimension `d`, compute `cos(p * inv_freq[d])` and `sin(p * inv_freq[d])` in FP64 for precision.
  ##
  ##  3. Duplicate the half table to cover all `rotary_dim` positions:
  ##     `[f0 .. f_{m-1}, f0 .. f_{m-1}]` (m = rotary_dim/2) by
  ##     concatenating the table with itself along the dimension axis.
  ##
  ##  4. Scale by `attentionFactor`, cast to `dtype` (e.g., BF16), move to `device`.
  ##
  ## **Complexity**: O(max_seq_len * rotary_dim), done once per model load.
  ##
  let dim = if rotary_dim < 0: head_dim else: rotary_dim
  doAssert dim <= head_dim, "rotary_dim " & $dim & " exceeds head_dim " & $head_dim
  doAssert (dim mod 2) == 0, "rotary_dim must be even"
  let half_dim = dim div 2
  var inv_freq =
    if yarnFactor > 1.0:
      checkValue(yarnOriginalMaxPos > 0,
        "[ttt] RotaryPositionEmbedding: yarn needs original_max_position_embeddings")
      yarnInvFreq(dim, rope_theta, yarnFactor, yarnBetaFast, yarnBetaSlow,
        yarnOriginalMaxPos)
    else:
      F.pow(F.full(1, rope_theta, kFloat64),
        (F.arange(0, dim, 2).to(kFloat64) / dim.float64).neg())
  if activePairs >= 0:
    checkValue(activePairs <= half_dim,
      "[ttt] RotaryPositionEmbedding: activePairs " & $activePairs &
      " exceeds the pair count " & $half_dim)
    if activePairs < half_dim:
      # A 0/1 pair mask multiplies in, the tail pairs keep a zero
      # inv_freq and their rotation is the identity.
      # The mask computes on CPU with the table, MPS carries no float64.
      inv_freq = inv_freq * F.arange(0, half_dim, kFloat64)
        .`<.`(Scalar(activePairs.float64)).to(kFloat64)
  let angles = F.arange(0, max_seq_len, kFloat64).unsqueeze(1) * inv_freq.unsqueeze(0)
  let cos_half = angles.cos()   # (max_seq_len, rotary_dim/2)
  let sin_half = angles.sin()   # (max_seq_len, rotary_dim/2)
  new(result)
  result.head_dim = head_dim
  result.rotary_dim = dim
  result.max_seq_len = max_seq_len
  result.rope_theta = rope_theta
  # NEOX-style: [c0, c0, c1, c1, ...] to cover rotary_dim columns
  result.cos_cache = (F.cat([cos_half, cos_half], -1) * Scalar(attentionFactor)).to(dtype).to(device)
  result.sin_cache = (F.cat([sin_half, sin_half], -1) * Scalar(attentionFactor)).to(dtype).to(device)

proc ropeByPositions*(self: RotaryPositionEmbedding, position_ids: Tensor): (Tensor, Tensor) =
  ## Slice cos/sin cache using position_ids.
  ##
  ## **input_ids vs position_ids — they are NOT the same**:
  ##
  ##   - `input_ids`: Token IDs. *What* to compute (e.g., `[9707, 11, 1246]` = "Hello, how")
  ##   - `position_ids`: Absolute positions in the sequence. *Where* each token goes
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
  ##   (cos, sin) of shape (seq_len, rotary_dim), sliced from cache
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
    self: RotaryPositionEmbedding,
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

# ###########################################################################
# Rope policies
# ###########################################################################

type
  NoPe* = object
    ## Rope policy: no rotation applied, ever. A kpe plane is cached
    ## unrotated when the checkpoint carries one (Kimi-Linear MLA).

  FullRoPe* = object
    ## Rope policy: the whole kpe plane rotates (DeepSeek-V2/V3, GLM).

  PartialRoPe*[rotaryDim: static int] = object
    ## Rope policy: the first `rotaryDim` plane channels rotate,
    ## remaining channels pass through. `rotaryDim` must be even,
    ## never over the plane width, evenness enforced at compile time,
    ## the over-plane width refused at layer init against the runtime
    ## plane width.

# ###########################################################################
# Interleaved rotation kernel
# ###########################################################################

func rotateInterleaved(x: Tensor, cos, sin: Tensor): Tensor =
  ## GPT-J interleaved rotation of the last dimension: the channel pair
  ## (2i, 2i+1) rotates by the row-i angle.
  ##
  ## Args:
  ##   x: (batch, seq, heads, plane), even plane
  ##   cos, sin: (seq, plane div 2) f32 frequency tables
  ##
  ## Returns:
  ##   Rotated tensor, same shape and dtype as `x`. The pair products
  ##   and sums run in f32 and round once back to the storage dtype,
  ##   the same arithmetic and rounding as the complex-multiply form
  ##   produces.
  let batch = x.size(0)
  let seq = x.size(1)
  let heads = x.size(2)
  let plane = x.size(3)
  doAssert (plane mod 2) == 0, "rotateInterleaved: plane width must be even"
  let half = plane div 2
  doAssert cos.dim == 2 and sin.dim == 2, "rotateInterleaved: cos/sin must be 2D (seq, half)"
  doAssert cos.size(0) == seq and sin.size(0) == seq,
    "rotateInterleaved: cos/sin rows must match the sequence length"
  doAssert cos.size(1) == half and sin.size(1) == half,
    "rotateInterleaved: cos/sin columns must match plane/2"
  doAssert cos.scalarType() == kFloat32 and sin.scalarType() == kFloat32,
    "rotateInterleaved: cos/sin must be f32 frequency tables"

  let x32 = x.to(kFloat32)
  let pairs = x32.reshape([batch, seq, heads, half, 2])
  let even = pairs.narrow(4, 0, 1).reshape([batch, seq, heads, half])
  let odd = pairs.narrow(4, 1, 1).reshape([batch, seq, heads, half])
  let cosB = cos.unsqueeze(0).unsqueeze(2) # (1, seq, 1, half)
  let sinB = sin.unsqueeze(0).unsqueeze(2)
  let outEven = even * cosB - odd * sinB
  let outOdd = even * sinB + odd * cosB
  result = F.cat([outEven.unsqueeze(4), outOdd.unsqueeze(4)], 4)
    .reshape([batch, seq, heads, plane]).to(x.scalarType())

func applyRope*(qPe, kPe: Tensor, cos, sin: Tensor, R: typedesc[NoPe]): (Tensor, Tensor) =
  ## NoPe rope: identity. Nothing is computed, the call compiles out.
  (qPe, kPe)

func applyRope*(qPe, kPe: Tensor, cos, sin: Tensor, R: typedesc[FullRoPe]): (Tensor, Tensor) =
  ## FullRoPe: rotate the whole plane of q and k.
  (rotateInterleaved(qPe, cos, sin), rotateInterleaved(kPe, cos, sin))

func applyRope*(qPe, kPe: Tensor, cos, sin: Tensor,
    R: typedesc[PartialRoPe]): (Tensor, Tensor) =
  ## PartialRoPe[rotaryDim]: rotate the first `rotaryDim` plane
  ## channels of q and k, pass the rest through unchanged.
  const rotaryDim = R.rotaryDim
  let plane = qPe.size(3)
  doAssert rotaryDim <= plane,
    "PartialRoPe rotary width " & $rotaryDim & " exceeds the plane width " & $plane
  if rotaryDim == plane:
    return (rotateInterleaved(qPe, cos, sin), rotateInterleaved(kPe, cos, sin))
  else:
    # Frequency tables cover the whole plane (seq, plane div 2),
    # the rotation consumes the first rotaryDim div 2 columns.
    doAssert cos.size(1) == plane div 2 and sin.size(1) == plane div 2,
      "PartialRoPe expects full-plane frequency tables (seq, plane/2)"
    let half = rotaryDim div 2
    let cosRot = cos.narrow(1, 0, half)
    let sinRot = sin.narrow(1, 0, half)
    let qRot = rotateInterleaved(qPe.narrow(3, 0, rotaryDim), cosRot, sinRot)
    let kRot = rotateInterleaved(kPe.narrow(3, 0, rotaryDim), cosRot, sinRot)
    let qOut = F.cat([qRot, qPe.narrow(3, rotaryDim, plane - rotaryDim)], 3)
    let kOut = F.cat([kRot, kPe.narrow(3, rotaryDim, plane - rotaryDim)], 3)
    (qOut, kOut)
