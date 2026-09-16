# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Multi-head Latent Attention (MLA) mixers: the typed latent cache and
## the DeepSeek-V2/V3 form over the paged cache. Rope policies
## (`NoPe`/`FullRoPe`/`PartialRoPe`) live in `layers/rope.nim`.
##
## - `MlaRotary` bakes the per-model f32 frequency tables, plain theta
##   or the checkpoint's yarn parameters.
## - `MlaLatentCache` is the typed latent cache, K the normed latent,
##   V the kpe plane, the plane width runtime from the checkpoint
##   qk_rope_head_dim.
## - `MLAttention[QKNorm, RopePolicy]` spells the DeepSeek-V2/V3
##   attention body, prefill and decode share it, decode decompresses
##   per step exactly like the reference.
## - The cache write offset is `ctx.kv_position`, the paged lifecycle
##   is the shared one (startSequence / appendToken / endSequence).
##
## Glossary: `kpe` is the DeepSeek-V2 decoupled RoPE key plane:
## the qk_rope_head_dim per-head key channels the HF config names k_pe.
## Those channels run through RoPE, cached beside the normed latent K,
## and are re-attached to the decompressed keys at decode.

import
  std/math,
  std/options,
  workspace/libtorch as F,
  workspace/transformers/src/instrumentation,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/layers/rope,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context

{.experimental: "callOperator".}


# ###########################################################################
# Softmax scale
# ###########################################################################

proc mlaSoftmaxScale*(qkNopeHeadDim, qkRopeHeadDim: int): float64 =
  ## Softmax scale of an MLA attention body: 1/sqrt(qk_head_dim),
  ## qk_head_dim = qk_nope_head_dim + qk_rope_head_dim, the DeepSeek-V2
  ## paper convention (softmax(QK^T / sqrt(qk_head_dim))). One helper
  ## serves every MLA variant in this repo, no per-model copies. No
  ## mscale folds in on plain-rope checkpoints: the yarn mscale branch
  ## stays skipped there.
  pow(float64(qkNopeHeadDim + qkRopeHeadDim), -0.5)

# ###########################################################################
# Frequency tables
# ###########################################################################

type
  MlaRotary* = ref object
    ## Per-model f32 frequency tables for the MLA plane rotation.
    ## Immutable after init. `new` bakes `max_seq_len` positions,
    ## each forward pass slices the tables for the active position_ids.
    planeWidth*: int          ## qk_rope_head_dim, even
    maxSeqLen*: int
    cosCache*: Tensor         ## (max_seq_len, planeWidth div 2) f32
    sinCache*: Tensor         ## (max_seq_len, planeWidth div 2) f32

proc yarnInvFreq(planeWidth: int, theta: float64, factor: float64,
    betaFast, betaSlow: float64, originalMaxPos: int, device: DeviceKind): Tensor =
  ## Yarn-blended inverse frequencies, mirroring the reference yarn
  ## formula: extrapolation below the correction range, linear ramp
  ## inside, interpolation (factor division) above. Computed in f32.
  let dim = planeWidth
  let half = dim div 2
  let pos = F.arange(0, dim, 2, F.tensorOptions(F.kFloat32, device)) / float(dim)
  let plain = F.pow(F.full(1, theta, F.tensorOptions(F.kFloat32, device)), pos.neg()) # (half,)
  let interpolated = plain / Scalar(factor)

  # Correction range in dimension units, f64 math, floored and ceiled.
  let lowF = dim.float64 * ln(originalMaxPos.float64 / (betaFast * 2.0 * PI)) /
    (2.0 * ln(theta))
  let highF = dim.float64 * ln(originalMaxPos.float64 / (betaSlow * 2.0 * PI)) /
    (2.0 * ln(theta))
  let low = max(floor(lowF), 0.0)
  let high = min(ceil(highF), float(half - 1))

  let rampIdx = F.arange(0, half, F.tensorOptions(F.kFloat32, device))
  var ramp: Tensor
  if low == high:
    ramp = F.full(half, 0.0, F.tensorOptions(F.kFloat32, device))
  else:
    ramp = ((rampIdx - low) / (high - low)).clamp(0.0, 1.0)
  let extrapWeight = 1.0 - ramp # 1 below the range, 0 above
  plain * extrapWeight + interpolated * ramp

proc new*(_: type MlaRotary, planeWidth, maxSeqLen: int, theta: float64,
    device: DeviceKind,
    yarnFactor = 0.0'f64, yarnBetaFast = 32.0'f64, yarnBetaSlow = 1.0'f64,
    yarnOriginalMaxPos = 0): MlaRotary =
  ## Bake the plane frequency tables for positions 0 .. max_seq_len-1.
  ##
  ## Plain theta when `yarnFactor <= 1`: inv_freq = theta^(-2i/plane).
  ## With `yarnFactor > 1` the checkpoint yarn parameters blend
  ## interpolation and extrapolation per dimension (DeepSeek-V2-Lite).
  ## Angles and trigonometry run in f32, mirroring the reference.
  doAssert planeWidth > 0 and (planeWidth mod 2) == 0,
    "MlaRotary: planeWidth must be a positive even count"
  doAssert maxSeqLen > 0
  let opts = F.tensorOptions(F.kFloat32, device)
  let invFreq =
    if yarnFactor > 1.0:
      checkValue(yarnOriginalMaxPos > 0,
        "[ttt] MlaRotary: yarn needs original_max_position_embeddings")
      yarnInvFreq(planeWidth, theta, yarnFactor,
        yarnBetaFast, yarnBetaSlow, yarnOriginalMaxPos, device)
    else:
      let pos = F.arange(0, planeWidth, 2, opts) / float(planeWidth)
      F.pow(F.full(1, theta, F.tensorOptions(F.kFloat32, device)), pos.neg())
  let angles = F.arange(0, maxSeqLen, opts).unsqueeze(1) * invFreq.unsqueeze(0)
  new(result)
  result.planeWidth = planeWidth
  result.maxSeqLen = maxSeqLen
  result.cosCache = angles.cos()
  result.sinCache = angles.sin()

proc setMlaRopeForPositions*(ctx: var InferenceContext, rotary: MlaRotary) =
  ## Populate ctx.cos and ctx.sin from the MLA frequency tables,
  ## matching the active position_ids. Called once per forward pass
  ## by the model, the mirror of setRopeForPositions for the MLA mixers.
  ##
  ## Context contract: ctx.cos and ctx.sin are f32 (seq, planeWidth div 2)
  ## frequency pair tables (unique frequencies, interleaved rotation),
  ## one row per token of the active pass. In the supported families
  ## MLA and the NEOX GQA-style mixers never share one checkpoint.
  ## Both ctx.cos contracts therefore never collide.
  let posIds = ctx.position_ids.to(rotary.cosCache.deviceType())
  var flat = posIds
  if flat.dim == 2:
    flat = flat[0, _]
  ctx.cos = rotary.cosCache.index_select(0, flat)
  ctx.sin = rotary.sinCache.index_select(0, flat)

# ###########################################################################
# Typed latent cache
# ###########################################################################

type
  MlaLatentCache* = ref object
    ## Typed MLA latent cache over the paged KV pool.
    ##
    ## - K buffer: the normed compressed latent, kvLoraRank channels,
    ##   one head, written and gathered per layer.
    ## - V buffer: when `kpeWidth > 0` the V buffer carries the kpe
    ##   plane, `kpeWidth` channels, absent otherwise. Rope policy
    ##   never changes the width: a NoPe checkpoint caches its plane
    ##   unrotated.
    ## - The gather buffers hold one contiguous per-layer slab per forward
    ##   pass, sized maxSeqLen at init and shared across the layers.
    ##   Each layer overwrite completes before the next reads.
    kvLoraRank*: int
    kpeWidth*: int          ## qk_rope_head_dim, even, 0 when no plane exists
    maxSeqLen*: int
    latentGatherBuf: Tensor # (1, maxSeqLen, 1, kvLoraRank)
    kpeGatherBuf: Tensor    # (1, maxSeqLen, 1, kpeWidth), when kpeWidth > 0

func init*(_: type MlaLatentCache, kvLoraRank, kpeWidth, maxSeqLen: int,
    dtype: ScalarKind, device: DeviceKind): MlaLatentCache =
  ## Build one typed latent cache: `kpeWidth` is the checkpoint
  ## qk_rope_head_dim read from the config at parse, 0 when no rope
  ## plane exists.
  checkValue(kpeWidth >= 0,
    "[ttt] MlaLatentCache: kpe width is " & $kpeWidth &
    ", expected a non-negative channel count")
  checkValue(kpeWidth mod 2 == 0,
    "[ttt] MlaLatentCache: kpe width " & $kpeWidth &
    " is not an even channel count")
  checkValue(kvLoraRank > 0,
    "[ttt] MlaLatentCache: kv_lora_rank is " & $kvLoraRank &
    ", expected a positive count")
  checkValue(maxSeqLen > 0,
    "[ttt] MlaLatentCache: max_seq_len is " & $maxSeqLen &
    ", expected a positive count")
  let opts = F.tensorOptions(dtype, device)
  result = MlaLatentCache(
    kvLoraRank: kvLoraRank,
    kpeWidth: kpeWidth,
    maxSeqLen: maxSeqLen,
    latentGatherBuf: F.zeros(1, maxSeqLen, 1, kvLoraRank, opts),
  )
  if kpeWidth > 0:
    result.kpeGatherBuf = F.zeros(1, maxSeqLen, 1, kpeWidth, opts)

proc writeLatentPages(ctx: var InferenceContext, layerIdx: int,
    latent: Tensor, offset, seqLen: int) =
  ## Page write loop, the latent slab only (plane-less cache form).
  ## Plain procs: the fancy page indexing does not resolve inside generic
  ## proc bodies (generic sandwich), the typed forms stay wrappers.
  let writeStart = max(0, ctx.cached_tokens - offset)
  var t = writeStart
  while t < seqLen:
    let globalPos = offset + t
    let pageIdx = globalPos div TokensPerPage
    let withinPage = globalPos mod TokensPerPage
    let page = ctx.pages[pageIdx]
    let chunkLen = min(TokensPerPage - withinPage, seqLen - t)
    let chunkEnd = t + chunkLen
    page.k_view[layerIdx, withinPage ..< withinPage + chunkLen].copyFrom(
      latent[0, t ..< chunkEnd, _, _])
    t = chunkEnd

proc writeKpePages(ctx: var InferenceContext, layerIdx: int,
    latent, kpe: Tensor, offset, seqLen: int) =
  ## Page write loop for both slabs: the latent and the kpe plane,
  ## the plane-ful cache form. Same plain-proc rule as writeLatentPages.
  let writeStart = max(0, ctx.cached_tokens - offset)
  var t = writeStart
  while t < seqLen:
    let globalPos = offset + t
    let pageIdx = globalPos div TokensPerPage
    let withinPage = globalPos mod TokensPerPage
    let page = ctx.pages[pageIdx]
    let chunkLen = min(TokensPerPage - withinPage, seqLen - t)
    let chunkEnd = t + chunkLen
    page.k_view[layerIdx, withinPage ..< withinPage + chunkLen].copyFrom(
      latent[0, t ..< chunkEnd, _, _])
    page.v_view[layerIdx, withinPage ..< withinPage + chunkLen].copyFrom(
      kpe[0, t ..< chunkEnd, _, _])
    t = chunkEnd

proc write*(c: MlaLatentCache, ctx: var InferenceContext,
    layerIdx: int, latent, kpe: Tensor, offset, seqLen: int) =
  ## Write the normed latent and the rope'd (or unrotated) kpe plane
  ## for this layer's tokens into the borrowed pages. Plane-less
  ## caches have no kpe to store: `writeLatent` is their form.
  checkValue(c.kpeWidth > 0,
    "[ttt] MlaLatentCache write: the cache carries no kpe plane," &
    " a plane-less cache stores the latent through writeLatent")
  doAssert latent.size(2) == 1 and kpe.size(2) == 1,
    "latent cache write expects single-head (1, seq, 1, width) tensors"
  doAssert latent.size(3) == c.kvLoraRank,
    "latent width " & $latent.size(3) & " does not match kvLoraRank " & $c.kvLoraRank
  doAssert kpe.size(3) == c.kpeWidth,
    "kpe width " & $kpe.size(3) & " does not match the plane width " & $c.kpeWidth
  writeKpePages(ctx, layerIdx, latent, kpe, offset, seqLen)

proc writeLatent*(c: MlaLatentCache, ctx: var InferenceContext,
    layerIdx: int, latent: Tensor, offset, seqLen: int) =
  ## Write the normed latent of a plane-less cache (qk_rope_head_dim 0).
  ## Plane-ful caches must store their kpe plane beside the latent.
  checkValue(c.kpeWidth == 0,
    "[ttt] MlaLatentCache writeLatent: the cache carries a " & $c.kpeWidth &
    "-channel kpe plane, a plane-ful cache stores it through write")
  doAssert latent.size(2) == 1,
    "latent cache write expects single-head (1, seq, 1, width) tensors"
  doAssert latent.size(3) == c.kvLoraRank,
    "latent width " & $latent.size(3) & " does not match kvLoraRank " & $c.kvLoraRank
  writeLatentPages(ctx, layerIdx, latent, offset, seqLen)

proc gatherLatentPages(ctx: InferenceContext, layerIdx: int,
    latentBuf: var Tensor, offset, seqLen: int) =
  ## Page gather loop for the contiguous latent slab,
  ## the plane-less cache form. Plain proc, same rule as the write helpers.
  let total = offset + seqLen
  let numPages = ceilDiv(total, TokensPerPage)
  for p in 0 ..< numPages:
    let pageStart = p * TokensPerPage
    let pageEnd = min(pageStart + TokensPerPage, total)
    let pageValidLen = pageEnd - pageStart
    let page = ctx.pages[p]
    latentBuf[0, pageStart ..< pageEnd, _, _] =
      page.k_view[layerIdx, 0 ..< pageValidLen]

proc gatherKpePages(ctx: InferenceContext, layerIdx: int,
    latentBuf, kpeBuf: var Tensor, offset, seqLen: int) =
  ## Page gather loop for the contiguous latent and kpe slabs,
  ## the plane-ful cache form. Same plain-proc rule as the write helpers.
  let total = offset + seqLen
  let numPages = ceilDiv(total, TokensPerPage)
  for p in 0 ..< numPages:
    let pageStart = p * TokensPerPage
    let pageEnd = min(pageStart + TokensPerPage, total)
    let pageValidLen = pageEnd - pageStart
    let page = ctx.pages[p]
    latentBuf[0, pageStart ..< pageEnd, _, _] =
      page.k_view[layerIdx, 0 ..< pageValidLen]
    kpeBuf[0, pageStart ..< pageEnd, _, _] =
      page.v_view[layerIdx, 0 ..< pageValidLen]

proc gather*(c: MlaLatentCache, ctx: InferenceContext,
    layerIdx, offset, seqLen: int): (Tensor, Tensor) =
  ## Gather this layer's cached latent and kpe plane into contiguous
  ## (1, total, 1, width) slabs for the decompress step. Plane-less caches
  ## gather through `gatherLatent`.
  checkValue(c.kpeWidth > 0,
    "[ttt] MlaLatentCache gather: the cache carries no kpe plane," &
    " a plane-less cache gathers the latent through gatherLatent")
  checkValue(offset + seqLen <= c.maxSeqLen,
    "[ttt] MlaLatentCache gather range " & $(offset + seqLen) &
    " exceeds the cache max_seq_len " & $c.maxSeqLen &
    ", size the pool for the full context")
  gatherKpePages(ctx, layerIdx, c.latentGatherBuf, c.kpeGatherBuf, offset, seqLen)
  let total = offset + seqLen
  result = (
    c.latentGatherBuf.narrow(1, 0, total),
    c.kpeGatherBuf.narrow(1, 0, total))

proc gatherLatent*(c: MlaLatentCache, ctx: InferenceContext,
    layerIdx, offset, seqLen: int): Tensor =
  ## Gather the cached latent of a plane-less cache.
  checkValue(c.kpeWidth == 0,
    "[ttt] MlaLatentCache gatherLatent: the cache carries a " & $c.kpeWidth &
    "-channel kpe plane, a plane-ful cache gathers it through gather")
  checkValue(offset + seqLen <= c.maxSeqLen,
    "[ttt] MlaLatentCache gather range " & $(offset + seqLen) &
    " exceeds the cache max_seq_len " & $c.maxSeqLen &
    ", size the pool for the full context")
  gatherLatentPages(ctx, layerIdx, c.latentGatherBuf, offset, seqLen)
  result = c.latentGatherBuf.narrow(1, 0, offset + seqLen)

# MLAttention
# ###########################################################################

type
  MLAttention*[QKNorm, RopePolicy] = ref object
    ## Multi-head Latent Attention over the paged latent cache,
    ## the DeepSeek-V2/V3 form.
    ##
    ## - `QKNorm` names the query-compression norm class. `void` means
    ##   the checkpoint maps Q directly: q_lora_rank None,
    ##   DeepSeek-V2-Lite, Moonlight, Kimi. `RmsNorm` means Q travels
    ##   through the q_lora bottleneck (GLM, Ling).
    ## - `RopePolicy` is the rope policy, the kpe plane width comes from the cache
    ##   (the checkpoint qk_rope_head_dim). Policy-plane checks
    ##   run at init: every rope policy refuses a plane-less layer, the kpe
    ##   plane being part of the key math, NoPe caching its plane
    ##   unrotated, a latent-only layer being a separate mixer
    ##   type, PartialRoPe refusing a rotary width over the plane.
    ## - Prefill and decode share the body: tokens are written
    ##   compressed to the latent cache, normed latent plus plane,
    ##   and the keys and values decompress from the gathered cache
    ##   per forward pass. This mirrors the reference decode shape.
    ##   An absorbed form is a separate mixer type.
    layer_idx*: int             # Layer index, indexes the cache pages
    name*: string               # Safetensor key prefix (e.g. "model.layers.0.self_attn")
    numHeads: int
    qkNopeHeadDim: int          # per-head non-rope key/query width (128)
    kvLoraRank: int             # compressed latent width (512)
    vHeadDim: int               # per-head value width (128)
    softmaxScale: float64       # config-owned: qk_head_dim^-0.5, yarn mscale^2 folded in at wiring
    when QKNorm isnot void:
      q_a_proj: Linear          # [q_lora_rank, hidden]
      q_a_norm: QKNorm          # query bottleneck norm
      q_b_proj: Linear          # [numHeads * qkHeadDim, q_lora_rank]
    else:
      q_proj: Linear            # [numHeads * (qkNopeHeadDim + kpeWidth), hidden]
    kv_a_proj_with_mqa: Linear  # [kvLoraRank + kpeWidth, hidden]
    kv_a_layernorm: RmsNorm     # norm over the compressed latent
    kv_b_proj: Linear           # [numHeads * (qkNopeHeadDim + vHeadDim), kvLoraRank]
    o_proj: Linear              # [hidden, numHeads * vHeadDim]
    cache: MlaLatentCache

proc checkPolicyPlane(RopePolicy: typedesc, kpeWidth: int) {.inline.} =
  ## Init-time check: every rope policy requires a kpe plane
  ## (kpeWidth > 0), NoPe alone runs plane-less; a latent-only layer
  ## is a separate mixer type.
  checkValue(kpeWidth > 0,
    "[ttt] MLAttention refuses a plane-less layer: the kpe plane is" &
    " part of the key math on every rope policy, NoPe caches it" &
    " unrotated, and a latent-only layer is a separate mixer type")
  when RopePolicy is PartialRoPe:
    when (RopePolicy.rotaryDim mod 2) != 0:
      {.error: "PartialRoPe rotary width must be even".}
    checkValue(RopePolicy.rotaryDim <= kpeWidth,
      "[ttt] MLAttention: PartialRoPe rotary width " & $RopePolicy.rotaryDim &
      " exceeds the kpe plane width " & $kpeWidth)

func init*[QKNorm, RopePolicy](
    _: type MLAttention[QKNorm, RopePolicy],
    layer_idx: int,
    name: string,
    q_proj, o_proj: Linear,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    numHeads, qkNopeHeadDim, kvLoraRank, vHeadDim: int,
    softmaxScale: float64,
    cache: MlaLatentCache): MLAttention[QKNorm, RopePolicy] =
  ## Direct-Q layer (q_lora_rank None). Raises ValueError naming the layer
  ## key path when a projection width disagrees with the head dims.
  checkPolicyPlane(RopePolicy, cache.kpeWidth)
  let kpeWidth = cache.kpeWidth
  let qkHeadDim = qkNopeHeadDim + kpeWidth
  checkValue(q_proj.out_features == numHeads * qkHeadDim,
    "[ttt] " & name & ": q_proj out_features is " & $q_proj.out_features &
    ", expected num_heads * (qk_nope_head_dim + qk_rope_head_dim) = " & $qkHeadDim)
  checkValue(kv_a_proj_with_mqa.out_features == kvLoraRank + kpeWidth,
    "[ttt] " & name & ": kv_a_proj_with_mqa out_features is " &
    $kv_a_proj_with_mqa.out_features & ", expected kv_lora_rank + qk_rope_head_dim = " &
    $(kvLoraRank + kpeWidth))
  checkValue(kv_b_proj.in_features == kvLoraRank,
    "[ttt] " & name & ": kv_b_proj in_features is " & $kv_b_proj.in_features &
    ", expected kv_lora_rank " & $kvLoraRank)
  checkValue(kv_b_proj.out_features == numHeads * (qkNopeHeadDim + vHeadDim),
    "[ttt] " & name & ": kv_b_proj out_features is " & $kv_b_proj.out_features &
    ", expected num_heads * (qk_nope_head_dim + v_head_dim)")
  checkValue(o_proj.in_features == numHeads * vHeadDim,
    "[ttt] " & name & ": o_proj in_features is " & $o_proj.in_features &
    ", expected num_heads * v_head_dim")
  checkValue(kv_a_layernorm.hidden_size == kvLoraRank,
    "[ttt] " & name & ": kv_a_layernorm width is " & $kv_a_layernorm.hidden_size &
    ", expected kv_lora_rank " & $kvLoraRank)
  checkValue(cache.kvLoraRank == kvLoraRank,
    "[ttt] " & name & ": cache kv_lora_rank is " & $cache.kvLoraRank &
    ", expected " & $kvLoraRank)
  MLAttention[QKNorm, RopePolicy](
    layer_idx: layer_idx,
    name: name,
    numHeads: numHeads,
    qkNopeHeadDim: qkNopeHeadDim,
    kvLoraRank: kvLoraRank,
    vHeadDim: vHeadDim,
    softmaxScale: softmaxScale,
    q_proj: q_proj,
    kv_a_proj_with_mqa: kv_a_proj_with_mqa,
    kv_a_layernorm: kv_a_layernorm,
    kv_b_proj: kv_b_proj,
    o_proj: o_proj,
    cache: cache
  )

func init*[QKNorm, RopePolicy](
    _: type MLAttention[QKNorm, RopePolicy],
    layer_idx: int,
    name: string,
    q_a_proj, q_b_proj: Linear,
    q_a_norm: QKNorm,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    numHeads, qkNopeHeadDim, kvLoraRank, vHeadDim: int,
    softmaxScale: float64,
    cache: MlaLatentCache): MLAttention[QKNorm, RopePolicy] =
  ## Compressed-Q layer (q_lora_rank not None) with the `QKNorm`
  ## bottleneck norm. Width checks mirror the direct-Q overload.
  checkPolicyPlane(RopePolicy, cache.kpeWidth)
  let kpeWidth = cache.kpeWidth
  let qkHeadDim = qkNopeHeadDim + kpeWidth
  checkValue(q_a_proj.out_features == q_b_proj.in_features,
    "[ttt] " & name & ": q_a_proj out_features " & $q_a_proj.out_features &
    " and q_b_proj in_features " & $q_b_proj.in_features & " disagree")
  checkValue(q_b_proj.out_features == numHeads * qkHeadDim,
    "[ttt] " & name & ": q_b_proj out_features is " & $q_b_proj.out_features &
    ", expected num_heads * (qk_nope_head_dim + qk_rope_head_dim) = " & $qkHeadDim)
  checkValue(kv_a_proj_with_mqa.out_features == kvLoraRank + kpeWidth,
    "[ttt] " & name & ": kv_a_proj_with_mqa out_features is " &
    $kv_a_proj_with_mqa.out_features & ", expected kv_lora_rank + qk_rope_head_dim = " &
    $(kvLoraRank + kpeWidth))
  checkValue(kv_b_proj.in_features == kvLoraRank,
    "[ttt] " & name & ": kv_b_proj in_features is " & $kv_b_proj.in_features &
    ", expected kv_lora_rank " & $kvLoraRank)
  checkValue(kv_b_proj.out_features == numHeads * (qkNopeHeadDim + vHeadDim),
    "[ttt] " & name & ": kv_b_proj out_features is " & $kv_b_proj.out_features &
    ", expected num_heads * (qk_nope_head_dim + v_head_dim)")
  checkValue(o_proj.in_features == numHeads * vHeadDim,
    "[ttt] " & name & ": o_proj in_features is " & $o_proj.in_features &
    ", expected num_heads * v_head_dim")
  checkValue(kv_a_layernorm.hidden_size == kvLoraRank,
    "[ttt] " & name & ": kv_a_layernorm width is " & $kv_a_layernorm.hidden_size &
    ", expected kv_lora_rank " & $kvLoraRank)
  checkValue(cache.kvLoraRank == kvLoraRank,
    "[ttt] " & name & ": cache kv_lora_rank is " & $cache.kvLoraRank &
    ", expected " & $kvLoraRank)
  MLAttention[QKNorm, RopePolicy](
    layer_idx: layer_idx,
    name: name,
    numHeads: numHeads,
    qkNopeHeadDim: qkNopeHeadDim,
    kvLoraRank: kvLoraRank,
    vHeadDim: vHeadDim,
    softmaxScale: softmaxScale,
    q_a_proj: q_a_proj,
    q_a_norm: q_a_norm,
    q_b_proj: q_b_proj,
    kv_a_proj_with_mqa: kv_a_proj_with_mqa,
    kv_a_layernorm: kv_a_layernorm,
    kv_b_proj: kv_b_proj,
    o_proj: o_proj,
    cache: cache
  )

func decompressLatent[QKNorm, RopePolicy](
    self: MLAttention[QKNorm, RopePolicy],
    latent: Tensor): (Tensor, Tensor) =
  ## Expand the gathered (1, seq, 1, kvLoraRank) latent slab into the key
  ## nope part and the values, the reference expand_kv form: one
  ## kv_b_proj pass, then the per-head split.
  let batch = latent.size(0)
  let seq = latent.size(1)
  let expanded = self.kv_b_proj.forward(latent.reshape([batch, seq, self.kvLoraRank]))
    .reshape([batch, seq, self.numHeads, self.qkNopeHeadDim + self.vHeadDim])
  let kNope = expanded.narrow(3, 0, self.qkNopeHeadDim)
  let v = expanded.narrow(3, self.qkNopeHeadDim, self.vHeadDim)
  (kNope, v)

proc latentAttention[QKNorm, RopePolicy](
    self: MLAttention[QKNorm, RopePolicy],
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Multi-head latent attention computation of the DeepSeek-V2 paper
  ## over the paged latent cache, everything through the flattened
  ## attention output, output projection NOT applied. The head-wise
  ## gated variant composes through this proc, its gate multiply sits
  ## before o_proj.
  ##
  ## Args:
  ##   ctx: InferenceContext with the active pages, the write cursor
  ##     (ctx.kv_position), and the f32 plane frequency tables inside
  ##     ctx.cos / ctx.sin (set per forward by setMlaRopeForPositions)
  ##   x: Input tensor of shape (batch, seq, hidden_size)
  ##
  ## Returns:
  ##   Tensor of shape (batch, seq, numHeads * vHeadDim)
  ##
  ## Body, in the paper vocabulary, mirroring the reference attention:
  ##   the queries run through the compressed bottleneck (c_Q) or project
  ##     directly, then split per head into q_nope and q_pe
  ##   the down-projection compresses the hidden into the latent c_KV
  ##     and the raw k_pe plane
  ##   the latent normalizes (kv_a_layernorm), q_pe and k_pe rotate
  ##     per the rope policy
  ##   both cache entries write compressed, then gather back
  ##   kv_b_proj decompresses the gathered latent into the k_nope
  ##   and value parts, the keys concatenate k_nope with the k_pe
  ##   plane broadcast per head
  ##   SDPA (causal when query and cache lengths agree) with the config
  ##     softmax scale
  let batch = x.size(0)
  checkValue(batch == 1,
    "[ttt] Paged KV attention currently supports batch_size == 1 only, got " & $batch)
  let seqLen = x.size(1)
  checkValue(seqLen == 1 or ctx.kv_position == 0,
    "[ttt] MLA mixer forward takes a multi-token pass only against an " &
    "empty cache: seqLen " & $seqLen & " with kv_position " & $ctx.kv_position &
    " would run non-causal attention over the prefix, chunked prefill " &
    "against a cached prefix is not a supported operation")
  let device = x.deviceType()
  let kpeWidth = self.cache.kpeWidth

  # RoPE tables for this pass.
  when RopePolicy isnot NoPe:
    let planeHalf = kpeWidth div 2
    doAssert ctx.cos.dim == 2 and ctx.sin.dim == 2,
      "MLA rope expects 2D (seq, plane/2) f32 frequency tables in ctx"
    doAssert ctx.cos.scalarType() == kFloat32 and ctx.sin.scalarType() == kFloat32,
      "MLA rope expects f32 frequency tables in ctx, set them with setMlaRopeForPositions"
    doAssert ctx.cos.size(0) == seqLen and ctx.sin.size(0) == seqLen,
      "MLA rope tables must cover this pass, got " & $ctx.cos.size(0) & " rows for " &
      $seqLen & " tokens"
    doAssert ctx.cos.size(1) == planeHalf,
      "MLA rope tables must be (seq, plane/2) wide, got " & $ctx.cos.size(1) &
      " columns for plane " & $kpeWidth

  # Query.
  when QKNorm isnot void:
    let qBottleneck = self.q_a_norm.forward(self.q_a_proj.forward(x))
    let q = self.q_b_proj.forward(qBottleneck)
  else:
    let q = self.q_proj.forward(x)
  let qHeads = q.reshape([batch, seqLen, self.numHeads,
    self.qkNopeHeadDim + kpeWidth])
  let qNope = qHeads.narrow(3, 0, self.qkNopeHeadDim)
  let qPe = qHeads.narrow(3, self.qkNopeHeadDim, kpeWidth)

  # Compressed latent and raw plane.
  let compressed = self.kv_a_proj_with_mqa.forward(x)
  let latentRaw = compressed.narrow(2, 0, self.kvLoraRank)
  let latent = self.kv_a_layernorm.forward(latentRaw).unsqueeze(2)
  let kPeRaw = compressed.narrow(2, self.kvLoraRank, kpeWidth).unsqueeze(2)

  # Rope policy.
  when RopePolicy is NoPe:
    let qPeRot = qPe
    let kPeRot = kPeRaw
  else:
    let (qPeRot, kPeRot) = applyRope(qPe, kPeRaw, ctx.cos, ctx.sin, RopePolicy)

  # Compressed cache round trip.
  let offset = ctx.kv_position
  self.cache.write(ctx, self.layer_idx, latent, kPeRot, offset, seqLen)
  let (latentCached, kpeCached) = self.cache.gather(ctx, self.layer_idx, offset, seqLen)

  # Decompress per pass, the reference expand_kv shape: keys concatenate
  # the nope part with the single-head plane broadcast per head.
  let (kNope, v) = self.decompressLatent(latentCached)
  let kpeExpanded = kpeCached.expand([1, kNope.size(1), self.numHeads, kpeWidth])
  let keyStates = F.cat([kNope, kpeExpanded], 3)
  let queryStates = F.cat([qNope, qPeRot], 3)

  # Attention. is_causal only when query and cache lengths agree,
  # decode (seqLen 1 over a longer cache) attends to every cached position.
  let qAttn = queryStates.permute([0, 2, 1, 3])
  let kAttn = keyStates.permute([0, 2, 1, 3])
  let vAttn = v.permute([0, 2, 1, 3])
  let doCausal = qAttn.size(2) == kAttn.size(2)
  let attnOut = F.scaled_dot_product_attention(
    qAttn, kAttn, vAttn,
    is_causal = doCausal,
    scale = some(self.softmaxScale))
    .permute([0, 2, 1, 3])

  attnOut.reshape([batch, seqLen, self.numHeads * self.vHeadDim])

proc forward[QKNorm, RopePolicy](
    self: MLAttention[QKNorm, RopePolicy],
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass for one MLA layer over the paged latent cache:
  ## the latent attention computation, then the output projection.
  self.o_proj.forward(self.latentAttention(ctx, x))

template `()`*[QKNorm, RopePolicy](layer: MLAttention[QKNorm, RopePolicy],
    ctx: var InferenceContext,
    x: Tensor): untyped =
  layer.forward(ctx, x)

type
  HeadwiseGatedMLAttention*[QKNorm, RopePolicy] = ref object
    ## Head-wise gated MLA: the attention output of every head scales by
    ## the sigmoid of a per-head gate projection BEFORE the output
    ## projection (the checkpoints' gated_attention_proj_granularity_type
    ## head_wise).
    ##   gate = sigmoid(g_proj(x).f32).to(x.dtype)   # (batch, seq, heads)
    ##   out   = o_proj((b, s, h, v) attn * gate[:, :, :, None])
    ## A distinct composed type, never a runtime flag: it wraps one full
    ## MLAttention instance plus the gate projection, and the gate
    ## multiply sits at the latent attention seam before the output
    ## projection. Element-wise granularity would be its own composed
    ## type, not built.
    attn: MLAttention[QKNorm, RopePolicy]
    g_proj: Linear            ## (numHeads, hidden), bias-free

func init*[QKNorm, RopePolicy](
    _: type HeadwiseGatedMLAttention[QKNorm, RopePolicy],
    layer_idx: int,
    name: string,
    q_a_proj, q_b_proj: Linear,
    q_a_norm: QKNorm,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    g_proj: Linear,
    numHeads, qkNopeHeadDim, kvLoraRank, vHeadDim: int,
    softmaxScale: float64,
    cache: MlaLatentCache): HeadwiseGatedMLAttention[QKNorm, RopePolicy] =
  ## Compressed-Q gated layer (q_lora_rank not None): the inner
  ## MLAttention carries the body width checks, the gate projection
  ## must carry exactly one row per head and feed from the residual
  ## width.
  checkValue(g_proj.out_features == numHeads,
    "[ttt] " & name & ": g_proj out_features is " & $g_proj.out_features &
    ", expected the head count " & $numHeads & " (head-wise granularity)")
  checkValue(g_proj.in_features == o_proj.out_features,
    "[ttt] " & name & ": g_proj in_features is " & $g_proj.in_features &
    ", expected the residual width " & $o_proj.out_features)
  HeadwiseGatedMLAttention[QKNorm, RopePolicy](
    attn: MLAttention[QKNorm, RopePolicy].init(
      layer_idx, name,
      q_a_proj, q_b_proj, q_a_norm,
      kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj,
      numHeads, qkNopeHeadDim, kvLoraRank, vHeadDim,
      softmaxScale, cache),
    g_proj: g_proj)

proc forward[QKNorm, RopePolicy](
    self: HeadwiseGatedMLAttention[QKNorm, RopePolicy],
    ctx: var InferenceContext,
    x: Tensor): Tensor =
  ## Forward pass for one gated MLA layer: latent attention, head-wise
  ## sigmoid gate, then output projection.
  let attnFlat = self.attn.latentAttention(ctx, x)
  let batch = attnFlat.size(0)
  let seqLen = attnFlat.size(1)
  let gate = F.sigmoid(self.g_proj.forward(x).to(kFloat32))
    .to(attnFlat.scalarType())
  let gated = attnFlat.reshape([batch, seqLen, self.attn.numHeads,
      self.attn.vHeadDim]) * gate.unsqueeze(-1)
  self.attn.o_proj.forward(gated.reshape(
    [batch, seqLen, self.attn.numHeads * self.attn.vHeadDim]))

template `()`*[QKNorm, RopePolicy](
    layer: HeadwiseGatedMLAttention[QKNorm, RopePolicy],
    ctx: var InferenceContext,
    x: Tensor): untyped =
  layer.forward(ctx, x)
