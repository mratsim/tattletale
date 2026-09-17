# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Model-free analytic suite for the MLA rope policies, frequency tables
## and the typed latent cache over the paged pool, on seeded synthetic
## stimulus with closed-form references.
##
## Checks:
## - NoPe rope is the identity, FullRoPe and PartialRoPe preserve pair norms
## - plain frequency tables match the closed form
## - the yarn blend stays inside its bounds and zones
## - setMlaRopeForPositions gathers the exact table rows
## - cache round trips across pages, layers and decode offsets
## - the LPM skip, the plane-less form and the runtime width guards
##
## Run:
##   nim cpp -d:release --stackTrace:on --debugger:native --passC:"-std=c++20" --verbosity:0 \
##     --hints:off --warnings:off --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_unit_mla.nim

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/layers/rope,
  workspace/transformers/src/layers/attn_ssm/multi_head_latent_attention,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/orchestrator {.all.}

from workspace/libtorch/src/raw_libtorch import manual_seed

type RopeGeom = object
  ## Rope stimulus geometry, main binds every value and each helper reads
  ## the dimensions through the parameter.
  batch, seq, heads, plane: int
  theta: float64

type TableGeom = object
  ## Frequency-table geometry, plane width even, theta the base.
  plane, seqLen: int
  theta: float64

type CacheGeom = object
  ## Paged-cache geometry of the plane-ful checks.
  kvLoraRank, plane, maxSeq, numPages, cacheLayers: int

template orRaise(cond: bool; msg: string) =
  ## Suite enforcement form, cond false raises ValueError carrying msg.
  if not cond:
    raise newException(ValueError, msg)

template mustRefuse(why: string; body: untyped) =
  ## Each guarded call must reject its input, the body raises ValueError
  ## and any silent outcome raises with why.
  var fired = false
  try:
    body
  except ValueError:
    fired = true
  if not fired:
    raise newException(ValueError, "no rejection: " & why)

func bf16Opts(): F.TensorOptions =
  F.tensorOptions(F.kBFloat16, F.kCPU)

func f32Opts(): F.TensorOptions =
  F.tensorOptions(F.kFloat32, F.kCPU)

func i64Opts(): F.TensorOptions =
  F.tensorOptions(F.kInt64, F.kCPU)

func maxDrift(a, b: F.Tensor): float64 =
  ## Largest absolute element difference of two tensors compared as f32,
  ## the drift quantity of every value comparison in this suite.
  let diff = (a.to(F.kFloat32) - b.to(F.kFloat32)).abs()
  result = diff.max().item(float32).float64

func withinBand(actual, expected: F.Tensor, rtol, abstol: float64): float32 =
  ## Largest tolerance violation of the band |actual - expected| <=
  ## abstol + rtol * |expected|, a non-positive result means the band
  ## holds over every element.
  let a = actual.to(F.kFloat32)
  let e = expected.to(F.kFloat32)
  let diff = (a - e).abs()
  let tol = e.abs() * rtol + abstol
  result = (diff - tol).max().item(float32)

func pairNorm(x: F.Tensor): F.Tensor =
  ## Per-channel-pair magnitude over the interleaved plane pairs, shape
  ## (batch, seq, heads, plane div 2).
  let half = x.size(3) div 2
  result = x.reshape([x.size(0), x.size(1), x.size(2), half, 2])
    .pow(2.0).sum(4).sqrt()

func plainAngle(plane, i: int, theta: float32): float32 =
  ## Closed-form plain angle per dimension, theta^(-2i/plane) radians for one position step.
  pow(theta, -(2.0'f32 * i.float32 / plane.float32))

proc randnRope(geom: RopeGeom, seed: uint64): F.Tensor =
  ## Seeded stimulus of the rope geometry, one table per call.
  Torch.manual_seed(seed)
  F.randn(geom.batch, geom.seq, geom.heads, geom.plane, f32Opts())

proc noPeIdentity(geom: RopeGeom, seed: uint64) =
  ## NoPe rope returns the input values unchanged, no rotation is computed
  ## on either side.
  let q = randnRope(geom, seed)
  let k = randnRope(geom, seed + 1'u64)
  let cos = F.zeros(geom.seq, geom.plane div 2, f32Opts())
  let sin = F.zeros(geom.seq, geom.plane div 2, f32Opts())
  let (qRot, kRot) = applyRope(q, k, cos, sin, NoPe)
  orRaise(maxDrift(qRot, q) == 0.0, "NoPe rope output vs input drift on q")
  orRaise(maxDrift(kRot, k) == 0.0, "NoPe rope output vs input drift on k")

proc fullRoPePairNorm(geom: RopeGeom, seed: uint64) =
  ## FullRoPe interleaved rotation preserves the per-channel-pair plane
  ## norm and actually rotates, positions past 0 differ from the input.
  let rotary = MlaRotary.new(geom.plane, geom.seq, geom.theta, F.kCPU)
  let q = randnRope(geom, seed)
  let k = randnRope(geom, seed + 1'u64)
  let (qRot, kRot) = applyRope(q, k, rotary.cosCache, rotary.sinCache, FullRoPe)
  orRaise(qRot.size(0) == geom.batch and qRot.size(1) == geom.seq and
      qRot.size(2) == geom.heads and qRot.size(3) == geom.plane,
    "FullRoPe output shape (" & $qRot.size(0) & ", " & $qRot.size(1) & ", " &
      $qRot.size(2) & ", " & $qRot.size(3) & ")")

  let violQ = withinBand(pairNorm(qRot), pairNorm(q), 1e-5, 1e-6)
  orRaise(violQ <= 0.0'f32,
    "FullRoPe q pair norm violation past the band " & $violQ)
  let violK = withinBand(pairNorm(kRot), pairNorm(k), 1e-5, 1e-6)
  orRaise(violK <= 0.0'f32,
    "FullRoPe k pair norm violation past the band " & $violK)

  let qTail = qRot.narrow(1, 1, geom.seq - 1)
  let qTailIn = q.narrow(1, 1, geom.seq - 1)
  let drift = maxDrift(qTail, qTailIn)
  orRaise(drift > 1e-6,
    "FullRoPe left positions 1.. unchanged, drift " & $drift)

proc partialRoPeSplit(geom: RopeGeom, seed: uint64) =
  ## PartialRoPe[32] over the 64-wide plane rotates the first 32 channels
  ## and passes channels 32..63 through unchanged, the rotated half keeps
  ## its pair norm over the frequency tables sliced to that half.
  ##
  ## The odd rotary width refusal stays a compile-time contract.
  let rotary = MlaRotary.new(geom.plane, geom.seq, geom.theta, F.kCPU)
  let q = randnRope(geom, seed)
  let (qRot, kRot) = applyRope(q, q, rotary.cosCache, rotary.sinCache,
    PartialRoPe[32])
  let passIn = q.narrow(3, 32, geom.plane - 32)
  orRaise(maxDrift(qRot.narrow(3, 32, geom.plane - 32), passIn) == 0.0,
    "PartialRoPe q tail channels moved")
  orRaise(maxDrift(kRot.narrow(3, 32, geom.plane - 32), passIn) == 0.0,
    "PartialRoPe k tail channels moved")
  let rotQ = qRot.narrow(3, 0, 32)
  let drift = maxDrift(rotQ, q.narrow(3, 0, 32))
  orRaise(drift > 1e-6, "PartialRoPe left the q rotated half unchanged")

  let viol = withinBand(pairNorm(rotQ), pairNorm(q.narrow(3, 0, 32)), 1e-5, 1e-6)
  orRaise(viol <= 0.0'f32,
    "PartialRoPe rotated-half pair norm violation past the band " & $viol)

proc plainTables(geom: TableGeom) =
  ## Plain frequency tables carry their shape and value contract.
  ##
  ## Contract:
  ## - both tables rank 2 (maxSeqLen, plane div 2) at f32
  ## - position 0 reads cos 1 and sin 0 exactly
  ## - position 1 matches the closed-form angles within 2e-6
  ## - cos decreases per dimension, strict inside the f32 resolution
  let rotary = MlaRotary.new(geom.plane, geom.seqLen, geom.theta, F.kCPU)
  let half = geom.plane div 2
  orRaise(rotary.planeWidth == geom.plane and rotary.maxSeqLen == geom.seqLen,
    "MlaRotary geometry plane " & $rotary.planeWidth & " seq " &
      $rotary.maxSeqLen)
  orRaise(rotary.cosCache.dim == 2 and rotary.cosCache.size(0) == geom.seqLen and
      rotary.cosCache.size(1) == half,
    "cos table shape (" & $rotary.cosCache.size(0) & ", " &
      $rotary.cosCache.size(1) & ")")
  orRaise(rotary.cosCache.scalarType() == F.kFloat32, "cos table dtype")
  orRaise(rotary.sinCache.scalarType() == F.kFloat32, "sin table dtype")

  let cosTable = rotary.cosCache
  let sinTable = rotary.sinCache
  for i in 0 ..< half:
    orRaise(cosTable[0, i].item(float32) == 1.0'f32,
      "position 0 cos value at dimension " & $i)
    orRaise(sinTable[0, i].item(float32) == 0.0'f32,
      "position 0 sin value at dimension " & $i)

  for i in 0 ..< half:
    let angle = plainAngle(geom.plane, i, geom.theta.float32)
    orRaise(abs(cosTable[1, i].item(float32) - cos(angle)) <= 2e-6,
      "position 1 cos vs closed form at dimension " & $i)
    orRaise(abs(sinTable[1, i].item(float32) - sin(angle)) <= 2e-6,
      "position 1 sin vs closed form at dimension " & $i)

  for i in 1 ..< half:
    let cosCur = cosTable[1, i].item(float32)
    let cosPrev = cosTable[1, i - 1].item(float32)
    orRaise(cosCur >= cosPrev,
      "cos frequency ordering broke at dimension " & $i)
    if i <= 15:
      orRaise(cosCur > cosPrev,
        "cos strict monotonicity broke at dimension " & $i)

proc yarnBlend(geom: TableGeom, yarnFactor, betaFast, betaSlow: float64,
    originalMaxPos, plainZoneEnd, interpZoneStart: int) =
  ## Yarn tables blend interpolation and extrapolation per dimension.
  ##
  ## Zones:
  ## - the yarn cosine stays inside the plain-to-scaled bounds
  ## - dimensions below the ramp floor keep the plain value
  ## - dimensions past the ramp ceil match the scaled value
  ## - plain construction ignores yarn parameters at factor <= 1
  let yarn = MlaRotary.new(geom.plane, geom.seqLen, geom.theta, F.kCPU,
    yarnFactor = yarnFactor, yarnBetaFast = betaFast,
    yarnBetaSlow = betaSlow, yarnOriginalMaxPos = originalMaxPos)
  let plain = MlaRotary.new(geom.plane, geom.seqLen, geom.theta, F.kCPU)

  let yarnCos = yarn.cosCache
  let plainCos = plain.cosCache
  let half = geom.plane div 2
  for i in 0 ..< half:
    let invFreq = plainAngle(geom.plane, i, geom.theta.float32)
    let cosLow = cos(invFreq / yarnFactor.float32)
    let cosYarn = yarnCos[1, i].item(float32)
    let cosHigh = plainCos[1, i].item(float32)
    # Cosine decreases on [0, 1] rad, so the yarn angle inside
    # [plain / factor, plain] keeps cosYarn in [cosHigh, cosLow].
    orRaise(cosYarn >= cosHigh - 2e-6,
      "yarn cos below the plain bound at dimension " & $i)
    orRaise(cosYarn <= cosLow + 2e-6,
      "yarn cos above the scaled bound at dimension " & $i)
    if i < plainZoneEnd:
      orRaise(abs(cosYarn - cosHigh) <= 2e-6,
        "extrapolation zone left the plain value at dimension " & $i)
    if i >= interpZoneStart:
      orRaise(abs(cosYarn - cosLow) <= 2e-6,
        "interpolation zone left the scaled value at dimension " & $i)

  let plainAgain = MlaRotary.new(geom.plane, geom.seqLen, geom.theta, F.kCPU,
    yarnFactor = 1.0)
  orRaise(maxDrift(plainAgain.cosCache, plain.cosCache) == 0.0,
    "factor <= 1 construction moved the plain frequency table")

proc setMlaRopeGather(geom: TableGeom, seqCount, ctxMaxSeq, ctxHeadDim: int) =
  ## setMlaRopeForPositions gathers the exact table rows for the active
  ## position_ids, the 1-D and the 2-D form.
  let rotary = MlaRotary.new(geom.plane, geom.seqLen, geom.theta, F.kCPU)
  var ctx = InferenceContext.init(1, 1, 1, ctxMaxSeq, ctxHeadDim)
  ctx.position_ids = F.arange(0, seqCount, i64Opts())
  ctx.setMlaRopeForPositions(rotary)
  orRaise(ctx.cos.dim == 2 and ctx.cos.size(0) == seqCount and
      ctx.cos.size(1) == geom.plane div 2,
    "gathered cos table shape (" & $ctx.cos.size(0) & ", " &
      $ctx.cos.size(1) & ")")
  let gathered = ctx.cos
  let source = rotary.cosCache
  for i in 0 ..< seqCount:
    orRaise(maxDrift(gathered[i, _], source[i, _]) == 0.0,
      "cos row gather moved row " & $i)

  ctx.position_ids = ctx.position_ids.unsqueeze(0)
  ctx.setMlaRopeForPositions(rotary)
  orRaise(ctx.cos.size(0) == seqCount, "2-D position_ids gather row count")
  orRaise(maxDrift(ctx.cos[2, _], source[2, _]) == 0.0,
    "2-D position_ids cos row gather moved row 2")

proc makePagedCtx(geom: CacheGeom,
    tokenCount: int): tuple[orc: Orchestrator, ctx: InferenceContext] =
  ## Fresh orchestrator over the MLA split geometry and tokenCount prompt
  ## tokens with a zero-initialized pool, untouched slots reading zeros.
  var ids: seq[uint32]
  for i in 1 .. tokenCount:
    ids.add(uint32(i))
  result.orc = Orchestrator.init(
    num_layers = geom.cacheLayers,
    batch_size = 1,
    k_kv_heads = 1,
    k_head_dim = geom.kvLoraRank,
    v_kv_heads = 1,
    v_head_dim = geom.plane,
    max_seq = geom.maxSeq,
    num_pages = geom.numPages,
    dtype = kBFloat16,
    device = kCPU)
  result.orc.startSequence(ids)
  result.ctx = result.orc.getInferenceContextMut()

proc cachePlaneRoundTrip(geom: CacheGeom, tokenCount: int, seed: uint64) =
  ## Plane-ful cache write and gather across a page boundary and per layer,
  ## both buffers keep their own width and the layer slabs stay distinct.
  Torch.manual_seed(seed)
  var (orc, ctx) = makePagedCtx(geom, tokenCount)
  let cache = MlaLatentCache.init(geom.kvLoraRank, geom.plane, geom.maxSeq,
    kBFloat16, kCPU)
  let latent0 = F.randn(1, tokenCount, 1, geom.kvLoraRank, bf16Opts())
  let kpe0 = F.randn(1, tokenCount, 1, geom.plane, bf16Opts())
  let latent1 = F.randn(1, tokenCount, 1, geom.kvLoraRank, bf16Opts())
  let kpe1 = F.randn(1, tokenCount, 1, geom.plane, bf16Opts())

  cache.write(ctx, 0, latent0, kpe0, 0, tokenCount)
  let (gotLatent0, gotKpe0) = cache.gather(ctx, 0, 0, tokenCount)
  orRaise(gotLatent0.size(0) == 1 and gotLatent0.size(1) == tokenCount and
      gotLatent0.size(2) == 1 and gotLatent0.size(3) == geom.kvLoraRank,
    "latent gather shape (" & $gotLatent0.size(0) & ", " &
      $gotLatent0.size(1) & ", " & $gotLatent0.size(2) & ", " &
      $gotLatent0.size(3) & ")")
  orRaise(gotKpe0.size(3) == geom.plane,
    "kpe gather width " & $gotKpe0.size(3))
  orRaise(maxDrift(gotLatent0, latent0) == 0.0,
    "latent gather layer 0 moved values")
  orRaise(maxDrift(gotKpe0, kpe0) == 0.0,
    "kpe gather layer 0 moved values")

  cache.write(ctx, 1, latent1, kpe1, 0, tokenCount)
  let (gotLatent1, gotKpe1) = cache.gather(ctx, 1, 0, tokenCount)
  orRaise(maxDrift(gotLatent1, latent1) == 0.0,
    "latent gather layer 1 moved values")
  orRaise(maxDrift(gotKpe1, kpe1) == 0.0,
    "kpe gather layer 1 moved values")

  # Layer slabs stay distinct, layer 0 reads back its own rows.
  let (lat0, _) = cache.gather(ctx, 0, 0, tokenCount)
  orRaise(maxDrift(lat0, latent0) == 0.0,
    "layer 0 slab changed under the layer 1 writes")

proc cacheDecodeOffset(geom: CacheGeom, prefillLen: int, seed: uint64) =
  ## Decode-shaped write of one token at offset prefillLen lands at global
  ## position prefillLen, the gather returns the full prefill and decode
  ## continuity in one slab.
  Torch.manual_seed(seed)
  var (orc, ctx) = makePagedCtx(geom, prefillLen + 1)
  let cache = MlaLatentCache.init(geom.kvLoraRank, geom.plane, geom.maxSeq,
    kBFloat16, kCPU)
  let prefill = F.randn(1, prefillLen, 1, geom.kvLoraRank, bf16Opts())
  let prefillKpe = F.randn(1, prefillLen, 1, geom.plane, bf16Opts())
  let decodeLatent = F.randn(1, 1, 1, geom.kvLoraRank, bf16Opts())
  let decodeKpe = F.randn(1, 1, 1, geom.plane, bf16Opts())

  cache.write(ctx, 0, prefill, prefillKpe, 0, prefillLen)
  cache.write(ctx, 0, decodeLatent, decodeKpe, prefillLen, 1)

  let (gotLatent, gotKpe) = cache.gather(ctx, 0, 0, prefillLen + 1)
  orRaise(gotLatent.size(1) == prefillLen + 1,
    "gathered continuity length " & $gotLatent.size(1))
  orRaise(maxDrift(gotLatent.narrow(1, 0, prefillLen), prefill) == 0.0,
    "prefill rows changed under the decode write")
  orRaise(maxDrift(gotLatent.narrow(1, prefillLen, 1), decodeLatent) == 0.0,
    "decode latent landed off global position " & $prefillLen)
  orRaise(maxDrift(gotKpe.narrow(1, prefillLen, 1), decodeKpe) == 0.0,
    "decode kpe landed off global position " & $prefillLen)

proc cacheLpmSkip(geom: CacheGeom, passLen, cachedCount: int, seed: uint64) =
  ## cached_tokens from LPM skips the already-cached prefix of the pass,
  ## the write starts at cached_tokens - offset and the fresh pool leaves
  ## the skipped slots zeroed.
  Torch.manual_seed(seed)
  var (orc, ctx) = makePagedCtx(geom, passLen)
  let cache = MlaLatentCache.init(geom.kvLoraRank, geom.plane, geom.maxSeq,
    kBFloat16, kCPU)
  ctx.cached_tokens = cachedCount

  let pass = F.randn(1, passLen, 1, geom.kvLoraRank, bf16Opts())
  let passKpe = F.randn(1, passLen, 1, geom.plane, bf16Opts())
  cache.write(ctx, 0, pass, passKpe, 0, passLen)

  let (gotLatent, gotKpe) = cache.gather(ctx, 0, 0, passLen)
  let zeros = F.zeros(1, cachedCount, 1, geom.kvLoraRank, bf16Opts())
  orRaise(maxDrift(gotLatent.narrow(1, 0, cachedCount), zeros) == 0.0,
    "cached prefix slots carry written values")
  orRaise(
    maxDrift(gotLatent.narrow(1, cachedCount, passLen - cachedCount),
      pass.narrow(1, cachedCount, passLen - cachedCount)) == 0.0,
    "uncached latent tail moved")
  orRaise(
    maxDrift(gotKpe.narrow(1, cachedCount, passLen - cachedCount),
      passKpe.narrow(1, cachedCount, passLen - cachedCount)) == 0.0,
    "uncached kpe tail moved")

proc cachePlaneless(maxSeq: int, seed: uint64) =
  ## Plane-less cache (qk_rope_head_dim 0), writeLatent and gatherLatent
  ## round trip unchanged, no kpe plane exists.
  Torch.manual_seed(seed)
  var orc = Orchestrator.init(
    num_layers = 1, batch_size = 1, k_kv_heads = 1, k_head_dim = 16,
    v_kv_heads = 1, v_head_dim = 16, max_seq = maxSeq,
    num_pages = 2, dtype = kBFloat16, device = kCPU)
  orc.startSequence(@[1'u32, 2, 3])
  var ctx = orc.getInferenceContextMut()
  let cache = MlaLatentCache.init(16, 0, maxSeq, kBFloat16, kCPU)
  let latent = F.randn(1, 3, 1, 16, bf16Opts())
  cache.writeLatent(ctx, 0, latent, 0, 3)
  let got = cache.gatherLatent(ctx, 0, 0, 3)
  orRaise(got.size(0) == 1 and got.size(1) == 3 and got.size(2) == 1 and
      got.size(3) == 16,
    "plane-less latent gather shape (" & $got.size(0) & ", " &
      $got.size(1) & ", " & $got.size(2) & ", " & $got.size(3) & ")")
  orRaise(maxDrift(got, latent) == 0.0,
    "plane-less latent round trip moved values")

proc cacheRuntimeRules(kvLoraRank, planeFul, maxSeq: int, seed: uint64) =
  ## Kpe-plane presence guards run against the runtime width.
  ##
  ## Refusals:
  ## - a plane-ful cache refuses the latent-only mixer forms
  ## - a plane-less cache refuses the kpe plane forms
  ## - odd or negative widths fail at init
  Torch.manual_seed(seed)
  let cache0 = MlaLatentCache.init(kvLoraRank, 0, maxSeq, kBFloat16, kCPU)
  let cache64 = MlaLatentCache.init(kvLoraRank, planeFul, maxSeq,
    kBFloat16, kCPU)
  var orc = Orchestrator.init(
    num_layers = 2, batch_size = 1, k_kv_heads = 1, k_head_dim = 16,
    v_kv_heads = 1, v_head_dim = planeFul, max_seq = maxSeq, num_pages = 2,
    dtype = kBFloat16, device = kCPU)
  orc.startSequence(@[1'u32])
  var ctx = orc.getInferenceContextMut()
  let latent = F.zeros(1, 1, 1, kvLoraRank, bf16Opts())
  let kpe = F.zeros(1, 1, 1, planeFul, bf16Opts())

  # Positive controls, each cache answers with its own plane width.
  cache64.write(ctx, 0, latent, kpe, 0, 1)
  let got = cache64.gather(ctx, 0, 0, 1)
  orRaise(got[0].size(3) == kvLoraRank and got[1].size(3) == planeFul,
    "plane-ful cache gather widths (" & $got[0].size(3) & ", " &
      $got[1].size(3) & ")")
  cache0.writeLatent(ctx, 0, latent, 0, 1)
  let gotLatent = cache0.gatherLatent(ctx, 0, 0, 1)
  orRaise(gotLatent.size(3) == kvLoraRank,
    "plane-less cache gather width " & $gotLatent.size(3))

  # Cross-form rejections raise on the runtime width guard.
  mustRefuse("plane-less cache write with a kpe plane accepted"):
    cache0.write(ctx, 0, latent, kpe, 0, 1)
  mustRefuse("plane-less cache kpe gather accepted"):
    let sink = cache0.gather(ctx, 0, 0, 1)
  mustRefuse("plane-ful cache latent-only write accepted"):
    cache64.writeLatent(ctx, 0, latent, 0, 1)
  mustRefuse("plane-ful cache latent-only gather accepted"):
    let sink = cache64.gatherLatent(ctx, 0, 0, 1)

  # Width rules at init.
  mustRefuse("odd kpe width accepted at init"):
    let sink = MlaLatentCache.init(kvLoraRank, 1, maxSeq, kBFloat16, kCPU)
  mustRefuse("negative kpe width accepted at init"):
    let sink = MlaLatentCache.init(kvLoraRank, -2, maxSeq, kBFloat16, kCPU)

proc main() =
  let ropeGeom = RopeGeom(batch: 1, seq: 4, heads: 2, plane: 64,
    theta: 10000.0)
  let plainGeom = TableGeom(plane: 64, seqLen: 8, theta: 10000.0)
  let gatherGeom = TableGeom(plane: 64, seqLen: 128, theta: 10000.0)
  let cacheGeom = CacheGeom(kvLoraRank: 512, plane: 64, maxSeq: 4096,
    numPages: 4, cacheLayers: 2)

  noPeIdentity(ropeGeom, seed = 7'u64)
  fullRoPePairNorm(ropeGeom, seed = 9'u64)
  partialRoPeSplit(ropeGeom, seed = 11'u64)
  plainTables(plainGeom)
  # Yarn zones carry the DeepSeek-V2-Lite blend values, factor 40, betas
  # 32 and 1, original positions 4096, ramp floor 10 and ramp ceil 23.
  yarnBlend(plainGeom, yarnFactor = 40.0, betaFast = 32.0, betaSlow = 1.0,
    originalMaxPos = 4096, plainZoneEnd = 10, interpZoneStart = 23)
  setMlaRopeGather(gatherGeom, seqCount = 3, ctxMaxSeq = 128, ctxHeadDim = 64)

  cachePlaneRoundTrip(cacheGeom, tokenCount = 300, seed = 31'u64)
  cacheDecodeOffset(cacheGeom, prefillLen = 3, seed = 33'u64)
  cacheLpmSkip(cacheGeom, passLen = 4, cachedCount = 2, seed = 35'u64)
  cachePlaneless(maxSeq = 4096, seed = 37'u64)
  cacheRuntimeRules(kvLoraRank = 16, planeFul = 64, maxSeq = 128,
    seed = 39'u64)

  # Out of scope for this suite, the MLAttention init rules and forward
  # passes. The projections are checkpoint layers, only the deserialization
  # loaders build those over real weights, synthetic stimulus carries none.

when isMainModule:
  main()
