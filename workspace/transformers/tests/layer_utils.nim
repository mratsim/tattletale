# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Standardized helpers for the fixture-replay suites.
##
## One home for everything a suite repeats, split by consumer:
##
## - config document and weights shard of one model directory
## - rotary table geometry read off the config text section
## - the layer loads, composed over src/deserialization.nim
##
## - the chain's plain input-tensor preparation of the 04 suites
## - per-layer fixture opens and checkpoint tensor counts of the 01/03 suites
## - deterministic bf16-grid stimulus tensors and tensor comparisons

import
  std/importutils,
  std/math,
  std/os,
  std/strutils,
  std/tables,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/safetensors/src/safetensors {.all.},
  workspace/safetensors/src/safetensors_libtorch,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/layers/rope,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/src/layers/attn_ssm/multi_head_latent_attention,
  workspace/transformers/src/quantizations/datatypes

privateAccess(SafetensorObj)

proc setup*(cfg: JsonNode, T: typedesc[RotaryPositionEmbedding],
    maxSeqLen: int, dtype = F.kBFloat16,
    device = F.kCPU): RotaryPositionEmbedding =
  ## Builds the rotary table of one checkpoint over the first `maxSeqLen` positions, geometry read off the config text section.
  ##
  ## Contract:
  ## - head_dim and rope_parameters.rope_theta name the table shape
  ## - a partial rotary factor scales the rotating width down from head_dim
  ## - the factor absent means full rotation
  let t = if cfg.hasKey("text_config"): cfg{"text_config"} else: cfg
  let headDim = t{"head_dim"}.getInt()
  let thetaSrc = if cfg.hasKey("rope_parameters"): t{"rope_parameters"} else: t
  let theta = thetaSrc{"rope_theta"}.getFloat(1e6)
  let factor = thetaSrc{"partial_rotary_factor"}.getFloat(1.0)
  let rotaryDim =
    if factor > 0.0 and factor < 1.0:
      round(headDim.float64 * factor).int
    else:
      headDim
  RotaryPositionEmbedding.new(headDim, maxSeqLen, theta, dtype, device,
    rotary_dim = rotaryDim)

proc setup*(cfg: JsonNode,
    T: typedesc[RopeElementWiseGatedAttention[RmsNormOne]],
    weights: SafetensorsCollection, layerStem: string, layerIdx: int,
    rotary: RotaryPositionEmbedding,
    device = F.kCPU): RopeElementWiseGatedAttention[RmsNormOne] =
  ## Loads checkpoint layer `layerIdx` as a RopeElementWiseGatedAttention
  ## over RmsNormOne q/k norms, geometry read off the config text section.
  ##
  ## Expected input:
  ## - layerStem, the layer path up to and including the dot separator,
  ##   shaped "model.language_model.layers." - rotary, the table the KV
  ##   context registers (the layer borrows it)
  let t = if cfg.hasKey("text_config"): cfg{"text_config"} else: cfg
  let prefix = layerStem & $layerIdx & ".self_attn"
  RopeElementWiseGatedAttention[RmsNormOne].load(weights, cfg, prefix,
    layerIdx,
    t{"num_attention_heads"}.getInt(), t{"num_key_value_heads"}.getInt(),
    t{"head_dim"}.getInt(), rotary, device)

proc setupGatedDeltaNet*[Decay: static DecayAxis,
    GateIn: FullRankGateIn | LowRankGateIn,
    Form: static GateForm](cfg: JsonNode,
    T: typedesc[GatedDeltaNet[Decay, GateIn, Form]],
    weights: SafetensorsCollection, layerStem: string, layerIdx: int,
    device = F.kCPU): GatedDeltaNet[Decay, GateIn, Form] =
  ## Loads checkpoint layer `layerIdx` as a GatedDeltaNet, recurrent
  ## geometry read off the config text section.
  ##
  ## Expected input:
  ## - layerStem, the dot-terminated layer path stem, shaped
  ##   model.language_model.layers. (weights load under linear_attn)
  ## - T, the concrete mixer variant, e.g.
  ##   GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus]
  ##   for the Qwen3.5-family GDN layers
  let t = if cfg.hasKey("text_config"): cfg{"text_config"} else: cfg
  let prefix = layerStem & $layerIdx & ".linear_attn"
  GatedDeltaNet[Decay, GateIn, Form].load(weights, cfg, prefix, layerIdx,
    t{"linear_num_key_heads"}.getInt(), t{"linear_num_value_heads"}.getInt(),
    t{"linear_key_head_dim"}.getInt(), t{"linear_value_head_dim"}.getInt(),
    t{"linear_conv_kernel_dim"}.getInt(), device)

func nextStepRow*(logits: F.Tensor, position: int): F.Tensor =
  ## Returns the [vocab] logit row at `position` of the sequence axis.
  ##
  ## Expected input:
  ## - logits, the [batch, seq, vocab] tensor of one forward call
  ## - the prefill pass hands prompt_len - 1, the last prompt position
  ## - a decode step hands 0, its logits carry the single generated position
  logits.narrow(1, position, 1).squeeze(1).squeeze(0)

proc setupLayerFixture*(fixtureDir: string, layerIdx: int): Table[string, F.Tensor] =
  ## Owned cpu tensor table of one per-layer `.safetensor` fixture, keyed by recorded tensor name.
  let fixturePath = fixtureDir / "layer-" & ($layerIdx).align(2, '0') & ".safetensor"
  var st = Safetensor.open(fixturePath)
  result = initTable[string, F.Tensor]()
  for name in st.tensors.keys():
    result[name] = st.getTensorOwned(name, F.kCPU)

proc setupLayerFixtureReader*(fixtureDir: string, layerIdx: int): Safetensor =
  ## Opens one per-layer `.safetensor` fixture reader.
  ## Contract, the returned reader owns its memory mapping, released after
  ## the last reference to it goes away.
  Safetensor.open(fixtureDir / "layer-" & ($layerIdx).align(2, '0') & ".safetensor")

func setupMatrixTensor*(rows: seq[seq[float64]]): F.Tensor =
  ## [rows, cols] fp32 tensor from equal-length numeric rows.
  var flat = newSeq[float32]()
  var cols = 0
  for row in rows:
    cols = row.len
    for cell in row:
      flat.add cell.float32
  F.toTensor(flat).reshape(rows.len, cols)

func setupIndicesTensor*(rows: seq[seq[int64]]): F.Tensor =
  ## [rows, cols] int64 tensor of expert ids.
  var flat = newSeq[int64]()
  var cols = 0
  for row in rows:
    cols = row.len
    flat.add row
  F.toTensor(flat).reshape(rows.len, cols)

proc ulpBf16*(m: float32): float32 {.inline.} =
  ## One bf16 ulp at magnitude m (7 significand bits), zero maps to zero.
  if m <= 0.0'f32:
    return 0.0'f32
  result = pow(2.0'f32, floor(log2(m)) - 7.0'f32)

# ── MLA (DeepSeek-style latent attention) ──────────────────────────────────

func mlaInterleaveLayout*(x: F.Tensor): F.Tensor =
  ## Recorded rotation output layout, the cat of the even and odd pair
  ## values over the last dimension, the interleaved pair layout.
  ##
  ## Expected input:
  ## - x, the (batch, seq, heads, plane) tensor whose plane channels
  ##   sit in the half-split layout, pair i at channels (i, plane/2 + i)
  ##
  ## Output:
  ## - the same shape with pair i at channels (2i, 2i + 1), pure data
  ##   movement over the pair values
  let d = x.size(3)
  let even = x.narrow(3, 0, d div 2)
  let odd = x.narrow(3, d div 2, d div 2)
  F.cat([even.unsqueeze(4), odd.unsqueeze(4)], 4).reshape(
    x.size(0), x.size(1), x.size(2), d)

proc setupMlaDirect*[Pe](modelDir, prefix: string, layerIdx, maxSeq: int,
    device = F.kCPU): MLAttention[void, Pe] =
  ## Direct-Q MLAttention load of checkpoint layer `layerIdx`
  ## over the typed latent cache, wiring mirrored from the model files:
  ## - geometry off the flat config.json section
  ## - latent norm at the bottleneck eps
  ## - softmax scale 1/sqrt(qk_head_dim)
  ##
  ## Expected input:
  ## - modelDir, the checkpoint directory (config.json plus weight shards)
  ## - prefix, the dot-terminated attention path, shaped "model.layers.N.self_attn."
  ## - Pe, the rope policy the caller locks (FullRoPe or NoPe)
  let cfgJson = packedjson.parseFile(modelDir / "config.json")
  let view = SafetensorsCollection.open(modelDir)
  let cache = MlaLatentCache.init(
    cfgJson{"kv_lora_rank"}.getInt(), cfgJson{"qk_rope_head_dim"}.getInt(),
    maxSeq, kBFloat16, device)
  MLAttention[void, Pe].init(
    layerIdx, prefix,
    Linear.load(view, cfgJson, prefix & ".q_proj", device),
    Linear.load(view, cfgJson, prefix & ".o_proj", device),
    Linear.load(view, cfgJson, prefix & ".kv_a_proj_with_mqa", device),
    RmsNorm.init(
      view.getTensorOwned(prefix & ".kv_a_layernorm.weight", device),
      qBF16, eps = BottleneckNormEps),
    Linear.load(view, cfgJson, prefix & ".kv_b_proj", device),
    numHeads = cfgJson{"num_attention_heads"}.getInt(),
    qkNopeHeadDim = cfgJson{"qk_nope_head_dim"}.getInt(),
    kvLoraRank = cfgJson{"kv_lora_rank"}.getInt(),
    vHeadDim = cfgJson{"v_head_dim"}.getInt(),
    softmaxScale = mlaSoftmaxScale(cfgJson{"qk_nope_head_dim"}.getInt(),
      cfgJson{"qk_rope_head_dim"}.getInt()),
    cache = cache)

proc setupMlaCompressed*(modelDir, prefix: string, layerIdx, maxSeq: int,
    device = F.kCPU): MLAttention[RmsNorm, FullRoPe] =
  ## Compressed-Q MLAttention load of checkpoint layer `layerIdx`
  ## over the typed latent cache, wiring mirrored from the model files:
  ## - q bottleneck plus both latent norms at the bottleneck eps
  ## - softmax scale 1/sqrt(qk_head_dim)
  ##
  ## Expected input:
  ## - modelDir, the checkpoint directory (config.json plus weight shards)
  ## - prefix, the dot-terminated attention path, shaped "model.layers.N.self_attn."
  let cfgJson = packedjson.parseFile(modelDir / "config.json")
  let view = SafetensorsCollection.open(modelDir)
  let cache = MlaLatentCache.init(
    cfgJson{"kv_lora_rank"}.getInt(), cfgJson{"qk_rope_head_dim"}.getInt(),
    maxSeq, kBFloat16, device)
  MLAttention[RmsNorm, FullRoPe].init(
    layerIdx, prefix,
    q_a_proj = Linear.load(view, cfgJson, prefix & ".q_a_proj", device),
    q_b_proj = Linear.load(view, cfgJson, prefix & ".q_b_proj", device),
    q_a_norm = RmsNorm.init(
      view.getTensorOwned(prefix & ".q_a_layernorm.weight", device),
      qBF16, eps = BottleneckNormEps),
    kv_a_proj_with_mqa = Linear.load(
      view, cfgJson, prefix & ".kv_a_proj_with_mqa", device),
    kv_a_layernorm = RmsNorm.init(
      view.getTensorOwned(prefix & ".kv_a_layernorm.weight", device),
      qBF16, eps = BottleneckNormEps),
    kv_b_proj = Linear.load(view, cfgJson, prefix & ".kv_b_proj", device),
    o_proj = Linear.load(view, cfgJson, prefix & ".o_proj", device),
    numHeads = cfgJson{"num_attention_heads"}.getInt(),
    qkNopeHeadDim = cfgJson{"qk_nope_head_dim"}.getInt(),
    kvLoraRank = cfgJson{"kv_lora_rank"}.getInt(),
    vHeadDim = cfgJson{"v_head_dim"}.getInt(),
    softmaxScale = mlaSoftmaxScale(cfgJson{"qk_nope_head_dim"}.getInt(),
      cfgJson{"qk_rope_head_dim"}.getInt()),
    cache = cache)

proc setupMlaGated*(modelDir, prefix: string, layerIdx, maxSeq: int,
    device = F.kCPU): HeadwiseGatedMLAttention[RmsNorm, FullRoPe] =
  ## Head-wise gated MLAttention load of checkpoint layer `layerIdx`
  ## over the typed latent cache, wiring mirrored from the model files:
  ## - compressed-Q bottleneck, both latent norms at the bottleneck eps
  ## - per-head sigmoid gate weight projection, the gate multiply
  ##   sits before the output projection
  ## - softmax scale 1/sqrt(qk_head_dim)
  ##
  ## Expected input:
  ## - modelDir, the checkpoint directory (config.json plus weight shards)
  ## - prefix, the dot-terminated attention path, shaped "model.layers.N.attention."
  let cfgJson = packedjson.parseFile(modelDir / "config.json")
  let view = SafetensorsCollection.open(modelDir)
  let cache = MlaLatentCache.init(
    cfgJson{"kv_lora_rank"}.getInt(), cfgJson{"qk_rope_head_dim"}.getInt(),
    maxSeq, kBFloat16, device)
  HeadwiseGatedMLAttention[RmsNorm, FullRoPe].init(
    layerIdx, prefix,
    q_a_proj = Linear.load(view, cfgJson, prefix & ".q_a_proj", device),
    q_b_proj = Linear.load(view, cfgJson, prefix & ".q_b_proj", device),
    q_a_norm = RmsNorm.init(
      view.getTensorOwned(prefix & ".q_a_layernorm.weight", device),
      qBF16, eps = BottleneckNormEps),
    kv_a_proj_with_mqa = Linear.load(
      view, cfgJson, prefix & ".kv_a_proj_with_mqa", device),
    kv_a_layernorm = RmsNorm.init(
      view.getTensorOwned(prefix & ".kv_a_layernorm.weight", device),
      qBF16, eps = BottleneckNormEps),
    kv_b_proj = Linear.load(view, cfgJson, prefix & ".kv_b_proj", device),
    o_proj = Linear.load(view, cfgJson, prefix & ".dense", device),
    g_proj = Linear.load(view, cfgJson, prefix & ".g_proj", device),
    numHeads = cfgJson{"num_attention_heads"}.getInt(),
    qkNopeHeadDim = cfgJson{"qk_nope_head_dim"}.getInt(),
    kvLoraRank = cfgJson{"kv_lora_rank"}.getInt(),
    vHeadDim = cfgJson{"v_head_dim"}.getInt(),
    softmaxScale = mlaSoftmaxScale(cfgJson{"qk_nope_head_dim"}.getInt(),
      cfgJson{"qk_rope_head_dim"}.getInt()),
    cache = cache)

proc scalarKindOfDtype(name: string): F.ScalarKind =
  ## Map a fixture metadata dtype string onto the libtorch scalar kind.
  case name
  of "bfloat16": F.kBFloat16
  of "float16": F.kFloat16
  of "float32": F.kFloat32
  of "uint8": F.kUint8
  else: raise newException(ValueError, "unsupported fixture segment dtype: " & name)  # noqa, an unknown recorded dtype is a fixture defect, not a checkable value

proc blobSegment*(st: Safetensor, layout: JsonNode, blob, segment: string,
    device = F.kCPU): F.Tensor =
  ## One segment of a blob tensor, typed and owned on `device`.
  ##
  ## Blob grammar of the Kimi KDA fixture file, one mixture group per blob:
  ##
  ## - the file carries the group as one uint8 blob tensor of raw little-endian bytes
  ## - each metadata `blobs.<blob>.segments` row carries dtype, shape and byte offset
  ## - every segment offset is a multiple of the element size, so each
  ##   segment view stays aligned
  ##
  ## Expected input:
  ## - st, the Safetensor reader of the fixture file, alive for the call
  ## - layout, the parsed metadata sidecar of the same fixture file
  ## - blob, segment, the blob tensor name and the segment key inside it
  ##
  ## Output:
  ## - an owned copy of the segment on `device`, per the metadata row,
  ##   shaped and typed, independent of the file mapping
  let info = layout{"blobs"}{blob}{"segments"}{segment}
  if info.kind == JNull:
    raise newException(KeyError,  # noqa, a missing segment names a malformed fixture metadata sidecar
      "no segment " & segment & " in blob " & blob & " of the fixture metadata")
  let off = info{"offset"}.getInt()
  var shape: seq[int]
  for dim in info{"shape"}.items():
    shape.add dim.getInt()
  let kind = scalarKindOfDtype(info{"dtype"}.getStr())
  let itemsize =
    case kind
    of F.kBFloat16, F.kFloat16: 2
    of F.kFloat32: 4
    of F.kUint8: 1
    else: raise newException(ValueError, "unsupported fixture segment kind")  # noqa, an untyped recorded segment is a fixture defect
  if off mod itemsize != 0:
    raise newException(ValueError,  # noqa, a misaligned segment offset is a fixture container defect
      "segment " & blob & "." & segment & " offset " & $off &
      " breaks the element-size alignment contract")
  let expected = shape.product() * itemsize
  let view = st.getMmapView(blob)
  if off + expected > view.size:
    raise newException(RangeDefect,  # noqa, a segment running past the blob payload is a fixture container defect
      "segment " & blob & "." & segment & " runs past the blob payload")
  let base = cast[pointer](cast[uint](view.data) + uint(off))
  from_blob(base, shape, kind).to(device, copy = true)
