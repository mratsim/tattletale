# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Standardized helpers for the fixture-replay suites.
##
## One home for everything a suite repeats:
## - config document and weights shard of one model directory
## - rotary table geometry read off the config text section
## - the layer loads, composed over src/deserialization.nim
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
  workspace/transformers/src/layers/attn_ssm/gated_delta_net

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

proc setup*(cfg: JsonNode, T: typedesc[GatedDeltaNet],
    weights: SafetensorsCollection, layerStem: string, layerIdx: int,
    device = F.kCPU): GatedDeltaNet =
  ## Loads checkpoint layer `layerIdx` as a GatedDeltaNet, recurrent
  ## geometry read off the config text section.
  ##
  ## Expected input:
  ## - layerStem, the dot-terminated layer path stem, shaped
  ##   model.language_model.layers. (weights load under linear_attn)
  let t = if cfg.hasKey("text_config"): cfg{"text_config"} else: cfg
  let prefix = layerStem & $layerIdx & ".linear_attn"
  GatedDeltaNet.load(weights, cfg, prefix, layerIdx,
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
