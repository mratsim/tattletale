# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Serialization dispatch — loads layer objects from safetensors.
##
## This is the ONLY file that bridges model loaders with quantization codecs.
## The quant format is detected from the model config.json, so callers
## stay quantization-blind.

import
  std/tables,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch,
  ./layers,
  ./quantizations/all_reexports,
  workspace/safetensors/src/collections,
  ./instrumentation

# ─── Quant method detection ────────────────────────────────────────────

proc detectQuantization*(cfg: JsonNode): QuantFormatKind =
  if cfg.hasKey("quantization_config"):
    let qm = cfg["quantization_config"]["quant_method"].getStr("")
    checkValue(qm == "exl3",
      "[ttt] Unsupported quant_method: '" & qm & "' (expected 'exl3' or no quantization_config)")
    qExl3
  else:
    qBF16

# var ... {.compileTime.} are removed from the runtime.
# We force materializing them as `const` at runtime by shadowing them with a const ... = static(...)
const QuantLoaderRegistry = static(QuantLoaderRegistry)

# ─── Activations ────────────────────────────────────────────────────────

proc getDeployDtype*(cfg: JsonNode): ScalarKind =
  QuantLoaderRegistry[detectQuantization(cfg)].deployDtype

# ─── Linear ─────────────────────────────────────────────────────────────

proc load*(_: type Linear, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): Linear =
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].linear
  checkValue(loader != nil, "[ttt] No linear loader for " & $quant)
  loader(view, prefix, cfg, device)

# ─── RmsNorm ───────────────────────────────────────────────────────────

proc loadRmsWeight(view: SafetensorsCollection, cfg: JsonNode, prefix: string,
                   device: DeviceKind): tuple[quant: QuantFormatKind, weight: Tensor, eps: float64] =
  let quant = detectQuantization(cfg)
  let weight = view.getTensorOwned(prefix & ".weight", device)
    .to(QuantLoaderRegistry[quant].deployDtype)
  let textCfg = cfg{"text_config"}
  let eps = textCfg{"rms_norm_eps"}.getFloat(cfg{"rms_norm_eps"}.getFloat(1e-6))
  (quant: quant, weight: weight, eps: eps)

proc load*(_: type RmsNorm, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): RmsNorm =
  let (quant, weight, eps) = loadRmsWeight(view, cfg, prefix, device)
  RmsNorm.init(weight, quant, eps)

proc load*(_: type RmsNormOne, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): RmsNormOne =
  let (quant, weight, eps) = loadRmsWeight(view, cfg, prefix, device)
  RmsNormOne.init(weight, quant, eps)

proc load*(_: type RmsNormGated, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): RmsNormGated =
  ## Checkpoints store this weight as F32 and it deploys as the format's dtype.
  let (_, weight, eps) = loadRmsWeight(view, cfg, prefix, device)
  RmsNormGated.init(weight, eps)

# ─── Embedding ──────────────────────────────────────────────────────────

proc load*(_: type Embedding, view: SafetensorsCollection, cfg: JsonNode, prefix: string, device = kCPU): Embedding =
  let quant = detectQuantization(cfg)
  let weight = view.getTensorOwned(prefix & ".weight", device)
    .to(QuantLoaderRegistry[quant].deployDtype)
  let textCfg = cfg{"text_config"}
  let vocab = textCfg{"vocab_size"}.getInt(cfg{"vocab_size"}.getInt())
  let hidden = textCfg{"hidden_size"}.getInt(cfg{"hidden_size"}.getInt())
  checkValue(
    weight.size(0) == vocab and weight.size(1) == hidden,
    "[ttt] " & prefix & ".weight is [" &
    $weight.size(0) & ", " & $weight.size(1) & "], expected [" &
    $vocab & ", " & $hidden & "]")
  Embedding.init(weight)

# ─── GatedDenseFFN ─────────────────────────────────────────────────────────

proc load*(_: type GatedDenseFFN, view: SafetensorsCollection, cfg: JsonNode,
           prefix: string, device = kCPU): GatedDenseFFN =
  let gate = Linear.load(view, cfg, prefix & ".gate_proj", device)
  let up = Linear.load(view, cfg, prefix & ".up_proj", device)
  let down = Linear.load(view, cfg, prefix & ".down_proj", device)
  GatedDenseFFN.init(gate, up, down)

# ─── GatedBlockSparseFFN ───────────────────────────────────────────────────

proc load*(_: type GatedBlockSparseFFN, view: SafetensorsCollection, cfg: JsonNode,
           prefix: string, numExpertsPerTok: int, device = kCPU): GatedBlockSparseFFN =
  let router = view.getTensorOwned(prefix & ".gate.weight", device)
  let gateUp = view.getTensorOwned(prefix & ".experts.gate_up_proj", device)
  let down = view.getTensorOwned(prefix & ".experts.down_proj", device)
  let shared = GatedDenseFFN.load(view, cfg, prefix & ".shared_expert", device)
  let sharedGate = view.getTensorOwned(prefix & ".shared_expert_gate.weight", device)
  GatedBlockSparseFFN.init(gateUp, down, router, shared, sharedGate, numExpertsPerTok)

# ─── GatedDeltaNet ─────────────────────────────────────────────────────────

proc load*(_: type GatedDeltaNet, view: SafetensorsCollection, cfg: JsonNode,
           prefix: string, layerIdx: int,
           numKHeads, numVHeads, headKDim, headVDim, convKernelSize: int,
           device = kCPU): GatedDeltaNet =
  let qkvProj = Linear.load(view, cfg, prefix & ".in_proj_qkv", device)
  let zProj = Linear.load(view, cfg, prefix & ".in_proj_z", device)
  let aProj = Linear.load(view, cfg, prefix & ".in_proj_a", device)
  let bProj = Linear.load(view, cfg, prefix & ".in_proj_b", device)
  let convWeight = view.getTensorOwned(prefix & ".conv1d.weight", device)
  let aLog = view.getTensorOwned(prefix & ".A_log", device)
  let dtBias = view.getTensorOwned(prefix & ".dt_bias", device)
  let norm = RmsNormGated.load(view, cfg, prefix & ".norm", device)
  let outProj = Linear.load(view, cfg, prefix & ".out_proj", device)
  GatedDeltaNet.init(layerIdx, prefix,
    qkvProj, zProj, aProj, bProj,
    convWeight, aLog, dtBias, norm, outProj,
    numKHeads, numVHeads, headKDim, headVDim, convKernelSize)

# ─── Grouped Query Attention ───────────────────────────────────────────────

proc load*[QKNorm](_: type RopeGQAttention[QKNorm], view: SafetensorsCollection,
                   cfg: JsonNode, prefix: string, layerIdx: int,
                   numQoHead, numKvHead, headDim: int,
                   rotary: RotaryPositionEmbedding,
                   device = kCPU): RopeGQAttention[QKNorm] =
  let qProj = Linear.load(view, cfg, prefix & ".q_proj", device)
  let kProj = Linear.load(view, cfg, prefix & ".k_proj", device)
  let vProj = Linear.load(view, cfg, prefix & ".v_proj", device)
  let oProj = Linear.load(view, cfg, prefix & ".o_proj", device)
  let qNorm = QKNorm.load(view, cfg, prefix & ".q_norm", device)
  let kNorm = QKNorm.load(view, cfg, prefix & ".k_norm", device)
  RopeGQAttention[QKNorm].init(layerIdx, prefix,
    qProj, kProj, vProj, oProj,
    numQoHead, numKvHead, headDim, rotary,
    q_norm = qNorm, k_norm = kNorm)

# ─── Gated Attention ───────────────────────────────────────────────────────

proc load*[QKNorm](_: type RopeElementWiseGatedAttention[QKNorm],
                   view: SafetensorsCollection, cfg: JsonNode, prefix: string,
                   layerIdx: int, numQoHead, numKvHead, headDim: int,
                   rotary: RotaryPositionEmbedding,
                   device = kCPU): RopeElementWiseGatedAttention[QKNorm] =
  let qProj = Linear.load(view, cfg, prefix & ".q_proj", device)
  let kProj = Linear.load(view, cfg, prefix & ".k_proj", device)
  let vProj = Linear.load(view, cfg, prefix & ".v_proj", device)
  let oProj = Linear.load(view, cfg, prefix & ".o_proj", device)
  let qNorm = QKNorm.load(view, cfg, prefix & ".q_norm", device)
  let kNorm = QKNorm.load(view, cfg, prefix & ".k_norm", device)
  RopeElementWiseGatedAttention[QKNorm].init(layerIdx, prefix,
    qProj, kProj, vProj, oProj,
    numQoHead, numKvHead, headDim, rotary,
    q_norm = qNorm, k_norm = kNorm)

# ─── LMHead ────────────────────────────────────────────────────────────────

proc load*(_: type LMHead, view: SafetensorsCollection, cfg: JsonNode, embedTokens: Embedding, device = kCPU): LMHead =
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].lmHead
  checkValue(not loader.isNil(), "[ttt] No LMHead loader for " & $quant)
  let lmHead = loader(view, device)
  if not lmHead.isNil():
    return lmHead
  let textCfg = cfg{"text_config"}
  let tied = textCfg{"tie_word_embeddings"}.getBool(cfg{"tie_word_embeddings"}.getBool(true))
  if not tied:
    raise newException(IOError,
      "[ttt] LMHead.load: no lm_head tensors in the checkpoint and " &
      "tie_word_embeddings is false")
  LMHead.initTied(embedTokens)
