# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Serialization dispatch — loads layer objects from safetensors.
##
## This is the ONLY file that bridges model loaders with quantization codecs.
## Model loaders call `Linear.load(st, cfg, prefix, device)`, `RmsNorm.load(st, cfg, prefix, device)`,
## etc. without any knowledge of which quant format is being used.
##
## The quant format is detected from the model config.json (`cfg`).

import
  std/tables,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch,
  ./layers,
  ./quantizations/all_reexports

# ─── Quant method detection ────────────────────────────────────────────

proc detectQuantization*(cfg: JsonNode): QuantFormatKind =
  ## Detect quantization method from parsed config.json.
  ## Future: accept `prefix` for per-layer mixed quantization.
  if cfg.hasKey("quantization_config") and
     cfg["quantization_config"]["quant_method"].getStr("") == "exl3":
    qExl3
  else:
    qBF16

# var ... {.compileTime.} are removed from the runtime.
# We force materializing them as `const` at runtime by shadowing them with a const ... = static(...)
const QuantLoaderRegistry = static(QuantLoaderRegistry)

# ─── Activations ────────────────────────────────────────────────────────

proc activationDtype*(cfg: JsonNode): ScalarKind =
  ## Return the activation dtype for the quant format detected from config.
  QuantLoaderRegistry[detectQuantization(cfg)].activationDtype

# ─── Linear ─────────────────────────────────────────────────────────────

proc load*(_: type Linear, st: Safetensor, cfg: JsonNode, prefix: string, device = kCPU): Linear =
  ## Load a linear layer from safetensors, dispatching to the right codec.
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].linear
  if loader == nil:
    raise newException(ValueError, "[ttt] No linear loader for " & $quant)
  loader(st, prefix, cfg, device)

# ─── RmsNorm ───────────────────────────────────────────────────────────

proc load*(_: type RmsNorm, st: Safetensor, cfg: JsonNode, prefix: string, device = kCPU): RmsNorm =
  ## Load RMS norm layer from safetensors, dispatching to the right codec.
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].rmsNorm
  if loader == nil:
    raise newException(ValueError, "[ttt] No RMSNorm loader for " & $quant)
  let weight = loader(st, prefix, cfg, device)
  RmsNorm.init(weight, quant, cfg["rms_norm_eps"].getFloat(1e-6))

# ─── Embedding ──────────────────────────────────────────────────────────

proc load*(_: type Embedding, st: Safetensor, cfg: JsonNode, prefix: string, device = kCPU): Tensor =
  ## Load embedding weights from safetensors, dispatching to the right codec.
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].embedding
  if loader == nil:
    raise newException(ValueError, "[ttt] No embedding loader for " & $quant)
  loader(st, prefix, cfg, device)

# ─── LMHead ──────────────────────────────────────────────────────────

proc load*(_: type LMHead, st: Safetensor, cfg: JsonNode, embedTokens: Embedding, device = kCPU): LMHead =
  ## Load LMHead from safetensors, dispatching to the right codec.
  ## Codec returns nil when weight is tied — falls back to initTied.
  let quant = detectQuantization(cfg)
  let loader = QuantLoaderRegistry[quant].lmHead
  if loader.isNil():
    # 1. Check if misconfigured coded
    raise newException(ValueError, "[ttt] No LMHead loader for " & $quant)
  let lmHead = loader(st, device)
  if lmHead.isNil():
    # 2. Check for tied embeddings
    #
    # The flow is a bit strange due to tied_embeddings
    # we want to isolate models from quantization peculiarities
    # but LMHead and Embeddings can share weight
    # but even if mentioned that they share weights, they might be quantized differently
    # and so are not shareable
    LMHead.initTied(embedTokens)
  else:
    return lmHead
