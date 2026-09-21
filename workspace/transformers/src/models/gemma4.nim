# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/os,
  std/tables,
  pkg/packedjson,
  workspace/libtorch,
  workspace/safetensors,
  ../instrumentation,
  ./all_interfaces,
  ./gemma4e2b,
  ./gemma4_12b,
  ./gemma4_26b

export gemma4e2b, gemma4_12b, gemma4_26b

proc loadGemma4Model(modelPath: string, device: DeviceKind): AnyModel =
  ## Architecture dispatcher for the shared Gemma4ForConditionalGeneration
  ## checkpoint family. The architecture string covers several
  ## member checkpoints, each port picked by its text_config block layout:
  ##
  ## - per-layer embeddings (hidden_size_per_layer_input > 0), the E2B shape
  ## - the routed block (enable_moe_block), the 26B-A4B shape
  let cfg = (modelPath / "config.json").parseFile()
  let tc = cfg{"text_config"}
  if tc{"hidden_size_per_layer_input"}.getInt(0) > 0:
    loadGemma4E2BModel(modelPath, device)
  elif tc{"enable_moe_block"}.getBool(false):
    loadGemma4Text26BModel(modelPath, device)
  else:
    raise newException(ValueError,
      "[ttt] loadGemma4Model: the checkpoint carries neither per-layer " &
      "embeddings nor a routed block, no port instantiates this Gemma4" &
      "ForConditionalGeneration layout")

static:
  # Register the gemma-4 family dispatcher under the shared architecture key
  ModelRegistry["Gemma4ForConditionalGeneration"] = loadGemma4Model
