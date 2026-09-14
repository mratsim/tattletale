# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-0 Gated DeltaNet of the Qwen3.6-35B-A3B checkpoint, replayed
## against the recorded prefill fixture, seq 5, chunk_size 64.
## Requires the local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_01_layer_internals_gdn.nim

import
  std/os,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils

{.experimental: "callOperator".}

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "Qwen3.6-35B-A3B-layer-0"

proc main() =
  ## Replay contract of the layer-0 linear attention block:
  ##
  ## - the recorded fixture input tensor drives one forward on the suite device, an exact replay
  ## - the block output meets the recorded output_chunked form, the reduction bands at depth 1
  ## - the recurrent-rule record output_seq stays unread, the chunked form is the production recording
  ##
  ## Stage records without an assert, conv_output, q, k, v, g, beta:
  ##
  ## - the layer forward holds every pre-delta-rule stage inline
  ## - no stage value is extractable from the layer
  ## - the stage asserts skip, the suite never re-derives layer math
  ## - the recording is CPU-side, the generator locks device cpu
  ## - the replay runs on the host cpu, the recording's own device
  echo "    devices: ", deviceName(F.kCPU)

  let cfgJson = (ModelDir / "config.json").parseFile()
  let tc = cfgJson{"text_config"}
  let view = SafetensorsCollection.open(ModelDir)
  let gdn = cfgJson.setup(GatedDeltaNet, view,
    "model.language_model.layers.", 0, device = F.kCPU)

  var ctx = InferenceContext.init(
    num_layers = 1, batch_size = 1,
    kv_heads = tc{"num_key_value_heads"}.getInt(), max_seq = 512,
    head_dim = tc{"linear_key_head_dim"}.getInt())

  var st = Safetensor.open(
    FixtureDir / "gdn-Qwen3.6-35B-A3B-00.safetensor")

  let x = st.getTensorOwned("input").to(F.kCPU)   # (1, 5, 2048) bf16
  let layerOut = gdn(ctx, x)
  assertStats(layerOut, FixtureDir / "gdn-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst", "output_chunked", kReduction, depth = 1, msg = "gdn chunked forward output")

when isMainModule:
  main()
