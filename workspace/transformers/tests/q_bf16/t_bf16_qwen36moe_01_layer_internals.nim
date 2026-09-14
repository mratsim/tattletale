# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full decoder-layer blocks of the Qwen3.6-35B-A3B checkpoint, replayed
## on cpu against the layer-0 and layer-3 fixture payloads.
## Requires the local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_01_layer_internals.nim

import
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/safetensors/src/collections,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/attn_ssm/grouped_query_attention {.all.},
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(RopeElementWiseGatedAttention[RmsNormOne])

const
  Layer0FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-0"
  Layer3FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-3"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  WeightsFile3 = ModelDir / "model-00003-of-00026.safetensors"
  Layer0Prefix = "model.language_model.layers.0"
  Layer3Prefix = "model.language_model.layers.3"
  Layer0Router = Layer0Prefix & ".mlp.gate.weight"

proc main() =
  let cfgJson = (ModelDir / "config.json").parseFile()
  let tc = cfgJson{"text_config"}
  let numExpertsPerTok = tc{"num_experts_per_tok"}.getInt()

  # Full decoder layer 0 (GDN + MoE) vs fixture
  block:
    echo "    devices: ", deviceName(F.kCPU)
    let view = SafetensorsCollection.open(ModelDir)
    let gdn = cfgJson.setup(GatedDeltaNet, view,
      "model.language_model.layers.", 0)
    var ctx = InferenceContext.init(num_layers = 1, batch_size = 1,
        kv_heads = tc{"num_key_value_heads"}.getInt(), max_seq = 512,
        head_dim = tc{"linear_key_head_dim"}.getInt())
    let inputLN = RmsNormOne.load(view, cfgJson, Layer0Prefix & ".input_layernorm")
    let postLN = RmsNormOne.load(view, cfgJson, Layer0Prefix & ".post_attention_layernorm")
    var st = Safetensor.open(Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor")
    let x = st.getTensorOwned("layer_input")
    let hNorm = inputLN.forward(x)
    assertStats(hNorm, Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst", "input_layernorm_output", kElementwise, msg = "layer-0 input layernorm output")
    let gdnOut = gdn(ctx, hNorm)
    assertStats(gdnOut, Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst", "gdn_block_output_chunked", kReduction, msg = "layer-0 gdn block output")
    let h1 = x + gdnOut
    let h2 = postLN.forward(h1)
    assertStats(h2, Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst", "post_attention_layernorm_output", kElementwise, msg = "layer-0 post-attention layernorm output")
    let ffn = GatedBlockSparseFFN.load(view, cfgJson, Layer0Prefix & ".mlp", numExpertsPerTok)
    let h2Flat = h2.reshape(h2.size(1), tc{"hidden_size"}.getInt())
    let (gotIndices, gotWeights) =
      routeToExperts(h2Flat, view.getTensorOwned(Layer0Router), numExpertsPerTok)
    let _ = gotIndices
    let moeOut = ffn.forward(h2)
    assertStats(gotWeights, Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst", "routing_weights", kElementwise, msg = "layer-0 routing weights")
    assertStats(moeOut, Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst", "moe_output", kReduction, msg = "layer-0 moe output")
    let layerOut = h1 + moeOut
    assertStats(layerOut, Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.stats.json.zst", "layer_output_chunked", kReduction, msg = "layer-0 post-residual output")

  # Full decoder layer 3 (attention + MoE) vs fixture
  block:
    echo "    devices: ", deviceName(F.kCPU)
    let weights = SafetensorsCollection.open(WeightsFile3)
    let inputLN = RmsNormOne.load(weights, cfgJson, Layer3Prefix & ".input_layernorm")
    let postLN = RmsNormOne.load(weights, cfgJson, Layer3Prefix & ".post_attention_layernorm")
    let ffn = GatedBlockSparseFFN.load(weights, cfgJson, Layer3Prefix & ".mlp", numExpertsPerTok)
    let rotary = cfgJson.setup(RotaryPositionEmbedding,
      tc{"max_position_embeddings"}.getInt())
    let attn = cfgJson.setup(RopeElementWiseGatedAttention[RmsNormOne],
      weights, "model.language_model.layers.", 3, rotary)
    var (ctx, pool) = newKVContext(numLayers = tc{"num_hidden_layers"}.getInt(),
        kvHeads = tc{"num_key_value_heads"}.getInt(),
        headDim = tc{"head_dim"}.getInt())
    var st = Safetensor.open(Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor")
    let x = st.getTensorOwned("layer_input")
    let posIds = st.getTensorOwned("position_ids")
    let ctxPosIds = posIds[0]
    ctx.position_ids = ctxPosIds
    ctx.setRopeForPositions(attn.rotary)
    let hNorm = inputLN.forward(x)
    assertStats(hNorm, Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor" & ".stats", "input_layernorm_output", kElementwise, msg = "layer-3 input layernorm output")
    let attnOut = attn(ctx, hNorm)
    assertStats(attnOut, Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor" & ".stats", "attn_mixer_output", kReduction, msg = "layer-3 attn mixer output")
    let h1 = x + attnOut
    let h2 = postLN.forward(h1)
    let h2Flat = h2.reshape(h2.size(1), tc{"hidden_size"}.getInt())
    let (gotIndices, gotWeights) =
      routeToExperts(h2Flat, weights.getTensorOwned(Layer3Prefix & ".mlp.gate.weight"),
        numExpertsPerTok)
    let _ = gotIndices
    let moeOut = ffn.forward(h2)
    assertStats(gotWeights, Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor" & ".stats", "routing_weights", kElementwise, msg = "layer-3 routing weights")
    assertStats(moeOut, Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor" & ".stats", "moe_output", kReduction, msg = "layer-3 moe output")
    let layerOut = h1 + moeOut
    assertStats(layerOut, Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor" & ".stats", "layer_output", kReduction, msg = "layer-3 post-residual output")

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
