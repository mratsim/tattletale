# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Layer-0 unit replay of the Qwen3.6-35B-A3B checkpoint, one assertion
## block per mixture of the single fixture file.
##
## - the gdn (Gated DeltaNet prefill), layer and moe (routed block) mixtures
## - replay on testDevice(), Metal on this host, the fixture recording
##   staying torch-side cpu
##
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
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  # workspace/transformers/src/models/qwen35_moe,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils

{.experimental: "callOperator".}
const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "Qwen3.6-35B-A3B-layer-0" / "layer0-Qwen3.6-35B-A3B-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"
  Layer0Prefix = "model.language_model.layers.0"
  Layer0Router = Layer0Prefix & ".mlp.gate.weight"

proc main() =
  # The Metal backend falls back to the cpu kernels where the device
  # kernels are missing, the mixtures replay on whatever testDevice()
  # resolves without a hard device requirement.
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let tc = cfgJson{"text_config"}
  let numExpertsPerTok = tc{"num_experts_per_tok"}.getInt()
  let hiddenSize = tc{"hidden_size"}.getInt()
  let view = SafetensorsCollection.open(ModelDir)
  let gdn = cfgJson.setupGatedDeltaNet(
        GatedDeltaNet[perHead, FullRankGateIn, GateForm.softplus], view,
    "model.language_model.layers.", 0, device = dev)
  let inputLN = RmsNormOne.load(view, cfgJson, Layer0Prefix & ".input_layernorm", dev)
  let postLN = RmsNormOne.load(view, cfgJson, Layer0Prefix & ".post_attention_layernorm", dev)
  let ffn = GatedBlockSparseFFN.load(view, cfgJson, Layer0Prefix & ".mlp",
      numExpertsPerTok, dev)
  let routerWeight = view.getTensorOwned(Layer0Router, dev)
  var st = Safetensor.open(FixturePath)

  # Mixture gdn, the layer-0 Gated DeltaNet prefill at seq 5, the suite
  # asserting the recorded chunked form (the production recording) while
  # the recurrent-rule record output stays unread.
  block:
    var ctx = InferenceContext.init(
      num_layers = 1, batch_size = 1,
      kv_heads = tc{"num_key_value_heads"}.getInt(), max_seq = 512,
      head_dim = tc{"linear_key_head_dim"}.getInt())
    let x = st.getTensorOwned("gdn.input", dev)   # (1, 5, 2048) bf16
    let gdnOut = gdn(ctx, x)
    assertStats(gdnOut, StatsPath, "gdn.output_chunked", kReduction,
      depth = 1, msg = "gdn chunked forward output")

  # Mixture layer, the full decoder layer 0 chain (GDN mixer + routed MoE)
  # per the installed Qwen3_5MoeDecoderLayer.forward:
  # local residual, input layernorm, token mixer, residual,
  # post-attention layernorm, routed block, residual.
  block:
    var ctx = InferenceContext.init(
      num_layers = 1, batch_size = 1,
      kv_heads = tc{"num_key_value_heads"}.getInt(), max_seq = 512,
      head_dim = tc{"linear_key_head_dim"}.getInt())
    let x = st.getTensorOwned("layer.layer_input", dev)   # (1, 6, 2048) bf16
    let hNorm = inputLN.forward(x)
    assertStats(hNorm, StatsPath, "layer.input_layernorm_output", kElementwise,
      msg = "layer-0 input layernorm output")
    let gdnOut = gdn(ctx, hNorm)
    assertStats(gdnOut, StatsPath, "layer.gdn_block_output_chunked", kReduction,
      msg = "layer-0 gdn block output")
    let h1 = x + gdnOut
    let h2 = postLN.forward(h1)
    assertStats(h2, StatsPath, "layer.post_attention_layernorm_output", kElementwise,
      msg = "layer-0 post-attention layernorm output")
    let h2Flat = h2.reshape(h2.size(1), hiddenSize)
    let (gotIndices, gotWeights) = routeToExperts(h2Flat, routerWeight, numExpertsPerTok)
    let _ = gotIndices
    let moeOut = ffn.forward(h2)
    assertStats(gotWeights, StatsPath, "layer.routing_weights", kElementwise,
      msg = "layer-0 routing weights")
    assertStats(moeOut, StatsPath, "layer.moe_output", kReduction,
      msg = "layer-0 moe output")
    let layerOut = h1 + moeOut
    assertStats(layerOut, StatsPath, "layer.layer_output_chunked", kReduction,
      msg = "layer-0 post-residual output")

  # Mixture moe, the routed block on the recorded T=6 tokens, the router
  # selection over rank-2 hidden states, the routed experts and the final
  # gated shared expert sum.
  block:
    let h3d = st.getTensorOwned("moe.h", dev)   # (1, 6, 2048) bf16
    let h = h3d.reshape(h3d.size(1), hiddenSize)
    let (topkIndices, routingWeights) = routeToExperts(h, routerWeight, numExpertsPerTok)
    let moeOutput = ffn.forward(h)
    assertStats(routingWeights, StatsPath, "moe.routing_weights", kElementwise,
      msg = "routing weights")
    assertStats(moeOutput, StatsPath, "moe.moe_output", kReduction,
      msg = "moe output")
    assertStats(topkIndices.to(kFloat32), StatsPath, "moe.topk_indices", kElementwise,
      msg = "topk expert ids")

when isMainModule:
  main()
