# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-1 routed MoE block of the GLM-4.7-Flash checkpoint, replayed on cpu
## against the recorded moe payloads (prefill seq8, decode pos8).
## Requires the local model at tests/hf_models/GLM-4.7-Flash (gitignored).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_glm47flash_01_layer_internals_moe.nim

import
  std/os,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers/ffn,
  workspace/transformers/src/layers/moe_router,
  workspace/transformers/src/deserialization,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "GLM-4.7-Flash-layer-1"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "GLM-4.7-Flash"
  LayerPrefix = "model.layers.1.mlp"

proc main() =
  # The routed block loads exactly as the GLM-4.7-Flash model file
  # wires it. Router constants arrive from the checkpoint config,
  # the expert stack and shared tail through the loader.
  let cfgJson = (ModelDir / "config.json").parseFile()
  let numExpertsPerTok = cfgJson{"num_experts_per_tok"}.getInt()
  let hiddenSize = cfgJson{"hidden_size"}.getInt()
  let view = SafetensorsCollection.open(ModelDir)
  let router = NoauxTcRouter.init(
    view.getTensorOwned(LayerPrefix & ".gate.weight"),
    view.getTensorOwned(LayerPrefix & ".gate.e_score_correction_bias"),
    numExpertsPerTok, cfgJson{"n_group"}.getInt(),
    cfgJson{"topk_group"}.getInt(),
    cfgJson{"routed_scaling_factor"}.getFloat(),
    normTopkProb = cfgJson{"norm_topk_prob"}.getBool())
  let ffn = BlockSparseFFN.load(view, cfgJson, LayerPrefix, router)

  # Per-payload replay. Case 0 carries the prefill seq8 pass, case 1
  # the decode pass at position 8, each payload its own stats frame.
  for caseNum in 0 ..< 2:
    let payload = FixtureDir / ("moe-GLM-4.7-Flash-0" & $caseNum &
      ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    let stage =
      if caseNum == 0: "prefill"
      else: "decode"
    echo "    devices: ", deviceName(F.kCPU), " (", stage, ")"

    let hidden = st.getTensorOwned("hidden_states").reshape(
      [st.getTensorOwned("hidden_states").numel() div hiddenSize, hiddenSize])
    let decision = router.route(hidden)
    assertStats(decision.logits, statsPath, "router_logits", kReduction,
      msg = stage & " router scoring logits")
    assertStats(decision.weights, statsPath, "topk_weights", kReduction,
      msg = stage & " renormalized top-k weights")

    let moeOutput = ffn.forward(hidden)
    # The routed block output carries one linear boundary step, the single
    # weighted accumulation reduction class.
    assertStats(moeOutput, statsPath, "moe_output", kReduction,
      depth = 1, msg = stage & " routed block output")

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
