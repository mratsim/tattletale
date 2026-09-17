# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-1 noaux_tc router of the Moonlight-16B-A3B checkpoint, replayed
## on cpu against the recorded layer-1 gate weight fixture (prefill plus decode rows).
## Requires the local model at tests/hf_models/Moonlight-16B-A3B (gitignored).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_moonlight_01_layer_internals_router.nim

import
  std/os,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers/moe_router,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

const
  FixturePath = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Moonlight-16B-A3B-layer-1" /
    "gate-Moonlight-16B-A3B-00.safetensor"
  GateSplitAPath = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Moonlight-16B-A3B-layer-1" /
    "gate-Moonlight-16B-A3B-00-gate-a.safetensor"
  GateSplitBPath = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Moonlight-16B-A3B-layer-1" /
    "gate-Moonlight-16B-A3B-00-gate-b.safetensor"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Moonlight-16B-A3B"
  LayerPrefix = "model.layers.1.mlp.gate"

proc main() =
  # Routing constants arrive from the checkpoint config, the router
  # gate weight plus bias buffer arrive from the loader.
  let cfgJson = (ModelDir / "config.json").parseFile()
  let numExperts = cfgJson{"n_routed_experts"}.getInt()
  let topk = cfgJson{"num_experts_per_tok"}.getInt()
  let nGroup = cfgJson{"n_group"}.getInt()
  let topkGroup = cfgJson{"topk_group"}.getInt()
  let scale = cfgJson{"routed_scaling_factor"}.getFloat()
  let normTopkProb = cfgJson{"norm_topk_prob"}.getBool()

  let view = SafetensorsCollection.open(ModelDir)
  let router = NoauxTcRouter.init(
    view.getTensorOwned(LayerPrefix & ".weight"),
    view.getTensorOwned(LayerPrefix & ".e_score_correction_bias"),
    topk, nGroup, topkGroup, scale, normTopkProb)

  # The recording split the gate weight rows into two side payloads
  # over the file cap, one [E/2, H] row block per half file.
  let gateWeight = view.getTensorOwned(LayerPrefix & ".weight")
  var stA = Safetensor.open(GateSplitAPath)
  var stB = Safetensor.open(GateSplitBPath)
  let halfARows = stA.getTensorOwned("gate_weight").size(0)
  let halfA = gateWeight.narrow(0, 0, halfARows)
  let halfB = gateWeight.narrow(0, halfARows, gateWeight.size(0) - halfARows)

  var st = Safetensor.open(FixturePath)
  let statsPath = FixturePath & ".stats.json.zst"

  # Gate weight fixture replay. The prefill and decode routing run
  # over the recorded real-hidden rows, every compared surface arriving
  # from the payload's own stats frame.
  block:
    echo "    devices: ", deviceName(F.kCPU)
    let prefill = router.route(st.getTensorOwned("hidden_states"))
    assertStats(prefill.logits, statsPath, "router_logits", kReduction,
      msg = "router scoring logits, prefill")
    assertStats(prefill.weights, statsPath, "topk_weights", kReduction,
      msg = "renormalized top-k weights, prefill")
    let decode = router.routeDecode(st.getTensorOwned("hidden_step"))
    assertStats(decode.logits, statsPath, "router_logits_step", kReduction,
      msg = "router scoring logits, decode")
    assertStats(decode.weights, statsPath, "topk_weights_step", kReduction,
      msg = "renormalized top-k weights, decode")

  # Loader cross-check. The checkpoint gate weight rows equal
  # the recorded halves under pure data movement, elementwise model.
  block:
    assertStats(halfA, GateSplitAPath & ".stats.json.zst", "gate_weight",
      kElementwise, msg = "checkpoint gate rows, recorded half a")
    assertStats(halfB, GateSplitBPath & ".stats.json.zst", "gate_weight",
      kElementwise, msg = "checkpoint gate rows, recorded half b")

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
