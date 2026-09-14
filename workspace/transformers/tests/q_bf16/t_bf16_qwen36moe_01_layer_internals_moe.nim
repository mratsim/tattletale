# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Routed MoE block layer 0 of the Qwen3.6-35B-A3B checkpoint, replayed
## on cpu against the standalone routed-block fixture.
##
## Requires the local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_01_layer_internals_moe.nim

import
  std/os,
  pkg/packedjson,
  pkg/jsony,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils

from workspace/transformers/tests/harness import zstdReadFixture

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  FixturePath = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-0" / "moe_layer0_fixture.json.zst"
  MoeFramePath = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-0" / "moe_layer0_fixture.stats"
  Layer0Prefix = "model.language_model.layers.0"
  Layer0Router = Layer0Prefix & ".mlp.gate.weight"

type
  ## Recorded routed-block fixture from the layer-0 recording environment.
  MoEFixture = object
    h: seq[seq[float64]]
    moe_output: seq[seq[float64]]
    routing_weights: seq[seq[float64]]
    topk_indices: seq[seq[int64]]

proc main() =
  let cfgJson = (ModelDir / "config.json").parseFile()
  let tc = cfgJson{"text_config"}
  let numExpertsPerTok = tc{"num_experts_per_tok"}.getInt()
  let view = SafetensorsCollection.open(ModelDir)
  let ffn = GatedBlockSparseFFN.load(view, cfgJson, Layer0Prefix & ".mlp",
      numExpertsPerTok)

  # Routed block forward vs fixture
  block:
    echo "    devices: ", deviceName(F.kCPU)
    let fixture = zstdReadFixture(FixturePath).fromJson(MoEFixture)
    let h = setupMatrixTensor(fixture.h).to(kBfloat16)
    let (topkIndices, routingWeights) = routeToExperts(h,
      view.getTensorOwned(Layer0Router), numExpertsPerTok)
    let moeOutput = ffn.forward(h)

    let fixtureWeights = setupMatrixTensor(fixture.routing_weights).to(kBfloat16)
    assertStats(routingWeights, MoeFramePath, "routing_weights", kElementwise, msg = "routing weights")
    let fixtureOutput = setupMatrixTensor(fixture.moe_output).to(kBfloat16)
    assertStats(moeOutput, MoeFramePath, "moe_output", kReduction, msg = "moe output")
    let fixtureIndices = setupIndicesTensor(fixture.topk_indices).to(kFloat32)
    assertStats(topkIndices.to(kFloat32), MoeFramePath, "topk_indices", kElementwise, msg = "topk expert ids")

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
