# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off \
##   --outdir:build/tests/t_bf16_qwen36moe_01_moe --nimcache:nimcache/tests/t_bf16_qwen36moe_01_moe \
##   workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_01_moe.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/options,
  std/os,
  std/strutils,
  pkg/jsony,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/ffn,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/models/loading/generation_config,
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

const ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
const WeightsFile1 = ModelDir / "model-00001-of-00026.safetensors"
const WeightsFile2 = ModelDir / "model-00002-of-00026.safetensors"
const FixturePath = currentSourcePath().parentDir() / ".." / "fixtures" /
  "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-0" / "moe_layer0_fixture.json.zst"

const Tokens = 6
const TopK = 8
const NumExperts = 256
const RoutedWidth = 512
const Hidden = 2048

# Recording environment of the committed routed-block fixture. The suite
# compares the fixture's meta versions against these consts, so an artifact
# regenerated in a foreign environment fails the comparison.
const TorchRecordingVersion = "2.14.0"
const TransformersRecordingVersion = "5.16.1"

type
  MoEFixtureBands = object
    output_band: float64
    router_logits_band: float64
    routing_weights_band: float64
    shared_gate_band: float64
  MoEFixtureMargins = object
    topk_inner_gap_min: float64
    topk_margin_min: float64
  MoEFixtureMeta = object
    torch_version: string
    transformers_version: string
    num_experts_per_tok: int
  MoEFixture = object
    meta: MoEFixtureMeta
    bands: MoEFixtureBands
    margins: MoEFixtureMargins
    h: seq[seq[float64]]
    moe_output: seq[seq[float64]]
    routing_weights: seq[seq[float64]]
    topk_indices: seq[seq[int64]]

func matrixTensor(rows: seq[seq[float64]]): Tensor =
  ## [rows, cols] fp32 tensor from equal-length numeric rows.
  var flat = newSeq[float32]()
  var cols = 0
  for row in rows:
    cols = row.len
    for cell in row:
      flat.add cell.float32
  flat.toTensor().reshape(rows.len, cols)

func indicesTensor(rows: seq[seq[int64]]): Tensor =
  ## [rows, cols] int64 tensor of expert ids.
  var flat = newSeq[int64]()
  var cols = 0
  for row in rows:
    cols = row.len
    flat.add row
  flat.toTensor().reshape(rows.len, cols)

proc main() =





  # Opening the two weight files by name below is a suite-only convenience.
  # The upcoming loader work replaces it with the index-backed weight_map
  # view.

  runCppTest "Real layer-0 MoE tensors: ranks and shapes":
    proc(): bool =
      var st1 = Safetensor.open(WeightsFile1)
      var st2 = Safetensor.open(WeightsFile2)

      let gateUpProj = st1.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.gate_up_proj", kCPU)
      let downProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.down_proj", kCPU)
      let routerWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.gate.weight", kCPU)
      let sharedGateWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert_gate.weight", kCPU)
      let sharedGateProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", kCPU)
      let sharedUpProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", kCPU)
      let sharedDownProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", kCPU)

      # Fused gate/up weight of the routed bodies: gate rows 0:I, up rows I:2I
      doAssert gateUpProj.dim() == 3
      doAssert gateUpProj.size(0) == NumExperts
      doAssert gateUpProj.size(1) == 2 * RoutedWidth
      doAssert gateUpProj.size(2) == Hidden
      doAssert downProj.dim() == 3
      doAssert downProj.size(0) == NumExperts
      doAssert downProj.size(1) == Hidden
      doAssert downProj.size(2) == RoutedWidth
      doAssert routerWeight.dim() == 2
      doAssert routerWeight.size(0) == NumExperts
      doAssert routerWeight.size(1) == Hidden
      doAssert sharedGateWeight.dim() == 2
      doAssert sharedGateWeight.size(0) == 1
      doAssert sharedGateWeight.size(1) == Hidden
      doAssert sharedGateProj.dim() == 2
      doAssert sharedGateProj.size(0) == RoutedWidth
      doAssert sharedGateProj.size(1) == Hidden
      doAssert sharedUpProj.dim() == 2
      doAssert sharedUpProj.size(0) == RoutedWidth
      doAssert sharedUpProj.size(1) == Hidden
      doAssert sharedDownProj.dim() == 2
      doAssert sharedDownProj.size(0) == Hidden
      doAssert sharedDownProj.size(1) == RoutedWidth
      true

  runCppTest "Routed block forward vs fixture within recorded bands":
    proc(): bool =
      echo "    devices: ", compareLine(splitFile(FixturePath).dir, F.kCPU)
      var st1 = Safetensor.open(WeightsFile1)
      var st2 = Safetensor.open(WeightsFile2)

      let gateUpProj = st1.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.gate_up_proj", kCPU)
      let downProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.experts.down_proj", kCPU)
      let routerWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.gate.weight", kCPU)
      let sharedGateWeight = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert_gate.weight", kCPU)
      let sharedGateProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", kCPU)
      let sharedUpProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", kCPU)
      let sharedDownProj = st2.getTensorOwned(
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", kCPU)

      let sharedExpert = GatedDenseFFN.init(
        sharedGateProj, sharedUpProj, sharedDownProj)

      let fixture = zstdReadFixture(FixturePath).fromJson(MoEFixture)
      doAssert fixture.meta.torch_version == TorchRecordingVersion
      doAssert fixture.meta.transformers_version == TransformersRecordingVersion
      let numExpertsPerTok = fixture.meta.num_experts_per_tok
      let ffn = GatedBlockSparseFFN.init(
        gateUpProj, downProj, routerWeight, sharedExpert, sharedGateWeight,
        numExpertsPerTok)

      let h = matrixTensor(fixture.h).to(kBfloat16)
      let (topkIndices, routingWeights) = routeToExperts(h, routerWeight, numExpertsPerTok)
      let moeOutput = ffn.forward(h)

      # Bands the generator recorded from bf16 ulp arithmetic
      let outputBand = fixture.bands.output_band
      let weightsBand = fixture.bands.routing_weights_band

      let fixtureOutput = matrixTensor(fixture.moe_output)
      let fixtureWeights = matrixTensor(fixture.routing_weights)

      doAssert maxAbsDiff(moeOutput, fixtureOutput) <= outputBand
      doAssert maxAbsDiff(routingWeights, fixtureWeights) <= weightsBand

      # The fp32 renorm is internal to routeToExperts and reaches the fixture
      # through the weights band, asserted together with its output dtype
      doAssert routingWeights.scalarType() == kBfloat16,
        "routing weights must be cast to the hidden-state dtype"

      # Smallest top-8 margin and smallest adjacent gap are both positive:
      # the top-8 order is uniquely determined by value
      doAssert fixture.margins.topk_margin_min > 0.0
      doAssert fixture.margins.topk_inner_gap_min > 0.0

      let fixtureIndices = indicesTensor(fixture.topk_indices)
      for tok in 0 ..< Tokens:
        for pos in 0 ..< TopK:
          let nimIdx = topkIndices[tok, pos]
          let fixIdx = fixtureIndices[tok, pos]
          doAssert nimIdx.item(int64) == fixIdx.item(int64),
            "topk index mismatch at [" & $tok & ", " & $pos & "]"
      true

  runCppTest "Routed block forward, cross-device variant (device-pair report)":
    proc(): bool =
      let runDev = testDevice()
      echo "    devices: ", compareLine(splitFile(FixturePath).dir, runDev)
      if runDev == F.kCPU:
        echo "    the run device matches the recorded device, the reference variant carries the replay"
        return true
      # The recorded output and routing-weight bands were measured cpu
      # against cpu by the generator. The layers family carries
      # no PROVENANCE.md manifest and states no recorded device, leaving
      # row-class selection unavailable: the suite skips on the run device.
      echo "    no cross-device drift row applies: the recorded bands are",
        " cpu class and the family manifest is absent, the suite skips on ",
        deviceName(runDev)
      return true

when isMainModule:
  main()
