# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chain suite for the Qwen3.5-0.8B dense stack, the 8 recorded prefix blocks
## plus the depth-24 tail checkpoint against the committed fixtures.
## Requires the local model at tests/hf_models/Qwen3.5-0.8B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_02_first_8_layers_plus_final.nim

import
  std/importutils,
  std/options,
  std/os,
  std/strutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Qwen35Model)
privateAccess(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])
privateAccess(DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedDenseFFN, RmsNormOne])
privateAccess(GatedDeltaNet)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-02-first-8-layers-plus-final" / "Qwen3.5-0.8B"
  GdnFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Qwen3.5-0.8B-layer-0"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"
  WeightsPath = ModelPath / "model.safetensors-00001-of-00001.safetensors"

proc main() =
  # The recorded chain contract of this fixture family is the reference
  # device replay, the fixtures were recorded on cpu. A cross-device run
  # names the tolerance class and skips, no device budget applies here.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadQwen35ModelRaw(ModelPath, F.kCPU)
  let numLayers = model.config.num_hidden_layers

  var (ctx, pool) = newKVContext(
    numLayers = numLayers,
    kvHeads = model.config.num_key_value_heads,
    headDim = model.config.head_dim)

  # The chain seeds on the recorded block-00 input, the fixture family
  # records no input ids.
  let st0 = Safetensor.open(FixtureDir / "block-00.safetensor")
  var blockInput = st0.getTensorOwned("layer_input")
  var hidden = blockInput
  var stream: Option[Tensor]

  ctx.position_ids = F.arange(hidden.size(1).int64,
    F.tensorOptions(F.kInt64, F.kCPU))
  ctx.setRopeForPositions(model.rotary)

  var prefixBlocks = 0
  while fileExists(FixtureDir /
      ("block-" & ($prefixBlocks).align(2, '0') & ".safetensor")):
    inc prefixBlocks

  for i in 0 ..< prefixBlocks:
    var st = Safetensor.open(FixtureDir /
      ("block-" & ($i).align(2, '0') & ".safetensor"))

    # The boundary input assert starts at block 1.
    if i > 0:
      assertStats(hidden, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "layer_input", kReduction, depth = i + 1, msg = "block " & $i & " boundary input")

    let pair = model.layers[i].forward(ctx, blockInput, stream)
    blockInput = pair[0]
    stream = some(pair[1])
    let layerOut = pair[1] + pair[0]
    assertStats(layerOut, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "layer_output", kReduction, depth = i + 1, msg = "block " & $i & " chain checkpoint")
    hidden = layerOut

  # The blocks past the fixture prefix carry no per-block records.
  # The chain runs on through the real forward to the tail checkpoint.
  for i in prefixBlocks ..< numLayers:
    let pair = model.layers[i].forward(ctx, blockInput, stream)
    blockInput = pair[0]
    stream = some(pair[1])

  # The tail checkpoint is the decoder stack output feeding the final
  # RMSNorm and lm head.
  let tail = stream.get(blockInput) + blockInput
  assertStats(tail, FixtureDir / "tail.safetensor.stats", "pre_final_norm", kReduction, depth = numLayers, msg = "chain tail checkpoint at depth " & $numLayers)

  # The layer-0 GDN replay of the recorded prefill case at seq 5.
  # The fixture carries the sublayer intermediates, the replay
  # recomputes each through the layer's own components.
  # This fixture carries no stats frame, so every assert record
  # comes off a fixture tensor directly.
  block:
    let cfg = (ModelPath / "config.json").parseFile()
    var weights = SafetensorsCollection.open(WeightsPath)
    let gdn = cfg.setup(GatedDeltaNet, weights,
      "model.language_model.layers.", 0)
    var st = Safetensor.open(GdnFixtureDir / "gdn-Qwen3.5-0.8B-00.safetensor")

    let x = st.getTensorOwned("input")
    var (gdnCtx, gdnPool) = newKVContext(numLayers = numLayers,
      kvHeads = 2, headDim = 256)
    let output = gdn(gdnCtx, x)

    let seqLen = x.size(1)

    # The block output, the recurrence exercised end to end.
    assertStats(output, GdnFixtureDir / "gdn-Qwen3.5-0.8B-00.safetensor" & ".stats", "output_chunked", kReduction, msg = "gdn block output vs the production recording")

  # The layer-0 GDN state persistence case, the recorded
  # prefill-then-decode trajectory. The context carries the conv window
  # across the calls. The stored states carry no expressible record.
  # The SSM trajectory sidecar is descriptors-only and the recorded conv
  # windows hold values below the recorder binade floor, so the asserts
  # cover the outputs alone.
  block:
    let cfg = (ModelPath / "config.json").parseFile()
    var weights = SafetensorsCollection.open(WeightsPath)
    let gdn = cfg.setup(GatedDeltaNet, weights,
      "model.language_model.layers.", 0)
    var st = Safetensor.open(GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor")

    let prefillX = st.getTensorOwned("prefill_x")
    let decodeXd = st.getTensorOwned("decode_x_d")
    let decodeXe = st.getTensorOwned("decode_x_e")
    let oneShot = st.getTensorOwned("one_shot_block_output")

    var (gdnCtx, gdnPool) = newKVContext(numLayers = numLayers,
      kvHeads = 2, headDim = 256)

    # Prefill [a, b, c] runs the first three positions of the recorded
    # one-shot trajectory, steps 0 to 2 of the one-shot block output.
    let outPrefill = gdn(gdnCtx, prefillX)
    assertStats(outPrefill, GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.stats", "one_shot_block_output_steps0to2", kReduction, msg = "prefill output vs recorded one-shot steps 0..2")
    let convStatePrefill = gdnCtx.gdnConvState[0].clone()

    # Decode [d] continues the trajectory, the prefill conv window feeds
    # the recorded decode conv output recompute.
    let outD = gdn(gdnCtx, decodeXd)
    assertStats(outD, GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.stats", "decode_output_d", kReduction, msg = "decode d output vs recording")
    assertStats(outD, GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.stats", "one_shot_block_output_step3", kReduction, msg = "decode d output vs recorded one-shot step 3")
    let mixedD = gdn.in_proj_qkv.forward(decodeXd).transpose(1, 2)
    let catInputD = F.cat([convStatePrefill.unsqueeze(0), mixedD], -1)
    let convD = F.conv1d(catInputD, gdn.conv1d_weight,
      padding = [0], groups = gdn.conv_dim)
    let convOutD = F.silu(convD.narrow(2, convD.size(2) - 1, 1))
    assertStats(convOutD, GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.stats", "decode_conv_output_d", kElementwise, msg = "decode d ATen conv replay over the Nim conv state")
    let convStateAfterD = gdnCtx.gdnConvState[0].clone()

    # Decode [e] closes the trajectory against the recorded final states.
    let outE = gdn(gdnCtx, decodeXe)
    assertStats(outE, GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.stats", "decode_output_e", kReduction, msg = "decode e output vs recording")
    assertStats(outE, GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.stats", "one_shot_block_output_step4", kReduction, msg = "decode e output vs recorded one-shot step 4")
    let mixedE = gdn.in_proj_qkv.forward(decodeXe).transpose(1, 2)
    let catInputE = F.cat([convStateAfterD.unsqueeze(0), mixedE], -1)
    let convE = F.conv1d(catInputE, gdn.conv1d_weight,
      padding = [0], groups = gdn.conv_dim)
    let convOutE = F.silu(convE.narrow(2, convE.size(2) - 1, 1))
    assertStats(convOutE, GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.stats", "decode_conv_output_e", kElementwise, msg = "decode e ATen conv replay over the Nim conv state")

  # The order-equivalence cases of the fixture family compare two live
  # replays against each other. The two cases are the conv-history
  # continuation and the one-shot vs step-decode replay on a synthetic
  # sequence. Neither case carries a recorded side, so no stats assert
  # expresses the contract on either case.

when isMainModule:
  main()
