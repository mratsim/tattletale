# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the Qwen3.5-0.8B dense stack, the 24
## decoder blocks replayed layer by layer against the committed fixtures.
##
## Every case enforces through assertStats or assertArgMax:
## - each layer boundary compares the replayed hidden against the fixture's
##   sequential reference, records from the fixture payloads
## - the deciding cases consume the committed decision frame per position
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_03_full_forward_to_logits.nim

import
  std/os,
  std/options,
  std/strutils,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/zstd/zstd_highlevel,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Qwen35Model)
privateAccess(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])
privateAccess(DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedDenseFFN, RmsNormOne])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "Qwen3.5-0.8B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"


proc main() =
  # The recorded chain contract of this fixture family is the reference
  # device replay, the fixtures were recorded on cpu. A cross-device run
  # names the tolerance class and skips, no device budget applies here.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadQwen35ModelRaw(ModelPath, F.kCPU)

  var (ctx, pool) = newKVContext(numLayers = 24, kvHeads = 2, headDim = 256)

  # "Hello, how are you?" as 6 token ids, matching the fixture metadata.
  let inputIds = @[9419'i64, 11, 1204, 513, 488, 30].toTensor().unsqueeze(0)

  ctx.position_ids = F.arange(6, F.tensorOptions(F.kInt64, F.kCPU))
  ctx.setRopeForPositions(model.rotary)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"

  # Manual layer loop mirroring the model forward and capturing each
  # hidden state, so every layer boundary compares against its fixture.
  # The recorded input of layer i+1 is the output of layer i, the chained
  # assert reads the next fixture's sequential record.
  var h = model.embedTokens(inputIds)
  var blockInput = h
  var stream: Option[Tensor]
  for layerIdx in 0 ..< model.layers.len:
    var st = setupLayerFixtureReader(FixtureDir, layerIdx)

    let chunkedRef = st.getTensorOwned("layer_input")
    assertStats(h, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "layer_input", kReduction, depth = max(1, layerIdx), msg = "layer " & $layerIdx & " input, chunked reference")

    let pair = model.layers[layerIdx].forward(ctx, blockInput, stream)
    blockInput = pair[0]
    stream = some(pair[1])
    h = pair[1] + pair[0]

    if layerIdx < model.layers.len - 1:
      var next = setupLayerFixtureReader(FixtureDir, layerIdx + 1)
      assertStats(h, FixtureDir / ("layer-" & ($(layerIdx + 1)).align(2, '0') & ".safetensor.stats"), "layer_input", kReduction, depth = layerIdx + 1, msg = "layer " & $layerIdx & " output, chained input (chunked)")

  let normed = model.norm(blockInput + stream.get(blockInput))
  let logits = model.lmHead(normed)

  var flipCount = 0

  for pos in 0 ..< logits.size(1):
    assertArgMax(logits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "Qwen3.5-0.8B final logits position " & $pos)

when isMainModule:
  main()
