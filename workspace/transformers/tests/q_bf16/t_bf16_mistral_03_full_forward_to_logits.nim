# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the Mistral-7B-v0.1 stack, the 32
## all-sliding decoder blocks replayed layer by layer over the long
## residual stream, one theta roping every layer.
##
## Requires the local model at tests/hf_models/Mistral-7B-v0.1 (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_mistral_03_full_forward_to_logits.nim

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/src/layers,
  workspace/transformers/src/models/mistral {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(MistralModel)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "Mistral-7B-v0.1"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Mistral-7B-v0.1"

proc main(): bool =
  # The fixtures were recorded on mps, the replay resolves GPU over CPU
  # through testDevice(), Metal on this host, the same device class.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadMistralModelRaw(ModelPath, runDev)

  let meta = parseJson(readFile(
    FixtureDir / "layer-00.safetensor.metadata.json.zst").zstdDecompress(string))
  var tokenIds: seq[int64] = @[]
  for tj in items(meta{"input_tokens"}):
    tokenIds.add tj.getInt()
  let seqLen = tokenIds.len

  var (ctx, pool) = newKVContext(
    numLayers = model.config.numHiddenLayers,
    kvHeads = model.config.numKeyValueHeads,
    headDim = model.config.headDim, device = runDev)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, runDev)).unsqueeze(0)

  var hidden = model.embedTokens(tokenIds.toTensor().unsqueeze(0))
  var residual: Option[Tensor] = none(Tensor)

  for layerIdx in 0 ..< model.layers.len:
    # The boundary input is the embedding output at layer 0 and the hidden
    # plus residual sum at layers 1+.
    let nimInput =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    let inputKind = if layerIdx == 0: kElementwise else: kReduction
    # Composed depth = the layers composed so far, the boundary after N
    # blocks carries N composed layers, the same derivation the tier-03
    # dense suites carry.
    assertStats(nimInput, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "layer_input_seq", inputKind,
      depth = max(1, layerIdx), msg = "layer " & $layerIdx & " boundary input")

    ctx.kv_position = 0
    # Single-theta rope over every layer of the all-sliding shape.
    ctx.setRopeForPositions(model.rotary)

    let (output, newResidual) = model.layers[layerIdx].forward(ctx, hidden, residual)

    # The boundary sum feeds the next layer's recorded input, the last
    # layer's sum is the recorded layer_output_seq.
    let nimSum = output + newResidual
    if layerIdx < model.layers.len - 1:
      assertStats(nimSum, FixtureDir / ("layer-" & ($(layerIdx + 1)).align(2, '0') & ".safetensor.stats"), "layer_input_seq", kReduction,
        depth = layerIdx + 1, msg = "layer " & $layerIdx & " output plus residual, chained input")
    else:
      assertStats(nimSum, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "layer_output_seq", kReduction,
        depth = model.layers.len, msg = "layer " & $layerIdx & " output plus residual")

    hidden = output
    residual = some(newResidual)

  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm(hidden + finalResidual)
  let finalLogits = model.lmHead(finalNorm)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"

  var flipCount = 0

  # The recorded decisions frame carries one step per prompt position,
  # the argmax rows asserted over the whole prefill.
  for pos in 0 ..< finalLogits.size(1):
    assertArgMax(finalLogits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "Mistral-7B-v0.1 final logits position " & $pos)
  result = true

when isMainModule:
  runCppTest("mistral full forward to logits", main)
