# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the gemma-3-270m-it stack, the 18
## sandwich decoder blocks replayed layer by layer, dual-theta rope
## re-selected per layer kind.
##
## Requires the local model at tests/hf_models/gemma-3-270m-it (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_gemma3270m_03_full_forward_to_logits.nim

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/src/layers,
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/models/gemma3 {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Gemma3Model)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "gemma-3-270m-it"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "gemma-3-270m-it"

proc main(): bool =
  # The fixtures were recorded on mps, the replay resolves GPU over CPU
  # through testDevice(), Metal on this host, the same device class.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadGemma3ModelRaw(ModelPath, runDev)

  let meta = parseJson(readFile(
    FixtureDir / "layer-00.safetensor.metadata.json.zst").zstdDecompress(string))
  var tokenIds: seq[int64] = @[]
  for tj in items(meta{"input_tokens"}):
    tokenIds.add tj.getInt()
  let seqLen = tokenIds.len
  # The prefill asserts the next-token positions, prompt_len - 1 rows,
  # one argmax decision per recorded frame step.
  let decodedPositions = seqLen - 1

  var (ctx, pool) = newKVContext(
    numLayers = model.config.num_hidden_layers,
    kvHeads = model.config.num_key_value_heads,
    headDim = model.config.head_dim, device = runDev)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, runDev)).unsqueeze(0)

  # The embedding output carries the Gemma3TextScaledWordEmbedding scale,
  # the layer-0 recorded input.
  var hidden = model.embedTokens(tokenIds.toTensor().unsqueeze(0)) * model.embedScale
  var residual: Option[Tensor] = none(Tensor)

  for layerIdx in 0 ..< model.layers.len:
    # The boundary input is the embedding output at layer 0 and the hidden
    # plus residual sum at layers 1+.
    let nimInput =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    # Layer 0's input is the embedding output, an elementwise chain, while
    # the layers 1+ inputs are the previous block output plus
    # its residual, a row fed by the whole composing chain, the reduction class.
    let inputKind = if layerIdx == 0: kElementwise else: kReduction
    assertStats(nimInput, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "layer_input_seq", inputKind, msg = "layer " & $layerIdx & " boundary input")

    ctx.kv_position = 0
    # Dual-theta rope, each layer kind ropes with its own table.
    ctx.setRopeForPositions(
      if model.config.layerKinds[layerIdx] == alkSlidingAttention:
        model.rotarySliding
      else:
        model.rotaryFull)

    let (output, newResidual) = model.layers[layerIdx].forward(ctx, hidden, residual)

    # The boundary sum feeds the next layer's recorded input, the last
    # layer's sum is the recorded layer_output_seq.
    let nimSum = output + newResidual
    if layerIdx < model.layers.len - 1:
      assertStats(nimSum, FixtureDir / ("layer-" & ($(layerIdx + 1)).align(2, '0') & ".safetensor.stats"), "layer_input_seq", kReduction, msg = "layer " & $layerIdx & " output plus residual, chained input")
    else:
      assertStats(nimSum, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "layer_output_seq", kReduction, msg = "layer " & $layerIdx & " output plus residual")

    hidden = output
    residual = some(newResidual)

  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm(hidden + finalResidual)
  let finalLogits = model.lmHead(finalNorm)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"

  var flipCount = 0

  for pos in 0 ..< decodedPositions:
    assertArgMax(finalLogits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "gemma-3-270m-it final logits position " & $pos)
  result = true

when isMainModule:
  runCppTest("gemma3 270m full forward to logits", main)
