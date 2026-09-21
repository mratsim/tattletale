# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the gemma-4-E2B-it stack, the 35 sandwich decoder
## blocks replayed layer by layer, the per-layer embedding tail and the layer scalar between the blocks.
##
## Dual-theta rope re-selects per layer kind, the kv-sharing layers gather their keys
## and values through the page pool, the pool slots carry the widest kv width.
##
## Requires the local model at tests/hf_models/gemma-4-E2B-it (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_gemma4e2b_03_full_forward_to_logits.nim

import
  std/math,
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/positron,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/src/layers,
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/models/gemma4e2b {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Gemma4E2BModel)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "gemma-4-E2B-it"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "gemma-4-E2B-it"

proc main(): bool =
  # The fixtures were recorded on mps, the replay resolves GPU over CPU
  # through testDevice(), Metal on this host, the same device class.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadGemma4E2BModelRaw(ModelPath, runDev)

  let meta = parseJson(readFile(
    FixtureDir / "layer-00.safetensor.metadata.json.zst").zstdDecompress(string))
  var tokenIds: seq[int64] = @[]
  for tj in items(meta{"input_tokens"}):
    tokenIds.add tj.getInt()
  let seqLen = tokenIds.len
  # The prefill asserts the next-token positions, prompt_len - 1 rows,
  # one argmax decision per recorded frame step.
  let decodedPositions = seqLen - 1

  # The pool slots carry the widest kv width, the full_attention layers
  # run head_dim 512, the sliding layers 256 and narrow their views.
  var (ctx, pool) = newKVContext(
    numLayers = model.config.num_hidden_layers,
    kvHeads = model.config.num_key_value_heads,
    headDim = model.config.global_head_dim, device = runDev)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, runDev)).unsqueeze(0)

  let batch = 1
  let numLayers = model.config.num_hidden_layers
  let pleDim = model.config.hidden_size_per_layer_input

  # The embedding output carries the Gemma4TextScaledWordEmbedding scale,
  # the layer-0 recorded input.
  var x = model.embedTokens(tokenIds.toTensor().unsqueeze(0)) * model.embedScale
  assertStats(x, FixtureDir / "layer-00.safetensor.stats", "layer_input_seq",
    kElementwise, msg = "layer 0 boundary input")

  # Per-Layer Embeddings (token-identity and context-aware halves),
  # combined once and sliced per layer in the loop.
  let pleTok = model.pleEmbedTokens(tokenIds.toTensor().unsqueeze(0)).reshape(
    [batch, seqLen, numLayers, pleDim]) * model.pleTokScale
  let pleProjRaw = model.perLayerModelProjection.forward(x) *
    Scalar(pow(model.config.hidden_size.float64, -0.5))
  let pleProj = model.perLayerProjectionNorm.forward(
    pleProjRaw.reshape([batch, seqLen, numLayers, pleDim]))
  let combined = (pleProj + pleTok) * Scalar(pow(2.0, -0.5))

  for layerIdx in 0 ..< model.layers.len:
    ctx.kv_position = 0
    # Dual-theta rope, each layer kind ropes with its own table.
    ctx.setRopeForPositions(
      if model.config.layerKinds[layerIdx] == alkSlidingAttention:
        model.rotarySliding
      else:
        model.rotaryFull)

    # The sandwich block consumes the previous layer's full output, no
    # residual crosses a layer boundary.
    let (post, h1) = model.layers[layerIdx].forward(ctx, x, none(Tensor))
    var h = h1 + post

    # Per-layer-embedding tail in the reference order, the tanh-activated
    # input rows multiply the PLE product, the projection and norm follow,
    # then the residual add and the layer scalar.
    let pleInput = combined[_, _, layerIdx, _]
    let gated = gelu_tanh(model.perLayerInputGate[layerIdx].forward(h)) * pleInput
    h = h + model.postPleNorm[layerIdx].forward(
      model.perLayerProjection[layerIdx].forward(gated))
    x = h * model.layerScalars[layerIdx]

    # The scaled layer output feeds the next block directly, the last
    # layer's output is the recorded layer_output_seq.
    if layerIdx < numLayers - 1:
      assertStats(x, FixtureDir / ("layer-" & ($(layerIdx + 1)).align(2, '0') & ".safetensor.stats"), "layer_input_seq", kReduction, depth = 2, msg = "layer " & $layerIdx & " scaled output, chained input")
    else:
      assertStats(x, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "layer_output_seq", kReduction, depth = 2, msg = "layer " & $layerIdx & " scaled output")

  let normed = model.norm.forward(x)
  var finalLogits = model.lmHead.forward(normed)
  if model.config.final_logit_softcapping > 0.0:
    # The reference softcap, divide by the cap, tanh, multiply back.
    let cap = Scalar(model.config.final_logit_softcapping)
    finalLogits = (finalLogits / cap).tanh() * cap

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"

  var flipCount = 0

  for pos in 0 ..< decodedPositions:
    assertArgMax(finalLogits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "gemma-4-E2B-it final logits position " & $pos)
  result = true

when isMainModule:
  runCppTest("gemma-4-E2B full forward to logits", main)
