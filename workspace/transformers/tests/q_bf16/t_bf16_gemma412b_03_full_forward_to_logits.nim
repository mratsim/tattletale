# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the gemma-4-12B-it stack, the 48 sandwich decoder blocks replayed layer by layer.
##
## Dual-theta rope re-selects per layer kind, the KV-tied full layers
## derive their value rows from the shared k projection, the layer
## scalar scales the whole layer output between the blocks.
##
## Requires the local model at tests/hf_models/gemma-4-12B-it (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_gemma412b_03_full_forward_to_logits.nim

import
  std/math,
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/positron,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/src/layers,
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/models/gemma4_12b {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Gemma4Text12BModel)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "gemma-4-12B-it"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "gemma-4-12B-it"

proc main(): bool =
  # The fixtures were recorded on mps, the replay resolves GPU over CPU
  # through testDevice(), Metal on this host, the same device class.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadGemma4Text12BModelRaw(ModelPath, runDev)

  let meta = parseJson(zstdDecompress(readFile(
    FixtureDir / "layer-00.safetensor.metadata.json.zst"), string))
  var tokenIds: seq[int64] = @[]
  for tj in items(meta{"input_tokens"}):
    tokenIds.add tj.getInt()
  let seqLen = tokenIds.len

  # The pool slots carry the widest kv geometry, the full_attention layers
  # run head_dim 512 over 1 kv head, the sliding layers 256 over 8 kv
  # heads and narrow their views.
  var (ctx, pool) = newKVContext(
    numLayers = model.config.num_hidden_layers,
    kvHeads = model.config.num_key_value_heads,
    headDim = model.config.global_head_dim, device = runDev)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, runDev)).unsqueeze(0)

  # The embedding output carries the Gemma4TextScaledWordEmbedding scale,
  # the layer-0 recorded input.
  var x = model.embedTokens(tokenIds.toTensor().unsqueeze(0)) * model.embedScale
  assertStats(x, FixtureDir / "layer-00.safetensor.stats", "layer_input_seq",
    kElementwise, msg = "layer 0 boundary input")

  for layerIdx in 0 ..< model.layers.len:
    ctx.kv_position = 0
    # Dual-theta rope, each layer kind ropes with its own table.
    ctx.setRopeForPositions(
      if model.config.layerKinds[layerIdx] == alkSlidingAttention:
        model.rotarySliding
      else:
        model.rotaryFull)

    # The sandwich block consumes the previous layer's full output, no
    # residual crosses a layer boundary. The layer scalar scales the whole block output.
    let (post, h1) = model.layers[layerIdx].forward(ctx, x, none(Tensor))
    x = (h1 + post) * model.layerScalars[layerIdx]

    # The scaled layer output feeds the next block directly, the last
    # layer's output is the recorded layer_output_seq.
    if layerIdx < model.config.num_hidden_layers - 1:
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

  # One argmax decision per recorded frame step, the full prompt seqLen.
  for pos in 0 ..< seqLen:
    assertArgMax(finalLogits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "gemma-4-12B-it final logits position " & $pos)
  result = true

when isMainModule:
  runCppTest("gemma-4-12B full forward to logits", main)
