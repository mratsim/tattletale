# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the Qwen3.6-35B-A3B MoE stack, the 40
## decoder blocks replayed layer by layer over the long residual stream.
## Requires the local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_03_full_forward_to_logits.nim

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/ffn {.all.},
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Qwen35MoeModel)
privateAccess(DecoderLayer[GatedDeltaNet, GatedBlockSparseFFN, RmsNormOne])
privateAccess(DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedBlockSparseFFN, RmsNormOne])
privateAccess(GatedBlockSparseFFN)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "Qwen3.6-35B-A3B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"

proc main() =
  # The recorded chain contract of this fixture family is the reference
  # device replay, the fixtures were recorded on cpu. A cross-device run
  # names the tolerance class and skips, no device budget applies here.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadQwen35MoeModelRaw(ModelPath, F.kCPU)

  let meta = parseJson(readFile(
    FixtureDir / "layer-00.safetensor.metadata.json.zst").zstdDecompress(string))
  var tokenIds: seq[int64] = @[]
  for tj in items(meta{"input_tokens"}):
    tokenIds.add tj.getInt()
  let seqLen = tokenIds.len

  var (ctx, pool) = newKVContext(
    numLayers = model.config.numHiddenLayers,
    kvHeads = model.config.numKeyValueHeads,
    headDim = model.config.headDim)
  ctx.position_ids = F.arange(seqLen.int64, F.tensorOptions(F.kInt64, F.kCPU))
  ctx.setRopeForPositions(model.rotary)

  # The router parity replay visits a second context layer by layer, each
  # mixer layer writes its own state slots, so the replay sees the same
  # fresh state the model forward saw.
  var (checkCtx, checkPool) = newKVContext(
    numLayers = model.config.numHiddenLayers,
    kvHeads = model.config.numKeyValueHeads,
    headDim = model.config.headDim)
  checkCtx.position_ids = ctx.position_ids
  checkCtx.setRopeForPositions(model.rotary)

  var h = model.embedTokens(tokenIds.toTensor().unsqueeze(0))
  var blockInput = h
  var stream: Option[Tensor]

  for layerIdx in 0 ..< model.layers.len:
    let layerIn = h
    let pair = model.layers[layerIdx].forward(ctx, blockInput, stream)
    blockInput = pair[0]
    stream = some(pair[1])
    h = pair[1] + pair[0]

    # Router parity against the recorded weights, the mixer output is
    # replayed through the layer's own components on the parity context,
    # the renormalized top-k weights compare under the elementwise model.
    let gdnLayer = model.layers[layerIdx].to(
      DecoderLayer[GatedDeltaNet, GatedBlockSparseFFN, RmsNormOne])
    let attnLayer = model.layers[layerIdx].to(
      DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedBlockSparseFFN, RmsNormOne])
    let inputNorm =
      if gdnLayer != nil: gdnLayer.input_layernorm
      else: attnLayer.input_layernorm
    let normedIn = inputNorm.forward(layerIn)
    let mixerOut =
      if gdnLayer != nil:
        gdnLayer.sequence_mixer(checkCtx, normedIn)
      else:
        attnLayer.sequence_mixer(checkCtx, normedIn)
    let h1 = layerIn + mixerOut
    let postNorm =
      if gdnLayer != nil: gdnLayer.post_attention_layernorm
      else: attnLayer.post_attention_layernorm
    let normedH1 = postNorm.forward(h1)
    let ffn =
      if gdnLayer != nil: gdnLayer.hidden_mixer
      else: attnLayer.hidden_mixer
    let (gotIndices, gotWeights) = routeToExperts(
      normedH1.reshape(seqLen, model.config.hiddenSize),
      ffn.routerWeight, ffn.numExpertsPerTok)
    let _ = gotIndices
    assertStats(gotWeights, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "routing_weights", kElementwise, msg = "layer " & $layerIdx & " routing weights")

  let normed = model.norm(blockInput + stream.get(blockInput))
  let logits = model.lmHead(normed)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"
  var flipCount = 0

  for pos in 0 ..< logits.size(1):
    assertArgMax(logits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "Qwen3.6-35B-A3B final logits position " & $pos)

when isMainModule:
  main()
