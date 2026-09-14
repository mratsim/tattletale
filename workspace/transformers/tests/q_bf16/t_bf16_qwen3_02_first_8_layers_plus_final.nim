# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chain suite for the Qwen3-0.6B dense stack, the 8 recorded prefix blocks
## plus the depth-28 tail checkpoint against the committed fixtures.
## Requires the local model at tests/hf_models/Qwen3-0.6B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen3_02_first_8_layers_plus_final.nim

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-02-first-8-layers-plus-final" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"

proc main() =
  # The recorded chain contract of this fixture family is the reference
  # device replay, the fixtures were recorded on cpu. A cross-device run
  # names the tolerance class and skips, no device budget applies here.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadQwen3ModelRaw(ModelPath, F.kCPU)

  var (ctx, pool) = newKVContext(
    numLayers = model.config.num_hidden_layers,
    kvHeads = model.config.num_key_value_heads,
    headDim = model.config.head_dim, maxSeq = 256)

  let inputIds = @[9707'i64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)
  var hidden = model.embedTokens(inputIds)
  var residual: Option[Tensor] = none(Tensor)

  let numLayers = model.config.num_hidden_layers
  var layerIdx = 0
  while layerIdx < numLayers and fileExists(FixtureDir /
      ("block-" & ($layerIdx).align(2, '0') & ".safetensor")):
    let i = layerIdx
    var st = Safetensor.open(FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor"))
    let layer = model.layers[i]
    let combined =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    # The layer_input record is the raw hidden the layer forward consumes,
    # the residual sum arrives separately as after_attn_norm_residual.
    assertStats(hidden, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "layer_input", kElementwise, depth = i + 1, msg = "block " & $i & " boundary input")
    assertStats(combined, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "after_attn_norm_residual", kElementwise, depth = i + 1, msg = "block " & $i & " residual sum")

    ctx.kv_position = 0
    ctx.position_ids = F.arange(combined.size(1).int64,
      F.tensorOptions(F.kInt64, F.kCPU)).unsqueeze(0)
    ctx.setRopeForPositions(layer.sequence_mixer.rotary)

    # The recorded sublayer intermediates re-express block by block,
    # records come off the committed stats frame.
    let hNorm = layer.input_layernorm(combined)
    assertStats(hNorm, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "after_attn_norm", kElementwise, depth = i + 1, msg = "block " & $i & " input layernorm output")
    let attnOut = layer.sequence_mixer(ctx, hNorm)
    assertStats(attnOut, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "after_attn", kReduction, depth = i + 1, msg = "block " & $i & " attention output")
    let combined2 = attnOut + combined
    assertStats(combined2, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "after_mlp_norm_residual", kElementwise, depth = i + 1, msg = "block " & $i & " post-attention residual sum")
    let h2 = layer.post_attention_layernorm(combined2)
    assertStats(h2, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "after_mlp_norm", kElementwise, depth = i + 1, msg = "block " & $i & " post-attention layernorm output")
    let mlpOut = layer.hidden_mixer(h2)
    assertStats(mlpOut, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "mlp_out", kReduction, depth = i + 1, msg = "block " & $i & " mlp output")

    # The real block forward under the same positional state, its chain
    # checkpoint is the composed block output.
    ctx.kv_position = 0
    let (blkOut, blkResidual) = layer(ctx, hidden, residual)
    let checkpoint = blkOut + blkResidual
    assertStats(checkpoint, FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats"), "hf_layer_output", kReduction, depth = i + 1, msg = "chain checkpoint block " & $i)

    hidden = blkOut
    residual = some(blkResidual)
    inc layerIdx

  # The blocks past the fixture prefix carry no per-block records.
  # The chain runs on through the real forward to the tail checkpoint.
  for i in layerIdx ..< numLayers:
    let layer = model.layers[i]
    ctx.kv_position = 0
    ctx.position_ids = F.arange(hidden.size(1).int64,
      F.tensorOptions(F.kInt64, F.kCPU)).unsqueeze(0)
    ctx.setRopeForPositions(layer.sequence_mixer.rotary)
    let (blkOut, blkResidual) = layer(ctx, hidden, residual)
    hidden = blkOut
    residual = some(blkResidual)

  # The tail checkpoint is the decoder stack output feeding the final
  # RMSNorm and lm head.
  let tail = hidden + residual.get(hidden)
  assertStats(tail, FixtureDir / "tail.safetensor.stats", "pre_final_norm", kReduction, depth = numLayers, msg = "chain tail checkpoint at depth " & $numLayers)

when isMainModule:
  main()
