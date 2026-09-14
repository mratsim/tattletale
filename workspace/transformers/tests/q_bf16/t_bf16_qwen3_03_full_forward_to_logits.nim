# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the Qwen3-0.6B dense stack, the 28
## decoder blocks replayed layer by layer over the long residual stream.
## Requires the local model at tests/hf_models/Qwen3-0.6B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen3_03_full_forward_to_logits.nim

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/zstd/zstd_highlevel,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"


proc main() =
  # The recorded chain contract of this fixture family is the reference
  # device replay, the fixtures were recorded on cpu. A cross-device run
  # names the tolerance class and skips, no device budget applies here.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadQwen3ModelRaw(ModelPath, F.kCPU)

  var ctx = InferenceContext.init(
    num_layers = model.config.num_hidden_layers,
    batch_size = 1, kv_heads = model.config.num_key_value_heads,
    max_seq = 4096, head_dim = model.config.head_dim)

  let pool = PagePool.init(
    64, num_layers = model.config.num_hidden_layers,
    kv_heads = model.config.num_key_value_heads,
    head_dim = model.config.head_dim,
    dtype = F.kBFloat16, device = F.kCPU)
  let numPages = ceilDiv(4096, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())

  # Input tokens are the tokenizer ids of "Hello, how are you?"
  let inputIds = @[9707'i64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)

  var hidden = model.embedTokens(inputIds)
  var residual: Option[Tensor] = none(Tensor)

  for layerIdx in 0 ..< model.layers.len:
    let hfFixture = setupLayerFixture(FixtureDir, layerIdx)
    let layer = model.layers[layerIdx]

    # The boundary input is the embedding output at layer 0 and the hidden
    # plus residual sum at layers 1+.
    let nimInput =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    assertStats(nimInput, FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats"), "layer_input", kElementwise, msg = "layer " & $layerIdx & " boundary input")

    ctx.kv_position = 0
    ctx.position_ids = F.arange(
      hidden.size(1).int64, F.tensorOptions(F.kInt64, F.kCPU)).unsqueeze(0)
    ctx.setRopeForPositions(layer.sequence_mixer.rotary)

    # Forward through layer (long residual stream pattern)
    let (output, newResidual) = layer(ctx, hidden, residual)

    # The boundary sum feeds the next layer's recorded input.
    let nimSum = output + newResidual
    if layerIdx < model.layers.len - 1:
      let nextFixture = setupLayerFixture(FixtureDir, layerIdx + 1)
      assertStats(nimSum, FixtureDir / ("layer-" & ($(layerIdx + 1)).align(2, '0') & ".safetensor.stats"), "layer_input", kElementwise, msg = "layer " & $layerIdx & " output plus residual, chained input")

    hidden = output
    residual = some(newResidual)

  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm(hidden + finalResidual)
  let finalLogits = model.lmHead(finalNorm)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"

  var flipCount = 0

  for pos in 0 ..< finalLogits.size(1):
    assertArgMax(finalLogits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "Qwen3-0.6B final logits position " & $pos)

when isMainModule:
  main()
