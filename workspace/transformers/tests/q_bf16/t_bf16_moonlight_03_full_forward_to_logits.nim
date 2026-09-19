# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the Moonlight-16B-A3B MoE stack.
## All 27 decoder blocks replay layer by layer over the long residual stream.
## Requires the local model at tests/hf_models/Moonlight-16B-A3B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_moonlight_03_full_forward_to_logits.nim

import
  std/importutils,
  std/math,
  std/options,
  std/os,
  std/strutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/attn_ssm/multi_head_latent_attention,
  workspace/transformers/src/models/moonlight {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils

{.experimental: "callOperator".}

privateAccess(MoonlightModel)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "Moonlight-16B-A3B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Moonlight-16B-A3B"

proc main(): bool =
  # The recorded chain contract of this fixture family is the reference
  # device replay, the fixtures were recorded on cpu. A cross-device run
  # names the tolerance class and skips, no device budget applies here.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadMoonlightModelRaw(ModelPath, F.kCPU)

  let meta = parseJson(readFile(
    FixtureDir / "layer-00.safetensor.metadata.json.zst").zstdDecompress(string))
  var tokenIds: seq[int64] = @[]
  for tj in items(meta{"input_tokens"}):
    tokenIds.add tj.getInt()
  let seqLen = tokenIds.len

  # MLA pool geometry of the recorded chain:
  # - the K buffer runs the compressed latent width, single head
  # - the V buffer runs the kpe plane width, single head
  # Pool pages stay borrowed while the context runs.
  let numPages = ceilDiv(seqLen, TokensPerPage)
  var ctx = InferenceContext.init(
    num_layers = model.config.numHiddenLayers, batch_size = 1, kv_heads = 1,
    max_seq = seqLen, head_dim = model.config.kvLoraRank)
  let pool = PagePool.init(
    numPages, num_layers = model.config.numHiddenLayers,
    k_kv_heads = 1, k_head_dim = model.config.kvLoraRank,
    v_kv_heads = 1, v_head_dim = model.config.qkRopeHeadDim,
    dtype = F.kBFloat16, device = F.kCPU)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())
  ctx.position_ids = F.arange(seqLen.int64, F.tensorOptions(F.kInt64, F.kCPU))
  ctx.setMlaRopeForPositions(model.rotary)

  # Block forward over the long residual stream:
  # the mixer input (blockInput) and the residual stream stay
  # separate values, their sum enters the next block.
  # Boundary checkpoints compare the block-entry hidden.
  # The recorded layer_input_seq rows carry the same local-residual quantity.
  var blockInput = model.embedTokens(tokenIds.toTensor().unsqueeze(0))
  var stream: Option[Tensor]

  for layerIdx in 0 ..< model.layers.len:
    let statsPath = FixtureDir /
      ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats")

    # Boundary checkpoint entering this block:
    # - layer 0 carries the embedding output, pure data movement
    # - later layers carry the previous block output plus residual
    # Composed-chain depth grows with the layer index:
    # each routed block adds one reduction class of drift per stage.
    if layerIdx == 0:
      assertStats(blockInput, statsPath, "layer_input_seq", kElementwise,
        msg = "layer 0 boundary input")
    else:
      # The composed depth counts the measured per-op classes of the 01
      # suites. Each sequence mixer composes three accumulation steps,
      # each routed block output composes two past the grouped_mm record.
      #
      # The leading dense block output composes one, so the boundary
      # after N blocks carries depth 5N - 1.
      let composedDepth = 5 * layerIdx - 1
      assertStats(blockInput + stream.get(blockInput), statsPath,
        "layer_input_seq", kReduction, depth = composedDepth,
        msg = "layer " & $layerIdx & " boundary input")

    # This suite asserts no routing weights. Recorded fixture anchors
    # carry no router_input payload, so a single-op router replay is
    # not possible. Eager-vs-grouped_mm mixer comparison falls outside
    # the f32 tolerance grid, and the routed output class stays covered
    # by the 01 moe suite.

    let layerOut = model.layers[layerIdx].forward(ctx, blockInput, stream)
    blockInput = layerOut[0]
    stream = some(layerOut[1])

    # Final boundary at the decoder stack output, feeding the final RMSNorm:
    # recorded on the last layer fixture, the composed depth 5N - 1 counts
    # the whole chain.
    if layerIdx == model.layers.len - 1:
      assertStats(blockInput + stream.get(blockInput), statsPath,
        "layer_output_seq", kReduction, depth = 5 * model.layers.len - 1,
        msg = "decoder stack output boundary")

  let normed = model.norm.forward(blockInput + stream.get(blockInput))
  let logits = model.lmHead.forward(normed)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"
  var flipCount = 0

  # The logits compose the stack boundary chain plus the final norm plus
  # the head projection, two more stages past depth 5N - 1.
  for pos in 0 ..< logits.size(1):
    assertArgMax(logits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = 5 * model.layers.len + 1,
      msg = "Moonlight-16B-A3B final logits position " & $pos)
  result = true

when isMainModule:
  runCppTest("moonlight full forward to logits", main)
