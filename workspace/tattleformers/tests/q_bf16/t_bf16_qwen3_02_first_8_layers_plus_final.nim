# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chain suite for the Qwen3-0.6B dense stack, the 8 recorded prefix blocks
## plus the depth-28 tail checkpoint against the committed fixtures.
##
## Test that the ported block chain matches the recorded stats frames
## on Metal with the per-block assertStats instruments at depth i+1,
## plus the tail checkpoint at depth 28.
##
## Recorded frames (Python testgen, consumed in place):
##   workspace/transformers/tests/fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B
## Local model: /Users/pi/Documents/Programming/MODELS/Qwen3-0.6B
## (TTT_QWEN3_PATH env override).
##
## The fixture guard FAILS the run when any of the 8 recorded block frames or the tail frame is missing.
## A missing fixture family never degrades into silently skipped asserts.
##
## Readback protocol (harness header): the sublayer chain and the real
## block forward build one block's graph. ONE eval per block boundary
## walks the graph, then every assertStats readback runs on the path.
##
## Run from the worktree root:
##   nim c -r -d:release --stackTrace:on --lineTrace:on --lineDir:on \
##     --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM09suite02 --outdir:build/wipM09suite02 \
##     workspace/tattleformers/tests/q_bf16/t_bf16_qwen3_02_first_8_layers_plus_final.nim

import
  std/os,
  std/options,
  std/importutils,
  std/sequtils,
  std/strutils,
  workspace/tensorcery/src/tensorcery,
  workspace/tattleformers/src/layers/embedding,
  workspace/tattleformers/src/layers/linear,
  workspace/tattleformers/src/layers/norm,
  workspace/tattleformers/src/layers/rope,
  workspace/tattleformers/src/layers/ffn,
  workspace/tattleformers/src/layers/lmhead,
  workspace/tattleformers/src/layers/attention,
  workspace/tattleformers/src/layers/decoder_layers,
  workspace/tattleformers/src/stateful/orchestrator,
  workspace/tattleformers/src/stateful/inference_context,
  workspace/tattleformers/src/models,
  workspace/tattleformers/src/models/qwen3 {.all.},
  workspace/tattleformers/tests/harness/harness

const
  FixtureDir = currentSourcePath().parentDir().parentDir().parentDir().parentDir() /
    "transformers" / "tests" / "fixtures" /
    "bf16-02-first-8-layers-plus-final" / "Qwen3-0.6B"
  ModelPathDefault = "/Users/pi/Documents/Programming/MODELS/Qwen3-0.6B"
  MaxSeq = 256
    ## Pool sizing override, the recorded chain spans 6 positions:
    ## the config's 4096-token pool would eager-allocate
    ## a multi-gigabyte Metal buffer (the models.nim generate override rationale).
  RecordedPrefixBlocks = 8
    ## Block count the committed fixture family records: block-00 to block-07 plus the tail frame, verified by the fixture guard below.

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])

proc blockStatsPath(i: int): string =
  ## Sidecar stats frame of one recorded block, block-NN aligned to 2 digits.
  FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats")

proc chainInputKind(i: int): RoundingErrorSourceKind =
  ## Error-model class of the assert kinds, per the harness kind table:
  ## a tensor whose producing chain reduces (attention, mixer, norm) replays under kReduction.
  ## The block-0 chain input is a pure embedding gather, a reduction-free sequence.
  if i == 0:
    kElementwise
  else:
    kReduction

proc main(modelPath: string): bool =
  # Fixture guard: every recorded frame must exist. A missing frame fails the run, no assert silently skips.
  block:
    var missing: seq[string]
    for i in 0 ..< RecordedPrefixBlocks:
      let p = blockStatsPath(i) & ".json.zst"
      if not fileExists(p):
        missing.add p
    let tailPath = FixtureDir / "tail.safetensor.stats.json.zst"
    if not fileExists(tailPath):
      missing.add tailPath
    doAssert missing.len == 0,
      "recorded fixtures missing under " & FixtureDir & ": " & missing.join(", ")

  let model = loadModel(modelPath)
  let cfg = model.config

  let numPoolPages = computeNumPages(MaxSeq, concurrentRequests = 1)
  var orc = Orchestrator.init(cfg.num_hidden_layers,
    cfg.num_key_value_heads, MaxSeq, cfg.head_dim, numPoolPages)
  defer: orc.endSequence()

  let inputIds = @[9707, 11, 1246, 525, 498, 30]
  orc.startSequence(inputIds.mapIt(uint32(it)))
  var ctx = orc.getInferenceContextMut()
  ctx.setRopeForPositions(model.rotary)

  var hidden = model.embedTokens.forward(inputIds, @[1, inputIds.len])
  var residual = none(Tensor)

  let numLayers = cfg.num_hidden_layers
  for i in 0 ..< RecordedPrefixBlocks:
    let layer = model.layers[i]
    let combined =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    # layer_input records the raw hidden the layer forward consumes, the residual sum is recorded separately as after_attn_norm_residual.
    let hNorm = layer.input_layernorm.forward(combined)
    let attnOut = layer.sequence_mixer.forward(ctx, hNorm)
    let combined2 = attnOut + combined
    let h2 = layer.post_attention_layernorm.forward(combined2)
    let mlpOut = layer.hidden_mixer.forward(h2)

    # Real block forward under the same positional state, the chain checkpoint being the composed block output.
    let (blkOut, blkResidual) = layer.forward(ctx, hidden, residual)
    let checkpoint = blkOut + blkResidual

    # One eval per block boundary walks the sublayer chain and the real
    # forward in one batch, the asserts read back on the evaluated path.
    eval(mlpOut, checkpoint)

    assertStats(hidden, blockStatsPath(i), "layer_input", chainInputKind(i), depth = i + 1, msg = "block " & $i & " boundary input")
    assertStats(combined, blockStatsPath(i), "after_attn_norm_residual", chainInputKind(i), depth = i + 1, msg = "block " & $i & " residual sum")
    # Recorded sublayer intermediates re-express block by block, records come off the committed stats frame.
    assertStats(hNorm, blockStatsPath(i), "after_attn_norm", kReduction, depth = i + 1, msg = "block " & $i & " input layernorm output")
    assertStats(attnOut, blockStatsPath(i), "after_attn", kReduction, depth = i + 1, msg = "block " & $i & " attention output")
    assertStats(combined2, blockStatsPath(i), "after_mlp_norm_residual", kReduction, depth = i + 1, msg = "block " & $i & " post-attention residual sum")
    assertStats(h2, blockStatsPath(i), "after_mlp_norm", kReduction, depth = i + 1, msg = "block " & $i & " post-attention layernorm output")
    assertStats(mlpOut, blockStatsPath(i), "mlp_out", kReduction, depth = i + 1, msg = "block " & $i & " mlp output")
    assertStats(checkpoint, blockStatsPath(i), "hf_layer_output", kReduction, depth = i + 1, msg = "chain checkpoint block " & $i)

    hidden = blkOut
    residual = some(blkResidual)

  # Blocks after the fixture prefix carry no per-block records, the chain runs on through the real forward to the tail checkpoint.
  for i in RecordedPrefixBlocks ..< numLayers:
    let layer = model.layers[i]
    let (blkOut, blkResidual) = layer.forward(ctx, hidden, residual)
    eval(blkOut, blkResidual)
    hidden = blkOut
    residual = some(blkResidual)

  # Tail checkpoint: the decoder stack output feeding the final
  # RMSNorm and lm head.
  let tail = hidden + residual.get(hidden)
  eval(tail)
  assertStats(tail, FixtureDir / "tail.safetensor.stats", "pre_final_norm", kReduction, depth = numLayers, msg = "chain tail checkpoint at depth " & $numLayers)
  result = true

when isMainModule:
  let modelPath = getEnv("TTT_QWEN3_PATH", ModelPathDefault)
  if main(modelPath):
    stderr.writeLine "qwen3 first 8 layers plus final (Metal): all asserts green"
