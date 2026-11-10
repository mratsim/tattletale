# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the Qwen3-0.6B dense stack:
## the 28 decoder blocks replayed layer by layer over the long residual stream.
##
## Test that the full 28-layer forward matches the recorded frames
## on Metal: per-layer boundary assertStats, the chained-input asserts,
## and assertArgMax at the 6 recorded logits positions with depth 28.
##
## Recorded frames (Python testgen, consumed in place):
##   workspace/transformers/tests/fixtures/bf16-03-full-forward-to-logits/Qwen3-0.6B
## Local model: /Users/pi/Documents/Programming/MODELS/Qwen3-0.6B
## (TTT_QWEN3_PATH env override).
##
## Readback protocol (harness header): each layer builds the boundary add
## and the layer graph, ONE eval per layer walks it, then the boundary
## asserts read back on the evaluated path. The final logits eval once
## before the six assertArgMax readbacks.
##
## Run from the worktree root:
##   nim c -r -d:release --stackTrace:on --lineTrace:on --lineDir:on \
##     --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM09suite03 --outdir:build/wipM09suite03 \
##     workspace/tattleformers/tests/q_bf16/t_bf16_qwen3_03_full_forward_to_logits.nim

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  std/sequtils,
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

privateAccess(Qwen3Model)

const
  FixtureDir = currentSourcePath().parentDir().parentDir().parentDir().parentDir() /
    "transformers" / "tests" / "fixtures" /
    "bf16-03-full-forward-to-logits" / "Qwen3-0.6B"
  ModelPathDefault = "/Users/pi/Documents/Programming/MODELS/Qwen3-0.6B"
  MaxSeq = 256
    ## Pool sizing override, the recorded chain spans 6 positions:
    ## the config's 4096-token pool would eager-allocate
    ## a multi-gigabyte Metal buffer (the models.nim generate override rationale).

# Error-model classes of the assert kinds, measured against the recorded
# tensors on this stack (suite 02 block 0):
# - the layer 0 boundary input is the embedding gather, no reduction,
#   its replay drift is 0 over the 2-step elementwise band, kElementwise
# - every later boundary input embeds the attention and FFN reductions
#   of the previous layer, small-magnitude elements churn
#   histogram buckets under drift, so the composed chains map to kReduction
# - the reference kElementwise choice for those chains was invisible
#   on a zero-drift stack

proc layerStatsPath(layerIdx: int): string =
  ## Sidecar stats frame of one recorded layer, layer-NN aligned to 2 digits.
  FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats")

proc main(modelPath: string): bool =
  doAssert fileExists(FixtureDir / "final_logits.decisions.json.zst"),
    "recorded fixtures missing under " & FixtureDir

  let model = loadModel(modelPath)
  let cfg = model.config

  let numPoolPages = computeNumPages(MaxSeq, concurrentRequests = 1)
  var orc = Orchestrator.init(cfg.num_hidden_layers,
    cfg.num_key_value_heads, MaxSeq, cfg.head_dim, numPoolPages)
  defer: orc.endSequence()

  # Input tokens are the tokenizer ids of "Hello, how are you?"
  let inputIds = @[9707, 11, 1246, 525, 498, 30]
  orc.startSequence(inputIds.mapIt(uint32(it)))
  var ctx = orc.getInferenceContextMut()
  ctx.setRopeForPositions(model.rotary)

  var hidden = model.embedTokens.forward(inputIds, @[1, inputIds.len])
  var residual = none(Tensor)

  for layerIdx in 0 ..< model.layers.len:
    let layer = model.layers[layerIdx]

    # Boundary input: the embedding output at layer 0, the hidden plus
    # residual sum at layers 1+.
    let nimInput =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    eval(nimInput)
    # Layer 0 boundary input: the embedding output, a pure gather.
    # Later boundary inputs carry the attention and FFN reductions
    # of the previous layer: the composed-chain class.
    let inputKind = if layerIdx == 0: kElementwise else: kReduction
    assertStats(nimInput, layerStatsPath(layerIdx), "layer_input", inputKind, msg = "layer " & $layerIdx & " boundary input")

    # Forward through layer (long residual stream pattern)
    let (output, newResidual) = layer.forward(ctx, hidden, residual)

    # Boundary sum feeds the next layer's recorded input.
    let nimSum = output + newResidual
    if layerIdx < model.layers.len - 1:
      eval(nimSum)
      assertStats(nimSum, layerStatsPath(layerIdx + 1), "layer_input", kReduction, msg = "layer " & $layerIdx & " output plus residual, chained input")

    hidden = output
    residual = some(newResidual)

  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm.forward(hidden + finalResidual)
  let finalLogits = model.lmHead.forward(finalNorm)
  eval(finalLogits)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"

  var flipCount = 0

  for pos in 0 ..< inputIds.len:
    assertArgMax(finalLogits.narrow(1, pos, 1), decisionsPath, pos,
      kReduction, flipCount, depth = cfg.num_hidden_layers,
      msg = "Qwen3-0.6B final logits position " & $pos)
  stderr.writeLine "tie flips across the 6 positions: " & $flipCount
  result = true

when isMainModule:
  let modelPath = getEnv("TTT_QWEN3_PATH", ModelPathDefault)
  if main(modelPath):
    stderr.writeLine "qwen3 full forward to logits (Metal): all asserts green"
