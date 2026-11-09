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
  workspace/crucible,
  workspace/ceramic/src/tensors,
  workspace/ceramic/src/layout_algebra,
  workspace/tensorcery/src/device_view,
  workspace/tensorcery/src/tensor_datatypes,
  workspace/tattleformers/src/layers/embedding,
  workspace/tattleformers/src/layers/linear,
  workspace/tattleformers/src/layers/norm,
  workspace/tattleformers/src/layers/rope,
  workspace/tattleformers/src/layers/ffn,
  workspace/tattleformers/src/layers/lmhead,
  # Import order (registration locking, carried by every op file):
  # - the rmsnorm op's walked chain registers first
  # - reliable f32-view launch bindings across the other ops require
  #   first instantiation of the rmsnorm chain
  workspace/tattleformers/src/ops_1i1o/rmsnorm,
  workspace/tattleformers/src/ops_2i1o/add,
  workspace/tattleformers/src/ops_2i1o/silu_mul,
  workspace/tattleformers/src/ops_Ni1o/attention,
  workspace/tattleformers/src/layers/attention,
  workspace/tattleformers/src/layers/decoder_layers,
  workspace/tattleformers/src/stateful/page_pool,
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
    ## the reference suite's 4096-token pool would eager-allocate
    ## a multi-gigabyte Metal buffer (the models.nim generate override rationale).

# Error-model classes of the assert kinds, measured against the recorded
# tensors on this stack (poc elementwise test, layer 0):
# - the layer 0 boundary input is the embedding gather, no reduction,
#   its replay drift is 0 over the 2-step elementwise band, kElementwise
# - every later boundary input embeds the attention and FFN
#   reductions of the previous layer: max absolute drift 1 step
#   at the recorded max magnitude, 0 elements over the elementwise band
# - small-magnitude elements churn histogram buckets under drift,
#   total variation 0.1407, elementwise floor 0.1263 exceeded,
#   Reduction floor 0.1786 respected
# Composed chains map to kReduction per the harness kind table, the reference kElementwise choice was invisible on a zero-drift stack.

proc k_primer*(z, x: distinct DeviceView, shift: static int) =
  ## z = x + shift, one element.
  ##
  ## Why this kernel exists:
  ## - the compilation must hold one enqueue before the first erased-Tensor call
  ## - Tensor views dispatch through the erased interface, whose vtable
  ##   instantiation compiles eval and its batch submit
  ## - submit's finalize macro is a compile-time check erroring on an empty
  ##   codegen frontend
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let s = xv[0].to(float32) + float32(shift)
  zv[0] = s.to(bfloat16)

proc layerStatsPath(layerIdx: int): string =
  ## Sidecar stats frame of one recorded layer, layer-NN aligned to 2 digits.
  FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats")

proc main[E: HwEngine](engine: E, modelPath: string): bool =
  # Primer block first: the primer enqueue must textually precede
  # every erased-Tensor call below. Erased vtable instantiation
  # compiles the submit finalize on that first call.
  block:
    let
      zDev = engine.allocDevice(bfloat16, 1)
      xDev = engine.allocDevice(bfloat16, 1)
      zb = TensorBridge[typeof(zDev), (int,), (int,)](
        data: zDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
      xb = TensorBridge[typeof(xDev), (int,), (int,)](
        data: xDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
    var batch = engine.recordBatch()
    batch.enqueue<<(1, 1)>> k_primer(zb, xb, 3)
    batch.submit()

  doAssert fileExists(FixtureDir / "final_logits.decisions.json.zst"),
    "recorded fixtures missing under " & FixtureDir

  let model = loadModel(engine, modelPath)
  let cfg = model.config

  let pool = PagePool.init(engine, 2,
    num_layers = cfg.num_hidden_layers,
    kv_heads = cfg.num_key_value_heads,
    head_dim = cfg.head_dim)
  var ctx = InferenceContext[uint32, Page].init(
    cfg.num_hidden_layers, batch_size = 1,
    kv_heads = cfg.num_key_value_heads,
    max_seq = MaxSeq, head_dim = cfg.head_dim)
  ctx.pool = pool
  for i in 0 ..< 2:
    ctx.pages.add(pool.borrow())

  # Input tokens are the tokenizer ids of "Hello, how are you?"
  let inputIds = @[9707, 11, 1246, 525, 498, 30]

  var hidden = model.embedTokens.forward(inputIds, @[1, inputIds.len])
  var residual = none(Tensor)

  ctx.setPositionIdsArange(inputIds.len)
  ctx.setRopeForPositions(model.rotary)

  for layerIdx in 0 ..< model.layers.len:
    let layer = model.layers[layerIdx]

    # Boundary input: the embedding output at layer 0, the hidden plus
    # residual sum at layers 1+.
    let nimInput =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    var inputRoots = @[nimInput.node]
    engine.eval(inputRoots.toSpan())
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
      var sumRoots = @[nimSum.node]
      engine.eval(sumRoots.toSpan())
      assertStats(nimSum, layerStatsPath(layerIdx + 1), "layer_input", kReduction, msg = "layer " & $layerIdx & " output plus residual, chained input")

    hidden = output
    residual = some(newResidual)

  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm.forward(hidden + finalResidual)
  let finalLogits = model.lmHead.forward(finalNorm)
  var logitsRoots = @[finalLogits.node]
  engine.eval(logitsRoots.toSpan())

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
  if main(gMetalEngine(), modelPath):
    stderr.writeLine "qwen3 full forward to logits (Metal): all asserts green"
