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
## on Metal with 7 assertStats instruments per block at depth i+1,
## plus the tail checkpoint at depth 28.
##
## Recorded frames (Python testgen, consumed in place):
##   workspace/transformers/tests/fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B
## Local model: /Users/pi/Documents/Programming/MODELS/Qwen3-0.6B
## (TTT_QWEN3_PATH env override).
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

const
  FixtureDir = currentSourcePath().parentDir().parentDir().parentDir().parentDir() /
    "transformers" / "tests" / "fixtures" /
    "bf16-02-first-8-layers-plus-final" / "Qwen3-0.6B"
  ModelPathDefault = "/Users/pi/Documents/Programming/MODELS/Qwen3-0.6B"
  MaxSeq = 256
    ## Pool sizing override, the recorded chain spans 6 positions:
    ## the reference suite's 4096-token pool would eager-allocate
    ## a multi-gigabyte Metal buffer (the models.nim generate override rationale).

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

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

proc blockStatsPath(i: int): string =
  ## Sidecar stats frame of one recorded block, block-NN aligned to 2 digits.
  FixtureDir / ("block-" & ($i).align(2, '0') & ".safetensor.stats")

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

  doAssert fileExists(FixtureDir / "tail.safetensor.stats.json.zst"),
    "recorded fixtures missing under " & FixtureDir

  let model = loadModel(engine, modelPath)
  let cfg = model.config

  let pool = PagePool.init(engine, 2,
    num_layers = cfg.num_hidden_layers,
    kv_heads = cfg.num_key_value_heads,
    head_dim = cfg.head_dim)
  var (ctx, _) = (
    InferenceContext[uint32, Page].init(
      cfg.num_hidden_layers, batch_size = 1,
      kv_heads = cfg.num_key_value_heads,
      max_seq = MaxSeq, head_dim = cfg.head_dim),
    pool)
  ctx.pool = pool
  for i in 0 ..< 2:
    ctx.pages.add(pool.borrow())

  let inputIds = @[9707, 11, 1246, 525, 498, 30]
  var hidden = model.embedTokens.forward(inputIds, @[1, inputIds.len])
  var residual = none(Tensor)

  ctx.setPositionIdsArange(inputIds.len)
  ctx.setRopeForPositions(model.rotary)

  let numLayers = cfg.num_hidden_layers
  var layerIdx = 0
  while layerIdx < numLayers and fileExists(blockStatsPath(layerIdx)):
    let i = layerIdx
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
    var roots = @[mlpOut.node, checkpoint.node]
    engine.eval(roots.toSpan())

    assertStats(hidden, blockStatsPath(i), "layer_input", kElementwise, depth = i + 1, msg = "block " & $i & " boundary input")
    assertStats(combined, blockStatsPath(i), "after_attn_norm_residual", kElementwise, depth = i + 1, msg = "block " & $i & " residual sum")
    # Recorded sublayer intermediates re-express block by block, records come off the committed stats frame.
    assertStats(hNorm, blockStatsPath(i), "after_attn_norm", kElementwise, depth = i + 1, msg = "block " & $i & " input layernorm output")
    assertStats(attnOut, blockStatsPath(i), "after_attn", kReduction, depth = i + 1, msg = "block " & $i & " attention output")
    assertStats(combined2, blockStatsPath(i), "after_mlp_norm_residual", kElementwise, depth = i + 1, msg = "block " & $i & " post-attention residual sum")
    assertStats(h2, blockStatsPath(i), "after_mlp_norm", kElementwise, depth = i + 1, msg = "block " & $i & " post-attention layernorm output")
    assertStats(mlpOut, blockStatsPath(i), "mlp_out", kReduction, depth = i + 1, msg = "block " & $i & " mlp output")
    assertStats(checkpoint, blockStatsPath(i), "hf_layer_output", kReduction, depth = i + 1, msg = "chain checkpoint block " & $i)

    hidden = blkOut
    residual = some(blkResidual)
    inc layerIdx

  # Blocks after the fixture prefix carry no per-block records, the chain runs on through the real forward to the tail checkpoint.
  for i in layerIdx ..< numLayers:
    let layer = model.layers[i]
    let (blkOut, blkResidual) = layer.forward(ctx, hidden, residual)
    var roots = @[blkOut.node, blkResidual.node]
    engine.eval(roots.toSpan())
    hidden = blkOut
    residual = some(blkResidual)

  # Tail checkpoint: the decoder stack output feeding the final
  # RMSNorm and lm head.
  let tail = hidden + residual.get(hidden)
  var tailRoots = @[tail.node]
  engine.eval(tailRoots.toSpan())
  assertStats(tail, FixtureDir / "tail.safetensor.stats", "pre_final_norm", kReduction, depth = numLayers, msg = "chain tail checkpoint at depth " & $numLayers)
  result = true

when isMainModule:
  let modelPath = getEnv("TTT_QWEN3_PATH", ModelPathDefault)
  if main(gMetalEngine(), modelPath):
    stderr.writeLine "qwen3 first 8 layers plus final (Metal): all asserts green"
