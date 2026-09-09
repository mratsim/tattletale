# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   TTT_TEST_ON=cpu nim test_tf_bf16_qwen3_02_first_8_layers_plus_final

import
  std/importutils,
  std/math,
  std/options,
  std/os,
  std/strformat,
  std/strutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

from workspace/libtorch/src/raw_libtorch import manual_seed

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-02-first-8-layers-plus-final" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"
  PrefixBlocks = 8
  ChainDepth = 28
  ChainSeqLen = 6
  ChainMaxSeq = 256

proc replayBlockManual(layer: Qwen3DecoderLayer, ctx: var InferenceContext,
    hidden: Tensor, residual: Option[Tensor], fixture: Safetensor) =
  ## One block of the long residual stream, recomputed step by step and compared
  ## bit-exactly against the recorded intermediates.
  let combined =
    if residual.isSome:
      hidden + residual.unsafeGet
    else:
      hidden
  assertAllClose(combined, fixture.getTensorOwned("after_attn_norm_residual"),
    rtol = 0.0, abstol = 0.0, msg = "block residual sum")
  let h = layer.input_layernorm(combined)
  assertAllClose(h, fixture.getTensorOwned("after_attn_norm"),
    rtol = 0.0, abstol = 0.0, msg = "input layernorm output")
  let attnOut = layer.sequence_mixer(ctx, h)
  assertAllClose(attnOut, fixture.getTensorOwned("after_attn"),
    rtol = 0.0, abstol = 0.0, msg = "attention output")
  let combined2 = attnOut + combined
  assertAllClose(combined2, fixture.getTensorOwned("after_mlp_norm_residual"),
    rtol = 0.0, abstol = 0.0, msg = "post-attention residual sum")
  let h2 = layer.post_attention_layernorm(combined2)
  assertAllClose(h2, fixture.getTensorOwned("after_mlp_norm"),
    rtol = 0.0, abstol = 0.0, msg = "post-attention layernorm output")
  let mlpOut = layer.hidden_mixer(h2)
  assertAllClose(mlpOut, fixture.getTensorOwned("mlp_out"),
    rtol = 0.0, abstol = 0.0, msg = "mlp output")
  # The long residual stream invariant: mlp_out + residual == the local
  # residual stream output, bit-exact on the reference device.
  let invariant = mlpOut + combined2
  assertAllClose(invariant, fixture.getTensorOwned("hf_layer_output"),
    rtol = 0.0, abstol = 0.0, msg = "long residual stream invariant")

proc newContext(model: Qwen3Model, device: F.DeviceKind,
    poolLayers: int): tuple[ctx: InferenceContext, pool: PagePool] =
  ## Inference context plus page pool sized for one prefill of the fixture.
  result.ctx = InferenceContext.init(
    num_layers = model.config.num_hidden_layers,
    batch_size = 1, kv_heads = model.config.num_key_value_heads,
    max_seq = ChainMaxSeq, head_dim = model.config.head_dim)
  result.pool = PagePool.init(
    64, num_layers = poolLayers,
    kv_heads = model.config.num_key_value_heads,
    head_dim = model.config.head_dim,
    dtype = F.kBFloat16, device = device)
  let numPages = ceilDiv(ChainMaxSeq, TokensPerPage)
  for _ in 0 ..< numPages:
    result.ctx.pages.add(result.pool.borrow())

proc positionRope(ctx: var InferenceContext, rotary: RotaryPositionEmbedding) =
  ## Reset the positional state for one prefill of the fixture blocks.
  ctx.kv_position = 0
  ctx.position_ids = F.arange(ChainSeqLen,
    F.tensorOptions(F.kInt64, F.kCPU))
  ctx.setRopeForPositions(rotary)

proc runTwoSidedDriftScaling() =
  ## Two-sided verification of the drift-scaling check itself: bounded
  ## band widths get accepted over the 8+1 point count, and compounding
  ## growth gets rejected at the checkpoint where it outruns the flat law.
  # Accept: the measured Metal drift of the committed fixtures.
  checkDriftScaling(@[0.31, 0.52, 0.46])
  # Accept: depth-flat band widths across all 9 checkpoints.
  checkDriftScaling(@[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
  # Accept: honest spread inside the flat law (ratio 2.2 < slack 3).
  checkDriftScaling(@[1.0, 1.6, 2.2])
  # Reject: band width growing linearly with the checkpoint index.
  try:
    checkDriftScaling(@[0.31, 0.62, 0.93, 1.24])
    doAssert false, "linear band width growth was accepted"
  except HarnessCheckError:
    discard
  # The tail value must sit clearly past the bound: the flat-law bound is the first
  # checkpoint times the slack, so a sequence ending exactly at the bound stays accepted
  # and the rejection needs the tail over the bound, not at it.
  try:
    checkDriftScaling(@[1.0, 2.0, 3.5])
    doAssert false, "pure linear band width growth was accepted"
  except HarnessCheckError:
    discard
  # Reject: compounding growth shaped like sqrt(depth), the signature
  # a per-block bias that compounds. Honest drift stays near the first
  # checkpoint, so the sequence passes every prefix checkpoint and gets
  # rejected exactly at the depth-28 tail.
  var sqrtGrowth: seq[float64] = @[]
  for depth in [1, 2, 3, 4, 5, 6, 7, 8, 28]:
    sqrtGrowth.add 0.31 * sqrt(depth.float64)
  try:
    checkDriftScaling(sqrtGrowth)
    doAssert false, "sqrt-growing band width (linear absolute drift) was accepted"
  except HarnessCheckError:
    discard

proc assertStatsCheckpoint(checkpoint: Tensor, statsPath, tensorName: string) =
  ## Check the checkpoint against the recorded sidecar: the order
  ## statistics and histogram of the recorded checkpoint against the
  ## computed value.
  let statsFile = loadFingerprintStats(FixtureDir / statsPath)
  let budget = ToleranceBudget(tier: ctDistribution, maxUlp: 2,
    maxMismatchFrac: 0.0, histL1: 0.01)
  assertStats(checkpoint, statsFile.statsTensor(tensorName),
    budget, msg = "chain checkpoint " & tensorName)

proc makeConstDelta(n: int, c: float64): Tensor =
  ## A constant bf16 delta of magnitude c on every element.
  var s = newSeq[float32](n)
  for i in 0 ..< n:
    s[i] = c.float32
  F.toTensor(s).to(F.kBFloat16)

proc makeAlternatingDelta(n: int, c: float64): Tensor =
  ## A zero-mean bf16 delta: +c on even elements, -c on odd elements.
  var s = newSeq[float32](n)
  for i in 0 ..< n:
    s[i] = (if i mod 2 == 0: c.float32 else: -c.float32)
  F.toTensor(s).to(F.kBFloat16)

proc runTwoSidedMeanDrift(reductionLen: int) =
  ## Two-sided verification of the mean-drift check itself. Zero-mean
  ## reordering-shaped drift gets accepted even at double the bound
  ## per element, a reordering-legal constant bias at half the bound gets
  ## accepted, and a constant bias past the bound gets rejected.
  Torch.manual_seed(0x5EED'u64)
  let expected = F.randn(6144, F.tensorOptions(F.kBFloat16, F.kCPU))
  let bound = chainMeanDriftBound(meanAbsValue(expected), reductionLen)
  let n = expected.numel()
  assertChainMeanDrift(expected + makeAlternatingDelta(n, 2.0 * bound),
    expected, reductionLen, msg = "zero-mean synthetic drift")
  assertChainMeanDrift(expected + makeConstDelta(n, 0.5 * bound),
    expected, reductionLen, msg = "bias half the reordering bound")
  try:
    assertChainMeanDrift(expected + makeConstDelta(n, 1.5 * bound),
      expected, reductionLen, msg = "bias past the reordering bound")
    doAssert false, "coherent bias beyond the reordering bound was accepted"
  except HarnessCheckError:
    discard

proc replayChain(model: Qwen3Model, ctx: var InferenceContext,
    device: F.DeviceKind, reductionLen: int): seq[float64] =
  ## Replay the full 28-layer chain and return the per-checkpoint band
  ## widths: the 8 prefix block checkpoints plus the tail checkpoint,
  ## the decoder stack output taken pre-final-norm. The recorded/run device
  ## comparison selects the tolerance class: a run on the recorded device
  ## compares the checkpoints bit-exactly, a run on any other device
  ## compares under the chain checkpoint budget.
  let referenceReplay =
    compareClass(recordedDevice(recordedFrom(FixtureDir)), device) ==
    sameDeviceBitExact
  let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)
  var hidden = model.embedTokens(inputIds)
  var residual: Option[Tensor] = none(Tensor)

  for layerIdx in 0 ..< PrefixBlocks:
    var fixture = Safetensor.open(FixtureDir / &"block-{layerIdx:02d}.safetensor")
    let layer = model.layers[layerIdx]
    positionRope(ctx, layer.sequence_mixer.rotary)

    if referenceReplay:
      # Manual decomposition, bit-exact against the recording.
      replayBlockManual(layer, ctx, hidden, residual, fixture)

    # The real block forward must agree with the manual decomposition.
    ctx.kv_position = 0
    let (blkOut, res) = layer(ctx, hidden, residual)
    if referenceReplay:
      assertAllClose(blkOut, fixture.getTensorOwned("mlp_out"),
        rtol = 0.0, abstol = 0.0,
        msg = "layer forward != manual decomposition, block " & $layerIdx)
      assertAllClose(res, fixture.getTensorOwned("after_mlp_norm_residual"),
        rtol = 0.0, abstol = 0.0,
        msg = "layer forward residual != manual decomposition, block " & $layerIdx)

    let checkpoint = blkOut + res
    if referenceReplay:
      assertStatsCheckpoint(checkpoint,
        &"block-{layerIdx:02d}.safetensor.stats", "hf_layer_output")

    # Measured drift of the checkpoint under the chain tolerance,
    # zero on the reference device. The measured values feed
    # the drift-scaling check.
    let mw = assertChainCheckpoint(checkpoint,
      fixture.getTensorOwned("hf_layer_output"), depth = layerIdx + 1,
      reductionLen = reductionLen,
      msg = "chain checkpoint block " & $layerIdx)
    echo "    " & $device & " block " & $layerIdx & ": worst band width " &
      formatBiggestFloat(mw.worst, ffDecimal, 3) & ", violations " & $mw.violations &
      ", mean drift " & formatBiggestFloat(mw.meanDrift, ffScientific, 2)
    result.add mw.worst

    hidden = blkOut
    residual = some(res)

  # Blocks 8..27 carry no per-block fixtures: the chain continues
  # through the real forward to the tail checkpoint.
  for layerIdx in PrefixBlocks ..< ChainDepth:
    let layer = model.layers[layerIdx]
    positionRope(ctx, layer.sequence_mixer.rotary)
    let (blkOut, res) = layer(ctx, hidden, residual)
    hidden = blkOut
    residual = some(res)

  # The tail checkpoint: the decoder stack output feeding the final
  # RMSNorm and lm_head.
  let tail = hidden + residual.get(hidden)
  var tailFixture = Safetensor.open(FixtureDir / "tail.safetensor")
  let tailRecorded = tailFixture.getTensorOwned("pre_final_norm")
  if referenceReplay:
    assertAllClose(tail, tailRecorded,
      rtol = 0.0, abstol = 0.0, msg = "tail checkpoint pre-final-norm")
    assertStatsCheckpoint(tail, "tail.safetensor.stats", "pre_final_norm")

  let mw = assertChainCheckpoint(tail, tailRecorded, depth = ChainDepth,
    reductionLen = reductionLen,
    msg = "chain checkpoint tail at depth " & $ChainDepth)
  echo "    " & $device & " tail (depth " & $ChainDepth & "): worst band width " &
    formatBiggestFloat(mw.worst, ffDecimal, 3) & ", violations " & $mw.violations
  result.add mw.worst

proc main() =
  assertTorchStamp(FixtureDir)
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "Drift-scaling check discriminates flat band widths from growth":
    proc(): bool =
      runTwoSidedDriftScaling()
      true

  runCppTest "Mean-drift check discriminates coherent bias from reordering noise":
    proc(): bool =
      runTwoSidedMeanDrift(3072)
      true

  runCppTest "Qwen3-0.6B 9-checkpoint chain, CPU variant bit-exact with sidecar checkpoints":
    proc(): bool =
      echo "    devices: ", compareReport(FixtureDir, F.kCPU)
      let model = loadQwen3ModelRaw(ModelPath, F.kCPU)
      var (ctx, pool) = newContext(model, F.kCPU, poolLayers = ChainDepth)
      let drifts = replayChain(model, ctx, F.kCPU,
        model.config.intermediate_size)
      checkDriftScaling(drifts)
      echo "    cpu variant drift band widths: ", drifts
      true

  runCppTest "Qwen3-0.6B 9-checkpoint chain, cross-device variant drift band and scaling check":
    proc(): bool =
      # The recorded/run device comparison selects the drift budget: the
      # recorded_from manifest value stays fixed while the run-time device
      # follows TTT_TEST_ON. Under TTT_TEST_ON=cpu the comparison
      # degenerates to the recorded device, the reference budgets apply and
      # the reference variant above carries the replay.
      let runDev = testDevice()
      echo "    devices: ", compareReport(FixtureDir, runDev)
      if compareClass(recordedDevice(recordedFrom(FixtureDir)), runDev) ==
          sameDeviceBitExact:
        echo "    the device comparison selects the reference budgets, the reference variant carries the replay"
        return true
      let model = loadQwen3ModelRaw(ModelPath, runDev)
      var (ctx, pool) = newContext(model, runDev, poolLayers = ChainDepth)
      let drifts = replayChain(model, ctx, runDev,
        model.config.intermediate_size)
      checkDriftScaling(drifts)
      true

when isMainModule:
  main()
