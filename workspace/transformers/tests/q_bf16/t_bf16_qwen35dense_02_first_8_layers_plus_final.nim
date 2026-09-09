# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chain suite for the Qwen3.5-0.8B dense stack: the GDN long residual stream replay
## at the 8+1 chain checkpoints against the recorded fixtures. The checkpoints are
## the outputs of decoder blocks 0..7 and the tail, the decoder stack output taken
## pre-final-norm at depth 24. The layer pattern is period-4 with three linear-attention
## blocks and one full-attention block per period, so the 8-block prefix is two full
## periods and class-complete, and the tail validates the depth extrapolation over
## the full stack. Every prefix block carries a sidecar checkpoint. The suite also
## checks the GDN state boundary paths:
## - block decomposition
## - state persistence
## - conv history
##
## The suite carries two variants over the same committed fixtures:
##
## - CPU variant (reference device): the sequential chain compares bit-exactly
##   against the recorded fixtures at every checkpoint, prefix blocks and tail
##   alike, and each checkpoint also passes the sidecar stats assert.
##   On the reference device both sides call the same ATen kernels in one call order,
##   so any drift at all is a bug by definition.
## - MPS variant (cross-device): the replay runs on Metal against the cpu
##   recording. Each checkpoint passes the chain budget:
##   the obChainCheckpoint elementwise cap in tolerance.nim.
##   Its absolute term scales with the depth and the checkpoint bulk.
##   Nine measured per-checkpoint band widths feed the drift-scaling check.
##
## Build:
##   nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on
##     --debugger:native --hints:off --warnings:off --passC:"-std=c++20"
##     --outdir:build/tests/rel --nimcache:nimcache/tests/rel
##     workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_02_first_8_layers_plus_final.nim

import
  std/importutils,
  std/math,
  std/options,
  std/os,
  std/strformat,
  std/strutils,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/tests/q_bf16/kvcontext,
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

from workspace/libtorch/src/raw_libtorch import manual_seed

{.experimental: "callOperator".}

privateAccess(Qwen35Model)
privateAccess(GatedDeltaNet)
privateAccess(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])

const
  PrefixBlocks = 8
  ChainDepth = 24
  ChainSeqLen = 4
  ChainFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-02-first-8-layers-plus-final" / "Qwen3.5-0.8B"
  GdnFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Qwen3.5-0.8B-layer-0"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc openModelWeights(): SafetensorsCollection =
  ## Read the real Qwen3.5-0.8B safetensor file for targeted weight
  ## loads. The result owns its memory mapping for its whole lifetime,
  ## released when the value goes out of scope.
  SafetensorsCollection.open(ModelDir / "model.safetensors-00001-of-00001.safetensors")

proc loadGdn(view: SafetensorsCollection, cfgJson: JsonNode, layerIdx: int): GatedDeltaNet =
  ## Load the real layer-`layerIdx` Gated DeltaNet weights into a layer.
  let tc = cfgJson{"text_config"}
  let lp = "model.language_model.layers." & $layerIdx & ".linear_attn."
  let qkvProj = Linear.load(view, cfgJson, lp & "in_proj_qkv")
  let zProj = Linear.load(view, cfgJson, lp & "in_proj_z")
  let aProj = Linear.load(view, cfgJson, lp & "in_proj_a")
  let bProj = Linear.load(view, cfgJson, lp & "in_proj_b")
  let convWeight = view.getTensorOwned(lp & "conv1d.weight")
  let aLog = view.getTensorOwned(lp & "A_log")
  let dtBias = view.getTensorOwned(lp & "dt_bias")
  let gdnNorm = RmsNormGated.load(view, cfgJson, lp & "norm")
  let outProj = Linear.load(view, cfgJson, lp & "out_proj")
  result = GatedDeltaNet.init(
    layerIdx, lp[0 .. ^2],
    qkvProj, zProj, aProj, bProj,
    convWeight, aLog, dtBias, gdnNorm, outProj,
    tc{"linear_num_key_heads"}.getInt().int,
    tc{"linear_num_value_heads"}.getInt().int,
    tc{"linear_key_head_dim"}.getInt().int,
    tc{"linear_value_head_dim"}.getInt().int,
    tc{"linear_conv_kernel_dim"}.getInt().int)

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
  ## Two-sided verification of the mean-drift check itself:
  ## - zero-mean drift shaped like honest rounding from a different
  ##   addition order gets accepted even at double the bound per element
  ## - a constant bias at half the chain drift bound gets accepted
  ## - a constant bias past the bound gets rejected
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

proc main() =
  assertTorchStamp(ChainFixtureDir)
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  runCppTest "Mean-drift check discriminates coherent bias from reordering noise":
    proc(): bool =
      runTwoSidedMeanDrift(3584)
      true

  # Two-sided verification of the drift-scaling check itself:
  # - honest bounded band widths get accepted
  # - compounding growth gets rejected at the checkpoint where it
  #   crosses the flat bound
  runCppTest "Drift-scaling check discriminates bounded drift from compounding growth":
    proc(): bool =
      checkDriftScaling(@[0.31, 0.52, 0.46])
      checkDriftScaling(@[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
      checkDriftScaling(@[1.0, 1.6, 2.2])
      try:
        checkDriftScaling(@[0.31, 0.62, 0.93, 1.24])
        doAssert false, "compounding band width growth was accepted"
      except HarnessCheckError:
        discard
      try:
        checkDriftScaling(@[1.0, 2.0, 3.0, 4.0])
        doAssert false, "pure compounding growth was accepted"
      except HarnessCheckError:
        discard
      # The mild sqrt-shaped compounding growth passes every prefix
      # checkpoint and is rejected exactly at the depth-28 tail.
      var sqrtGrowth: seq[float64] = @[]
      for depth in [1, 2, 3, 4, 5, 6, 7, 8, 28]:
        sqrtGrowth.add 0.31 * sqrt(depth.float64)
      try:
        checkDriftScaling(sqrtGrowth)
        doAssert false, "sqrt-shaped compounding was accepted at the tail"
      except HarnessCheckError:
        discard
      true

  runCppTest "Qwen3.5-0.8B 9-checkpoint chain, CPU variant bit-exact with sidecar checkpoints":
    proc(): bool =
      echo "    devices: ", compareReport(ChainFixtureDir, F.kCPU)
      let model = loadQwen35ModelRaw(ModelDir, F.kCPU)
      doAssert model.layers.len == ChainDepth
      var (ctx, pool) = newKVContext(numLayers = ChainDepth, kvHeads = 2,
        headDim = 256)
      ctx.position_ids = F.arange(ChainSeqLen, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(model.rotary)

      var st0 = Safetensor.open(ChainFixtureDir / "block-00.safetensor")
      var hidden = st0.getTensorOwned("layer_input_seq")
      var blockInput = hidden
      var stream: Option[Tensor]

      var drifts: seq[float64] = @[]
      let reductionLen = model.config.intermediate_size
      let budget = ToleranceBudget(tier: ctDistribution, maxUlp: 2,
        maxMismatchFrac: 0.0, histL1: 0.01)
      for i in 0 ..< PrefixBlocks:
        var st = Safetensor.open(ChainFixtureDir / ("block-0" & $i & ".safetensor"))
        #  Sequential chain inputs/outputs at 0.00, vendored chunked chain
        #  within the measured layer delta the recording carries between its
        #  sequential chain and its single-layer entries. The single-chunk T=4 fixtures of the first
        #  blocks have chunked == sequential bit-exact, so those band asserts
        #  are degenerate. They keep the chunked tensors loaded and compared.
        assertAllClose(hidden, st.getTensorOwned("layer_input_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " input mismatch")
        let inputBand = maxAbsDiff(st.getTensorOwned("layer_input_seq"),
                                   st.getTensorOwned("layer_input"))
        assertAllClose(hidden, st.getTensorOwned("layer_input"),
          rtol = 0.0, abstol = inputBand,
          msg = "chain layer " & $i & " chunked input mismatch (measured layer delta " & $inputBand & ")")
        let pair = model.layers[i].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])
        let layerOut = pair[1] + pair[0]
        assertAllClose(layerOut, st.getTensorOwned("layer_output_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential output mismatch")
        let outputBand = maxAbsDiff(st.getTensorOwned("layer_output_seq"),
                                    st.getTensorOwned("layer_output"))
        assertAllClose(layerOut, st.getTensorOwned("layer_output"),
          rtol = 0.0, abstol = outputBand,
          msg = "chain layer " & $i & " chunked output mismatch (measured layer delta " & $outputBand & ")")

        #  Local residual invariant on the Gated DeltaNet blocks: block
        #  output = input + attn delta + mlp delta. The deltas get recomputed
        #  through the layer components, and the sum is compared against the sequential
        #  fixture at 0.00 and against the layer forward at 0.00, one op
        #  order). A fresh context per block keeps the GDN state at the sequence
        #  start, matching the fixture layout. Full-attention blocks skip the decomposition:
        #  their bit-exact forward compare above carries them, and the all-attention
        #  0.6B chain exercises the same decomposition on every one of its blocks.
        let layer = model.layers[i].to(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])
        if layer != nil:
          var invCtx = InferenceContext.init(24, 1, 2, 512, 256)
          let hNorm = layer.input_layernorm(hidden)
          let attnOut = layer.sequence_mixer(invCtx, hNorm)
          let h1 = hidden + attnOut
          let hNorm2 = layer.post_attention_layernorm(h1)
          let mlpOut = layer.hidden_mixer.forward(hNorm2)
          let deltaSum = h1 + mlpOut
          assertAllClose(deltaSum, st.getTensorOwned("layer_output_seq"),
            rtol = 0.0, abstol = 0.0,
            msg = "local residual invariant mismatch, layer " & $i)
          assertAllClose(deltaSum, layerOut,
            rtol = 0.0, abstol = 0.0,
            msg = "layer forward != input + attn + mlp deltas, layer " & $i)

        # Sidecar checkpoint: recorded order statistics and histogram
        # of the block output, asserted against the computed checkpoint.
        let statsFile = loadFingerprintStats(
          ChainFixtureDir / ("block-0" & $i & ".safetensor.stats"))
        assertStats(layerOut, statsFile.statsTensor("layer_output_seq"),
          budget, msg = "chain checkpoint block " & $i)

        # Measured drift of the checkpoint under the chain budget,
        # zero on the reference device. The measured values feed
        # the drift-scaling check.
        let mw = assertChainCheckpoint(layerOut,
          st.getTensorOwned("layer_output_seq"), depth = i + 1,
          reductionLen = reductionLen,
          msg = "chain checkpoint block " & $i)
        drifts.add mw.worst

        hidden = layerOut

      # Blocks 8..23 carry no per-block fixtures: the chain continues
      # through the real forward to the tail checkpoint.
      for i in PrefixBlocks ..< ChainDepth:
        let pair = model.layers[i].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])

      # The tail checkpoint: the decoder stack output feeding the final
      # RMSNorm and lm head, recorded on the sequential chain at 0.00,
      # with the chunked chain alongside.
      let tail = stream.get(blockInput) + blockInput
      var tailFixture = Safetensor.open(ChainFixtureDir / "tail.safetensor")
      let tailRecorded = tailFixture.getTensorOwned("pre_final_norm")
      let tailChunked = tailFixture.getTensorOwned("pre_final_norm_chunked")
      let tailBand = maxAbsDiff(tailRecorded, tailChunked)
      doAssert tailBand > 0.0,
        "sequential and chunked tails are identical, band not exercised"
      assertAllClose(tail, tailRecorded,
        rtol = 0.0, abstol = 0.0, msg = "tail checkpoint pre-final-norm")
      let tailStats = loadFingerprintStats(
        ChainFixtureDir / "tail.safetensor.stats")
      assertStats(tail, tailStats.statsTensor("pre_final_norm"),
        budget, msg = "chain checkpoint tail")

      let mw = assertChainCheckpoint(tail, tailRecorded, depth = ChainDepth,
        reductionLen = reductionLen,
        msg = "chain checkpoint tail at depth " & $ChainDepth)
      echo "    cpu tail (depth " & $ChainDepth & "): worst band width " &
        formatBiggestFloat(mw.worst, ffDecimal, 3) & ", violations " & $mw.violations
      drifts.add mw.worst
      checkDriftScaling(drifts)
      echo "    cpu variant drift band widths: ", drifts
      true

  runCppTest "Qwen3.5-0.8B 9-checkpoint chain, cross-device variant drift band and scaling check":
    proc(): bool =
      # The device comparison selects the tolerance class: the recorded
      # device comes from the recorded_from manifest value while the run
      # device follows TTT_TEST_ON. Under TTT_TEST_ON=cpu both name the
      # same device, the reference tolerances apply and the reference
      # variant above carries the replay.
      let runDev = testDevice()
      echo "    devices: ", compareReport(ChainFixtureDir, runDev)
      if compareClass(recordedDevice(recordedFrom(ChainFixtureDir)),
          runDev) == sameDeviceBitExact:
        echo "    the pair selects the reference rows, the reference variant carries the replay"
        return true
      let model = loadQwen35ModelRaw(ModelDir, runDev)
      var (ctx, pool) = newKVContext(numLayers = ChainDepth, kvHeads = 2,
        headDim = 256, device = runDev)
      ctx.position_ids = F.arange(ChainSeqLen, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(model.rotary)

      var st0 = Safetensor.open(ChainFixtureDir / "block-00.safetensor")
      var blockInput = st0.getTensorOwned("layer_input_seq").to(runDev)
      var stream: Option[Tensor]

      var drifts: seq[float64] = @[]
      let reductionLen = model.config.intermediate_size
      for i in 0 ..< PrefixBlocks:
        var st = Safetensor.open(ChainFixtureDir / ("block-0" & $i & ".safetensor"))
        let pair = model.layers[i].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])
        let layerOut = pair[1] + pair[0]
        let mw = assertChainCheckpoint(layerOut,
          st.getTensorOwned("layer_output_seq"), depth = i + 1,
          reductionLen = reductionLen,
          msg = "chain checkpoint block " & $i)
        echo "    " & deviceName(runDev) & " block " & $i & ": worst band width " &
          formatBiggestFloat(mw.worst, ffDecimal, 3) & ", violations " & $mw.violations &
          ", mean drift " & formatBiggestFloat(mw.meanDrift, ffScientific, 2)
        drifts.add mw.worst

      for i in PrefixBlocks ..< ChainDepth:
        let pair = model.layers[i].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])

      let tail = stream.get(blockInput) + blockInput
      var tailFixture = Safetensor.open(ChainFixtureDir / "tail.safetensor")
      let mw = assertChainCheckpoint(tail,
        tailFixture.getTensorOwned("pre_final_norm"), depth = ChainDepth,
        reductionLen = reductionLen,
        msg = "chain checkpoint tail at depth " & $ChainDepth)
      echo "    " & deviceName(runDev) & " tail (depth " & $ChainDepth & "): worst band width " &
        formatBiggestFloat(mw.worst, ffDecimal, 3) & ", violations " & $mw.violations
      drifts.add mw.worst
      checkDriftScaling(drifts)
      true

  runCppTest "GDN block layer 0 prefill (seq 5) vs fixture":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(weights, cfgJson, 0)

      var st = Safetensor.open(GdnFixtureDir / "gdn-Qwen3.5-0.8B-00.safetensor")

      let x = st.getTensorOwned("input")
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      let output = gdn(ctx, x)
      doAssert output.size(0) == 1 and output.size(1) == 5 and output.size(2) == 1024

      # Deterministic intermediates, recomputed through the layer's own
      # components and compared against the captured vendored values.
      let seqLen = x.size(1)
      let mixedQkv = gdn.in_proj_qkv.forward(x).transpose(1, 2)
      let conv = F.conv1d(mixedQkv, gdn.conv1d_weight,
        padding = [3], groups = gdn.conv_dim)
      let convOut = F.silu(conv.narrow(2, 0, seqLen))
      assertAllClose(convOut, st.getTensorOwned("conv_output"),
        rtol = 0.0, abstol = 0.0, msg = "conv output mismatch")

      let split = F.chunk(convOut.transpose(1, 2), 3, -1)
      let query = split[0].reshape([1, seqLen, gdn.num_k_heads, gdn.head_k_dim])
      let key = split[1].reshape([1, seqLen, gdn.num_k_heads, gdn.head_k_dim])
      let value = split[2].reshape([1, seqLen, gdn.num_v_heads, gdn.head_v_dim])
      assertAllClose(query, st.getTensorOwned("q"),
        rtol = 0.0, abstol = 0.0, msg = "q post-split mismatch")
      assertAllClose(key, st.getTensorOwned("k"),
        rtol = 0.0, abstol = 0.0, msg = "k post-split mismatch")
      assertAllClose(value, st.getTensorOwned("v"),
        rtol = 0.0, abstol = 0.0, msg = "v post-split mismatch")

      # Decay and beta coefficients: f32 exp/softplus and bf16 sigmoid.
      # Both sides call the same ATen ops, so bit-exact is expected
      # on the reference CPU. The 1e-4 bar guards against cross-platform
      # libm variance.
      let aProj = gdn.in_proj_a.forward(x)
      let aLogExp = gdn.a_log.to(kFloat32).exp()
      let aPlusBias = aProj.to(kFloat32) + gdn.dt_bias
      let g = aLogExp.neg() * F.softplus(aPlusBias, 1.0, 20.0)
      assertAllClose(g, st.getTensorOwned("g"),
        rtol = 1e-4, abstol = 1e-4, msg = "decay g mismatch")
      let beta = F.sigmoid(gdn.in_proj_b.forward(x))
      assertAllClose(beta, st.getTensorOwned("beta"),
        rtol = 1e-4, abstol = 1e-4, msg = "beta mismatch")

      # Layer output: 0.00 vs the sequential reference, 5e-3 vs the vendored
      # chunked forward. The recurrence is exercised end to end. The T=5
      # fixture is a single chunk, so chunked and sequential agree
      # within sub-bf16-ulp f32 distance with rounding-boundary flips.
      assertAllClose(output, st.getTensorOwned("output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "sequential block output mismatch")
      assertAllClose(output, st.getTensorOwned("output_chunked"),
        rtol = 5e-3, abstol = 5e-3, msg = "chunked block output mismatch")
      true

  runCppTest "GDN state persistence: 2-step decode == one-shot prefill":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(weights, cfgJson, 0)

      var st = Safetensor.open(GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor")

      let prefillX = st.getTensorOwned("prefill_x")
      let decodeXd = st.getTensorOwned("decode_x_d")
      let decodeXe = st.getTensorOwned("decode_x_e")
      let oneShotLayer = st.getTensorOwned("one_shot_block_output")

      var ctx = InferenceContext.init(24, 1, 2, 512, 256)

      # The per-step f32 states left the payload under the fixture contract: their external
      # surface is the descriptor sidecar, one entry per recorded trajectory step, steps 3 to 5,
      # which is the minimum the prefill-then-decode property consumes. The state checks
      # re-express against it, assertStats carrying the fingerprint plus assertDescriptors carrying
      # the bulk, signed mean, tail and probe comparisons under the four fp32 ulp cap.
      let descFile = loadFingerprintStats(
        GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor.descriptors")
      proc assertStateDesc(live: Tensor, step: int, msg: string) =
        let entry = descFile.statsTensor("one_shot_ssm_states[" & $step & "]")
        assertStats(live, entry, descriptorStatsBudget(entry.probeMode),
          msg = msg)
        assertDescriptors(live, entry, msg = msg)

      # Prefill [a, b, c]: outputs and stored state must match the one-shot trajectory at steps
      # 0..2. Outputs compare bit-exactly: the CPU depthwise conv implementation computes
      # both sides, a same-kernel same-shape replay on the reference device,
      # so equal values are the expected result and any drift is a bug.
      # The stored state compares against the recorded
      # trajectory state with the descriptors: the two evaluation orders of the recurrence drift a measured
      # few fp32 ulps at the state max, inside the cap.
      let outPrefill = gdn(ctx, prefillX)
      assertAllClose(outPrefill, oneShotLayer.narrow(1, 0, 3),
        rtol = 0.0, abstol = 0.0, msg = "prefill output != one-shot steps 0..2")
      assertStateDesc(ctx.gdnSsmState[0], 3,
        "SSM state after prefill vs one-shot step 3")
      assertAllClose(ctx.gdnConvState[0],
        st.getTensorOwned("conv_state_after_prefill_tail3")[0],
        rtol = 0.0, abstol = 0.0, msg = "conv state after prefill mismatch")
      let convStatePrefill = ctx.gdnConvState[0].clone()

      # Decode [d]: the output must equal the decode fixture and also
      # the one-shot step 3, with the stored state feeding the conv
      # output check.
      let outD = gdn(ctx, decodeXd)
      assertAllClose(outD, st.getTensorOwned("decode_output_d"),
        rtol = 0.0, abstol = 0.0, msg = "decode d output mismatch")
      assertAllClose(outD, oneShotLayer.narrow(1, 3, 1),
        rtol = 0.0, abstol = 0.0, msg = "decode d != one-shot step 3")
      let mixedD = gdn.in_proj_qkv.forward(decodeXd).transpose(1, 2)
      let catInputD = F.cat([convStatePrefill.unsqueeze(0), mixedD], -1)
      let convD = F.conv1d(catInputD, gdn.conv1d_weight,
        padding = [0], groups = gdn.conv_dim)
      let convOutD = F.silu(convD.narrow(2, convD.size(2) - 1, 1))
      assertAllClose(convOutD, st.getTensorOwned("decode_conv_output_d"),
        rtol = 0.0, abstol = 0.0, msg = "decode d conv output mismatch")
      assertStateDesc(ctx.gdnSsmState[0], 4,
        "SSM state after decode d vs one-shot step 4")
      assertAllClose(ctx.gdnConvState[0],
        st.getTensorOwned("conv_state_after_d_tail3")[0],
        rtol = 0.0, abstol = 0.0, msg = "conv state after d mismatch")
      let convStateAfterD = ctx.gdnConvState[0].clone()

      # Decode [e]: same checks against step 4 and the final states.
      let outE = gdn(ctx, decodeXe)
      assertAllClose(outE, st.getTensorOwned("decode_output_e"),
        rtol = 0.0, abstol = 0.0, msg = "decode e output mismatch")
      assertAllClose(outE, oneShotLayer.narrow(1, 4, 1),
        rtol = 0.0, abstol = 0.0, msg = "decode e != one-shot step 4")
      let mixedE = gdn.in_proj_qkv.forward(decodeXe).transpose(1, 2)
      let catInputE = F.cat([convStateAfterD.unsqueeze(0), mixedE], -1)
      let convE = F.conv1d(catInputE, gdn.conv1d_weight,
        padding = [0], groups = gdn.conv_dim)
      let convOutE = F.silu(convE.narrow(2, convE.size(2) - 1, 1))
      assertAllClose(convOutE, st.getTensorOwned("decode_conv_output_e"),
        rtol = 0.0, abstol = 0.0, msg = "decode e conv output mismatch")
      assertStateDesc(ctx.gdnSsmState[0], 5,
        "final SSM state vs one-shot step 5")
      assertAllClose(ctx.gdnConvState[0],
        st.getTensorOwned("conv_state_after_e_tail3")[0],
        rtol = 0.0, abstol = 0.0, msg = "conv state after e mismatch")
      true

  runCppTest "GDN conv history: multi-token continuation == one-shot tail":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(weights, cfgJson, 0)

      # Deterministic tiny input: arange scaled to [0, 1), bf16.
      let n = 5 * 1024
      let arange = F.arange(n, F.tensorOptions(F.kInt64, F.kCPU)).to(kFloat32)
      let x = (arange * (1.0 / float64(n)))
        .reshape([1, 5, 1024])
        .to(kBFloat16)
      let firstCall = x.narrow(1, 0, 3)
      let secondCall = x.narrow(1, 3, 2)

      # Two-call continuation: prefill 3 tokens, then a second multi-token
      # prefill of 2 tokens on the same context. The second call must keep
      # the conv history written by the first, not re-zero the window.
      var ctx2 = InferenceContext.init(24, 1, 2, 512, 256)
      discard gdn(ctx2, firstCall)
      let outTwoCall = gdn(ctx2, secondCall)

      # One-shot prefill over the concatenated 5 tokens.
      var ctx1 = InferenceContext.init(24, 1, 2, 512, 256)
      let outOneShot = gdn(ctx1, x)

      # The second call's positions must be bit-identical to the one-shot
      # tail (the conv history of the first call feeds the second).
      # One conv implementation, the CPU depthwise form, computes every position
      # from the same kernel window and weights on one device, so equal
      # values are the expected result.
      assertAllClose(outTwoCall, outOneShot.narrow(1, 3, 2),
        rtol = 0.0, abstol = 0.0,
        msg = "multi-token continuation tail != one-shot tail")
      true

  runCppTest "GDN evaluation order: one-shot prefill == step-decode, synthetic T=70":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(weights, cfgJson, 0)

      # Tier-1 property check on a seeded synthetic input. The input
      # crosses the 64-token chunk boundary: the one-shot prefill runs the chunked kernel over the whole
      # sequence, the 70-step decode runs the step-wise recurrence. The equivalence of the two computation
      # orders on the same inputs must agree. The final f32 state drift accumulates
      # linearly across the decode steps, see assertSsmEvalOrder,
      # the budget-change record in SPEC.md. The bf16 block outputs stay
      # inside the bf16 rounding scale (a few bf16 ulps) of the two orders.
      Torch.manual_seed(0x5EEDC0DE'u64)
      let x = F.randn(70 * 1024, F.tensorOptions(F.kBFloat16, F.kCPU))
        .reshape([1, 70, 1024])

      var ctxOneShot = InferenceContext.init(24, 1, 2, 512, 256)
      let outOneShot = gdn(ctxOneShot, x)

      var ctxDecode = InferenceContext.init(24, 1, 2, 512, 256)
      var outDrift = 0.0'f64
      for t in 0 ..< 70:
        let outT = gdn(ctxDecode, x.narrow(1, t, 1))
        outDrift = max(outDrift, maxAbsDiff(outT, outOneShot.narrow(1, t, 1)))
      let outMax = outOneShot.to(F.kFloat32).abs().max().item(float64)
      let outUlps = outDrift / bf16UlpAt(outMax)
      echo "    evaluation-order output drift ", outDrift, " = ", outUlps,
        " bf16 ulps at output max ", outMax
      doAssert outDrift <= 4.0 * bf16UlpAt(outMax),
        "evaluation-order output drift " & $outUlps &
        " bf16 ulps at output max exceeds the 4 ulp cap"
      assertConvEvalOrder(ctxOneShot.gdnConvState[0], ctxDecode.gdnConvState[0],
        msg = "one-shot prefill vs step-decode conv state")
      assertSsmEvalOrder(ctxOneShot.gdnSsmState[0], ctxDecode.gdnSsmState[0],
        steps = 70,
        msg = "one-shot prefill vs step-decode final state")
      true

when isMainModule:
  main()
