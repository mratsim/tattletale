# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 layer-internals suite for Qwen3-0.6B-EXL3-5bpw: the linear, attention
## and decoder-block fixtures of the exl3-01-layer-internals family, compared
## through the fingerprint stats sidecars plus the elementwise match-rate
## against the raw payloads. The exl3 families use the bf16 bounds with the
## ulp unit taken in fp16, because EXL3 dequantizes to fp16.
##
## Run:
##   TTT_TEST_ON=cpu nim test_tf_exl3_qwen3_03_layers

import
  std/options,
  std/os,
  std/strformat,
  std/strutils,
  std/importutils,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-01-layer-internals" / "Qwen3-0.6B-EXL3-5bpw-layer-0"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  ModelName = "Qwen3-0.6B-EXL3-5bpw"

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])
privateAccess(GatedDenseFFN)

proc statsPath(fixturePath: string): string =
  ## The sidecar path as loadFingerprintStats expects it: the fixture path
  ## without the .json.zst suffix.
  fixturePath & ".stats"

proc elementwise(actual, expected: Tensor, sameDevice: bool, reductionLen: int,
    msg: string) =
  ## Elementwise comparison by device class: a run on the recording device
  ## (the exl3 payloads are recorded from the production CUDA kernel) takes
  ## the per-op ulp row, a cross-device run takes the chain checkpoint band
  ## (the SPEC cross-device class of composed paths). Chained stages
  ## accumulate drift linearly, so the band grows with the stage depth.
  if meanAbsValue(expected) == 0.0:
    # The zeros-input cases produce all-zero outputs, a pure function of the
    # recorded bytes on any device: compared value-equal.
    assertWithinBudget(actual, expected,
      ToleranceBudget(tier: ctBitExact), msg = msg)
  elif sameDevice:
    assertMatchRate(actual, expected, exl3Budget(obPostResidual), msg = msg)
  else:
    discard assertChainCheckpoint(actual, expected, 1,
      reductionLen = reductionLen, msg = msg, rtol = ChainCheckpointRtolF16)

proc main() =
  let dev = testDevice()
  let sameDevice = dev == F.kCUDA
  echo compareLine(FixtureDir, dev)

  let model = loadQwen3ModelRaw($ModelPath, dev)

  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 Linear layer fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "EXL3 Linear layer fixtures":
    proc(): bool =
      let layer = model.layers[0]
      let projNames = @[
        ("self_attn.q_proj", layer.sequence_mixer.q_proj),
        ("self_attn.k_proj", layer.sequence_mixer.k_proj),
        ("self_attn.v_proj", layer.sequence_mixer.v_proj),
        ("self_attn.o_proj", layer.sequence_mixer.o_proj),
        ("mlp.gate_proj", layer.hidden_mixer.gate_proj),
        ("mlp.up_proj", layer.hidden_mixer.up_proj),
        ("mlp.down_proj", layer.hidden_mixer.down_proj),
      ]

      for (projName, linear) in projNames:
        let fixturePrefix = &"linear-{projName}-{ModelName}"
        echo &"\n  {projName}: in={linear.in_features}, out={linear.out_features}, fmt={linear.quant_format}"

        for caseNum in 0..3:
          let fixturePath = FixtureDir / &"{fixturePrefix}-{caseNum:02d}.safetensor"
          if not fileExists(fixturePath): continue

          var st = Safetensor.open(fixturePath)

          let input = st.getTensorOwned("input", dev)
          let expectedOutput = st.getTensorOwned("output", dev)
          let output = linear(input)

          elementwise(output, expectedOutput, sameDevice, linear.in_features,
            &"Linear {projName} case {caseNum}")
          let statsFile = loadFingerprintStats(statsPath(fixturePath))
          if sameDevice:
            assertStats(output, statsFile.statsTensor("output"),
              exl3Budget(obPostResidual),
              msg = &"Linear {projName} case {caseNum} stats")
          else:
            assertStatsChainBand(output, statsFile.statsTensor("output"), 1,
    rtol = ChainCheckpointRtolF16,
              msg = &"Linear {projName} case {caseNum} stats")
          echo &"    case {caseNum}: PASSED"

      true

  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 Attention layer fixtures (RopeGQAttention)
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "EXL3 Attention layer fixtures":
    proc(): bool =
      let attn = model.layers[0].sequence_mixer
      let rotary = model.rotary

      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim)

      let pool = PagePool.init(
        64, num_layers = 1,
        kv_heads = model.config.num_key_value_heads,
        head_dim = model.config.head_dim,
        dtype = F.kFloat16, device = dev)
      let numPages = ceilDiv(4096, TokensPerPage)

      # Borrow pages once — reused across all batch items
      for i in 0 ..< numPages:
        ctx.pages.add(pool.borrow())

      for caseNum in 0..1:
        let fixturePath = FixtureDir / &"attn-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo &"    Skipping attn case {caseNum} (no fixture)"
          continue

        var st = Safetensor.open(fixturePath)

        let hiddenStates = st.getTensorOwned("hidden_states", dev)
        let expectedOutput = st.getTensorOwned("output", dev)
        let hfCos = st.getTensorOwned("cos", dev)
        let hfSin = st.getTensorOwned("sin", dev)
        let hfPosIds = st.getTensorOwned("position_ids", dev)

        let batch = hiddenStates.size(0)
        var outputs: seq[Tensor] = @[]
        for b in 0..<batch:
          # Reset positional state — keep pages
          ctx.kv_position = 0
          ctx.position_ids = hfPosIds[b]
          ctx.setRopeForPositions(rotary)

          let hfCos2d = if hfCos.dim == 3: hfCos[b] else: hfCos
          let hfSin2d = if hfSin.dim == 3: hfSin[b] else: hfSin
          assertAllClose(ctx.cos, hfCos2d, rtol = 1e-5, abstol = 1e-5,
            msg = &"RoPE cos/sin mismatch (case {caseNum}, batch {b})")
          assertAllClose(ctx.sin, hfSin2d, rtol = 1e-5, abstol = 1e-5,
            msg = &"RoPE cos/sin mismatch (case {caseNum}, batch {b})")

          let x = hiddenStates[b].unsqueeze(0)
          let o = attn(ctx, x)
          outputs.add(o)

        let finalOutput = F.cat(outputs)
        doAssert finalOutput.shape == expectedOutput.shape,
          &"Shape mismatch: {finalOutput.shape} vs {expectedOutput.shape}"

        elementwise(finalOutput, expectedOutput, sameDevice,
          model.config.hidden_size, &"Attention case {caseNum}")
        let statsFile = loadFingerprintStats(statsPath(fixturePath))
        if sameDevice:
          assertStats(finalOutput, statsFile.statsTensor("output"),
            exl3Budget(obAttention),
            msg = &"Attention case {caseNum} stats")
        else:
          assertStatsChainBand(finalOutput, statsFile.statsTensor("output"), 1,
    rtol = ChainCheckpointRtolF16,
            msg = &"Attention case {caseNum} stats")

        echo &"Attention case {caseNum} (batch={batch}, seq={hiddenStates.size(1)}): PASSED"

      true

  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 DecoderLayer fixtures (long residual stream)
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "EXL3 DecoderLayer fixtures":
    proc(): bool =
      let layer = model.layers[0]
      let rotary = model.rotary

      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim)

      let poolLayer = PagePool.init(
        64, num_layers = 1,
        kv_heads = model.config.num_key_value_heads,
        head_dim = model.config.head_dim,
        dtype = F.kFloat16, device = dev)
      let numPagesLayer = ceilDiv(4096, TokensPerPage)

      # Borrow pages once -- reused across all batch items
      for i in 0 ..< numPagesLayer:
        ctx.pages.add(poolLayer.borrow())

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"transformer-block-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo &"    Skipping block case {caseNum} (no fixture)"
          continue

        var st = Safetensor.open(fixturePath)

        let inputHiddenStates = st.getTensorOwned("input_hidden_states", dev)
        let expectedOutput = st.getTensorOwned("output", dev)
        let expectedOutputResidual = st.getTensorOwned("output_residual", dev)
        let hfPosIds = st.getTensorOwned("position_ids", dev)

        let batch = inputHiddenStates.size(0)
        var outputs: seq[Tensor] = @[]
        var outputResiduals: seq[Tensor] = @[]
        for b in 0..<batch:
          # Reset positional state -- keep pages
          ctx.kv_position = 0
          let x = inputHiddenStates[b].unsqueeze(0)
          ctx.position_ids = hfPosIds[b]
          ctx.setRopeForPositions(rotary)

          let residualTensor = st.getTensorOwned("residual", dev)
          let residual = some(residualTensor[b].unsqueeze(0))

          let (o, oRes) = layer(ctx, x, residual)
          outputs.add(o)
          outputResiduals.add(oRes)

        let finalOutput = F.cat(outputs)
        let finalOutputResidual = F.cat(outputResiduals)

        doAssert finalOutput.shape == expectedOutput.shape, &"Shape mismatch: {finalOutput.shape} vs {expectedOutput.shape}"
        doAssert finalOutputResidual.shape == expectedOutputResidual.shape, &"Shape mismatch: {finalOutputResidual.shape} vs {expectedOutputResidual.shape}"

        elementwise(finalOutput, expectedOutput, sameDevice,
          model.config.intermediate_size, &"Layer output case {caseNum}")
        elementwise(finalOutputResidual, expectedOutputResidual, sameDevice,
          model.config.hidden_size, &"Layer output_residual case {caseNum}")
        let statsFile = loadFingerprintStats(statsPath(fixturePath))
        if sameDevice:
          assertStats(finalOutput, statsFile.statsTensor("output"),
            exl3Budget(obPostResidual),
            msg = &"Layer output case {caseNum} stats")
          assertStats(finalOutputResidual, statsFile.statsTensor("output_residual"),
            exl3Budget(obPostResidual),
            msg = &"Layer output_residual case {caseNum} stats")
        else:
          assertStatsChainBand(finalOutput, statsFile.statsTensor("output"), 1,
    rtol = ChainCheckpointRtolF16,
            msg = &"Layer output case {caseNum} stats")
          assertStatsChainBand(finalOutputResidual,
            statsFile.statsTensor("output_residual"), 1,
            msg = &"Layer output_residual case {caseNum} stats")

        echo &"    Layer case {caseNum}: PASSED"

      true

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All EXL3 layer tests completed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
