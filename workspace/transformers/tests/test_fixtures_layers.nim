# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/options,
  std/os,
  std/memfiles,
  std/strformat,
  std/strutils,
  std/tables,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch/vendor/libtorch,
  workspace/transformers/src/layers/all,
  workspace/transformers/src/layers/rope {.all.},
  workspace/positron,
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / "fixtures" / "layers" / "Qwen3-0.6B-layer-8"
  EmbedLmHeadFixtureDir = currentSourcePath().parentDir() / "fixtures" / "layers" / "Qwen3-0.6B-embed-lmhead"
  TransformerBlockFixtureDir = currentSourcePath().parentDir() / "fixtures" / "layers" / "Qwen3-0.6B-block-8"
  ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B" / "model.safetensors"
  ModelName = "Qwen3-0.6B"
proc main() =
  runTest "RMSNorm layer fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let inputLnWeight = weightsSt.getTensorOwned("model.layers.8.input_layernorm.weight")
      let postAttnWeight = weightsSt.getTensorOwned("model.layers.8.post_attention_layernorm.weight")

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"norm-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        var st = safetensors.load(fixtureMemFile)

        let inputHiddenStates = st.getTensorOwned("input_hidden_states")
        let expectedOutput = st.getTensorOwned("output")

        let layerPath = st.metadata.unsafeGet().getOrDefault("layer", "")
        let normLayer =
          if layerPath.endsWith("post_attention_layernorm"):
            RmsNorm.init(postAttnWeight)
          elif layerPath.endsWith("input_layernorm"):
            RmsNorm.init(inputLnWeight)
          else:
            raise newException(ValueError, &"Invalid layer: '{layerPath}'")

        var output = normLayer.forward(inputHiddenStates)
        assertAllClose(output, expectedOutput, msg = "RMSNorm case " & $caseNum & " failed")
        echo "RMSNorm case ", caseNum, " PASSED"
      true

  runTest "MLP layer fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let gateWeight = weightsSt.getTensorOwned("model.layers.8.mlp.gate_proj.weight")
      let upWeight = weightsSt.getTensorOwned("model.layers.8.mlp.up_proj.weight")
      let downWeight = weightsSt.getTensorOwned("model.layers.8.mlp.down_proj.weight")

      let mlp = GatedMLP.init(gateWeight, upWeight, downWeight, kSilu)
      assertDefined(mlp.down_proj.weight)
      assertDefined(mlp.gate_up_proj.weight)

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"mlp-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)

        var st = safetensors.load(fixtureMemFile)

        let inputX = st.getTensorOwned("input_x")
        let expectedOutput = st.getTensorOwned("output")

        let output = mlp.forward(inputX)
        assertAllClose(output, expectedOutput)
        echo "MLP case ", caseNum, " PASSED"
      true

  runTest "Attention layer fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let inputLnWeight = weightsSt.getTensorOwned("model.layers.8.input_layernorm.weight")
      let qWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_proj.weight")
      let kWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_proj.weight")
      let vWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.v_proj.weight")
      let oWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.o_proj.weight")
      let qNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_norm.weight")
      let kNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_norm.weight")

      # input_layernorm should be applied BEFORE attention (matching HF)
      let inputLn = RmsNorm.init(inputLnWeight)

      let numQoHeads = 16
      let numKvHeads = 8
      let headDim = 128
      let ropeTheta = 1_000_000.0

      var rotary = RotaryPositionEmbedding.init(headDim, 40960, ropeTheta, F.kBFloat16, F.kCPU)

      var attn: RopeGQAttention
      attn = RopeGQAttention.init(qWeight, kWeight, vWeight, oWeight, qNormWeight, kNormWeight, numQoHeads, numKvHeads, headDim, rotary, rms_norm_eps = 1e-6)

      for caseNum in 0..1:
        let fixturePath = FixtureDir / &"attn-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)

        var st = safetensors.load(fixtureMemFile)

        let hiddenStates = st.getTensorOwned("hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        let cos = st.getTensorOwned("cos")
        let sin = st.getTensorOwned("sin")

        # Fixture was generated WITHOUT input_layernorm (Qwen3Attention receives raw hidden_state)
        # The input_layernorm is applied at the decoder layer level, not inside attention
        # Use pre-computed cos/sin from fixture for exact match
        attn.resetCache()
        attn.rotary.setCache(cos, sin)
        let output = attn.forward(hiddenStates)
        assertAllClose(output, expectedOutput, msg = "Attention case " & $caseNum & " failed")
        echo "Attention case ", caseNum, " PASSED"
      true

  runTest "Embedding + LMHead fixtures":
    proc(): bool =
      # Load embed_tokens.weight from main model
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let embedWeight = weightsSt.getTensorOwned("model.embed_tokens.weight")

      let embedding = Embedding.init(embedWeight)
      let lmhead = LMHead.initTied(embedding)

      for caseNum in 0..1:
        let fixturePath = EmbedLmHeadFixtureDir / &"embed-lmhead-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo "  Skipping: ", fixturePath
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        var st = safetensors.load(fixtureMemFile)

        # Test embedding
        let embedInput = st.getTensorOwned("embed_input_ids")
        let embedExpected = st.getTensorOwned("embed_output")
        let embedOutput = embedding.forward(embedInput)
        assertAllClose(embedOutput, embedExpected, msg = "Embedding case " & $caseNum & " failed")

        # Test LMHead
        let lmheadInput = st.getTensorOwned("lmhead_input")
        let lmheadExpected = st.getTensorOwned("lmhead_output")
        let lmheadOutput = lmhead.forward(lmheadInput)
        assertAllClose(lmheadOutput, lmheadExpected, msg = "LMHead case " & $caseNum & " failed")

        echo "Embedding + LMHead case ", caseNum, " PASSED"
      true

  runTest "TransformerBlock fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let inputLnWeight = weightsSt.getTensorOwned("model.layers.8.input_layernorm.weight")
      let postAttnWeight = weightsSt.getTensorOwned("model.layers.8.post_attention_layernorm.weight")
      let qWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_proj.weight")
      let kWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_proj.weight")
      let vWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.v_proj.weight")
      let oWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.o_proj.weight")
      let qNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_norm.weight")
      let kNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_norm.weight")
      let gateWeight = weightsSt.getTensorOwned("model.layers.8.mlp.gate_proj.weight")
      let upWeight = weightsSt.getTensorOwned("model.layers.8.mlp.up_proj.weight")
      let downWeight = weightsSt.getTensorOwned("model.layers.8.mlp.down_proj.weight")

      let numQoHeads = 16
      let numKvHeads = 8
      let headDim = 128
      let ropeTheta = 1_000_000.0

      var rotary = RotaryPositionEmbedding.init(headDim, 40960, ropeTheta, F.kBFloat16, F.kCPU)

      # Initialize sublayers
      let attn_norm = RmsNorm.init(inputLnWeight)
      var attn = RopeGQAttention.init(qWeight, kWeight, vWeight, oWeight, qNormWeight, kNormWeight, numQoHeads, numKvHeads, headDim, rotary, rms_norm_eps = 1e-6)
      let mlp_norm = RmsNorm.init(postAttnWeight)
      let mlp = GatedMLP.init(gateWeight, upWeight, downWeight, kSilu)

      # Create TransformerBlock
      # Create TransformerBlock
      var transBlock = TransformerBlock.init(attn_norm, attn, mlp_norm, mlp)

      for caseNum in 0..3:
        let fixturePath = TransformerBlockFixtureDir / &"transformer-block-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)

        var st = safetensors.load(fixtureMemFile)

        let inputHiddenStates = st.getTensorOwned("input_hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        let expectedOutputResidual = st.getTensorOwned("output_residual")
        let cos = st.getTensorOwned("cos")
        let sin = st.getTensorOwned("sin")

        # Handle residual: fixture always has it, but for "no residual" cases it's a clone of input
        let residualOpt = some(st.getTensorOwned("residual"))

        # Reset cache and set RoPE cos/sin from fixture
        transBlock.attn.resetCache()
        transBlock.attn.rotary.setCache(cos, sin)

        # Run forward pass
        let (output, outputResidual) = transBlock.forward(inputHiddenStates, residualOpt)

        # Validate outputs within BF16 tolerance
        assertAllClose(output, expectedOutput, rtol = 5e-2, abstol = 5e-2, msg = "TransformerBlock output case " & $caseNum & " failed")
        assertAllClose(outputResidual, expectedOutputResidual, rtol = 5e-2, abstol = 5e-2, msg = "TransformerBlock output_residual case " & $caseNum & " failed")

        echo "TransformerBlock case ", caseNum, " PASSED"
      true
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All tests completed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
