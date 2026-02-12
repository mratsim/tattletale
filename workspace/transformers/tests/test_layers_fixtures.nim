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
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch/vendor/libtorch,
  workspace/transformers/src/layers/all,
  ./common_utils

const
  FixtureDir = currentSourcePath().parentDir() / "fixtures" / "layers" / "Qwen3-0.6B-layer-8"
  WeightsFile = FixtureDir / "Weights-Qwen3-0.6B-layer-8.safetensor"
  ModelName = "Qwen3-0.6B"

proc main() =
  # runTest "RMSNorm layer fixtures":
  #   proc(): bool =
  #     echo "  File: ", WeightsFile
  #     var weightsMemFile = memFiles.open(WeightsFile, mode = fmRead)
  #     defer: close(weightsMemFile)

  #     let (weightsSt, _) = safetensors.load(weightsMemFile)
  #     let inputLnWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "input_layernorm.weight")
  #     let postAttnWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "post_attention_layernorm.weight")

  #     for caseNum in 0..3:
  #       let fixturePath = FixtureDir / &"norm-{ModelName}-{caseNum:02d}.safetensor"
  #       if not fileExists(fixturePath):
  #         continue

  #       var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
  #       let (st, dataOffset) = safetensors.load(fixtureMemFile)

  #       let inputHiddenStates = ST.getTensor(st, fixtureMemFile, dataOffset, "input_hidden_states")
  #       let expectedOutput = ST.getTensor(st, fixtureMemFile, dataOffset, "output")

  #       let layerPath = st.metadata.get().getOrDefault("layer", "")
  #       let normLayer =
  #         if layerPath.endsWith("post_attention_layernorm"):
  #           RmsNorm.init(postAttnWeight)
  #         elif layerPath.endsWith("input_layernorm"):
  #           RmsNorm.init(inputLnWeight)
  #         else:
  #           raise newException(ValueError, &"Invalid layer: '{layerPath}'")

  #       # Note: Transformers does the hidden_state + residual while we don't
  #       var output = normLayer.forward(inputHiddenStates)
  #       output += inputHiddenStates
  #       let allClose = F.allClose(output, expectedOutput, rtol = 1e-3, abstol = 1e-4)
  #       echo "allClose: ", allClose
  #       doAssert allClose, block:
  #         "RMSNorm case " & $caseNum & " failed\n" &
  #         "--------------------------------------------\n" &
  #         "Output[0, 0..<5, 0..<5]:\n" & $output[0, 0..<5, 0..<5] &
  #         "\n--------------------------------------------\n" &
  #         "Expected[0, 0..<5, 0..<5]:\n" & $expectedOutput[0, 0..<5, 0..<5] &
  #         "\n--------------------------------------------\n"
  #       close(fixtureMemFile)

  runTest "MLP layer fixtures":
    proc(): bool =
      var weightsMemFile = memFiles.open(WeightsFile, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let gateWeight = weightsSt.getTensor("mlp.gate_proj.weight")
      let upWeight = weightsSt.getTensor("mlp.up_proj.weight")
      let downWeight = weightsSt.getTensor("mlp.down_proj.weight")

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

        let inputX = st.getTensor("input_x")
        let expectedOutput = st.getTensor("output")

        let output = mlp.forward(inputX)
        assertAllClose(output, expectedOutput)
        echo "MLP case ", caseNum, " PASSED"
      true

  runTest "Attention layer fixtures":
    proc(): bool =
      var weightsMemFile = memFiles.open(WeightsFile, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let qWeight = weightsSt.getTensor("self_attn.q_proj.weight")
      let kWeight = weightsSt.getTensor("self_attn.k_proj.weight")
      let vWeight = weightsSt.getTensor("self_attn.v_proj.weight")
      let oWeight = weightsSt.getTensor("self_attn.o_proj.weight")

      let numQoHeads = 16
      let numKvHeads = 8
      let headDim = 128
      let ropeTheta = 1_000_000.0

      var rotary = RotaryPositionEmbedding.init(headDim, 40960, ropeTheta, F.kFloat32, F.kCPU)

      var attn: RopeMHAttention
      attn = RopeMHAttention.init(qWeight, kWeight, vWeight, oWeight, numQoHeads, numKvHeads, headDim, rotary, rms_norm_eps = 1e-6)

      for caseNum in 0..1:
        let fixturePath = FixtureDir / &"attn-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        var st = safetensors.load(fixtureMemFile)

        let hiddenStates = st.getTensor("hidden_states")
        let expectedOutput = st.getTensor("output")

        let batchSize = hiddenStates.size(0).int
        let seqLen = hiddenStates.size(1).int

        let basePos = F.arange(seqLen.int64, F.kInt64)
        let positions = basePos.unsqueeze(0).expand([batchSize.int64, seqLen.int64])
        let output = attn.forward(hiddenStates, positions, use_cache = false)
        assertAllClose(output, expectedOutput, msg = "Attention case " & $caseNum & " failed")
        close(fixtureMemFile)
      true

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All tests completed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
