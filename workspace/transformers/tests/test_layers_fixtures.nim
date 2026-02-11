# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/unittest,
  std/options,
  std/os,
  std/memfiles,
  std/strformat,
  std/strutils,
  std/tables,
  workspace/safetensors as ST,
  workspace/libtorch as F,
  workspace/transformers/src/layers/all

const
  FixtureDir = currentSourcePath().parentDir() / "fixtures" / "layers"
  WeightsFile = currentSourcePath().parentDir() / "Qwen3-0.6B.model.layers.8.safetensors"
  ModelName = "Qwen3-0.6B"

proc main() =
  suite "Qwen3-0.6B layer fixture tests":
    test "RMSNorm layer fixtures":
      var weightsMemFile = memFiles.open(WeightsFile, mode = fmRead)
      defer: close(weightsMemFile)

      let (weightsSt, _) = safetensors.load(weightsMemFile)
      let inputLnWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "input_layernorm.weight")
      let postAttnWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "post_attention_layernorm.weight")

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"norm-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        let (st, dataOffset) = safetensors.load(fixtureMemFile)

        let inputHiddenStates = ST.getTensor(st, fixtureMemFile, dataOffset, "input_hidden_states")
        let expectedOutput = ST.getTensor(st, fixtureMemFile, dataOffset, "output")

        let layerPath = st.metadata.get().getOrDefault("layer", "")
        let normLayer =
          if layerPath.endsWith("post_attention_layernorm"):
            RmsNorm.init(postAttnWeight)
          elif layerPath.endsWith("input_layernorm.weight"):
            RmsNorm.init(inputLnWeight)
          else:
            raise newException(ValueError, &"Invalid layer: '{layerPath}'")

        let output = normLayer.forward(inputHiddenStates)
        check F.allClose(output, expectedOutput, rtol = 1e-3, abstol = 1e-4)
        close(fixtureMemFile)

    test "MLP layer fixtures":
      var weightsMemFile = memFiles.open(WeightsFile, mode = fmRead)
      defer: close(weightsMemFile)

      let (weightsSt, _) = safetensors.load(weightsMemFile)
      let gateWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "mlp.gate_proj.weight")
      let upWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "mlp.up_proj.weight")
      let downWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "mlp.down_proj.weight")

      let mlp = GatedMLP.init(gateWeight, upWeight, downWeight, kSilu)

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"mlp-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        let (st, dataOffset) = safetensors.load(fixtureMemFile)

        let inputX = ST.getTensor(st, fixtureMemFile, dataOffset, "input_x")
        let expectedOutput = ST.getTensor(st, fixtureMemFile, dataOffset, "output")

        let output = mlp.forward(inputX)
        check F.allClose(output, expectedOutput, rtol = 1e-3, abstol = 1e-4)
        close(fixtureMemFile)

    test "Attention layer fixtures":
      var weightsMemFile = memFiles.open(WeightsFile, mode = fmRead)
      defer: close(weightsMemFile)

      let (weightsSt, _) = safetensors.load(weightsMemFile)
      let qWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "self_attn.q_proj.weight")
      let kWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "self_attn.k_proj.weight")
      let vWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "self_attn.v_proj.weight")
      let oWeight = ST.getTensor(weightsSt, weightsMemFile, 0, "self_attn.o_proj.weight")

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
        let (st, dataOffset) = safetensors.load(fixtureMemFile)

        let hiddenStates = ST.getTensor(st, fixtureMemFile, dataOffset, "hidden_states")
        let expectedOutput = ST.getTensor(st, fixtureMemFile, dataOffset, "output")

        let batchSize = hiddenStates.size(0).int
        let seqLen = hiddenStates.size(1).int

        let basePos = F.arange(seqLen.int64, F.kInt64)
        let positions = basePos.unsqueeze(0).expand([batchSize.int64, seqLen.int64])
        let output = attn.forward(hiddenStates, positions, use_cache = false)

        check F.allClose(output, expectedOutput, rtol = 1e-3, abstol = 1e-4)
        close(fixtureMemFile)

when isMainModule:
  main()
