# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# STANDALONE REPRODUCTION: Tensor aliasing bug
#
# This test demonstrates tensor memory corruption.
# EXACT reproduction of the buggy pattern from test_fixtures_layers.nim

import
  std/memfiles,
  std/options,
  std/os,
  std/strformat,
  std/strutils,
  std/tables,
  workspace/libtorch as F,
  workspace/libtorch/src/abi/neural_nets,
  workspace/libtorch_testutils,
  workspace/safetensors

const
  FixtureDir = "/home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/transformers/tests/fixtures/layers/Qwen3-0.6B-layer-8"
  ModelPath = "/home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/transformers/tests/hf_models/Qwen3-0.6B/model.safetensors"

type
  NormLayer = object
    weight*: TorchTensor
    eps*: float
    hidden_size*: int

func init(_: type NormLayer, weight: TorchTensor, eps: float = 1e-6): NormLayer =
  let hidden_size = weight.size(0)
  NormLayer(weight: weight, eps: eps, hidden_size: hidden_size)

proc forward(self: NormLayer, hidden_state: TorchTensor): TorchTensor =
  let normalized_shape = asTorchView(self.hidden_size)
  rms_norm(hidden_state, normalized_shape, self.weight, self.eps)

proc main() =
  runTest "Tensor aliasing bug reproduction":
    proc(): bool =
      echo "=== STANDALONE TENSOR ALIASING REPRODUCTION ==="
      echo ""
      echo "EXACT reproduction of the buggy pattern from test_fixtures_layers.nim"
      echo ""

      # EXACT pattern from test_fixtures_layers.nim (buggy version)
      # Load weights ONCE outside the loop
      echo "Loading weights once..."
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let inputLnWeight = weightsSt.getTensorOwned("model.layers.8.input_layernorm.weight")
      let postAttnWeight = weightsSt.getTensorOwned("model.layers.8.post_attention_layernorm.weight")

      echo "Loaded weights:"
      echo "  inputLnWeight.shape = ", inputLnWeight.shape
      echo "  postAttnWeight.shape = ", postAttnWeight.shape
      echo ""

      for caseNum in 0..3:
        let fixturePath = FixtureDir / fmt"norm-Qwen3-0.6B-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        echo "--- Case ", caseNum, " ---"

        # Load fixture
        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)

        var st = safetensors.load(fixtureMemFile)
        let inputHiddenStates = st.getTensorOwned("input_hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        let layerPath = st.metadata.unsafeGet().getOrDefault("layer", "")

        echo "Before init: inputLnWeight.shape = ", inputLnWeight.shape

        # Create norm layer
        let normLayer =
          if layerPath.endsWith("post_attention_layernorm"):
            NormLayer.init(postAttnWeight)
          elif layerPath.endsWith("input_layernorm"):
            NormLayer.init(inputLnWeight)
          else:
            raise newException(ValueError, fmt"Invalid layer: '{layerPath}'")

        echo "After init: inputLnWeight.shape = ", inputLnWeight.shape

        # Call forward - THIS is what triggers the corruption
        echo "Calling forward (rms_norm)..."
        var output = normLayer.forward(inputHiddenStates)

        echo "After forward: inputLnWeight.shape = ", inputLnWeight.shape

        # ASSERTION: Check if shape changed
        let actualShape = inputLnWeight.shape
        if actualShape.len != 1 or actualShape[0] != 1024:
          echo ""
          echo "❌ BUG DETECTED: inputLnWeight.shape changed from [1024] to ", actualShape
          echo ""
          echo "ROOT CAUSE:"
          echo "  1. inputLnWeight and normLayer.weight share the same tensor"
          echo "  2. rms_norm modifies the weight tensor"
          echo "  3. The modification affects BOTH variables"
          echo ""
          return false
        echo ""

      echo "✅ Test passed - no shape corruption detected"
      true

when isMainModule:
  main()