# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed under MIT or Apache v2
#
# STANDALONE REPRODUCTION: Tensor aliasing bug
#
# EXACT REPRODUCTION of the buggy test pattern from test_fixtures_layers.nim

import
  std/options,
  std/os,
  std/memfiles,
  std/strformat,
  std/strutils,
  std/tables,
  workspace/safetensors,
  workspace/libtorch,
  workspace/libtorch/src/abi/neural_nets

const
  ModelPath = "/home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/transformers/tests/hf_models/Qwen3-0.6B/model.safetensors"
  FixtureDir = "/home/beta/Programming/Perso/workspace-tattletale/tattletale/workspace/transformers/tests/fixtures/layers/Qwen3-0.6B-layer-8"

type
  NormLayer = object
    weight*: TorchTensor
    eps*: float
    hidden_size*: int

func init(_: type NormLayer, weight: TorchTensor, eps: float = 1e-6): NormLayer =
  let hidden_size = weight.size(weight.dim()-1)
  NormLayer(weight: weight, eps: eps, hidden_size: hidden_size)

proc forward(self: NormLayer, hidden_state: TorchTensor): TorchTensor =
  let normalized_shape = asTorchView(self.hidden_size)
  rms_norm(hidden_state, normalized_shape, self.weight, self.eps)

proc main() =
  echo "=== EXACT REPRODUCTION OF BUGGY TEST PATTERN ==="
  echo ""

  # EXACT pattern from test_fixtures_layers.nim (buggy version)
  # Load weights ONCE outside the loop
  var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
  defer: close(weightsMemFile)

  var weightsSt = safetensors.load(weightsMemFile)
  let inputLnWeight = weightsSt.getTensorOwned("model.layers.8.input_layernorm.weight")
  let postAttnWeight = weightsSt.getTensorOwned("model.layers.8.post_attention_layernorm.weight")

  echo "Loaded weights once:"
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

    # Create norm layer - stores weight by reference
    let normLayer =
      if layerPath.endsWith("post_attention_layernorm"):
        NormLayer.init(postAttnWeight)
      elif layerPath.endsWith("input_layernorm"):
        NormLayer.init(inputLnWeight)
      else:
        raise newException(ValueError, fmt"Invalid layer: '{layerPath}'")

    echo "After init: inputLnWeight.shape = ", inputLnWeight.shape

    # Call forward
    var output = normLayer.forward(inputHiddenStates)

    echo "After forward: inputLnWeight.shape = ", inputLnWeight.shape
    echo ""

  echo "=== If inputLnWeight.shape changed, we reproduced the bug ==="

when isMainModule:
  main()