# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test Qwen3-0.6B full inference intermediates against HF fixtures.
##
## Strategy:
## - ``layer_input``: should match exactly (same embedding)
## - ``layer_output + layer_residual`` (Nim) vs ``layer_output`` (HF): should match exactly
##   (proven invariant: ``y_long + r_long == x_local``)
## - Sublayer intermediates: EXPECTED to differ (norms see different inputs)
##
## This isolates whether the 0.1875 diff comes from:
## 1. Weight loading (if layer_input mismatches)
## 2. Sublayer implementation (if output + residual mismatches)
## 3. Expected residual pattern difference (if only sublayer norms differ)

import
  std/memfiles,
  std/strformat,
  std/tables,
  std/os,
  std/options,
  std/importutils,
  workspace/libtorch,
  workspace/safetensors,
  workspace/safetensors/src/safetensors_libtorch,
  workspace/transformers/src/layers,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / "fixtures" / "full-inference" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B"

proc loadLayerFixture(layerIdx: int): Table[string, Tensor] =
  ## Load HF layer intermediates from safetensor fixture.
  let fixturePath = FixtureDir / &"layer-{layerIdx:02d}.safetensor"
  var memFile = memFiles.open(fixturePath, mode = fmRead)
  defer: close(memFile)
  let st = safetensors.load(memFile)
  result = initTable[string, Tensor]()
  for name in st.tensors.keys():
    result[name] = st.getTensorOwned(name, kCPU)

proc main() =
  runTest "Qwen3-0.6B full inference - long residual stream vs HF":
    proc(): bool =
      ## Strategy:
      ## - layer_input: should match exactly (same embedding)
      ## - layer_output + layer_residual (Nim) vs layer_output (HF): should match exactly
      ##   (proven invariant: y_long + r_long == x_local)
      ## - after_attn_norm, after_attn, after_mlp: EXPECTED to differ
      ##   (norms see different inputs: N(x+r) vs N(x))

      const tol = 1e-5

      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)

      # Input tokens: "Hello, how are you?"
      let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)

      # Embedding pass
      let x = model.embedTokens.forward(inputIds)
      var hidden = x
      var residual: Option[Tensor] = none(Tensor)

      echo "Comparing layer-by-layer intermediates..."
      echo "================================================================="

      for layerIdx in 0..<model.layers.len:
        let hfFixture = loadLayerFixture(layerIdx)
        var layer = model.layers[layerIdx]

        # Compare layer_input
        # For layer 0: hidden is the embedding output
        # For layers 1+: hidden + residual is the boundary sum (matches HF layer_input)
        let nimInput = if residual.isSome():
          hidden + residual.unsafeGet()
        else:
          hidden
        let inputDiff = (nimInput.to(kFloat32) - hfFixture["layer_input"].to(kFloat32)).abs().max().item(float)
        echo &"Layer {layerIdx:02d}: input_diff={inputDiff:.2e}"

        if inputDiff > tol:
          raise newException(ValueError, &"Layer {layerIdx:02d}: layer_input diff = {inputDiff:.6e}")

        # Forward through layer (long residual stream pattern)
        let (output, newResidual) = layer.forward(hidden, residual)

        # Compare: Nim (output + residual) vs HF (layer_output)
        let nimSum = output + newResidual
        let outputDiff = (hfFixture["layer_output"].to(kFloat32) - nimSum.to(kFloat32)).abs().max().item(float)
        echo &"  output + residual diff={outputDiff:.2e}"

        # Update state
        hidden = output
        residual = some(newResidual)

        if outputDiff > tol:
          raise newException(ValueError, &"Layer {layerIdx:02d}: output + residual diff = {outputDiff:.6e}")

      # Final logits comparison
      echo "================================================================="
      echo "Final logits:"
      let finalResidual = residual.get(hidden)
      let finalNorm = model.norm.forward(hidden + finalResidual)
      let finalLogits = model.lmHead.forward(finalNorm)
      echo &"  Nim logits mean: {finalLogits.mean().item(float):.6f}"

      echo "✓ PASS: All layers match within tolerance (", tol, ")"
      true

when isMainModule:
  main()
