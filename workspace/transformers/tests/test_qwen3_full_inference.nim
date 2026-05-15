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
## All layers should match with tolerance 1e-5.

import
  std/memfiles,
  std/strformat,
  std/tables,
  std/os,
  std/options,
  std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(TransformerBlock)
privateAccess(RopeGQAttention)

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

      # InferenceContext for stateful attention
      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim,
        dtype = F.kBFloat16, device = F.kCPU
      )

      # Input tokens: "Hello, how are you?"
      let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)

      # Embedding pass
      let x = model.embedTokens(inputIds)
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

        # Prepare InferenceContext for this layer
        ctx.reset()
        let pos_ids = arange(hidden.size(1)).unsqueeze(0).to(kInt64)
        ctx.position_ids = pos_ids
        ctx.setRopeForPositions(layer.self_attn.rotary)

        # Forward through layer (long residual stream pattern)
        let (output, newResidual) = layer(ctx, hidden, residual)

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
      let finalNorm = model.norm(hidden + finalResidual)
      let finalLogits = model.lmHead(finalNorm)
      echo &"  Nim logits mean: {finalLogits.mean().item(float):.6f}"
      echo &"  Nim logits shape: {finalLogits.shape}"

      # Load HF logits fixture
      let logitsFixturePath = FixtureDir / "final_logits.safetensor"
      var logitsMemFile = memFiles.open(logitsFixturePath, mode = fmRead)
      defer: close(logitsMemFile)
      let logitsSt = safetensors.load(logitsMemFile)
      let hfLogits = logitsSt.getTensorOwned("logits", kCPU)
      echo &"  HF  logits mean: {hfLogits.mean().item(float):.6f}"
      echo &"  HF  logits shape: {hfLogits.shape}"

      let logitsDiff = (finalLogits.to(kFloat32) - hfLogits.to(kFloat32)).abs().max().item(float)
      echo &"  max_diff: {logitsDiff:.6e}"

      if logitsDiff > tol:
        raise newException(ValueError, &"Logits diff = {logitsDiff:.6e} (tol={tol})")

      echo "✓ PASS: All layers + logits match within tolerance (" & $tol & ")"
      true

when isMainModule:
  main()
