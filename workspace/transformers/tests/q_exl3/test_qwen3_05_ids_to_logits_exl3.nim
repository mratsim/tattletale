# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test Qwen3-0.6B-EXL3-5bpw: token IDs to logit inference with layer intermediates
## checked against EXL3-specific fixtures.
## Due to floating point associativity issue, rounding and
## warp-shuffle reduction, the tests cannot match on CPU
## and tests against EXL3 fixtures MUST be done with Cuda backend.

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
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(TransformerBlock)
privateAccess(RopeGQAttention)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-ids-inference" / "Qwen3-0.6B-EXL3-5bpw"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"

proc loadLayerFixture(layerIdx: int): Table[string, Tensor] =
  ## Load EXL3 layer intermediates from safetensor fixture.
  let fixturePath = FixtureDir / &"layer-{layerIdx:02d}.safetensor"
  var memFile = memFiles.open(fixturePath, mode = fmRead)
  defer: close(memFile)
  let st = safetensors.load(memFile)
  result = initTable[string, Tensor]()
  for name in st.tensors.keys():
    result[name] = st.getTensorOwned(name, kCuda)

proc main() =
  runTest "Qwen3-0.6B-EXL3-5bpw: ids-to-logits — long residual stream vs EXL3 fixtures":
    proc(): bool =
      ## Strategy:
      ## - layer_input: should match (same embedding)
      ## - layer_output + layer_residual (Nim) vs layer_output (HF): should match
      ## - EXL3 tolerance: 1e-4 (Must use Cuda due to RMSNorm warp-shuffle)

      const tol = 1e-4

      let model = loadQwen3ModelRaw($ModelPath, kCuda)

      # InferenceContext for stateful attention
      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim)

      let pool = PagePool.init(
        64, num_layers = model.config.num_hidden_layers,
        kv_heads = model.config.num_key_value_heads,
        head_dim = model.config.head_dim,
        dtype = F.kFloat16, device = F.kCuda)
      let numPages = ceilDiv(4096, TokensPerPage)

      # Borrow pages once — reused across all layers (each writes to own layerIdx slice)
      for i in 0 ..< numPages:
        ctx.pages.add(pool.borrow())

      # Input tokens: "Hello, how are you?"
      let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0).to(kCuda)

      # Embedding pass
      let x = model.embedTokens(inputIds)
      var hidden = x
      var residual: Option[Tensor] = none(Tensor)

      echo "Comparing layer-by-layer EXL3 intermediates..."
      echo "================================================================="

      for layerIdx in 0..<model.layers.len:
        let hfFixture = loadLayerFixture(layerIdx)
        var layer = model.layers[layerIdx]

        # Compare layer_input
        let nimInput = if residual.isSome():
          hidden + residual.unsafeGet()
        else:
          hidden
        let inputDiff = (nimInput.to(kFloat32) - hfFixture["layer_input"].to(kFloat32)).abs().max().item(float)
        echo &"Layer {layerIdx:02d}: input_diff={inputDiff:.2e}"

        if inputDiff > tol:
          raise newException(ValueError, &"Layer {layerIdx:02d}: layer_input diff = {inputDiff:.6e}")

        # Prepare InferenceContext for this layer — reuse pages, reset positional state
        ctx.kv_position = 0
        ctx.position_ids = nil
        let pos_ids = arange(hidden.size(1)).unsqueeze(0).to(kInt64).to(kCuda)
        ctx.position_ids = pos_ids
        ctx.setRopeForPositions(layer.self_attn.rotary)

        # Forward through layer (long residual stream pattern)
        let (output, newResidual) = layer(ctx, hidden, residual)

        # Compare: Nim (output + residual) vs EXL3 fixture (layer_output)
        let nimSum = output + newResidual
        let outputDiff = (hfFixture["layer_output"].to(kFloat32) - nimSum.to(kFloat32)).abs().max().item(float)
        echo &"  output + residual diff={outputDiff:.2e}"

        # Update state
        hidden = output
        residual = some(newResidual)

        if outputDiff > tol:
          raise newException(ValueError, &"Layer {layerIdx:02d}: output + residual diff = {outputDiff:.6e} (tol={tol})")

      # Final logits comparison
      echo "================================================================="
      echo "Final logits:"
      let finalResidual = residual.get(hidden)
      let finalNorm = model.norm(hidden + finalResidual)
      let finalLogits = model.lmHead(finalNorm)
      # Load EXL3 logits fixture to CPU (save GPU memory)
      let logitsFixturePath = FixtureDir / "final_logits.safetensor"
      var logitsMemFile = memFiles.open(logitsFixturePath, mode = fmRead)
      defer: close(logitsMemFile)
      let logitsSt = safetensors.load(logitsMemFile)
      let hfLogits = logitsSt.getTensorOwned("logits", kCPU)
      # Compare on CPU to avoid GPU OOM
      let finalLogits_cpu = finalLogits.to(kCPU).to(kFloat32)
      let hfLogits_cpu = hfLogits.to(kCPU).to(kFloat32)  # already CPU but no-op is fine
      echo &"  Nim logits mean: {finalLogits_cpu.mean().item(float):.6f}"
      echo &"  Nim logits shape: {finalLogits_cpu.shape}"
      echo &"  Fixture logits mean: {hfLogits_cpu.mean().item(float):.6f}"
      echo &"  Fixture logits shape: {hfLogits_cpu.shape}"
      let logitsDiff = (finalLogits_cpu - hfLogits_cpu).abs().max().item(float)
      echo &"  max_diff: {logitsDiff:.6e}"

      if logitsDiff > tol:
        raise newException(ValueError, &"Logits diff = {logitsDiff:.6e} (tol={tol})")

      echo "✓ PASS: All layers + logits match within EXL3 tolerance (" & $tol & ")"
      true

when isMainModule:
  main()
