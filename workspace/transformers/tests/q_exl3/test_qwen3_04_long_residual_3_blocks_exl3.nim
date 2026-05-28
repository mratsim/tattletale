## Test Qwen3 EXL3: 3-block long residual stream chain.
##
## Chains 3 EXL3 transformer blocks sequentially vs fixtures.

import
  std/memfiles,
  std/strformat,
  std/tables,
  std/os,
  std/options,
  std/importutils,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(TransformerBlock)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-ids-inference" / "Qwen3-0.6B-EXL3-5bpw"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  Tol = 1e-2

proc main() =
  # Due to floating point associativity issue, rounding and
  # warp-shuffle reduction, the tests cannot match on CPU
  # and tests against EXL3 fixtures MUST be done with Cuda backend.

  runTest "Qwen3-0.6B-EXL3-5bpw: 3-block long residual chain":
    proc(): bool =
      let model = loadQwen3ModelRaw($ModelPath, kCuda)

      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim)

      let pool = PagePool.init(
        64, num_layers = 3,
        kv_heads = model.config.num_key_value_heads,
        head_dim = model.config.head_dim,
        dtype = F.kFloat16, device = F.kCuda)
      let numPages = ceilDiv(4096, TokensPerPage)

      # Borrow pages once — reused across all layers (each writes to own layerIdx slice)
      for i in 0 ..< numPages:
        ctx.pages.add(pool.borrow())

      let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0).to(kInt64).to(kCuda)
      let x = model.embedTokens(inputIds)
      var hidden = x
      var residual: Option[Tensor] = none(Tensor)

      echo "Comparing 3-block EXL3 long residual chain..."
      echo "================================================================="

      for layerIdx in 0..2:
        let fixturePath = FixtureDir / &"layer-{layerIdx:02d}.safetensor"
        var memFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(memFile)
        let st = safetensors.load(memFile)
        let fixtureInput = st.getTensorOwned("layer_input", kCuda)
        let fixtureOutput = st.getTensorOwned("layer_output", kCuda)

        let layer = model.layers[layerIdx]

        let nimInput = if residual.isSome:
          hidden + residual.unsafeGet
        else:
          hidden
        let inputDiff = (nimInput.to(kFloat32) - fixtureInput.to(kFloat32)).abs().max().item(float)
        echo &"Layer {layerIdx:02d}: input_diff={inputDiff:.2e}"

        if inputDiff > Tol:
          raise newException(ValueError,
            &"Layer {layerIdx:02d}: input diff = {inputDiff:.6e} (tol={Tol})")

        # Reset positional state — keep pages
        ctx.kv_position = 0
        ctx.position_ids = nil
        let pos_ids = arange(hidden.size(1)).unsqueeze(0).to(kInt64).to(kCuda)
        ctx.position_ids = pos_ids
        ctx.setRopeForPositions(model.rotary)

        let (output, newResidual) = layer(ctx, hidden, residual)
        let nimSum = output + newResidual
        let outputDiff = (fixtureOutput.to(kFloat32) - nimSum.to(kFloat32)).abs().max().item(float)
        echo &"  output + residual diff={outputDiff:.2e}"

        if outputDiff > Tol:
          raise newException(ValueError,
            &"Layer {layerIdx:02d}: output + residual diff = {outputDiff:.6e} (tol={Tol})")

        hidden = output
        residual = some(newResidual)

      echo "================================================================="
      echo &"✓ 3 blocks PASSED within tolerance ({Tol})"
      true

when isMainModule:
  main()
