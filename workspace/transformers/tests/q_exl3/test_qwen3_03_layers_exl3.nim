# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 layer tests for Qwen3-0.6B-EXL3-5bpw.
##
## Runs on CUDA via LD_PRELOAD of libtorch_cuda.so.
## Analogous to test_qwen3_03_layers.nim (bf16) but uses EXL3-quantized weights.

import
  std/options,
  std/os,
  std/memfiles,
  std/strformat,
  std/strutils,
  std/importutils,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch/vendor/libtorch,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-layers" / "Qwen3-0.6B-EXL3-5bpw-layer-0"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  ModelName = "Qwen3-0.6B-EXL3-5bpw"

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(TransformerBlock)
privateAccess(RopeGQAttention)
privateAccess(GatedMLP)

const Tol = 1e-4
const TolAttn = 5e-3  # SDPA differs from production kernel by ~0.001 in fp16 softmax
const TolBlock = 5e-3  # Compounds 7 linears + SDPA + RoPE + RMSNorm + SiLU

proc main() =
  # Model loaded once on CUDA — all layers already on the right device
  let model = loadQwen3ModelRaw($ModelPath, kCPU)

  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 Linear layer fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runTest "EXL3 Linear layer fixtures":
    proc(): bool =
      let layer = model.layers[0]
      let projNames = @[
        ("self_attn.q_proj", layer.self_attn.q_proj),
        ("self_attn.k_proj", layer.self_attn.k_proj),
        ("self_attn.v_proj", layer.self_attn.v_proj),
        ("self_attn.o_proj", layer.self_attn.o_proj),
        ("mlp.gate_proj", layer.mlp.gate_proj),
        ("mlp.up_proj", layer.mlp.up_proj),
        ("mlp.down_proj", layer.mlp.down_proj),
      ]

      for (projName, linear) in projNames:
        let fixturePrefix = &"linear-{projName}-{ModelName}"
        echo &"\n  {projName}: in={linear.in_features}, out={linear.out_features}, fmt={linear.quant_format}"

        for caseNum in 0..3:
          let fixturePath = FixtureDir / &"{fixturePrefix}-{caseNum:02d}.safetensor"
          if not fileExists(fixturePath): continue

          var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
          defer: close(fixtureMemFile)
          let st = safetensors.load(fixtureMemFile)

          let input = st.getTensorOwned("input")
          let expectedOutput = st.getTensorOwned("output")
          let output = linear(input)

          assertAllClose(output, expectedOutput, rtol = Tol, abstol = Tol,
            msg = &"Linear {projName} case {caseNum} failed")
          echo &"    case {caseNum}: PASSED"

      true

  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 Attention layer fixtures (RopeGQAttention)
  # ──────────────────────────────────────────────────────────────────────────
  runTest "EXL3 Attention layer fixtures":
    proc(): bool =
      let attn = model.layers[0].self_attn
      let rotary = model.rotary

      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim,
        dtype = F.kFloat16, device = F.kCPU
      )

      for caseNum in 0..1:
        let fixturePath = FixtureDir / &"attn-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo &"    Skipping attn case {caseNum} (no fixture)"
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        let st = safetensors.load(fixtureMemFile)

        let hiddenStates = st.getTensorOwned("hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        let hfCos = st.getTensorOwned("cos")
        let hfSin = st.getTensorOwned("sin")
        let hfPosIds = st.getTensorOwned("position_ids")

        let batch = hiddenStates.size(0)
        var outputs: seq[Tensor] = @[]

        for b in 0..<batch:
          ctx.reset()
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

        assertAllClose(finalOutput, expectedOutput, rtol = TolAttn, abstol = TolAttn,
          msg = &"Attention case {caseNum} failed")

        echo &"Attention case {caseNum} (batch={batch}, seq={hiddenStates.size(1)}): PASSED"

      true

  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 TransformerBlock fixtures (long residual stream)
  # ──────────────────────────────────────────────────────────────────────────
  runTest "EXL3 TransformerBlock fixtures":
    proc(): bool =
      let layer = model.layers[0]
      let rotary = model.rotary

      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim,
        dtype = F.kFloat16, device = F.kCPU
      )

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"transformer-block-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo &"    Skipping block case {caseNum} (no fixture)"
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        let st = safetensors.load(fixtureMemFile)

        let inputHiddenStates = st.getTensorOwned("input_hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        let expectedOutputResidual = st.getTensorOwned("output_residual")
        let hfPosIds = st.getTensorOwned("position_ids")

        let batch = inputHiddenStates.size(0)
        var outputs: seq[Tensor] = @[]
        var outputResiduals: seq[Tensor] = @[]

        for b in 0..<batch:
          ctx.reset()
          let x = inputHiddenStates[b].unsqueeze(0)
          ctx.position_ids = hfPosIds[b]
          ctx.setRopeForPositions(rotary)

          let residualTensor = st.getTensorOwned("residual")
          let residual = some(residualTensor[b].unsqueeze(0))

          let (o, oRes) = layer(ctx, x, residual)
          outputs.add(o)
          outputResiduals.add(oRes)

        let finalOutput = F.cat(outputs)
        let finalOutputResidual = F.cat(outputResiduals)

        doAssert finalOutput.shape == expectedOutput.shape, &"Shape mismatch: {finalOutput.shape} vs {expectedOutput.shape}"
        doAssert finalOutputResidual.shape == expectedOutputResidual.shape, &"Shape mismatch: {finalOutputResidual.shape} vs {expectedOutputResidual.shape}"

        assertAllClose(finalOutput, expectedOutput, rtol = TolBlock, abstol = TolBlock,
          msg = &"Block output case {caseNum} failed")
        assertAllClose(finalOutputResidual, expectedOutputResidual, rtol = TolBlock, abstol = TolBlock,
          msg = &"Block output_residual case {caseNum} failed")

        echo &"    Block case {caseNum}: PASSED"

      true

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All EXL3 layer tests completed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
