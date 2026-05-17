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
  std/tables,
  std/importutils,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch/vendor/libtorch,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/linear {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/src/deserialization,
  workspace/transformers/src/quantizations/datatypes,
  workspace/positron/src/activations,
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-layers" / "Qwen3-0.6B-EXL3-5bpw-layer-0"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw" / "model.safetensors"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  ModelName = "Qwen3-0.6B-EXL3-5bpw"

{.experimental: "callOperator".}
privateAccess(Linear)
const Tol = 1e-4
const TolAttn = 5e-3  # SDPA (F.scaled_dot_product_attention) differs from production kernel by ~0.001 in fp16 softmax
const TolBlock = 5e-3  # Compounds 7 linears + SDPA + RoPE + RMSNorm + SiLU; SDPA varies from production kernel

proc linearToDevice(l: Linear) =
  ## Move Linear layer tensors to CUDA.
  l.weight = l.weight.cuda()
  case l.quant_format
  of qBF16:
    if l.bias.isSome:
      l.bias.get() = l.bias.get().cuda()
  of qExl3:
    l.suh = l.suh.cuda()
    l.svh = l.svh.cuda()
    if l.bias.isSome:
      l.bias.get() = l.bias.get().cuda()

proc main() =
  let cfgJson = (ModelDir / "config.json").parseFile()

  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 Linear layer fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runTest "EXL3 Linear layer fixtures":
    proc(): bool =
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)
      var weightsSt = safetensors.load(weightsMemFile)

      let projNames = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
      ]

      for projName in projNames:
        let fixturePrefix = &"linear-{projName}-{ModelName}"
        var linear = Linear.load(weightsSt, cfgJson, "model.layers.0." & projName)
        linearToDevice(linear)

        echo &"\n  {projName}: in={linear.in_features}, out={linear.out_features}, fmt={linear.quant_format}"

        for caseNum in 0..3:
          let fixturePath = FixtureDir / &"{fixturePrefix}-{caseNum:02d}.safetensor"
          if not fileExists(fixturePath): continue

          var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
          defer: close(fixtureMemFile)
          var st = safetensors.load(fixtureMemFile)

          var input = st.getTensorOwned("input").cuda()
          var expectedOutput = st.getTensorOwned("output").cuda()
          let output = linear(input)

          assertAllClose(output, expectedOutput, rtol = Tol, abstol = Tol,
            msg = &"Linear {projName} case {caseNum} failed")
          echo &"    case {caseNum}: PASSED"

      true


  # ──────────────────────────────────────────────────────────────────────────
  # EXL3 Attention layer fixtures (RopeGQAttention)
  #
  # Tests RopeGQAttention with InferenceContext, KV cache, and RoPE.
  #
  # Pattern (matches bf16 test):
  #   1. Get RoPE from model level (shared across all layers)
  #   2. Set position_ids on InferenceContext
  #   3. Compute cos/sin via rotary.ropeByPositions(ctx.position_ids)
  #   4. Verify cos/sin match fixture reference
  #   5. Pass cos/sin to attn(ctx, x)
  #
  # TODO: Batch processing — currently one batch at a time
  # ──────────────────────────────────────────────────────────────────────────
  runTest "EXL3 Attention layer fixtures":
    proc(): bool =
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)
      var weightsSt = safetensors.load(weightsMemFile)

      let model = loadQwen3ModelRaw($ModelDir, kCUDA)
      privateAccess(Qwen3Model)
      let rotary = model.rotary

      let numQoHeads = model.config.num_attention_heads
      let numKvHeads = model.config.num_key_value_heads
      let headDim = model.config.head_dim

      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = numKvHeads,
        max_seq = 4096, head_dim = headDim,
        dtype = F.kFloat16, device = F.kCUDA
      )

      let qNormWeight = RmsNorm.load(weightsSt, cfgJson, "model.layers.0.self_attn.q_norm")
      let kNormWeight = RmsNorm.load(weightsSt, cfgJson, "model.layers.0.self_attn.k_norm")

      var qProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.q_proj")
      var kProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.k_proj")
      var vProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.v_proj")
      var oProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.o_proj")
      linearToDevice(qProj); linearToDevice(kProj); linearToDevice(vProj); linearToDevice(oProj)

      var attn = RopeGQAttention.init(
        0, "model.layers.0.self_attn",
        qProj, kProj, vProj, oProj,
        qNormWeight.cuda(), kNormWeight.cuda(),
        numQoHeads, numKvHeads, headDim, rotary,
        rms_norm_eps = model.config.rms_norm_eps
      )

      for caseNum in 0..1:
        let fixturePath = FixtureDir / &"attn-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo &"    Skipping attn case {caseNum} (no fixture)"
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        var st = safetensors.load(fixtureMemFile)

        let hiddenStates = st.getTensorOwned("hidden_states").cuda()
        let expectedOutput = st.getTensorOwned("output").cuda()
        let hfCos = st.getTensorOwned("cos").cuda()
        let hfSin = st.getTensorOwned("sin").cuda()
        let hfPosIds = st.getTensorOwned("position_ids").cuda()

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
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)
      var weightsSt = safetensors.load(weightsMemFile)

      let model = loadQwen3ModelRaw($ModelDir, kCUDA)
      privateAccess(Qwen3Model)
      let rotary = model.rotary

      let numQoHeads = model.config.num_attention_heads
      let numKvHeads = model.config.num_key_value_heads
      let headDim = model.config.head_dim

      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = numKvHeads,
        max_seq = 4096, head_dim = headDim,
        dtype = F.kFloat16, device = F.kCUDA
      )

      let inputLnWeight = RmsNorm.load(weightsSt, cfgJson, "model.layers.0.input_layernorm")
      let postAttnWeight = RmsNorm.load(weightsSt, cfgJson, "model.layers.0.post_attention_layernorm")
      let qNormWeight = RmsNorm.load(weightsSt, cfgJson, "model.layers.0.self_attn.q_norm")
      let kNormWeight = RmsNorm.load(weightsSt, cfgJson, "model.layers.0.self_attn.k_norm")

      var qProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.q_proj")
      var kProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.k_proj")
      var vProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.v_proj")
      var oProj = Linear.load(weightsSt, cfgJson, "model.layers.0.self_attn.o_proj")
      linearToDevice(qProj); linearToDevice(kProj); linearToDevice(vProj); linearToDevice(oProj)

      var gateProj = Linear.load(weightsSt, cfgJson, "model.layers.0.mlp.gate_proj")
      var upProj = Linear.load(weightsSt, cfgJson, "model.layers.0.mlp.up_proj")
      var downProj = Linear.load(weightsSt, cfgJson, "model.layers.0.mlp.down_proj")
      linearToDevice(gateProj); linearToDevice(upProj); linearToDevice(downProj)

      let attn_norm = RmsNorm.init(inputLnWeight.cuda())
      var attn = RopeGQAttention.init(
        0, "model.layers.0.self_attn",
        qProj, kProj, vProj, oProj,
        qNormWeight.cuda(), kNormWeight.cuda(),
        numQoHeads, numKvHeads, headDim, rotary,
        rms_norm_eps = model.config.rms_norm_eps
      )
      let mlp_norm = RmsNorm.init(postAttnWeight.cuda())
      let mlp = GatedMLP.init(gateProj, upProj, downProj, kSilu)

      var transformer = TransformerBlock.init(0, attn_norm, attn, mlp_norm, mlp)

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"transformer-block-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo &"    Skipping block case {caseNum} (no fixture)"
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        var st = safetensors.load(fixtureMemFile)

        let inputHiddenStates = st.getTensorOwned("input_hidden_states").cuda()
        let expectedOutput = st.getTensorOwned("output").cuda()
        let expectedOutputResidual = st.getTensorOwned("output_residual").cuda()
        let hfPosIds = st.getTensorOwned("position_ids").cuda()

        let batch = inputHiddenStates.size(0)
        var outputs: seq[Tensor] = @[]
        var outputResiduals: seq[Tensor] = @[]

        for b in 0..<batch:
          ctx.reset()
          let x = inputHiddenStates[b].unsqueeze(0)
          ctx.position_ids = hfPosIds[b]
          ctx.setRopeForPositions(rotary)

          let residualTensor = st.getTensorOwned("residual").cuda()
          let residual = some(residualTensor[b].unsqueeze(0))

          let (o, oRes) = transformer(ctx, x, residual)
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
