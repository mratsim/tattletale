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
  std/tables,
  std/importutils,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/libtorch/vendor/libtorch,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/positron,
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3-0.6B-layer-8"
  EmbedLmHeadFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3-0.6B-embed-lmhead"
  TransformerBlockFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3-0.6B-block-8"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B" / "model.safetensors"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"
  ModelName = "Qwen3-0.6B"

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # RMSNorm layer fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RMSNorm layer fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let inputLnWeight = weightsSt.getTensorOwned("model.layers.8.input_layernorm.weight")
      let postAttnWeight = weightsSt.getTensorOwned("model.layers.8.post_attention_layernorm.weight")

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"norm-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        var st = safetensors.load(fixtureMemFile)

        let inputHiddenStates = st.getTensorOwned("input_hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        # Load metadata from separate JSON file (deterministic)
        let metadataPath = fixturePath & ".metadata.json"
        let metadataJson = readFile(metadataPath)
        let metadata = parseJson(metadataJson)

        let layerPath = metadata{"layer"}.getStr()
        let normLayer =
          if layerPath.endsWith("post_attention_layernorm"):
            RmsNorm.init(postAttnWeight)
          elif layerPath.endsWith("input_layernorm"):
            RmsNorm.init(inputLnWeight)
          else:
            raise newException(ValueError, &"Invalid layer: '{layerPath}'")
        var output = normLayer(inputHiddenStates)
        assertAllClose(output, expectedOutput, msg = "RMSNorm case " & $caseNum & " failed")
        echo "RMSNorm case ", caseNum, " PASSED"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # MLP layer fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runTest "MLP layer fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let gateWeight = weightsSt.getTensorOwned("model.layers.8.mlp.gate_proj.weight")
      let upWeight = weightsSt.getTensorOwned("model.layers.8.mlp.up_proj.weight")
      let downWeight = weightsSt.getTensorOwned("model.layers.8.mlp.down_proj.weight")

      let mlp = GatedMLP.init(gateWeight, upWeight, downWeight, kSilu)

      for caseNum in 0..3:
        let fixturePath = FixtureDir / &"mlp-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)

        var st = safetensors.load(fixtureMemFile)

        let inputX = st.getTensorOwned("input_x")
        let expectedOutput = st.getTensorOwned("output")

        let output = mlp(inputX)
        assertAllClose(output, expectedOutput)
        echo "MLP case ", caseNum, " PASSED"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Attention layer fixtures
  #
  # Tests attention() with InferenceContext, KV cache, and RoPE.
  #
  # Pattern:
  #   1. Get RoPE from model level (shared across all layers)
  #   2. Set position_ids on InferenceContext
  #   3. Compute cos/sin via rotary.ropeByPositions(ctx.position_ids)
  #   4. Pass cos/sin to attn(ctx, x)
  #
  # The fixture stores raw HF values (3D cos/sin, 2D position_ids).
  #
  # TODO: Batch processing — currently one batch at a time
  # Each batch item may have different position_ids (ragged batches).
  # Proper batching requires allocating KV cache for max(batch_size) and
  # running all items together. For now, process sequentially.
  # ──────────────────────────────────────────────────────────────────────────
  runTest "Attention layer fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let qWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_proj.weight")
      let kWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_proj.weight")
      let vWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.v_proj.weight")
      let oWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.o_proj.weight")
      let qNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_norm.weight")
      let kNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_norm.weight")

      # Get RoPE from model level (shared across all layers)
      let model = loadQwen3ModelRaw(ModelDir, kCPU)
      privateAccess(Qwen3Model)
      let rotary = model.rotary

      # Read head dimensions from model.config (already populated at load time)
      let numQoHeads = model.config.num_attention_heads
      let numKvHeads = model.config.num_key_value_heads
      let headDim = model.config.head_dim

      # Create InferenceContext for all layers (layer 8 self-indexes via layer_idx)
      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = numKvHeads,
        max_seq = 4096, head_dim = headDim,
        dtype = F.kBFloat16, device = F.kCPU)

      var attn: RopeGQAttention
      attn = RopeGQAttention.init(8, "model.layers.8.self_attn", qWeight, kWeight, vWeight, oWeight, qNormWeight, kNormWeight, numQoHeads, numKvHeads, headDim, rotary, rms_norm_eps = 1e-6)

      for caseNum in 0..1:
        let fixturePath = FixtureDir / &"attn-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)

        var st = safetensors.load(fixtureMemFile)

        let hiddenStates = st.getTensorOwned("hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        # Raw HF cos/sin — 3D (batch, seq, head_dim), stored as-is from HF
        let hfCos = st.getTensorOwned("cos")
        let hfSin = st.getTensorOwned("sin")
        let hfPosIds = st.getTensorOwned("position_ids")

        let batch = hiddenStates.size(0)
        var outputs: seq[Tensor] = @[]

        # Process one batch at a time
        for b in 0..<batch:
          ctx.reset()
          ctx.position_ids = hfPosIds[b]  # (seq,) for this batch item
          ctx.setRopeForPositions(rotary)

          # Verify compute() produces the same cos/sin as HF reference (batch-independent)
          let hfCos2d = if hfCos.dim == 3: hfCos[b] else: hfCos
          let hfSin2d = if hfSin.dim == 3: hfSin[b] else: hfSin
          assertAllClose(ctx.cos, hfCos2d, rtol = 1e-5, abstol = 1e-5, msg = "RoPE cos/sin mismatch (case " & $caseNum & ", batch " & $b & ")")
          assertAllClose(ctx.sin, hfSin2d, rtol = 1e-5, abstol = 1e-5, msg = "RoPE cos/sin mismatch (case " & $caseNum & ", batch " & $b & ")")

          let x = hiddenStates[b].unsqueeze(0)
          let o = attn(ctx, x)
          outputs.add(o)

        let finalOutput = F.cat(outputs)
        assertAllClose(finalOutput, expectedOutput, msg = "Attention case " & $caseNum & " failed")
        echo "Attention case ", caseNum, " PASSED"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Embedding + LMHead fixtures
  # No changes needed — stateless, no RoPE/KV cache involved
  # ──────────────────────────────────────────────────────────────────────────
  runTest "Embedding + LMHead fixtures":
    proc(): bool =
      # Load embed_tokens.weight from main model
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let embedWeight = weightsSt.getTensorOwned("model.embed_tokens.weight")

      let embedding = Embedding.init(embedWeight)
      let lmhead = LMHead.initTied(embedding)

      for caseNum in 0..1:
        let fixturePath = EmbedLmHeadFixtureDir / &"embed-lmhead-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          echo "  Skipping: ", fixturePath
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)
        var st = safetensors.load(fixtureMemFile)

        # Test embedding
        let embedInput = st.getTensorOwned("embed_input_ids")
        let embedExpected = st.getTensorOwned("embed_output")
        let embedOutput = embedding(embedInput)
        assertAllClose(embedOutput, embedExpected, msg = "Embedding case " & $caseNum & " failed")

        # Test LMHead
        let lmheadInput = st.getTensorOwned("lmhead_input")
        let lmheadExpected = st.getTensorOwned("lmhead_output")
        let lmheadOutput = lmhead(lmheadInput)
        assertAllClose(lmheadOutput, lmheadExpected, msg = "LMHead case " & $caseNum & " failed")

        echo "Embedding + LMHead case ", caseNum, " PASSED"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # TransformerBlock fixtures
  #
  # Tests full block (attn_norm → attn → mlp_norm → mlp) with long residual.
  #
  # Pattern:
  #   1. Get RoPE from model level (shared across all layers)
  #   2. Set position_ids on InferenceContext
  #   3. Compute cos/sin via rotary.ropeByPositions(ctx.position_ids)
  #   4. Pass cos/sin to transformer(ctx, x, residual)
  #
  # The fixture stores raw HF values (3D cos/sin, 2D position_ids).
  #
  # TODO: Batch processing — currently one batch at a time
  # Each batch item may have different position_ids (ragged batches).
  # Proper batching requires allocating KV cache for max(batch_size) and
  # running all items together. For now, process sequentially.
  # ──────────────────────────────────────────────────────────────────────────
  runTest "TransformerBlock fixtures":
    proc(): bool =
      # Load weights from main model (space-saving approach)
      var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
      defer: close(weightsMemFile)

      var weightsSt = safetensors.load(weightsMemFile)
      let inputLnWeight = weightsSt.getTensorOwned("model.layers.8.input_layernorm.weight")
      let postAttnWeight = weightsSt.getTensorOwned("model.layers.8.post_attention_layernorm.weight")
      let qWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_proj.weight")
      let kWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_proj.weight")
      let vWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.v_proj.weight")
      let oWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.o_proj.weight")
      let qNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_norm.weight")
      let kNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_norm.weight")
      let gateWeight = weightsSt.getTensorOwned("model.layers.8.mlp.gate_proj.weight")
      let upWeight = weightsSt.getTensorOwned("model.layers.8.mlp.up_proj.weight")
      let downWeight = weightsSt.getTensorOwned("model.layers.8.mlp.down_proj.weight")

      # Get RoPE from model level (shared across all layers)
      let model = loadQwen3ModelRaw(ModelDir, kCPU)
      privateAccess(Qwen3Model)
      let rotary = model.rotary

      # Read head dimensions from model.config (already populated at load time)
      let numQoHeads = model.config.num_attention_heads
      let numKvHeads = model.config.num_key_value_heads
      let headDim = model.config.head_dim

      # Create InferenceContext for all layers (layer 8 self-indexes via layer_idx)
      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = numKvHeads,
        max_seq = 4096, head_dim = headDim,
        dtype = F.kBFloat16, device = F.kCPU)

      # Initialize sublayers
      let attn_norm = RmsNorm.init(inputLnWeight)
      var attn = RopeGQAttention.init(8, "model.layers.8.self_attn", qWeight, kWeight, vWeight, oWeight, qNormWeight, kNormWeight, numQoHeads, numKvHeads, headDim, rotary, rms_norm_eps = 1e-6)
      let mlp_norm = RmsNorm.init(postAttnWeight)
      let mlp = GatedMLP.init(gateWeight, upWeight, downWeight, kSilu)

      # Create TransformerBlock
      var transformer = TransformerBlock.init(8, attn_norm, attn, mlp_norm, mlp)

      for caseNum in 0..3:
        let fixturePath = TransformerBlockFixtureDir / &"transformer-block-{ModelName}-{caseNum:02d}.safetensor"
        if not fileExists(fixturePath):
          continue

        var fixtureMemFile = memFiles.open(fixturePath, mode = fmRead)
        defer: close(fixtureMemFile)

        var st = safetensors.load(fixtureMemFile)

        let inputHiddenStates = st.getTensorOwned("input_hidden_states")
        let expectedOutput = st.getTensorOwned("output")
        let expectedOutputResidual = st.getTensorOwned("output_residual")
        let residual = st.getTensorOwned("residual")
        # Raw HF cos/sin — 3D (batch, seq, head_dim), stored as-is from HF
        let hfCos = st.getTensorOwned("cos")
        let hfSin = st.getTensorOwned("sin")
        let hfPosIds = st.getTensorOwned("position_ids")

        let batch = inputHiddenStates.size(0)
        var outputs: seq[Tensor] = @[]
        var outputResiduals: seq[Tensor] = @[]

        # Process one batch at a time
        for b in 0..<batch:
          ctx.reset()
          ctx.position_ids = hfPosIds[b]  # (seq,) for this batch item
          ctx.setRopeForPositions(rotary)

          # Verify compute() produces the same cos/sin as HF reference (batch-independent)
          let hfCos2d = if hfCos.dim == 3: hfCos[b] else: hfCos
          let hfSin2d = if hfSin.dim == 3: hfSin[b] else: hfSin
          assertAllClose(ctx.cos, hfCos2d, rtol = 1e-5, abstol = 1e-5, msg = "RoPE cos/sin mismatch (case " & $caseNum & ", batch " & $b & ")")
          assertAllClose(ctx.sin, hfSin2d, rtol = 1e-5, abstol = 1e-5, msg = "RoPE cos/sin mismatch (case " & $caseNum & ", batch " & $b & ")")

          let x = inputHiddenStates[b].unsqueeze(0)     # (1, seq, hidden)
          let res = residual[b].unsqueeze(0)             # (1, seq, hidden)
          let (o, oRes) = transformer(ctx, x, some(res))
          outputs.add(o)
          outputResiduals.add(oRes)

        let finalOutput = F.cat(outputs)
        let finalOutputResidual = F.cat(outputResiduals)

        # Validate outputs within BF16 tolerance
        assertAllClose(finalOutput, expectedOutput, rtol = 1e-5, abstol = 1e-5, msg = "TransformerBlock output case " & $caseNum & " failed")
        assertAllClose(finalOutputResidual, expectedOutputResidual, rtol = 1e-5, abstol = 1e-5, msg = "TransformerBlock output_residual case " & $caseNum & " failed")

        echo "TransformerBlock case ", caseNum, " PASSED"
      true

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All tests completed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
