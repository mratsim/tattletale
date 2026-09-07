# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35dense-layers \
##   --nimcache:nimcache/tests/qwen35dense-layers \
##   workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_03_layers.nim

import
  std/options,
  std/os,
  std/importutils,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models,
  workspace/transformers/src/models/all_interfaces,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/tests/transformers_testutils,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen35Model)
privateAccess(GatedDeltaNet)
privateAccess(RopeElementWiseGatedAttention[RmsNormOne])
privateAccess(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])
privateAccess(DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedDenseFFN, RmsNormOne])

const
  GdnFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3.5-0.8B-layer-0"
  Layer3FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3.5-0.8B-layer-3"
  ChainFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "long-residual-3-block" / "Qwen3.5-0.8B"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc openModelWeights(): SafetensorsCollection =
  ## Read the real Qwen3.5-0.8B safetensor file for targeted weight
  ## loads. The result owns its memory mapping for its whole lifetime,
  ## released when the value goes out of scope.
  SafetensorsCollection.open(ModelDir / "model.safetensors-00001-of-00001.safetensors")

proc loadGdn(view: SafetensorsCollection, cfgJson: JsonNode, layerIdx: int): GatedDeltaNet =
  ## Load the real layer-`layerIdx` Gated DeltaNet weights into a layer.
  let tc = cfgJson{"text_config"}
  let lp = "model.language_model.layers." & $layerIdx & ".linear_attn."
  let qkvProj = Linear.load(view, cfgJson, lp & "in_proj_qkv")
  let zProj = Linear.load(view, cfgJson, lp & "in_proj_z")
  let aProj = Linear.load(view, cfgJson, lp & "in_proj_a")
  let bProj = Linear.load(view, cfgJson, lp & "in_proj_b")
  let convWeight = view.getTensorOwned(lp & "conv1d.weight")
  let aLog = view.getTensorOwned(lp & "A_log")
  let dtBias = view.getTensorOwned(lp & "dt_bias")
  let gdnNorm = RmsNormGated.load(view, cfgJson, lp & "norm")
  let outProj = Linear.load(view, cfgJson, lp & "out_proj")
  result = GatedDeltaNet.init(
    layerIdx, lp[0 .. ^2],
    qkvProj, zProj, aProj, bProj,
    convWeight, aLog, dtBias, gdnNorm, outProj,
    tc{"linear_num_key_heads"}.getInt().int,
    tc{"linear_num_value_heads"}.getInt().int,
    tc{"linear_key_head_dim"}.getInt().int,
    tc{"linear_value_head_dim"}.getInt().int,
    tc{"linear_conv_kernel_dim"}.getInt().int)

proc loadLayer(view: SafetensorsCollection, cfgJson: JsonNode, layerIdx: int): (DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne], DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedDenseFFN, RmsNormOne]) =
  ## Load one full decoder layer (norms, mlp, and the attention variant
  ## selected by the parsed kind at `layer_types[layerIdx]`) with real
  ## weights. Exactly one pairing of the returned tuple is non-nil.
  let lp = "model.language_model.layers." & $layerIdx & "."
  let inputLN = RmsNormOne.load(view, cfgJson, lp & "input_layernorm")
  let postLN = RmsNormOne.load(view, cfgJson, lp & "post_attention_layernorm")
  let gateProj = Linear.load(view, cfgJson, lp & "mlp.gate_proj")
  let upProj = Linear.load(view, cfgJson, lp & "mlp.up_proj")
  let downProj = Linear.load(view, cfgJson, lp & "mlp.down_proj")
  let mlp = GatedDenseFFN.init(gateProj, upProj, downProj)
  let tc = cfgJson{"text_config"}
  if parseAttnFromHfTransformers(tc{"layer_types"}[layerIdx].getStr(),
      "fixture layer_types") == alkGatedDeltaNet:
    let gdn = loadGdn(view, cfgJson, layerIdx)
    result[0] = DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne].init(
      input_layernorm = inputLN, sequence_mixer = gdn,
      post_attention_layernorm = postLN, hidden_mixer = mlp)
  else:
    let rotary = RotaryPositionEmbedding.new(
      tc{"head_dim"}.getInt().int,
      tc{"max_position_embeddings"}.getInt().int,
      tc{"rope_parameters"}{"rope_theta"}.getFloat(1e6),
      F.kBFloat16, F.kCPU,
      rotary_dim = int(tc{"head_dim"}.getInt().float64 *
        tc{"rope_parameters"}{"partial_rotary_factor"}.getFloat(1.0)))
    let sp = lp & "self_attn."
    let qProj = Linear.load(view, cfgJson, sp & "q_proj")
    let kProj = Linear.load(view, cfgJson, sp & "k_proj")
    let vProj = Linear.load(view, cfgJson, sp & "v_proj")
    let oProj = Linear.load(view, cfgJson, sp & "o_proj")
    let qNorm = RmsNormOne.load(view, cfgJson, sp & "q_norm")
    let kNorm = RmsNormOne.load(view, cfgJson, sp & "k_norm")
    let attn = RopeElementWiseGatedAttention[RmsNormOne].init(
      layerIdx, sp[0 .. ^2],
      qProj, kProj, vProj, oProj,
      tc{"num_attention_heads"}.getInt().int,
      tc{"num_key_value_heads"}.getInt().int,
      tc{"head_dim"}.getInt().int,
      rotary,
      q_norm = qNorm, k_norm = kNorm)
    result[1] = DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedDenseFFN, RmsNormOne].init(
      input_layernorm = inputLN, sequence_mixer = attn,
      post_attention_layernorm = postLN, hidden_mixer = mlp)

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # GDN block layer 0, prefill seq 5, vs the sequential + chunked fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN block layer 0 prefill (seq 5) vs fixture":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(weights, cfgJson, 0)

      var st = Safetensor.open(GdnFixtureDir / "gdn-Qwen3.5-0.8B-00.safetensor")

      let x = st.getTensorOwned("input")
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      let output = gdn(ctx, x)
      doAssert output.size(0) == 1 and output.size(1) == 5 and output.size(2) == 1024

      # Deterministic intermediates, recomputed through the layer's own
      # components and compared against the captured vendored values.
      let seqLen = x.size(1)
      let mixedQkv = gdn.in_proj_qkv.forward(x).transpose(1, 2)
      let conv = F.conv1d(mixedQkv, gdn.conv1d_weight,
        padding = [3], groups = gdn.conv_dim)
      let convOut = F.silu(conv.narrow(2, 0, seqLen))
      assertAllClose(convOut, st.getTensorOwned("conv_output"),
        rtol = 0.0, abstol = 0.0, msg = "conv output mismatch")

      let split = F.chunk(convOut.transpose(1, 2), 3, -1)
      let query = split[0].reshape([1, seqLen, gdn.num_k_heads, gdn.head_k_dim])
      let key = split[1].reshape([1, seqLen, gdn.num_k_heads, gdn.head_k_dim])
      let value = split[2].reshape([1, seqLen, gdn.num_v_heads, gdn.head_v_dim])
      assertAllClose(query, st.getTensorOwned("q"),
        rtol = 0.0, abstol = 0.0, msg = "q post-split mismatch")
      assertAllClose(key, st.getTensorOwned("k"),
        rtol = 0.0, abstol = 0.0, msg = "k post-split mismatch")
      assertAllClose(value, st.getTensorOwned("v"),
        rtol = 0.0, abstol = 0.0, msg = "v post-split mismatch")

      # Gates: f32 exp/softplus and bf16 sigmoid. Both sides call the same
      # ATen ops, so bit-exact is expected on the reference CPU. The 1e-4
      # bar is a portability guard for cross-platform libm variance.
      let aProj = gdn.in_proj_a.forward(x)
      let aLogExp = gdn.a_log.to(kFloat32).exp()
      let aPlusBias = aProj.to(kFloat32) + gdn.dt_bias
      let g = aLogExp.neg() * F.softplus(aPlusBias, 1.0, 20.0)
      assertAllClose(g, st.getTensorOwned("g"),
        rtol = 1e-4, abstol = 1e-4, msg = "decay g mismatch")
      let beta = F.sigmoid(gdn.in_proj_b.forward(x))
      assertAllClose(beta, st.getTensorOwned("beta"),
        rtol = 1e-4, abstol = 1e-4, msg = "beta mismatch")

      # Layer output: 0.00 vs the sequential reference, 5e-3 vs the vendored
      # chunked forward. The recurrence is exercised end to end. The per-step
      # core/state values are asserted by the state-persistence test.
      # The T=5 fixture is a single chunk, so chunked vs sequential agree to
      # ~1e-8 f32 (sub-bf16-ULP, bf16 rounding-boundary flips possible).
      # Multi-chunk prefills diverge ~1.5e-5. Both are expected, not defects.
      assertAllClose(output, st.getTensorOwned("output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "sequential block output mismatch")
      assertAllClose(output, st.getTensorOwned("output_chunked"),
        rtol = 5e-3, abstol = 5e-3, msg = "chunked block output mismatch")
      true


  # ──────────────────────────────────────────────────────────────────────────
  # Full decoder layer 0 (GDN), prefill seq 5
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Full decoder layer 0 (GDN) vs fixture":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let (gdnLayer, _) = loadLayer(weights, cfgJson, 0)

      var st = Safetensor.open(GdnFixtureDir / "layer-Qwen3.5-0.8B-00.safetensor")

      let x = st.getTensorOwned("layer_input")

      # Deterministic pieces on a fresh sequence, at 0.00 vs the fixture.
      doAssert gdnLayer != nil, "layer 0 is the Gated DeltaNet pairing"
      let hNorm = gdnLayer.input_layernorm(x)
      assertAllClose(hNorm, st.getTensorOwned("input_layernorm_output"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 input_layernorm mismatch")
      var ctxGdn = InferenceContext.init(24, 1, 2, 512, 256)
      let gdnOut = gdnLayer.sequence_mixer(ctxGdn, hNorm)
      assertAllClose(gdnOut, st.getTensorOwned("gdn_block_output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 gdn block output mismatch")
      let h1 = x + gdnOut
      let postLnOut = gdnLayer.post_attention_layernorm(h1)
      assertAllClose(postLnOut, st.getTensorOwned("post_attention_layernorm_output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 post_attention_layernorm mismatch")
      let mlpOut = gdnLayer.hidden_mixer.forward(postLnOut)
      assertAllClose(mlpOut, st.getTensorOwned("mlp_output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 mlp mismatch")

      # The layer forward itself: 0.00 vs the sequential replay, 5e-3 vs the
      # vendored chunked forward (single-chunk T=5 fixture, sub-ULP agree).
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      let firstLayer = gdnLayer.forward(ctx, x, none(Tensor))
      let output = firstLayer[1] + firstLayer[0]
      assertAllClose(output, st.getTensorOwned("layer_output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 sequential output mismatch")
      assertAllClose(output, st.getTensorOwned("layer_output"),
        rtol = 5e-3, abstol = 5e-3, msg = "layer 0 chunked output mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Full decoder layer 3 (full attention), prefill seq 5
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Full decoder layer 3 (full attention) vs fixture":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let (_, attnLayer) = loadLayer(weights, cfgJson, 3)

      var st = Safetensor.open(Layer3FixtureDir / "layer-Qwen3.5-0.8B-03.safetensor")

      let x = st.getTensorOwned("layer_input")
      let hfPosIds = st.getTensorOwned("position_ids")
      var (ctx, pool) = newKVContext(numLayers = 24, kvHeads = 2, headDim = 256)
      ctx.position_ids = hfPosIds[0]
      doAssert attnLayer != nil, "layer 3 is the output-gated attention pairing"
      ctx.setRopeForPositions(attnLayer.sequence_mixer.rotary)

      let thirdLayer = attnLayer.forward(ctx, x, none(Tensor))
      let output = thirdLayer[1] + thirdLayer[0]
      doAssert output.size(0) == 1 and output.size(1) == 5 and output.size(2) == 1024
      assertAllClose(output, st.getTensorOwned("layer_output"),
        rtol = 5e-3, abstol = 5e-3, msg = "layer 3 output mismatch")

      # Deterministic intermediates: input norm and the attention pre-SDPA
      # values (the SDPA itself carries the block-level tolerance above).
      let hNorm = attnLayer.input_layernorm(x)
      assertAllClose(hNorm, st.getTensorOwned("input_layernorm_output"),
        rtol = 0.0, abstol = 0.0, msg = "layer 3 input_layernorm mismatch")
      let seqLen = x.size(1)
      let gqa = attnLayer.sequence_mixer.gqa_attn
      # q_proj packs [q | gate] per head, so the head axis is 2 * head_dim.
      let qg = attnLayer.sequence_mixer.q_proj.forward(hNorm)
      let qgR = qg.reshape([1, seqLen, gqa.num_qo_head, 2 * gqa.head_dim])
      let queryR = qgR.narrow(3, 0, gqa.head_dim)
      let gateR = qgR.narrow(3, gqa.head_dim, gqa.head_dim)
      let gate = gateR.reshape([1, seqLen, gqa.num_qo_head * gqa.head_dim])
      let qNormed = attnLayer.sequence_mixer.q_norm.forward(queryR)
      let kReshaped = attnLayer.sequence_mixer.k_proj.forward(hNorm).reshape(
        [1, seqLen, gqa.num_kv_head, gqa.head_dim])
      let kNormed = attnLayer.sequence_mixer.k_norm.forward(kReshaped)
      assertAllClose(qNormed, st.getTensorOwned("q_normed"),
        rtol = 0.0, abstol = 0.0, msg = "layer 3 q_normed mismatch")
      assertAllClose(kNormed, st.getTensorOwned("k_normed"),
        rtol = 0.0, abstol = 0.0, msg = "layer 3 k_normed mismatch")
      assertAllClose(gate, st.getTensorOwned("gate"),
        rtol = 0.0, abstol = 0.0, msg = "layer 3 gate mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Long-residual 3-block chain (layers 0..2, all GDN)
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Long-residual 3-block chain (layers 0-2) vs fixture":
    proc(): bool =
      # The chain blocks come from the model loader, the same safetensor file weights
      # the single-block sections load directly, so the interface stack
      # is assembled only by the model module that owns the loader.
      let chainModel = loadQwen35ModelRaw(ModelDir, kCPU)
      let layers = chainModel.layers

      var st0 = Safetensor.open(ChainFixtureDir / "block-00.safetensor")
      let x = st0.getTensorOwned("layer_input_seq")

      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      var hidden = x
      var blockInput = x
      var stream: Option[Tensor]
      for i in 0 ..< 3:
        var st = Safetensor.open(ChainFixtureDir / ("block-0" & $i & ".safetensor"))
        # Sequential chain inputs/outputs at 0.00, vendored chunked chain
        # at 5e-3. The single-chunk T=4 fixture has chunked == sequential
        # bit-exact, so the 5e-3-vs-chunked asserts are degenerate.
        # Multi-chunk prefills diverge ~1.5e-5 (locked by the T=70 band
        # test above).
        assertAllClose(hidden, st.getTensorOwned("layer_input_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential input mismatch")
        assertAllClose(hidden, st.getTensorOwned("layer_input"),
          rtol = 5e-3, abstol = 5e-3, msg = "chain layer " & $i & " chunked input mismatch")
        let pair = layers[i].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])
        let layerOut = pair[1] + pair[0]
        assertAllClose(layerOut, st.getTensorOwned("layer_output_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential output mismatch")
        assertAllClose(layerOut, st.getTensorOwned("layer_output"),
          rtol = 5e-3, abstol = 5e-3, msg = "chain layer " & $i & " chunked output mismatch")
        hidden = layerOut
      true

  # ──────────────────────────────────────────────────────────────────────────
  # State persistence: two-step decode == one-shot prefill
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN state persistence: 2-step decode == one-shot prefill":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(weights, cfgJson, 0)

      var st = Safetensor.open(GdnFixtureDir / "gdn-Qwen3.5-0.8B-01.safetensor")

      let prefillX = st.getTensorOwned("prefill_x")
      let decodeXd = st.getTensorOwned("decode_x_d")
      let decodeXe = st.getTensorOwned("decode_x_e")
      let oneShotLayer = st.getTensorOwned("one_shot_block_output")

      var ctx = InferenceContext.init(24, 1, 2, 512, 256)

      # Prefill [a, b, c]: outputs and stored state must match the one-shot
      # trajectory at steps 0..2.
      let outPrefill = gdn(ctx, prefillX)
      assertAllClose(outPrefill, oneShotLayer.narrow(1, 0, 3),
        rtol = 0.0, abstol = 0.0, msg = "prefill output != one-shot steps 0..2")
      assertAllClose(ctx.gdnSsmState[0], st.getTensorOwned("ssm_state_after_prefill"),
        rtol = 0.0, abstol = 0.0, msg = "SSM state after prefill mismatch")
      assertAllClose(ctx.gdnSsmState[0], st.getTensorOwned("one_shot_ssm_states")[3],
        rtol = 0.0, abstol = 0.0, msg = "SSM state after prefill != one-shot step 3")
      assertAllClose(ctx.gdnConvState[0],
        st.getTensorOwned("conv_state_after_prefill_tail3")[0],
        rtol = 0.0, abstol = 0.0, msg = "conv state after prefill mismatch")
      let convStatePrefill = ctx.gdnConvState[0].clone()
      let ssmStatePrefill = ctx.gdnSsmState[0].clone()

      # Decode [d]: output must equal both the decode fixture and the
      # one-shot step 3, with the stored state feeding the conv input.
      let outD = gdn(ctx, decodeXd)
      assertAllClose(outD, st.getTensorOwned("decode_output_d"),
        rtol = 0.0, abstol = 0.0, msg = "decode d output mismatch")
      assertAllClose(outD, oneShotLayer.narrow(1, 3, 1),
        rtol = 0.0, abstol = 0.0, msg = "decode d != one-shot step 3")
      let mixedD = gdn.in_proj_qkv.forward(decodeXd).transpose(1, 2)
      let catInputD = F.cat([convStatePrefill.unsqueeze(0), mixedD], -1)
      assertAllClose(catInputD,
        st.getTensorOwned("decode_conv_input_d").narrow(2, 1, 4),
        rtol = 0.0, abstol = 0.0, msg = "decode d conv input mismatch")
      let convD = F.conv1d(catInputD, gdn.conv1d_weight,
        padding = [0], groups = gdn.conv_dim)
      let convOutD = F.silu(convD.narrow(2, convD.size(2) - 1, 1))
      assertAllClose(convOutD, st.getTensorOwned("decode_conv_output_d"),
        rtol = 0.0, abstol = 0.0, msg = "decode d conv output mismatch")
      assertAllClose(ctx.gdnSsmState[0], st.getTensorOwned("ssm_state_after_d"),
        rtol = 0.0, abstol = 0.0, msg = "SSM state after d mismatch")
      assertAllClose(ctx.gdnSsmState[0], st.getTensorOwned("one_shot_ssm_states")[4],
        rtol = 0.0, abstol = 0.0, msg = "SSM state after d != one-shot step 4")
      assertAllClose(ctx.gdnConvState[0],
        st.getTensorOwned("conv_state_after_d_tail3")[0],
        rtol = 0.0, abstol = 0.0, msg = "conv state after d mismatch")
      let convStateAfterD = ctx.gdnConvState[0].clone()

      # Decode [e]: same checks against step 4 and the final states.
      let outE = gdn(ctx, decodeXe)
      assertAllClose(outE, st.getTensorOwned("decode_output_e"),
        rtol = 0.0, abstol = 0.0, msg = "decode e output mismatch")
      assertAllClose(outE, oneShotLayer.narrow(1, 4, 1),
        rtol = 0.0, abstol = 0.0, msg = "decode e != one-shot step 4")
      let mixedE = gdn.in_proj_qkv.forward(decodeXe).transpose(1, 2)
      let catInputE = F.cat([convStateAfterD.unsqueeze(0), mixedE], -1)
      assertAllClose(catInputE,
        st.getTensorOwned("decode_conv_input_e").narrow(2, 1, 4),
        rtol = 0.0, abstol = 0.0, msg = "decode e conv input mismatch")
      let convE = F.conv1d(catInputE, gdn.conv1d_weight,
        padding = [0], groups = gdn.conv_dim)
      let convOutE = F.silu(convE.narrow(2, convE.size(2) - 1, 1))
      assertAllClose(convOutE, st.getTensorOwned("decode_conv_output_e"),
        rtol = 0.0, abstol = 0.0, msg = "decode e conv output mismatch")
      assertAllClose(ctx.gdnSsmState[0], st.getTensorOwned("ssm_state_after_e"),
        rtol = 0.0, abstol = 0.0, msg = "SSM state after e mismatch")
      assertAllClose(ctx.gdnSsmState[0], st.getTensorOwned("one_shot_ssm_states")[5],
        rtol = 0.0, abstol = 0.0, msg = "final SSM state != one-shot step 5")
      assertAllClose(ctx.gdnConvState[0],
        st.getTensorOwned("conv_state_after_e_tail3")[0],
        rtol = 0.0, abstol = 0.0, msg = "conv state after e mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Conv history across a multi-token continuation: two-call == one-shot
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN conv history: multi-token continuation == one-shot tail":
    proc(): bool =
      var weights = openModelWeights()
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(weights, cfgJson, 0)

      # Deterministic tiny input: arange scaled to [0, 1), bf16.
      let n = 5 * 1024
      let arange = F.arange(n, F.tensorOptions(F.kInt64, F.kCPU)).to(kFloat32)
      let x = (arange * (1.0 / float64(n)))
        .reshape([1, 5, 1024])
        .to(kBFloat16)
      let firstCall = x.narrow(1, 0, 3)
      let secondCall = x.narrow(1, 3, 2)

      # Two-call continuation: prefill 3 tokens, then a second multi-token
      # prefill of 2 tokens on the same context. The second call must keep
      # the conv history written by the first, not re-zero the window.
      var ctx2 = InferenceContext.init(24, 1, 2, 512, 256)
      discard gdn(ctx2, firstCall)
      let outTwoCall = gdn(ctx2, secondCall)

      # One-shot prefill over the concatenated 5 tokens.
      var ctx1 = InferenceContext.init(24, 1, 2, 512, 256)
      let outOneShot = gdn(ctx1, x)

      # The second call's positions must be bit-identical to the one-shot
      # tail (the conv history of the first call feeds the second).
      assertAllClose(outTwoCall, outOneShot.narrow(1, 3, 2),
        rtol = 0.0, abstol = 0.0,
        msg = "multi-token continuation tail != one-shot tail")
      true



  echo "\nAll Qwen3.5 GDN / layer / state tests passed!"

when isMainModule:
  main()
