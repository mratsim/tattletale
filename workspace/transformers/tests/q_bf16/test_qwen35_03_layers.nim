# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-layers \
##   --nimcache:nimcache/tests/qwen35-layers \
##   workspace/transformers/tests/q_bf16/test_qwen35_03_layers.nim

import
  std/options,
  std/memfiles,
  std/os,
  std/importutils,
  pkg/iface,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models,
  workspace/transformers/src/models/all_interfaces,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/src/model/loading/layer_kinds,
  workspace/transformers/tests/transformers_testutils,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(GatedDeltaNet)
privateAccess(RopeGQAttention)
privateAccess(Qwen35DecoderLayer)

const
  GdnFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3.5-0.8B-layer-0"
  Layer3FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3.5-0.8B-layer-3"
  ChainFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "long-residual-3-block" / "Qwen3.5-0.8B"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc openModelShard(): (MemFile, Safetensor) =
  ## Open the real Qwen3.5-0.8B shard for targeted weight loads.
  let memFile = memFiles.open(
    ModelDir / "model.safetensors-00001-of-00001.safetensors", mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc loadGdn(st: Safetensor, cfgJson: JsonNode, layerIdx: int): GatedDeltaNet =
  ## Load the real layer-`layerIdx` Gated DeltaNet weights into a layer.
  let tc = cfgJson{"text_config"}
  let lp = "model.language_model.layers." & $layerIdx & ".linear_attn."
  let qkvProj = Linear.load(st, cfgJson, lp & "in_proj_qkv")
  let zProj = Linear.load(st, cfgJson, lp & "in_proj_z")
  let aProj = Linear.load(st, cfgJson, lp & "in_proj_a")
  let bProj = Linear.load(st, cfgJson, lp & "in_proj_b")
  let convWeight = st.getTensorOwned(lp & "conv1d.weight")
  let aLog = st.getTensorOwned(lp & "A_log")
  let dtBias = st.getTensorOwned(lp & "dt_bias")
  let gdnNorm = RmsNormGated.load(st, cfgJson, lp & "norm")
  let outProj = Linear.load(st, cfgJson, lp & "out_proj")
  result = GatedDeltaNet.init(
    layerIdx, lp[0 .. ^2],
    qkvProj, zProj, aProj, bProj,
    convWeight, aLog, dtBias, gdnNorm, outProj,
    tc{"linear_num_key_heads"}.getInt().int,
    tc{"linear_num_value_heads"}.getInt().int,
    tc{"linear_key_head_dim"}.getInt().int,
    tc{"linear_value_head_dim"}.getInt().int,
    tc{"linear_conv_kernel_dim"}.getInt().int)

proc loadLayer(st: Safetensor, cfgJson: JsonNode, layerIdx: int): Qwen35DecoderLayer =
  ## Load one full decoder layer (norms, mlp, and the attention variant
  ## selected by the parsed kind at `layer_types[layerIdx]`) with real weights.
  let lp = "model.language_model.layers." & $layerIdx & "."
  let inputLN = RmsNorm.load(st, cfgJson, lp & "input_layernorm", constant_bias = 1.0)
  let postLN = RmsNorm.load(st, cfgJson, lp & "post_attention_layernorm", constant_bias = 1.0)
  let gateProj = Linear.load(st, cfgJson, lp & "mlp.gate_proj")
  let upProj = Linear.load(st, cfgJson, lp & "mlp.up_proj")
  let downProj = Linear.load(st, cfgJson, lp & "mlp.down_proj")
  let mlp = GatedMLP.init(gateProj, upProj, downProj)
  let tc = cfgJson{"text_config"}
  if parseAttnFromHfTransformers(tc{"layer_types"}[layerIdx].getStr(),
      "fixture layer_types") == alkGatedDeltaNet:
    let gdn = loadGdn(st, cfgJson, layerIdx)
    result = Qwen35DecoderLayer.init(alkGatedDeltaNet, inputLN, postLN, gdn, nil, mlp)
  else:
    let rotary = RotaryPositionEmbeddingRef.new(
      tc{"head_dim"}.getInt().int,
      tc{"max_position_embeddings"}.getInt().int,
      tc{"rope_parameters"}{"rope_theta"}.getFloat(1e6),
      F.kBFloat16, F.kCPU,
      rotary_dim = int(tc{"head_dim"}.getInt().float64 *
        tc{"rope_parameters"}{"partial_rotary_factor"}.getFloat(1.0)))
    let sp = lp & "self_attn."
    let qProj = Linear.load(st, cfgJson, sp & "q_proj")
    let kProj = Linear.load(st, cfgJson, sp & "k_proj")
    let vProj = Linear.load(st, cfgJson, sp & "v_proj")
    let oProj = Linear.load(st, cfgJson, sp & "o_proj")
    let qNorm = RmsNorm.load(st, cfgJson, sp & "q_norm", constant_bias = 1.0)
    let kNorm = RmsNorm.load(st, cfgJson, sp & "k_norm", constant_bias = 1.0)
    let attn = RopeGQAttention.init(
      layerIdx, sp[0 .. ^2],
      qProj, kProj, vProj, oProj,
      tc{"num_attention_heads"}.getInt().int,
      tc{"num_key_value_heads"}.getInt().int,
      tc{"head_dim"}.getInt().int,
      rotary,
      q_norm = some(qNorm), k_norm = some(kNorm), fused_gate = true)
    result = Qwen35DecoderLayer.init(alkAttention, inputLN, postLN, nil, attn, mlp)

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # GDN block layer 0, prefill seq 5, vs the sequential + chunked fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN block layer 0 prefill (seq 5) vs fixture":
    proc(): bool =
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(shardSt, cfgJson, 0)

      var (memFile, st) = openSafetensor(GdnFixtureDir, "gdn-Qwen3.5-0.8B-00.safetensor")
      defer: close(memFile)

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

      # Block output: 0.00 vs the sequential reference, 5e-3 vs the vendored
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
  # Multi-chunk GDN prefill T=70: sequential-vs-chunked divergence band
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN multi-chunk prefill T=70 divergence band (sequential vs chunked)":
    proc(): bool =
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(shardSt, cfgJson, 0)

      var (memFile, st) = openSafetensor(GdnFixtureDir, "gdn-Qwen3.5-0.8B-02.safetensor")
      defer: close(memFile)

      let x = st.getTensorOwned("input")
      let outputSeq = st.getTensorOwned("output_seq")
      let outputChunked = st.getTensorOwned("output_chunked")

      # The fixture's own band: the chunked and sequential rules diverge by
      # more than zero and stay under 1e-3 (f32 core ~1.5e-5, a few bf16
      # ULPs at the block output). The band is a property of the two rules,
      # locked here so a future change to either side cannot drift silently.
      let bandDiff = (outputSeq.to(kFloat32) - outputChunked.to(kFloat32)).abs().max().item(float64)
      doAssert bandDiff > 0.0, "sequential and chunked outputs are identical, band not exercised"
      doAssert bandDiff < 1e-3, "sequential-vs-chunked divergence exceeded the documented band"

      # The f32 recurrence core and SSM state carry the same band: the two
      # rules diverge by more than zero and stay sub-1e-5 (measured 9.5e-7
      # core / 1.5e-7 state on the committed fixture).
      let coreSeq = st.getTensorOwned("core_attn_out_seq")
      let coreChunked = st.getTensorOwned("core_attn_out_chunked")
      let ssmSeq = st.getTensorOwned("ssm_state_seq")
      let ssmChunked = st.getTensorOwned("ssm_state_chunked")
      let coreBand = (coreSeq.to(kFloat32) - coreChunked.to(kFloat32)).abs().max().item(float64)
      let ssmBand = (ssmSeq.to(kFloat32) - ssmChunked.to(kFloat32)).abs().max().item(float64)
      doAssert coreBand > 0.0 and coreBand < 1e-5, "core band outside (0, 1e-5)"
      doAssert ssmBand > 0.0 and ssmBand < 1e-6, "ssm band outside (0, 1e-6)"

      # The Nim layer is sequential: 0.00 against the sequential replay, and
      # inside the 5e-3 block bar against the vendored chunked forward.
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      let output = gdn(ctx, x)
      assertAllClose(output, outputSeq,
        rtol = 0.0, abstol = 0.0, msg = "multi-chunk sequential output mismatch")
      assertAllClose(output, outputChunked,
        rtol = 5e-3, abstol = 5e-3, msg = "multi-chunk chunked output mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Full decoder layer 0 (GDN), prefill seq 5
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Full decoder layer 0 (GDN) vs fixture":
    proc(): bool =
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let layer0 = loadLayer(shardSt, cfgJson, 0)

      var (memFile, st) = openSafetensor(GdnFixtureDir, "layer-Qwen3.5-0.8B-00.safetensor")
      defer: close(memFile)

      let x = st.getTensorOwned("layer_input")

      # Deterministic pieces on a fresh sequence, at 0.00 vs the fixture.
      let hNorm = layer0.input_layernorm(x)
      assertAllClose(hNorm, st.getTensorOwned("input_layernorm_output"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 input_layernorm mismatch")
      var ctxGdn = InferenceContext.init(24, 1, 2, 512, 256)
      let gdnOut = layer0.gdn(ctxGdn, hNorm)
      assertAllClose(gdnOut, st.getTensorOwned("gdn_block_output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 gdn block output mismatch")
      let h1 = x + gdnOut
      let postLnOut = layer0.post_attention_layernorm(h1)
      assertAllClose(postLnOut, st.getTensorOwned("post_attention_layernorm_output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 post_attention_layernorm mismatch")
      let mlpOut = layer0.mlp(postLnOut)
      assertAllClose(mlpOut, st.getTensorOwned("mlp_output_seq"),
        rtol = 0.0, abstol = 0.0, msg = "layer 0 mlp mismatch")

      # The layer forward itself: 0.00 vs the sequential replay, 5e-3 vs the
      # vendored chunked forward (single-chunk T=5 fixture, sub-ULP agree).
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      let output = layer0(ctx, x)
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
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let layer3 = loadLayer(shardSt, cfgJson, 3)

      var (memFile, st) = openSafetensor(Layer3FixtureDir, "layer-Qwen3.5-0.8B-03.safetensor")
      defer: close(memFile)

      let x = st.getTensorOwned("layer_input")
      let hfPosIds = st.getTensorOwned("position_ids")
      var (ctx, pool) = newKVContext(numLayers = 24, kvHeads = 2, headDim = 256)
      ctx.position_ids = hfPosIds[0]
      ctx.setRopeForPositions(layer3.attn.rotary)

      let output = layer3(ctx, x)
      doAssert output.size(0) == 1 and output.size(1) == 5 and output.size(2) == 1024
      assertAllClose(output, st.getTensorOwned("layer_output"),
        rtol = 5e-3, abstol = 5e-3, msg = "layer 3 output mismatch")

      # Deterministic intermediates: input norm and the attention pre-SDPA
      # values (the SDPA itself carries the block-level tolerance above).
      let hNorm = layer3.input_layernorm(x)
      assertAllClose(hNorm, st.getTensorOwned("input_layernorm_output"),
        rtol = 0.0, abstol = 0.0, msg = "layer 3 input_layernorm mismatch")
      let seqLen = x.size(1)
      let gqa = layer3.attn.gqa_attn
      # q_proj packs [q | gate] per head, so the head axis is 2 * head_dim.
      let qg = layer3.attn.q_proj.forward(hNorm)
      let qgR = qg.reshape([1, seqLen, gqa.num_qo_head, 2 * gqa.head_dim])
      let queryR = qgR.narrow(3, 0, gqa.head_dim)
      let gateR = qgR.narrow(3, gqa.head_dim, gqa.head_dim)
      let gate = gateR.reshape([1, seqLen, gqa.num_qo_head * gqa.head_dim])
      let qNormed = layer3.attn.q_norm.get().forward(queryR)
      let kReshaped = layer3.attn.k_proj.forward(hNorm).reshape(
        [1, seqLen, gqa.num_kv_head, gqa.head_dim])
      let kNormed = layer3.attn.k_norm.get().forward(kReshaped)
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
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      var layers = newSeq[Qwen35DecoderLayer](3)
      for i in 0 ..< 3:
        layers[i] = loadLayer(shardSt, cfgJson, i)

      var (memFile0, st0) = openSafetensor(ChainFixtureDir, "block-00.safetensor")
      defer: close(memFile0)
      let x = st0.getTensorOwned("layer_input_seq")

      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      var hidden = x
      for i in 0 ..< 3:
        var (memFile, st) = openSafetensor(ChainFixtureDir, "block-0" & $i & ".safetensor")
        defer: close(memFile)
        # Sequential chain inputs/outputs at 0.00, vendored chunked chain
        # at 5e-3. The single-chunk T=4 fixture has chunked == sequential
        # bit-exact, so the 5e-3-vs-chunked asserts are degenerate.
        # Multi-chunk prefills diverge ~1.5e-5 (locked by the T=70 band
        # test above).
        assertAllClose(hidden, st.getTensorOwned("layer_input_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential input mismatch")
        assertAllClose(hidden, st.getTensorOwned("layer_input"),
          rtol = 5e-3, abstol = 5e-3, msg = "chain layer " & $i & " chunked input mismatch")
        let layerOut = layers[i](ctx, hidden)
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
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(shardSt, cfgJson, 0)

      var (memFile, st) = openSafetensor(GdnFixtureDir, "gdn-Qwen3.5-0.8B-01.safetensor")
      defer: close(memFile)

      let prefillX = st.getTensorOwned("prefill_x")
      let decodeXd = st.getTensorOwned("decode_x_d")
      let decodeXe = st.getTensorOwned("decode_x_e")
      let oneShotBlock = st.getTensorOwned("one_shot_block_output")

      var ctx = InferenceContext.init(24, 1, 2, 512, 256)

      # Prefill [a, b, c]: outputs and stored state must match the one-shot
      # trajectory at steps 0..2.
      let outPrefill = gdn(ctx, prefillX)
      assertAllClose(outPrefill, oneShotBlock.narrow(1, 0, 3),
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
      assertAllClose(outD, oneShotBlock.narrow(1, 3, 1),
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
      assertAllClose(outE, oneShotBlock.narrow(1, 4, 1),
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
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(shardSt, cfgJson, 0)

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

  # ──────────────────────────────────────────────────────────────────────────
  # Dispatch: 18 linear + 6 full attention, plus a model-level run
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN/full-attention dispatch (18 linear + 6 full) + model run":
    proc(): bool =
      let cfg = loadQwen35Config(ModelDir / "config.json")
      var linearCount = 0
      var fullCount = 0
      for i in 0 ..< cfg.num_hidden_layers:
        if cfg.layer_types[i] == alkGatedDeltaNet:
          inc linearCount
          doAssert (i mod 4) != 3,
            "Gated DeltaNet layer at a full-attention position"
        else:
          inc fullCount
          doAssert cfg.layer_types[i] == alkAttention
          doAssert (i mod 4) == 3,
            "full-attention layer outside (i mod 4) == 3"
      doAssert linearCount == 18, "expected 18 Gated DeltaNet layers"
      doAssert fullCount == 6, "expected 6 full-attention layers"

      # Full forward on 4 tokens gives finite logits through all 24 layers.
      let model = loadQwen35ModelRaw(ModelDir, kCPU)
      var (ctx, pool) = newKVContext(numLayers = 24, kvHeads = 2, headDim = 256)
      ctx.position_ids = F.arange(4, F.tensorOptions(F.kInt64, F.kCPU))
      let inputIds = @[9707'i64, 11, 1246, 525].toTensor().unsqueeze(0)
      let logits = model.forward(ctx, inputIds)
      doAssert logits.size(0) == 1 and logits.size(1) == 4
      doAssert logits.size(2) == cfg.vocab_size
      let maxAbs = logits.abs().max().item(float64)
      doAssert maxAbs > 0.0 and maxAbs < 1e30, "logits not finite"

      # generate() runs the prefill + decode loop end to end.
      let text = model.to(Model).generate(
        "hi", temp = 1.0f, maxTokens = 3, maxContextLen = 512)
      doAssert text.len > 0
      true

  # ──────────────────────────────────────────────────────────────────────────
  # GDN layers allocate no KV pages
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN layers allocate no KV pages":
    proc(): bool =
      var (shardMem, shardSt) = openModelShard()
      defer: close(shardMem)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let gdn = loadGdn(shardSt, cfgJson, 0)

      # Fresh context with no pages borrowed: a GDN forward must not touch
      # ctx.pages, the conv + SSM state is the whole cache.
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)
      doAssert ctx.pages.len == 0
      let x = F.randn(1, 5, 1024, F.tensorOptions(F.kBFloat16, F.kCPU))
      let output = gdn(ctx, x)
      doAssert output.size(2) == 1024
      doAssert ctx.pages.len == 0, "GDN forward must not allocate KV pages"
      doAssert not ctx.gdnConvState[0].isNil
      doAssert not ctx.gdnSsmState[0].isNil
      true

  echo "\nAll Qwen3.5 GDN / layer / state tests passed!"

when isMainModule:
  main()
