# Tattletale
# Copyright (c) 2026 Moby André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Decoder layer 0 and 3 integration tests for the Qwen3.6-35B-A3B bf16 port:
## full decoder layers against recorded fixtures, the routed-block guards,
## and the wired model compared layer-wise against the wiring fixtures. Run command:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off
##     --outdir:build/tests/test_qwen36_03_layers --nimcache:nimcache/tests/test_qwen36_03_layers
##     workspace/transformers/tests/q_bf16/test_qwen36_03_layers.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)
#
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/options,
  std/memfiles,
  std/os,
  std/strutils,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/layers/mixtures_of_experts,
  workspace/transformers/src/safetensors/collection,
  workspace/transformers/src/layers/mlp,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/layers/attn_ssm/grouped_query_attention {.all.},
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/layers/attn_ssm/gated_delta_net {.all.},
  workspace/transformers/tests/transformers_testutils,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(RopeGQAttention)
privateAccess(Qwen35MoeModel)
privateAccess(LMHead)
privateAccess(Embedding)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "qwen36-layer"
  WiringFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "qwen36-wiring"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  Shard3 = ModelDir / "model-00003-of-00026.safetensors"
  GdnPrefix = "model.language_model.layers.0.linear_attn"
  Layer0Prefix = "model.language_model.layers.0"
  AttnPrefix = "model.language_model.layers.3.self_attn"
  Layer3Prefix = "model.language_model.layers.3"
  Layer0GateUp = Layer0Prefix & ".mlp.experts.gate_up_proj"
  Layer0Down = Layer0Prefix & ".mlp.experts.down_proj"
  Layer0Router = Layer0Prefix & ".mlp.gate.weight"
  Hidden = 2048
  NumKHeads = 16
  NumVHeads = 32
  HeadKDim = 128
  HeadVDim = 128
  ConvKernelSize = 4
  NumQoHeads = 16
  NumKvHeads = 2
  HeadDim = 256
  RotaryDim = 64
  MaxPositionEmbeddings = 262144
  RopeTheta = 1e7
  NumExperts = 256
  TopK = 8
  Tokens = 6

proc normEpsFromConfig(): float64 =
  ## rms_norm_eps of the checkpoint text config.
  (ModelDir / "config.json").parseFile(){"text_config"}{"rms_norm_eps"}.getFloat(1e-6)

proc buildGdnViaView(view: SafetensorsCollection): GatedDeltaNet =
  ## Real layer-0 GDN weights routed through the checkpoint index view.
  let normEps = normEpsFromConfig()
  let qkvProj = Linear.init(view.getTensor(GdnPrefix & ".in_proj_qkv.weight"))
  let zProj = Linear.init(view.getTensor(GdnPrefix & ".in_proj_z.weight"))
  let aProj = Linear.init(view.getTensor(GdnPrefix & ".in_proj_a.weight"))
  let bProj = Linear.init(view.getTensor(GdnPrefix & ".in_proj_b.weight"))
  let oProj = Linear.init(view.getTensor(GdnPrefix & ".out_proj.weight"))
  let convW = view.getTensor(GdnPrefix & ".conv1d.weight")
  let aLog = view.getTensor(GdnPrefix & ".A_log")
  let dtBias = view.getTensor(GdnPrefix & ".dt_bias")
  let normWeight = view.getTensor(GdnPrefix & ".norm.weight")
  let norm = RmsNormGated.init(normWeight, eps = normEps)
  GatedDeltaNet.init(
    0, GdnPrefix,
    qkvProj, zProj, aProj, bProj,
    convW, aLog, dtBias,
    norm, oProj,
    NumKHeads, NumVHeads, HeadKDim, HeadVDim, ConvKernelSize)

proc moeCfg(): Qwen35MoeConfig =
  ## Text config carrying the checkpoint routed-block geometry.
  new result
  result.numExperts = NumExperts
  result.numExpertsPerTok = TopK

proc main() =
  runCppTest "full decoder layer 0 (GDN + MoE) vs fixture":
    proc(): bool =
      var view = openSafetensorsCollection(ModelDir)
      defer:
        close(view)
      let gdn = buildGdnViaView(view)
      var ctx = InferenceContext.init(num_layers = 1, batch_size = 1,
          kv_heads = 2, max_seq = 512, head_dim = HeadKDim)
      let normEps = normEpsFromConfig()
      let inputLN = RmsNorm.init(view.getTensor(Layer0Prefix & ".input_layernorm.weight"),
        eps = normEps, constant_bias = 1.0)
      let postLN = RmsNorm.init(view.getTensor(Layer0Prefix & ".post_attention_layernorm.weight"),
        eps = normEps, constant_bias = 1.0)
      var (memFile, st) = openSafetensor(FixtureDir, "layer-Qwen3.6-35B-A3B-00.safetensor")
      defer:
        close(memFile)
      let x = st.getTensorOwned("layer_input")
      let hNorm = inputLN.forward(x)
      assertAllClose(hNorm, st.getTensorOwned("input_layernorm_output"), rtol = 0.0, abstol = 0.0)
      let gdnOut = gdn(ctx, hNorm)
      assertAllClose(gdnOut, st.getTensorOwned("gdn_block_output_seq"), rtol = 0.0, abstol = 0.0)
      let h1 = x + gdnOut
      let h2 = postLN.forward(h1)
      assertAllClose(h2, st.getTensorOwned("post_attention_layernorm_output"), rtol = 0.0, abstol = 0.0)
      let routerWeight = view.getTensor(Layer0Router)
      let gateUp = view.getTensor(Layer0GateUp)
      let down = view.getTensor(Layer0Down)
      let sharedG = view.getTensor(Layer0Prefix & ".mlp.shared_expert.gate_proj.weight")
      let sharedU = view.getTensor(Layer0Prefix & ".mlp.shared_expert.up_proj.weight")
      let sharedD = view.getTensor(Layer0Prefix & ".mlp.shared_expert.down_proj.weight")
      let sharedGateWeight = view.getTensor(Layer0Prefix & ".mlp.shared_expert_gate.weight")
      let experts = MixtureOfExperts.init(gateUp, down)
      let sharedExpert = GatedMLP.init(sharedG, sharedU, sharedD)
      let h2Flat = h2.reshape(Tokens, Hidden)
      let moe = sparseMoeForward(moeCfg().numExpertsPerTok, h2Flat, routerWeight, experts, sharedExpert, sharedGateWeight)
      let meta = parseFile(FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.metadata.json")
      let routerLogitsBand = meta{"bands", "router_logits_band"}.getFloat()
      let routingWeightsBand = meta{"bands", "routing_weights_band"}.getFloat()
      let sharedGateBand = meta{"bands", "shared_gate_band"}.getFloat()
      let moeOutputBand = meta{"bands", "moe_output_band"}.getFloat()
      let layerOutputBand = meta{"bands", "layer_output_band"}.getFloat()
      let topkMarginMin = meta{"margins", "topk_margin_min"}.getFloat()
      doAssert topkMarginMin > 0.0
      let routerLogitsDiff = maxAbsDiff(moe.routerLogits, st.getTensorOwned("router_logits"))
      doAssert routerLogitsDiff <= routerLogitsBand, "router_logits diff " & $routerLogitsDiff & " outside the recorded band"
      let routingWeightsDiff = maxAbsDiff(moe.routingWeights, st.getTensorOwned("routing_weights"))
      doAssert routingWeightsDiff <= routingWeightsBand, "routing_weights diff " & $routingWeightsDiff & " outside the recorded band"
      let sharedGateDiff = maxAbsDiff(moe.sharedGate, st.getTensorOwned("shared_gate"))
      doAssert sharedGateDiff <= sharedGateBand, "shared_gate diff " & $sharedGateDiff & " outside the recorded band"
      let moeOutputDiff = maxAbsDiff(moe.output, st.getTensorOwned("moe_output"))
      doAssert moeOutputDiff <= moeOutputBand, "moe_output diff " & $moeOutputDiff & " outside the recorded band"
      let gotIndices = moe.topkIndices
      let fixIndices = st.getTensorOwned("topk_indices")
      for tok in 0 ..< Tokens:
        for pos in 0 ..< TopK:
          let nimIdx = gotIndices[tok, pos]
          let fixIdx = fixIndices[tok, pos]
          doAssert nimIdx.item(int64) == fixIdx.item(int64)
      let layerOut = h1 + moe.output.reshape(1, Tokens, Hidden)
      let layerOutDiff = maxAbsDiff(layerOut, st.getTensorOwned("layer_output_seq"))
      doAssert layerOutDiff <= layerOutputBand, "layer_output diff " & $layerOutDiff & " outside the recorded band"
      true

  runCppTest "full decoder layer 3 (attn + MoE) vs fixture":
    proc(): bool =
      var weightsMemFile = memFiles.open(Shard3, fmRead)
      defer:
        close(weightsMemFile)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let weightsSt = safetensors.load(weightsMemFile)
      let qProj = Linear.load(weightsSt, cfgJson, AttnPrefix & ".q_proj")
      let kProj = Linear.load(weightsSt, cfgJson, AttnPrefix & ".k_proj")
      let vProj = Linear.load(weightsSt, cfgJson, AttnPrefix & ".v_proj")
      let oProj = Linear.load(weightsSt, cfgJson, AttnPrefix & ".o_proj")
      let qNorm = RmsNorm.load(weightsSt, cfgJson, AttnPrefix & ".q_norm", constant_bias = 1.0)
      let kNorm = RmsNorm.load(weightsSt, cfgJson, AttnPrefix & ".k_norm", constant_bias = 1.0)
      let inputLN = RmsNorm.load(weightsSt, cfgJson, Layer3Prefix & ".input_layernorm", constant_bias = 1.0)
      let postLN = RmsNorm.load(weightsSt, cfgJson, Layer3Prefix & ".post_attention_layernorm", constant_bias = 1.0)
      let routerWeight = weightsSt.getTensorOwned(Layer3Prefix & ".mlp.gate.weight")
      let gateUp = weightsSt.getTensorOwned(Layer3Prefix & ".mlp.experts.gate_up_proj")
      let down = weightsSt.getTensorOwned(Layer3Prefix & ".mlp.experts.down_proj")
      let sharedG = weightsSt.getTensorOwned(Layer3Prefix & ".mlp.shared_expert.gate_proj.weight")
      let sharedU = weightsSt.getTensorOwned(Layer3Prefix & ".mlp.shared_expert.up_proj.weight")
      let sharedD = weightsSt.getTensorOwned(Layer3Prefix & ".mlp.shared_expert.down_proj.weight")
      let sharedGateWeight = weightsSt.getTensorOwned(Layer3Prefix & ".mlp.shared_expert_gate.weight")
      let experts = MixtureOfExperts.init(gateUp, down)
      let sharedExpert = GatedMLP.init(sharedG, sharedU, sharedD)
      let rotary = RotaryPositionEmbeddingRef.new(HeadDim, MaxPositionEmbeddings, RopeTheta, F.kBFloat16, F.kCPU, rotary_dim = RotaryDim)
      let attn = RopeGQAttention.init(3, AttnPrefix, qProj, kProj, vProj, oProj,
          NumQoHeads, NumKvHeads, HeadDim, rotary,
          q_norm = some(qNorm), k_norm = some(kNorm), fused_gate = true)
      var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = NumKvHeads, headDim = HeadDim)
      var (memFile, st) = openSafetensor(FixtureDir, "layer-Qwen3.6-35B-A3B-03.safetensor")
      defer:
        close(memFile)
      let x = st.getTensorOwned("layer_input")
      let posIds = st.getTensorOwned("position_ids")
      let ctxPosIds = posIds[0]
      ctx.position_ids = ctxPosIds
      ctx.setRopeForPositions(attn.rotary)
      let hNorm = inputLN.forward(x)
      assertAllClose(hNorm, st.getTensorOwned("input_layernorm_output"), rtol = 0.0, abstol = 0.0)
      let attnOut = attn(ctx, hNorm)
      # Cross-version SDPA: this binary links libtorch 2.11 and the fixture
      # was recorded with torch 2.13, one bf16 ulp on the differing element.
      # The delta is invariant to stride and input provenance.
      assertAllClose(attnOut, st.getTensorOwned("attn_mixer_output"), rtol = 5e-3, abstol = 5e-3)
      let h1 = x + attnOut
      # SDPA cross-version noise reaches h2, so h2 has no bitwise compare.
      let h2 = postLN.forward(h1)
      let h2Flat = h2.reshape(Tokens, Hidden)
      let moe = sparseMoeForward(moeCfg().numExpertsPerTok, h2Flat, routerWeight, experts, sharedExpert, sharedGateWeight)
      let meta = parseFile(FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor.metadata.json")
      let routerLogitsBand = meta{"bands", "router_logits_band"}.getFloat()
      # The band absorbs MoE- and SDPA-inherited noise through the residual
      # add, the post-attention layernorm, the fp32 renorm and the bf16 cast:
      # two bf16 ulps at the fixture max, a factor four margin.
      const routingWeightsBand = 0.015625
      let sharedGateBand = meta{"bands", "shared_gate_band"}.getFloat()
      let moeOutputBand = meta{"bands", "moe_output_band"}.getFloat()
      let layerBand = meta{"bands", "layer_output_band"}.getFloat()
      let topkMarginMin = meta{"margins", "topk_margin_min"}.getFloat()
      doAssert topkMarginMin > 0.0
      let routerLogitsDiff = maxAbsDiff(moe.routerLogits, st.getTensorOwned("router_logits"))
      doAssert routerLogitsDiff <= routerLogitsBand, "router_logits diff " & $routerLogitsDiff & " outside the recorded band"
      let routingWeightsDiff = maxAbsDiff(moe.routingWeights, st.getTensorOwned("routing_weights"))
      doAssert routingWeightsDiff <= routingWeightsBand, "routing_weights diff " & $routingWeightsDiff & " outside the recorded band"
      let sharedGateDiff = maxAbsDiff(moe.sharedGate, st.getTensorOwned("shared_gate"))
      doAssert sharedGateDiff <= sharedGateBand, "shared_gate diff " & $sharedGateDiff & " outside the recorded band"
      let moeOutputDiff = maxAbsDiff(moe.output, st.getTensorOwned("moe_output"))
      doAssert moeOutputDiff <= moeOutputBand, "moe_output diff " & $moeOutputDiff & " outside the recorded band"
      let gotIndices = moe.topkIndices
      let fixIndices = st.getTensorOwned("topk_indices")
      for tok in 0 ..< Tokens:
        for pos in 0 ..< TopK:
          let nimIdx = gotIndices[tok, pos]
          let fixIdx = fixIndices[tok, pos]
          doAssert nimIdx.item(int64) == fixIdx.item(int64)
      let layerOut = h1 + moe.output.reshape(1, Tokens, Hidden)
      let layerOutDiff = maxAbsDiff(layerOut, st.getTensorOwned("layer_output"))
      doAssert layerOutDiff <= layerBand, "layer_output diff " & $layerOutDiff & " outside the recorded band"
      true

  runCppTest "MixtureOfExperts and sparse-forward guards raise ValueError":
    proc(): bool =
      let e = 5
      let h = 4
      let inter = 3
      let t = 2
      let rank2GateUp = newSeq[float32](e * 2 * inter).toTensor().reshape(e, 2 * inter)
      let rank4Down = F.zeros(e, h, inter, 1, kFloat32)
      let mismatchDown = F.zeros(e, h, inter + 1, kFloat32)
      let validDown = F.zeros(e, h, inter, kFloat32)
      let gateUpValid = F.zeros(e, 2 * inter, h, kFloat32)
      try:
        discard MixtureOfExperts.init(rank2GateUp, validDown)
        doAssert false, "rank-2 gate_up projection must be rejected"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
      try:
        discard MixtureOfExperts.init(gateUpValid, rank4Down)
        doAssert false, "rank-4 down projection must be rejected"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
      try:
        discard MixtureOfExperts.init(gateUpValid, mismatchDown)
        doAssert false, "down projection with mismatched intermediate width must be rejected"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
      let experts = MixtureOfExperts.init(gateUpValid, validDown)
      let states = F.zeros(t, h, kFloat32)
      # Row 0 repeats expert 0: the duplicate the forward guard rejects
      let idx = [0'i64, 0'i64, 1'i64, 2'i64].toTensor().reshape(t, 2)
      let weights = F.ones(t, 2, kFloat32)
      try:
        discard experts.forward(states, idx, weights)
        doAssert false, "duplicate expert ids within one token must be rejected"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
      let realExperts = MixtureOfExperts.init(F.zeros(NumExperts, 2 * inter, h, kFloat32), F.zeros(NumExperts, h, inter, kFloat32))
      let realSharedExpert = GatedMLP.init(F.zeros(inter, h, kFloat32), F.zeros(inter, h, kFloat32), F.zeros(h, inter, kFloat32))
      let realGateWeight = F.zeros(1, h, kFloat32)
      let realRouterWeight = F.zeros(NumExperts, h, kFloat32)
      let cfg = moeCfg()
      try:
        discard sparseMoeForward(cfg.numExpertsPerTok, F.zeros(h, kFloat32), realRouterWeight, realExperts, realSharedExpert, realGateWeight)
        doAssert false, "rank-1 hidden states must be rejected"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
      let badRouter = F.zeros(NumExperts, h + 1, kFloat32)
      try:
        discard sparseMoeForward(cfg.numExpertsPerTok, F.zeros(t, h, kFloat32), badRouter, realExperts, realSharedExpert, realGateWeight)
        doAssert false, "router weight width mismatch must be rejected"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
      let badSharedGate = F.zeros(1, h + 1, kFloat32)
      try:
        discard sparseMoeForward(cfg.numExpertsPerTok, F.zeros(t, h, kFloat32), realRouterWeight, realExperts, realSharedExpert, badSharedGate)
        doAssert false, "shared gate width mismatch must be rejected"
      except ValueError as err:
        doAssert err.msg.startsWith("[ttt]")
      true

  runCppTest "layer-wise logits through the wired 35B vs fixtures":
    proc(): bool =
      # One sequential replay pins every boundary at 0.00, the chunked
      # replay compares against the band recorded in the fixture metadata.
      # Prompt ids come from the metadata, different from the Qwen3.5-0.8B fixtures.
      # Router topk is bitwise-equal to the recorded
      # torch.topk choices, never margin-checked: fp32 ties at the top-k
      # boundary are structural.
      let model = loadQwen35MoeModelRaw(ModelDir, kCPU)
      doAssert model.layers.len == 40
      doAssert model.config.numHiddenLayers == 40
      doAssert model.loadedTensorCount == 693

      var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = NumKvHeads, headDim = HeadDim)

      let meta = parseFile(WiringFixtureDir / "layer-00.safetensor.metadata.json")
      let metaTokens = meta{"input_tokens"}
      doAssert metaTokens.kind == JArray
      doAssert metaTokens.len < 64
      var tokenIds: seq[int64] = @[]
      for i in 0 ..< metaTokens.len:
        tokenIds.add metaTokens[i].getInt().int64
      let seqLen = tokenIds.len
      let inputIds = tokenIds.toTensor().unsqueeze(0)
      ctx.position_ids = F.arange(seqLen.int64, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(model.rotary)

      # Upper rail for the recorded bands, the binding tolerance is each recorded band.
      const WiringBandGuard = 2.0

      var h = model.embedTokens(inputIds)
      var routed: SparseMoeResult = nil
      # An all-zero band everywhere would make the chunked compares vacuous, at least one band stays positive.
      var exercisedInputBand = false
      var exercisedOutputBand = false
      for layerIdx in 0 ..< model.layers.len:
        var memFile = memFiles.open(
          WiringFixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor"),
          mode = fmRead)
        defer: close(memFile)
        let st = safetensors.load(memFile)

        # Sequential replay: 0.00 everywhere. Chunked replay: the recorded
        # band, asserted equal to the measured band before use.
        let seqInput = st.getTensorOwned("layer_input_seq")
        let chunkedInput = st.getTensorOwned("layer_input")
        assertAllClose(h, seqInput,
          rtol = 0.0, abstol = 0.0,
          msg = "layer " & $layerIdx & " sequential input mismatch")
        let layerMeta = parseFile(
          WiringFixtureDir / ("layer-" & ($layerIdx).align(2, '0') &
          ".safetensor.metadata.json"))
        doAssert layerMeta{"bands"}.kind == JObject,
          "layer " & $layerIdx & " fixture metadata has no bands object"
        doAssert layerMeta{"bands"}{"input_band"}.kind == JFloat,
          "layer " & $layerIdx & " fixture metadata input_band is not a float"
        doAssert layerMeta{"bands"}{"output_band"}.kind == JFloat,
          "layer " & $layerIdx & " fixture metadata output_band is not a float"
        let recordedInputBand = layerMeta{"bands"}{"input_band"}.getFloat()
        let recordedOutputBand = layerMeta{"bands"}{"output_band"}.getFloat()
        let inputBand = maxAbsDiff(seqInput, chunkedInput)
        doAssert inputBand == recordedInputBand,
          "layer " & $layerIdx & " measured input band " & $inputBand &
          " disagrees with the recorded band " & $recordedInputBand
        if inputBand > 0.0:
          exercisedInputBand = true
        doAssert inputBand < WiringBandGuard,
          "layer " & $layerIdx & " fixture input band " & $inputBand &
          " exceeds the documented " & $WiringBandGuard & " generator guard"
        assertAllClose(h, chunkedInput,
          rtol = 0.0, abstol = recordedInputBand,
          msg = "layer " & $layerIdx & " chunked input mismatch (band " &
            $inputBand & ")")

        h = model.layers[layerIdx].forward(ctx, h, addr routed)

        let seqOutput = st.getTensorOwned("layer_output_seq")
        let chunkedOutput = st.getTensorOwned("layer_output")
        assertAllClose(h, seqOutput,
          rtol = 0.0, abstol = 0.0,
          msg = "layer " & $layerIdx & " sequential output mismatch")
        let outputBand = maxAbsDiff(seqOutput, chunkedOutput)
        doAssert outputBand == recordedOutputBand,
          "layer " & $layerIdx & " measured output band " & $outputBand &
          " disagrees with the recorded band " & $recordedOutputBand
        if outputBand > 0.0:
          exercisedOutputBand = true
        doAssert outputBand < WiringBandGuard,
          "layer " & $layerIdx & " fixture output band " & $outputBand &
          " exceeds the documented " & $WiringBandGuard & " generator guard"
        assertAllClose(h, chunkedOutput,
          rtol = 0.0, abstol = recordedOutputBand,
          msg = "layer " & $layerIdx & " chunked output mismatch (band " &
            $outputBand & ")")

        # Router parity against the recorded topk. Bitwise: the indices,
        # then the renormed weights. The fp32 renorm spelling round-trips
        # the bf16 cast identically on the same libtorch build.
        let fixIndices = st.getTensorOwned("topk_indices")
        doAssert routed != nil
        doAssert routed.topkIndices.dim == 2
        for tok in 0 ..< seqLen:
          for pos in 0 ..< TopK:
            doAssert routed.topkIndices[tok, pos].item(int64) ==
              fixIndices[tok, pos].item(int64),
              "layer " & $layerIdx & " token " & $tok & " slot " & $pos &
              " routed expert id disagrees with the recorded torch.topk choice"
        assertAllClose(routed.routingWeights, st.getTensorOwned("routing_weights"),
          rtol = 0.0, abstol = 0.0,
          msg = "layer " & $layerIdx & " routing weights mismatch")

      doAssert exercisedInputBand and exercisedOutputBand,
        "the seq-vs-chunked ladder recorded a bitwise-identical band everywhere; " &
        "the chunked band locks are not exercised"
      let normed = model.norm.forward(h)
      let logits = model.lmHead.forward(normed)

      var memFileF = memFiles.open(
        WiringFixtureDir / "final_logits.safetensor", mode = fmRead)
      defer: close(memFileF)
      let stF = safetensors.load(memFileF)

      # Same contract: 0.00 vs the sequential replay, plus the recorded
      # logits band, asserted equal to the measured band before use.
      let seqLogits = stF.getTensorOwned("logits_seq")
      let chunkedLogits = stF.getTensorOwned("logits")
      assertAllClose(logits, seqLogits,
        rtol = 0.0, abstol = 0.0, msg = "final logits vs sequential replay mismatch")
      let logitsMeta = parseFile(
        WiringFixtureDir / "final_logits.safetensor.metadata.json")
      doAssert logitsMeta{"bands"}.kind == JObject,
        "final logits fixture metadata has no bands object"
      doAssert logitsMeta{"bands"}{"logits_band"}.kind == JFloat,
        "final logits fixture metadata logits_band is not a float"
      let recordedLogitsBand = logitsMeta{"bands"}{"logits_band"}.getFloat()
      let logitsBand = maxAbsDiff(seqLogits, chunkedLogits)
      doAssert logitsBand == recordedLogitsBand,
        "measured final logits band " & $logitsBand &
        " disagrees with the recorded band " & $recordedLogitsBand
      const LogitsBandGuard = 4.0
      doAssert logitsBand < LogitsBandGuard,
        "final logits band " & $logitsBand & " exceeds the " & $LogitsBandGuard & " generator guard"
      assertAllClose(logits, chunkedLogits,
        rtol = 0.0, abstol = recordedLogitsBand,
        msg = "final logits vs chunked forward mismatch (band " & $logitsBand & ")")
      true

  echo "\nAll Qwen3.6 decoder-layer blocks PASS"

when isMainModule:
  main()
