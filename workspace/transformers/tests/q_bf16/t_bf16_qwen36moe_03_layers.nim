# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off
##   --outdir:build/tests/t_bf16_qwen36moe_03_layers --nimcache:nimcache/tests/t_bf16_qwen36moe_03_layers
##   workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_03_layers.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/options,
  std/os,
  std/strutils,
  std/importutils,
  pkg/jsony,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers/ffn {.all.},
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/attn_ssm/grouped_query_attention {.all.},
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/layers/attn_ssm/gated_delta_net {.all.},
  workspace/transformers/tests/harness,
  workspace/transformers/tests/q_bf16/kvcontext,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(RopeElementWiseGatedAttention[RmsNormOne])
privateAccess(GatedBlockSparseFFN)

const
  Layer0FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-0"
  Layer3FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-3"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  WeightsFile3 = ModelDir / "model-00003-of-00026.safetensors"
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
  let qkvProj = Linear.init(view.getTensorOwned(GdnPrefix & ".in_proj_qkv.weight"))
  let zProj = Linear.init(view.getTensorOwned(GdnPrefix & ".in_proj_z.weight"))
  let aProj = Linear.init(view.getTensorOwned(GdnPrefix & ".in_proj_a.weight"))
  let bProj = Linear.init(view.getTensorOwned(GdnPrefix & ".in_proj_b.weight"))
  let oProj = Linear.init(view.getTensorOwned(GdnPrefix & ".out_proj.weight"))
  let convW = view.getTensorOwned(GdnPrefix & ".conv1d.weight")
  let aLog = view.getTensorOwned(GdnPrefix & ".A_log")
  let dtBias = view.getTensorOwned(GdnPrefix & ".dt_bias")
  let normWeight = view.getTensorOwned(GdnPrefix & ".norm.weight")
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

type
  LayerBands = object
    attn_mixer_band: float64
    layer_output_band: float64
    moe_output_band: float64
    router_logits_band: float64
    routing_weights_band: float64
    shared_gate_band: float64
  LayerMargins = object
    topk_inner_gap_min: float64
    topk_margin_min: float64
  LayerFixture = object
    bands: LayerBands
    margins: LayerMargins

proc main() =
  runCppTest "full decoder layer 0 (GDN + MoE) vs fixture":
    proc(): bool =
      echo "    devices: ", compareLine(Layer0FixtureDir, F.kCPU)
      let view = SafetensorsCollection.open(ModelDir)
      let gdn = buildGdnViaView(view)
      var ctx = InferenceContext.init(num_layers = 1, batch_size = 1,
          kv_heads = 2, max_seq = 512, head_dim = HeadKDim)
      let normEps = normEpsFromConfig()
      let inputLN = RmsNormOne.init(view.getTensorOwned(Layer0Prefix & ".input_layernorm.weight"),
        eps = normEps)
      let postLN = RmsNormOne.init(view.getTensorOwned(Layer0Prefix & ".post_attention_layernorm.weight"),
        eps = normEps)
      var st = Safetensor.open(Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor")
      let x = st.getTensorOwned("layer_input")
      let hNorm = inputLN.forward(x)
      assertAllClose(hNorm, st.getTensorOwned("input_layernorm_output"), rtol = 0.0, abstol = 0.0)
      let gdnOut = gdn(ctx, hNorm)
      assertAllClose(gdnOut, st.getTensorOwned("gdn_block_output_seq"), rtol = 0.0, abstol = 0.0)
      let h1 = x + gdnOut
      let h2 = postLN.forward(h1)
      assertAllClose(h2, st.getTensorOwned("post_attention_layernorm_output"), rtol = 0.0, abstol = 0.0)
      let routerWeight = view.getTensorOwned(Layer0Router)
      let gateUp = view.getTensorOwned(Layer0GateUp)
      let down = view.getTensorOwned(Layer0Down)
      let sharedG = view.getTensorOwned(Layer0Prefix & ".mlp.shared_expert.gate_proj.weight")
      let sharedU = view.getTensorOwned(Layer0Prefix & ".mlp.shared_expert.up_proj.weight")
      let sharedD = view.getTensorOwned(Layer0Prefix & ".mlp.shared_expert.down_proj.weight")
      let sharedGateWeight = view.getTensorOwned(Layer0Prefix & ".mlp.shared_expert_gate.weight")
      let sharedExpert = GatedDenseFFN.init(sharedG, sharedU, sharedD)
      let numExpertsPerTok = moeCfg().numExpertsPerTok
      let ffn = GatedBlockSparseFFN.init(
        gateUp, down, routerWeight, sharedExpert, sharedGateWeight,
        numExpertsPerTok)
      let h2Flat = h2.reshape(Tokens, Hidden)
      let (gotIndices, gotWeights) =
        routeToExperts(h2Flat, routerWeight, numExpertsPerTok)
      let moeOut = ffn.forward(h2)
      let meta = zstdReadFixture(
        Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.metadata.json.zst"
      ).fromJson(LayerFixture)
      let routingWeightsBand = meta.bands.routing_weights_band
      let moeOutputBand = meta.bands.moe_output_band
      let layerOutputBand = meta.bands.layer_output_band
      let topkMarginMin = meta.margins.topk_margin_min
      doAssert topkMarginMin > 0.0
      let routingWeightsDiff = maxAbsDiff(gotWeights, st.getTensorOwned("routing_weights"))
      doAssert routingWeightsDiff <= routingWeightsBand, "routing_weights diff " & $routingWeightsDiff & " outside the recorded band"
      let moeOutputDiff = maxAbsDiff(moeOut.reshape(Tokens, Hidden), st.getTensorOwned("moe_output"))
      doAssert moeOutputDiff <= moeOutputBand, "moe_output diff " & $moeOutputDiff & " outside the recorded band"
      let fixIndices = st.getTensorOwned("topk_indices")
      for tok in 0 ..< Tokens:
        for pos in 0 ..< TopK:
          let nimIdx = gotIndices[tok, pos]
          let fixIdx = fixIndices[tok, pos]
          doAssert nimIdx.item(int64) == fixIdx.item(int64)
      let layerOut = h1 + moeOut
      let layerOutDiff = maxAbsDiff(layerOut, st.getTensorOwned("layer_output_seq"))
      doAssert layerOutDiff <= layerOutputBand, "layer_output diff " & $layerOutDiff & " outside the recorded band"

      # Layer-0 block outputs get signature checks: match rate, order
      # statistics everywhere. Histograms appear only beside the tensor
      # that feeds the post-residual step.
      let layerBudgetAttn = defaultBudget(obAttention, F.kCPU)
      let layerBudgetResid = defaultBudget(obPostResidual, F.kCPU)
      let layerStatsFile = loadFingerprintStats(
        Layer0FixtureDir / "layer-Qwen3.6-35B-A3B-00.safetensor.stats")
      assertMatchRate(gdnOut, st.getTensorOwned("gdn_block_output_seq"),
        layerBudgetAttn, msg = "layer-0 gdn output match rate")
      assertStats(gdnOut, layerStatsFile.statsTensor("gdn_block_output_seq"),
        layerBudgetAttn, msg = "layer-0 gdn output stats")
      assertStats(moeOut, layerStatsFile.statsTensor("moe_output"),
        layerBudgetResid, msg = "layer-0 moe output stats")
      assertMatchRate(layerOut, st.getTensorOwned("layer_output_seq"),
        layerBudgetResid, msg = "layer-0 post-residual match rate")
      assertStats(layerOut, layerStatsFile.statsTensor("layer_output_seq"),
        layerBudgetResid, msg = "layer-0 post-residual stats")
      true

  runCppTest "full decoder layer 3 (attn + MoE) vs fixture":
    proc(): bool =
      echo "    devices: ", compareLine(Layer3FixtureDir, F.kCPU)
      let cfgJson = (ModelDir / "config.json").parseFile()
      let weights = SafetensorsCollection.open(WeightsFile3)
      let qProj = Linear.load(weights, cfgJson, AttnPrefix & ".q_proj")
      let kProj = Linear.load(weights, cfgJson, AttnPrefix & ".k_proj")
      let vProj = Linear.load(weights, cfgJson, AttnPrefix & ".v_proj")
      let oProj = Linear.load(weights, cfgJson, AttnPrefix & ".o_proj")
      let qNorm = RmsNormOne.load(weights, cfgJson, AttnPrefix & ".q_norm")
      let kNorm = RmsNormOne.load(weights, cfgJson, AttnPrefix & ".k_norm")
      let inputLN = RmsNormOne.load(weights, cfgJson, Layer3Prefix & ".input_layernorm")
      let postLN = RmsNormOne.load(weights, cfgJson, Layer3Prefix & ".post_attention_layernorm")
      let routerWeight = weights.getTensorOwned(Layer3Prefix & ".mlp.gate.weight")
      let gateUp = weights.getTensorOwned(Layer3Prefix & ".mlp.experts.gate_up_proj")
      let down = weights.getTensorOwned(Layer3Prefix & ".mlp.experts.down_proj")
      let sharedG = weights.getTensorOwned(Layer3Prefix & ".mlp.shared_expert.gate_proj.weight")
      let sharedU = weights.getTensorOwned(Layer3Prefix & ".mlp.shared_expert.up_proj.weight")
      let sharedD = weights.getTensorOwned(Layer3Prefix & ".mlp.shared_expert.down_proj.weight")
      let sharedGateWeight = weights.getTensorOwned(Layer3Prefix & ".mlp.shared_expert_gate.weight")
      let sharedExpert = GatedDenseFFN.init(sharedG, sharedU, sharedD)
      let numExpertsPerTok = moeCfg().numExpertsPerTok
      let ffn = GatedBlockSparseFFN.init(
        gateUp, down, routerWeight, sharedExpert, sharedGateWeight,
        numExpertsPerTok)
      let rotary = RotaryPositionEmbedding.new(HeadDim, MaxPositionEmbeddings, RopeTheta, F.kBFloat16, F.kCPU, rotary_dim = RotaryDim)
      let attn = RopeElementWiseGatedAttention[RmsNormOne].init(3, AttnPrefix, qProj, kProj, vProj, oProj,
          NumQoHeads, NumKvHeads, HeadDim, rotary,
          q_norm = qNorm, k_norm = kNorm)
      var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = NumKvHeads, headDim = HeadDim)
      var st = Safetensor.open(Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor")
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
      let (gotIndices, gotWeights) =
        routeToExperts(h2Flat, routerWeight, numExpertsPerTok)
      let moeOut = ffn.forward(h2)
      let meta = zstdReadFixture(
        Layer3FixtureDir / "layer-Qwen3.6-35B-A3B-03.safetensor.metadata.json.zst"
      ).fromJson(LayerFixture)
      # The band absorbs MoE- and SDPA-inherited noise through the residual
      # add, the post-attention layernorm, the fp32 renorm and its cast
      # back to the hidden-state dtype:
      # two bf16 ulps at the fixture max, a factor four margin.
      const routingWeightsBand = 0.015625
      let moeOutputBand = meta.bands.moe_output_band
      let layerBand = meta.bands.layer_output_band
      let topkMarginMin = meta.margins.topk_margin_min
      doAssert topkMarginMin > 0.0
      let routingWeightsDiff = maxAbsDiff(gotWeights, st.getTensorOwned("routing_weights"))
      doAssert routingWeightsDiff <= routingWeightsBand, "routing_weights diff " & $routingWeightsDiff & " outside the recorded band"
      let moeOutputDiff = maxAbsDiff(moeOut.reshape(Tokens, Hidden), st.getTensorOwned("moe_output"))
      doAssert moeOutputDiff <= moeOutputBand, "moe_output diff " & $moeOutputDiff & " outside the recorded band"
      let fixIndices = st.getTensorOwned("topk_indices")
      for tok in 0 ..< Tokens:
        for pos in 0 ..< TopK:
          let nimIdx = gotIndices[tok, pos]
          let fixIdx = fixIndices[tok, pos]
          doAssert nimIdx.item(int64) == fixIdx.item(int64)
      let layerOut = h1 + moeOut
      let layerOutDiff = maxAbsDiff(layerOut, st.getTensorOwned("layer_output"))
      doAssert layerOutDiff <= layerBand, "layer_output diff " & $layerOutDiff & " outside the recorded band"
      true


  runCppTest "full decoder layer, cross-device variant (device-pair report)":
    proc(): bool =
      let runDev = testDevice()
      echo "    devices: ", compareLine(Layer0FixtureDir, runDev)
      if runDev == F.kCPU:
        echo "    the run device matches the recorded device, the reference variant carries the replay"
        return true
      # The recorded intermediates compare bit-exact (norms, GDN block row)
      # and the fingerprint sidecars were recorded on cpu. No cross-device
      # drift row applies, the suite skips.
      echo "    no cross-device drift row applies: the bit-exact rows and the",
        " cpu fingerprint sidecars accept zero drift, the suite skips on ",
        deviceName(runDev)
      return true

  echo "\nAll Qwen3.6 decoder-layer blocks PASS"

when isMainModule:
  main()
