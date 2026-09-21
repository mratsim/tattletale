# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer 0/1/4 unit replay of the Laguna-XS-2.1 checkpoint.
##
## The plain local-residual block normalizes, mixes the attention, adds
## the residual, renormalizes and mixes the dense or routed FFN.
##
## - the per-head softplus g_proj scalars multiply the attention rows before the o_proj
## - layer 0 runs the dense prefix block, layers 1/4 the 256-expert routed block
## - the recorded post_attention_layernorm_output row is the residual sum the norm consumes
##
## Replay runs on testDevice(), Metal here, the fixture recording torch-side mps.
##
## Requires the local model at tests/hf_models/Laguna-XS-2.1 (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_laguna_01_layer_internals.nim

import
  std/options,
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/moe_router,
  workspace/transformers/src/layers/rope,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

{.experimental: "callOperator".}

privateAccess(RopeGQAttention[RmsNorm])

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "Laguna-XS-2.1"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "Laguna-XS-2.1-layer-0-1-4" / "layer0-1-4-Laguna-XS-2.1-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"

proc main(): bool =
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let hiddenSize = cfgJson{"hidden_size"}.getInt()
  let kvHeads = cfgJson{"num_key_value_heads"}.getInt()
  let headDim = cfgJson{"head_dim"}.getInt()
  let numExpertsPerTok = cfgJson{"num_experts_per_tok"}.getInt()
  let window = cfgJson{"sliding_window"}.getInt()
  let actDtype = getDeployDtype(cfgJson)
  let view = SafetensorsCollection.open(ModelDir)

  var st = Safetensor.open(FixturePath)
  # The replay seq length reads off the recorded input tensor shape.
  let seqLen = st.getTensorOwned("layer0.input", dev).size(1)

  # Rope tables, one per layer kind:
  #
  # - the full-attention yarn form rotates half the head dim over
  #   the blended frequency table, the table carries the attention factor
  # - the sliding kind runs the default theta over the full head dim
  let ropeFull = cfgJson{"rope_parameters"}{"full_attention"}
  let rotaryDim = int(headDim.float64 *
    ropeFull{"partial_rotary_factor"}.getFloat(1.0))
  let rotaryYarn = RotaryPositionEmbedding.new(headDim, seqLen,
    ropeFull{"rope_theta"}.getFloat(), actDtype, dev,
    rotary_dim = rotaryDim,
    yarnFactor = ropeFull{"factor"}.getFloat(),
    yarnBetaFast = ropeFull{"beta_fast"}.getFloat(),
    yarnBetaSlow = ropeFull{"beta_slow"}.getFloat(),
    yarnOriginalMaxPos =
      ropeFull{"original_max_position_embeddings"}.getInt(),
    attentionFactor = ropeFull{"attention_factor"}.getFloat(1.0))
  let rotarySliding = RotaryPositionEmbedding.new(headDim, seqLen,
    cfgJson{"rope_parameters"}{"sliding_attention"}{"rope_theta"}.getFloat(),
    actDtype, dev)

  # Layer surfaces loading exactly as the Laguna model file wires them:
  #
  # - the attention mixer carries the per-head g_proj projection
  # - the routed blocks load the sigmoid top-k router with the selection
  #   bias over the hidden-dtype scoring GEMM
  # - the returned routing weights stay the unscaled renormalized scores,
  #   the routed scaling factor applies at the FFN routed output, layer 0
  #   loads its dense prefix block
  var routers: array[3, NoAuxTopCorr]
  var ffns: array[3, BlockSparseFFN]
  var attns: array[3, RopeGQAttention[RmsNorm]]
  var inputLNs: array[3, RmsNorm]
  var postLNs: array[3, RmsNorm]
  var denseFfn: GatedDenseFFN
  for li, layerIdx in [0, 1, 4]:
    let lp = "model.layers." & $layerIdx & "."
    let heads = cfgJson{"num_attention_heads_per_layer"}[layerIdx].getInt()
    let sliding =
      cfgJson{"layer_types"}[layerIdx].getStr() == "sliding_attention"
    let routed = layerIdx != 0
    attns[li] = RopeGQAttention[RmsNorm].load(view, cfgJson,
      lp & "self_attn", layerIdx, heads, kvHeads, headDim,
      if sliding: rotarySliding else: rotaryYarn, dev,
      window = if sliding: window else: FullVisibilityWindow,
      perHeadGate = true)
    inputLNs[li] = RmsNorm.load(view, cfgJson, lp & "input_layernorm", dev)
    postLNs[li] = RmsNorm.load(view, cfgJson,
      lp & "post_attention_layernorm", dev)
    if routed:
      routers[li] = NoAuxTopCorr.init(
        view.getTensorOwned(lp & "mlp.gate.weight", dev),
        view.getTensorOwned(lp & "mlp.experts.e_score_correction_bias", dev),
        numExpertsPerTok, 1, 1, 1.0,
        cfgJson{"norm_topk_prob"}.getBool(false),
        scalesWeights = false, scoreBf16 = true)
      ffns[li] = BlockSparseFFN.load(view, cfgJson, lp & "mlp", routers[li],
        dev, routedOutputScale =
          cfgJson{"moe_routed_scaling_factor"}.getFloat(1.0))
    else:
      denseFfn = GatedDenseFFN.load(view, cfgJson, lp & "mlp", dev)

  var ctx = InferenceContext.init(
    num_layers = 1, batch_size = 1, kv_heads = kvHeads,
    max_seq = seqLen, head_dim = headDim)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, dev)).unsqueeze(0)

  # Layer chains over the three mixtures, one manual replay
  # per mixture, asserted equal to the module forward before saving:
  #
  # - the attention op surface over one normalized layer input, one
  #   per-head softplus row from the g_proj output cast to the hidden
  #   dtype before the o_proj
  # - the rotation computes f32 over the bf16-grid rows, one bf16
  #   rounding at the rotation output
  # - the residual sum after the attention replays under the recorded
  #   post_attention_layernorm_output name, the input the second norm consumes
  for li, layerIdx in [0, 1, 4]:
    let layerTag = "layer" & $layerIdx
    let heads = cfgJson{"num_attention_heads_per_layer"}[layerIdx].getInt()
    let routed = layerIdx != 0
    let x = st.getTensorOwned(layerTag & ".input", dev)
    ctx.setRopeForPositions(
      if cfgJson{"layer_types"}[layerIdx].getStr() == "sliding_attention":
        rotarySliding
      else:
        rotaryYarn)
    assertStats(ctx.cos, StatsPath, layerTag & ".cos", kElementwise,
      msg = layerTag & " rope cos rows")
    assertStats(ctx.sin, StatsPath, layerTag & ".sin", kElementwise,
      msg = layerTag & " rope sin rows")
    let cosRows = ctx.cos.to(F.kFloat32)
    let sinRows = ctx.sin.to(F.kFloat32)
    let hNorm = inputLNs[li].forward(x)
    assertStats(hNorm, StatsPath,
      layerTag & ".layer.input_layernorm_output", kElementwise,
      msg = layerTag & " input layernorm output")
    let q = attns[li].q_proj.forward(hNorm).reshape(
      [1, seqLen, heads, headDim])
    let k = attns[li].k_proj.forward(hNorm).reshape(
      [1, seqLen, kvHeads, headDim])
    let v = attns[li].v_proj.forward(hNorm).reshape(
      [1, seqLen, kvHeads, headDim])
    let qNormed = attns[li].q_norm.forward(q)
    let kNormed = attns[li].k_norm.forward(k)
    let (qRot, kRot) = attns[li].rotary.applyRope(qNormed, kNormed,
      cosRows, sinRows)
    let qRotBf = qRot.to(F.kBFloat16)
    let kRotBf = kRot.to(F.kBFloat16)
    assertStats(qRotBf, StatsPath, layerTag & ".q_rot", kReduction,
      msg = layerTag & " rotated query rows")
    assertStats(kRotBf, StatsPath, layerTag & ".k_rot", kReduction,
      msg = layerTag & " rotated key rows")
    assertStats(v, StatsPath, layerTag & ".v", kReduction,
      msg = layerTag & " value projection output")
    let gate = F.softplus(
      attns[li].gProj.unsafeGet().forward(hNorm).to(F.kFloat32)
    ).to(F.kBFloat16)
    assertStats(gate, StatsPath, layerTag & ".gate", kElementwise,
      msg = layerTag & " per-head softplus row")
    let kExpanded = kRotBf.repeat_interleave(heads div kvHeads, 2).contiguous()
    let vExpanded = v.repeat_interleave(heads div kvHeads, 2).contiguous()
    assertStats(kExpanded, StatsPath, layerTag & ".k_expanded", kReduction,
      msg = layerTag & " expanded key rows")
    assertStats(vExpanded, StatsPath, layerTag & ".v_expanded", kReduction,
      msg = layerTag & " expanded value rows")
    let sdpaOut = attns[li].gqa_attn.forward(qRotBf, kExpanded, vExpanded,
      is_causal = true, enable_gqa = false)
    assertStats(sdpaOut, StatsPath, layerTag & ".sdpa_output", kReduction,
      depth = 2, msg = layerTag & " sdpa output")
    let gated = sdpaOut.reshape([1, seqLen, heads, headDim]) *
      gate.reshape([1, seqLen, heads, 1])
    assertStats(gated, StatsPath, layerTag & ".gated_output", kReduction,
      depth = 2, msg = layerTag & " gated attention rows")
    let attnOut = attns[li].o_proj.forward(
      gated.reshape([1, seqLen, heads * headDim]))
    assertStats(attnOut, StatsPath, layerTag & ".attn_output", kReduction,
      depth = 2, msg = layerTag & " o_proj output")
    let h1 = x + attnOut
    assertStats(h1, StatsPath,
      layerTag & ".layer.post_attention_layernorm_output", kReduction,
      msg = layerTag & " residual sum before the second norm")
    let hNorm2 = postLNs[li].forward(h1)
    let mlpOut =
      if routed: ffns[li].forward(hNorm2)
      else: denseFfn.forward(hNorm2)
    # The prefill pass composes the eager weighted accumulation over
    # the expert body reduction, one depth step past the record.
    assertStats(mlpOut, StatsPath, layerTag & ".layer.mlp_output",
      kReduction, depth = if routed: 2 else: 1,
      msg = layerTag & " FFN block output")
    let layerOut = h1 + mlpOut
    assertStats(layerOut, StatsPath, layerTag & ".layer.layer_output",
      kReduction, msg = layerTag & " post-residual output")

  # Mixture moe, the layer-1 routed block over the raw recorded hidden,
  # the router decision rows with the layer-1 and layer-4 surfaces
  # routing over their raw layer inputs.
  #
  # - the scoring logits record at f32, the router emits f32 logits
  # - the top-k weights record the unscaled renormalized scores cast
  #   to the hidden dtype
  block:
    let h3d = st.getTensorOwned("moe.h", dev)
    let hFlat = h3d.reshape([h3d.numel() div hiddenSize, hiddenSize])
    let decision = routers[1].route(hFlat)
    assertStats(decision.logits, StatsPath, "moe.router_logits",
      kReduction, msg = "router scoring logits")
    assertStats(decision.weights.to(F.kBFloat16), StatsPath,
      "moe.topk_weights", kReduction, msg = "sigmoid top-k weights")
    let moeOutput = ffns[1].forward(hFlat)
    assertStats(moeOutput, StatsPath, "moe.moe_output", kReduction,
      depth = 2, msg = "routed block output")
    assertStats(view.getTensorOwned("model.layers.1.mlp.gate.weight", dev),
      StatsPath, "moe.gate_weight", kElementwise,
      msg = "router gate weight rows")
    assertStats(view.getTensorOwned(
      "model.layers.1.mlp.experts.e_score_correction_bias", dev),
      StatsPath, "moe.bias", kElementwise,
      msg = "router selection bias rows")

    let h1Flat = st.getTensorOwned("layer1.input", dev).reshape(
      [seqLen, hiddenSize])
    let decision1 = routers[1].route(h1Flat)
    assertStats(decision1.logits, StatsPath, "layer1.moe.router_logits",
      kReduction, msg = "layer-1 router scoring logits")
    assertStats(decision1.weights.to(F.kBFloat16), StatsPath,
      "layer1.moe.topk_weights", kReduction,
      msg = "layer-1 sigmoid top-k weights")
    assertStats(ffns[1].forward(st.getTensorOwned("layer1.input", dev)),
      StatsPath, "layer1.moe.moe_output", kReduction, depth = 2,
      msg = "layer-1 routed block output")

    let h4Flat = st.getTensorOwned("layer4.input", dev).reshape(
      [seqLen, hiddenSize])
    let decision4 = routers[2].route(h4Flat)
    assertStats(decision4.logits, StatsPath, "layer4.moe.router_logits",
      kReduction, msg = "layer-4 router scoring logits")
    assertStats(decision4.weights.to(F.kBFloat16), StatsPath,
      "layer4.moe.topk_weights", kReduction,
      msg = "layer-4 sigmoid top-k weights")
    assertStats(ffns[2].forward(st.getTensorOwned("layer4.input", dev)),
      StatsPath, "layer4.moe.moe_output", kReduction, depth = 2,
      msg = "layer-4 routed block output")
  result = true

when isMainModule:
  runCppTest("laguna layer 0/1/4 internals", main)
