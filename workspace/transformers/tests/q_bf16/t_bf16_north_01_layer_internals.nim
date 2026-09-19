# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer 0/1/4 unit replay of the North-Mini-Code-1.0 checkpoint.
##
## The parallel block normalizes once, both mixers read the same
## normalized rows, the block output adds both contributions.
##
## - rope layers 0/1 seat the paired q/k projections, layer 4 is unrotated
## - routed blocks replay the degenerate one-group sigmoid top-k router
## - the moe surfaces replay over the raw layer inputs
##
## Replay runs on testDevice(), Metal here, the fixture recording torch-side mps.
##
## Requires the local model at tests/hf_models/North-Mini-Code-1.0 (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_north_01_layer_internals.nim

import
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
  workspace/transformers/src/models/north {.all.},
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

{.experimental: "callOperator".}

privateAccess(RopeGQAttention[void])

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "North-Mini-Code-1.0"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "North-Mini-Code-1.0-layer-0-1-4" / "layer0-1-4-North-Mini-Code-1.0-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"

proc main(): bool =
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let numHeads = cfgJson{"num_attention_heads"}.getInt()
  let kvHeads = cfgJson{"num_key_value_heads"}.getInt()
  let headDim = cfgJson{"head_dim"}.getInt()
  let hiddenSize = cfgJson{"hidden_size"}.getInt()
  let numExperts = cfgJson{"num_experts"}.getInt()
  let numExpertsPerTok = cfgJson{"num_experts_per_tok"}.getInt()
  let window = cfgJson{"sliding_window"}.getInt()
  let actDtype = getDeployDtype(cfgJson)
  let view = SafetensorsCollection.open(ModelDir)

  var st = Safetensor.open(FixturePath)
  # The replay seq length reads off the recorded input tensor shape.
  let x0 = st.getTensorOwned("layer0.input", dev)   # (1, 6, 2048) bf16
  let seqLen = x0.size(1)
  let rotary = RotaryPositionEmbedding.new(headDim, seqLen,
    cfgJson{"rope_theta"}.getFloat(), actDtype, dev)

  # The mixers load exactly as the North model file wires them.
  #
  # - rope-carrying layers seat the evens-then-odds paired q/k projection
  #   rows over the plain load
  # - the unrotated layer 4 keeps the checkpoint rows as is
  let attn0 = RopeGQAttention[void].load(view, cfgJson,
    "model.layers.0.self_attn", 0, numHeads, kvHeads, headDim, rotary, dev)
  attn0.q_proj = loadRopePairedProjection(view, cfgJson,
    "model.layers.0.self_attn.q_proj", numHeads, headDim, dev)
  attn0.k_proj = loadRopePairedProjection(view, cfgJson,
    "model.layers.0.self_attn.k_proj", kvHeads, headDim, dev)
  let attn1 = RopeGQAttention[void].load(view, cfgJson,
    "model.layers.1.self_attn", 1, numHeads, kvHeads, headDim, rotary, dev,
    window = window)
  attn1.q_proj = loadRopePairedProjection(view, cfgJson,
    "model.layers.1.self_attn.q_proj", numHeads, headDim, dev)
  attn1.k_proj = loadRopePairedProjection(view, cfgJson,
    "model.layers.1.self_attn.k_proj", kvHeads, headDim, dev)
  let attn4 = RopeGQAttention[void].load(view, cfgJson,
    "model.layers.4.self_attn", 4, numHeads, kvHeads, headDim, rotary, dev)

  let inputLN0 = RmsNorm.load(view, cfgJson, "model.layers.0.input_layernorm", dev)
  let ffn0 = GatedDenseFFN.load(view, cfgJson, "model.layers.0.mlp", dev)
  let inputLN1 = RmsNorm.load(view, cfgJson, "model.layers.1.input_layernorm", dev)
  let inputLN4 = RmsNorm.load(view, cfgJson, "model.layers.4.input_layernorm", dev)

  # The sigmoid top-k router at its degenerate grouping, the master
  # NoAuxTopCorr corner unchanged.
  #
  # - no selection bias
  # - one group, no renorm
  let router1 = NoAuxTopCorr.init(
    view.getTensorOwned("model.layers.1.mlp.gate.weight", dev),
    F.zeros(numExperts, F.tensorOptions(F.kFloat32, dev)),
    numExpertsPerTok, 1, 1, 1.0'f64,
    cfgJson{"norm_topk_prob"}.getBool(false))
  let ffn1 = BlockSparseFFN.load(view, cfgJson, "model.layers.1.mlp", router1, dev)
  let router4 = NoAuxTopCorr.init(
    view.getTensorOwned("model.layers.4.mlp.gate.weight", dev),
    F.zeros(numExperts, F.tensorOptions(F.kFloat32, dev)),
    numExpertsPerTok, 1, 1, 1.0'f64,
    cfgJson{"norm_topk_prob"}.getBool(false))
  let ffn4 = BlockSparseFFN.load(view, cfgJson, "model.layers.4.mlp", router4, dev)

  var ctx = InferenceContext.init(
    num_layers = 1, batch_size = 1, kv_heads = kvHeads,
    max_seq = seqLen, head_dim = headDim)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, dev)).unsqueeze(0)

  # The attention op surface over one normalized layer input, the rope
  # rows the caller seats.
  #
  # - the kv head repeat materializes through repeat_interleave
  # - sdpa runs is_causal
  # - the rotation computes f32 over the bf16-grid rows, one bf16
  #   rounding at the rotation output, the recorded rounding point
  block:
    let x1 = st.getTensorOwned("layer1.input", dev)   # (1, 6, 2048) bf16
    let x4 = st.getTensorOwned("layer4.input", dev)   # (1, 6, 2048) bf16

    # Mixture layer0, the dense prefix block, rope-carrying.
    ctx.setRopeForPositions(rotary)
    assertStats(ctx.cos, StatsPath, "layer0.cos", kElementwise,
      msg = "layer-0 rope cos rows")
    assertStats(ctx.sin, StatsPath, "layer0.sin", kElementwise,
      msg = "layer-0 rope sin rows")
    let cosRows0 = ctx.cos.to(F.kFloat32)
    let sinRows0 = ctx.sin.to(F.kFloat32)
    let hNorm0 = inputLN0.forward(x0)
    assertStats(hNorm0, StatsPath, "layer0.layer.input_layernorm_output",
      kElementwise, msg = "layer-0 input layernorm output")
    let q0 = attn0.q_proj.forward(hNorm0).reshape(
      [1, seqLen, numHeads, headDim])
    let k0 = attn0.k_proj.forward(hNorm0).reshape(
      [1, seqLen, kvHeads, headDim])
    let v0 = attn0.v_proj.forward(hNorm0).reshape(
      [1, seqLen, kvHeads, headDim])
    let (qRot0, kRot0) = attn0.rotary.applyRope(q0, k0, cosRows0, sinRows0)
    let qRotBf0 = qRot0.to(F.kBFloat16)
    let kRotBf0 = kRot0.to(F.kBFloat16)
    assertStats(qRotBf0, StatsPath, "layer0.q_rot", kReduction,
      msg = "layer-0 rotated query rows")
    assertStats(kRotBf0, StatsPath, "layer0.k_rot", kReduction,
      msg = "layer-0 rotated key rows")
    assertStats(v0, StatsPath, "layer0.v", kReduction,
      msg = "layer-0 value projection output")
    let kExpanded0 = kRotBf0.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    let vExpanded0 = v0.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    assertStats(kExpanded0, StatsPath, "layer0.k_expanded", kReduction,
      msg = "layer-0 expanded key rows")
    assertStats(vExpanded0, StatsPath, "layer0.v_expanded", kReduction,
      msg = "layer-0 expanded value rows")
    let sdpaOut0 = attn0.gqa_attn.forward(qRotBf0, kExpanded0, vExpanded0,
      is_causal = true, enable_gqa = false)
    assertStats(sdpaOut0, StatsPath, "layer0.sdpa_output", kReduction,
      depth = 2, msg = "layer-0 sdpa output")
    let attnOut0 = attn0.o_proj.forward(sdpaOut0)
    assertStats(attnOut0, StatsPath, "layer0.attn_output", kReduction,
      depth = 2, msg = "layer-0 o_proj output")
    let mlpOut0 = ffn0.forward(hNorm0)
    assertStats(mlpOut0, StatsPath, "layer0.layer.mlp_output", kReduction,
      msg = "layer-0 dense block output")
    let layerOut0 = attnOut0 + mlpOut0 + x0
    assertStats(layerOut0, StatsPath, "layer0.layer.layer_output", kReduction,
      msg = "layer-0 post-residual output")

    # Mixture layer1, the routed block, rope-carrying sliding layer.
    ctx.setRopeForPositions(rotary)
    assertStats(ctx.cos, StatsPath, "layer1.cos", kElementwise,
      msg = "layer-1 rope cos rows")
    assertStats(ctx.sin, StatsPath, "layer1.sin", kElementwise,
      msg = "layer-1 rope sin rows")
    let cosRows1 = ctx.cos.to(F.kFloat32)
    let sinRows1 = ctx.sin.to(F.kFloat32)
    let hNorm1 = inputLN1.forward(x1)
    assertStats(hNorm1, StatsPath, "layer1.layer.input_layernorm_output",
      kElementwise, msg = "layer-1 input layernorm output")
    let q1 = attn1.q_proj.forward(hNorm1).reshape(
      [1, seqLen, numHeads, headDim])
    let k1 = attn1.k_proj.forward(hNorm1).reshape(
      [1, seqLen, kvHeads, headDim])
    let v1 = attn1.v_proj.forward(hNorm1).reshape(
      [1, seqLen, kvHeads, headDim])
    let (qRot1, kRot1) = attn1.rotary.applyRope(q1, k1, cosRows1, sinRows1)
    let qRotBf1 = qRot1.to(F.kBFloat16)
    let kRotBf1 = kRot1.to(F.kBFloat16)
    assertStats(qRotBf1, StatsPath, "layer1.q_rot", kReduction,
      msg = "layer-1 rotated query rows")
    assertStats(kRotBf1, StatsPath, "layer1.k_rot", kReduction,
      msg = "layer-1 rotated key rows")
    assertStats(v1, StatsPath, "layer1.v", kReduction,
      msg = "layer-1 value projection output")
    let kExpanded1 = kRotBf1.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    let vExpanded1 = v1.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    assertStats(kExpanded1, StatsPath, "layer1.k_expanded", kReduction,
      msg = "layer-1 expanded key rows")
    assertStats(vExpanded1, StatsPath, "layer1.v_expanded", kReduction,
      msg = "layer-1 expanded value rows")
    let sdpaOut1 = attn1.gqa_attn.forward(qRotBf1, kExpanded1, vExpanded1,
      is_causal = true, enable_gqa = false)
    assertStats(sdpaOut1, StatsPath, "layer1.sdpa_output", kReduction,
      depth = 2, msg = "layer-1 sdpa output")
    let attnOut1 = attn1.o_proj.forward(sdpaOut1)
    assertStats(attnOut1, StatsPath, "layer1.attn_output", kReduction,
      depth = 2, msg = "layer-1 o_proj output")
    let mlpOut1 = ffn1.forward(hNorm1)
    # The prefill pass composes the eager weighted accumulation over
    # the expert body reduction, one depth step past the record.
    assertStats(mlpOut1, StatsPath, "layer1.layer.mlp_output", kReduction,
      depth = 2, msg = "layer-1 routed block output")
    let layerOut1 = attnOut1 + mlpOut1 + x1
    assertStats(layerOut1, StatsPath, "layer1.layer.layer_output", kReduction,
      msg = "layer-1 post-residual output")

    # Mixture layer4, the routed block on an unrotated layer where
    # zero-angle rows rotate nothing.
    ctx.cos = F.ones(seqLen, headDim,
      F.tensorOptions(F.kBFloat16, dev))
    ctx.sin = F.zeros(seqLen, headDim,
      F.tensorOptions(F.kBFloat16, dev))
    let hNorm4 = inputLN4.forward(x4)
    assertStats(hNorm4, StatsPath, "layer4.layer.input_layernorm_output",
      kElementwise, msg = "layer-4 input layernorm output")
    let q4 = attn4.q_proj.forward(hNorm4).reshape(
      [1, seqLen, numHeads, headDim])
    let k4 = attn4.k_proj.forward(hNorm4).reshape(
      [1, seqLen, kvHeads, headDim])
    let v4 = attn4.v_proj.forward(hNorm4).reshape(
      [1, seqLen, kvHeads, headDim])
    let (qRot4, kRot4) = attn4.rotary.applyRope(q4, k4, ctx.cos, ctx.sin)
    assertStats(qRot4, StatsPath, "layer4.q_rot", kReduction,
      msg = "layer-4 unrotated query rows")
    assertStats(kRot4, StatsPath, "layer4.k_rot", kReduction,
      msg = "layer-4 unrotated key rows")
    assertStats(v4, StatsPath, "layer4.v", kReduction,
      msg = "layer-4 value projection output")
    let kExpanded4 = kRot4.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    let vExpanded4 = v4.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    assertStats(kExpanded4, StatsPath, "layer4.k_expanded", kReduction,
      msg = "layer-4 expanded key rows")
    assertStats(vExpanded4, StatsPath, "layer4.v_expanded", kReduction,
      msg = "layer-4 expanded value rows")
    let sdpaOut4 = attn4.gqa_attn.forward(qRot4, kExpanded4, vExpanded4,
      is_causal = true, enable_gqa = false)
    assertStats(sdpaOut4, StatsPath, "layer4.sdpa_output", kReduction,
      depth = 2, msg = "layer-4 sdpa output")
    let attnOut4 = attn4.o_proj.forward(sdpaOut4)
    assertStats(attnOut4, StatsPath, "layer4.attn_output", kReduction,
      depth = 2, msg = "layer-4 o_proj output")
    let mlpOut4 = ffn4.forward(hNorm4)
    assertStats(mlpOut4, StatsPath, "layer4.layer.mlp_output", kReduction,
      depth = 2, msg = "layer-4 routed block output")
    let layerOut4 = attnOut4 + mlpOut4 + x4
    assertStats(layerOut4, StatsPath, "layer4.layer.layer_output", kReduction,
      msg = "layer-4 post-residual output")

  # Mixture moe, the layer-1 routed block over the raw recorded hidden.
  #
  # - the router scoring, the sigmoid top-k weights, the routed output
  # - the recorder stores this surface under the layer-1 names too
  # - the layer-4 surface replays the same raw-input form
  block:
    let h3d = st.getTensorOwned("moe.h", dev)   # (1, 6, 2048) bf16
    let hFlat = h3d.reshape([h3d.numel() div hiddenSize, hiddenSize])
    let decision = router1.route(hFlat)
    assertStats(decision.logits.to(F.kBFloat16), StatsPath,
      "moe.router_logits", kReduction, msg = "router scoring logits")
    assertStats(decision.weights.to(F.kBFloat16), StatsPath,
      "moe.topk_weights", kReduction, msg = "sigmoid top-k weights")
    let moeOutput = ffn1.forward(hFlat)
    assertStats(moeOutput, StatsPath, "moe.moe_output", kReduction,
      depth = 2, msg = "routed block output")
    assertStats(decision.logits.to(F.kBFloat16), StatsPath,
      "layer1.moe.router_logits", kReduction,
      msg = "layer-1 router scoring logits")
    assertStats(decision.weights.to(F.kBFloat16), StatsPath,
      "layer1.moe.topk_weights", kReduction,
      msg = "layer-1 sigmoid top-k weights")
    assertStats(moeOutput, StatsPath, "layer1.moe.moe_output", kReduction,
      depth = 2, msg = "layer-1 routed block output")

    let h4Flat = st.getTensorOwned("layer4.input", dev).reshape(
      [seqLen, hiddenSize])
    let decision4 = router4.route(h4Flat)
    assertStats(decision4.logits.to(F.kBFloat16), StatsPath,
      "layer4.moe.router_logits", kReduction,
      msg = "layer-4 router scoring logits")
    assertStats(decision4.weights.to(F.kBFloat16), StatsPath,
      "layer4.moe.topk_weights", kReduction,
      msg = "layer-4 sigmoid top-k weights")
    let moeOut4 = ffn4.forward(h4Flat)
    assertStats(moeOut4, StatsPath, "layer4.moe.moe_output", kReduction,
      depth = 2, msg = "layer-4 routed block output")
  result = true

when isMainModule:
  runCppTest("north layer 0/1/4 internals", main)
