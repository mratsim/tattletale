# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer 0/13/14/15/19 unit replay of the gemma-4-E2B-it checkpoint.
##
## The sandwich block runs two norms around attention and FFN, the per-layer embedding tail follows every block:
##
## - the kv-sharing layers 15/19 gather the stored post-rope keys, post-normed values from the last own-kv sliding layer 13
## - the recorded post_attention_layernorm_output row is the post-norm residual sum, unlike the Laguna pre-norm spelling
## - the layer scalar scales the layer output after the per-layer tail
##
## Replay runs on testDevice(), Metal here, the fixture recording torch-side mps.
##
## Requires the local model at tests/hf_models/gemma-4-E2B-it (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_gemma4e2b_01_layer_internals.nim

import
  std/math,
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/positron,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/rope,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/models/gemma4e2b {.all.},
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

{.experimental: "callOperator".}

privateAccess(RopeGQAttention[FusedRmsNorm])
privateAccess(Gemma4E2BConfig)

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "gemma-4-E2B-it"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "gemma-4-E2B-it-layer-0-13-14-15-19" /
    "layer0-13-14-15-19-gemma-4-E2B-it-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"

proc main(): bool =
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let config = loadGemma4E2BConfig(ModelDir / "config.json")
  let cfgJson = (ModelDir / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)
  let view = SafetensorsCollection.open(ModelDir)
  var st = Safetensor.open(FixturePath)
  # The replay seq length reads off the recorded input tensor shape.
  let seqLen = st.getTensorOwned("layer0.input", dev).size(1)
  let lp0 = "model.language_model."
  let firstShared = config.num_hidden_layers - config.num_kv_shared_layers

  # Rope tables, one per layer kind:
  #
  # - the full-attention kind runs the proportional zero-tail table over
  #   the global head dim, only the first quarter of the pairs rotate
  # - the sliding kind runs the default theta over the sliding head dim
  let rotaryFull = RotaryPositionEmbedding.new(
    config.global_head_dim, config.max_position_embeddings,
    config.ropeFullTheta, actDtype, dev,
    rotary_dim = config.ropeFullRotaryDim,
    activePairs = config.ropeFullActivePairs)
  let rotarySliding = RotaryPositionEmbedding.new(
    config.head_dim, config.max_position_embeddings,
    config.ropeSlidingTheta, actDtype, dev)

  # Layer surfaces loading exactly as the gemma-4-E2B model file wires them:
  #
  # - the attention mixer carries the per-head q/k norms and the value norm
  # - the kv-sharing layers skip their dead k/v projections and norms,
  #   the value norm of an own-kv layer stays the ones-weight form
  # - the FFN block loads the tanh-approximate GELU activation over the checkpoint weight shapes
  var inputLNs: array[5, FusedRmsNorm]
  var postAttnLNs: array[5, FusedRmsNorm]
  var preFfnLNs: array[5, FusedRmsNorm]
  var postFfnLNs: array[5, FusedRmsNorm]
  var attns: array[5, RopeGQAttention[FusedRmsNorm]]
  var mlps: array[5, GatedDenseFFN]
  var pleGates: array[5, Linear]
  var pleProjs: array[5, Linear]
  var pleNorms: array[5, FusedRmsNorm]
  var layerScalars: array[5, Tensor]
  for li, layerIdx in [0, 13, 14, 15, 19]:
    let lp = lp0 & "layers." & $layerIdx & "."
    let sliding = config.layerKinds[layerIdx] == alkSlidingAttention
    let headDim = if sliding: config.head_dim else: config.global_head_dim
    let shared = layerIdx >= firstShared
    inputLNs[li] = FusedRmsNorm.load(view, cfgJson, lp & "input_layernorm", dev)
    postAttnLNs[li] = FusedRmsNorm.load(view, cfgJson,
      lp & "post_attention_layernorm", dev)
    preFfnLNs[li] = FusedRmsNorm.load(view, cfgJson,
      lp & "pre_feedforward_layernorm", dev)
    postFfnLNs[li] = FusedRmsNorm.load(view, cfgJson,
      lp & "post_feedforward_layernorm", dev)
    let vNorm =
      if not shared:
        FusedRmsNorm.init(
          F.ones(headDim, F.tensorOptions(F.kBFloat16, dev)),
          eps = config.rms_norm_eps)
      else:
        nil
    attns[li] = RopeGQAttention[FusedRmsNorm].load(view, cfgJson,
      lp & "self_attn", layerIdx,
      config.num_attention_heads, config.num_key_value_heads, headDim,
      if sliding: rotarySliding else: rotaryFull, dev,
      window = if sliding: config.sliding_window else: FullVisibilityWindow,
      softmaxScale = 1.0,
      kvSourceLayer = if shared: 13 else: -1,
      vNorm = vNorm)
    mlps[li] = GatedDenseFFN.load(view, cfgJson, lp & "mlp", dev,
      activation = kGeluTanh)
    pleGates[li] = Linear.load(view, cfgJson,
      lp & "per_layer_input_gate", dev)
    pleProjs[li] = Linear.load(view, cfgJson,
      lp & "per_layer_projection", dev)
    pleNorms[li] = FusedRmsNorm.init(
      view.getTensorOwned(lp & "post_per_layer_input_norm.weight", dev),
      eps = config.rms_norm_eps)
    layerScalars[li] = view.getTensorOwned(lp & "layer_scalar", dev)

  let pleEmbedTable = view.getTensorOwned(
    lp0 & "embed_tokens_per_layer.weight", dev)
  let pleTokScale = F.full(1,
    sqrt(config.hidden_size_per_layer_input.float64),
    F.tensorOptions(F.kBFloat16, dev))
  let perLayerModelProjection = Linear.load(view, cfgJson,
    lp0 & "per_layer_model_projection", dev)
  let perLayerProjectionNorm = FusedRmsNorm.init(view.getTensorOwned(
    lp0 & "per_layer_projection_norm.weight", dev), eps = config.rms_norm_eps)

  var ctx = InferenceContext.init(
    num_layers = 1, batch_size = 1,
    kv_heads = config.num_key_value_heads,
    max_seq = seqLen, head_dim = config.head_dim)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, dev)).unsqueeze(0)

  # PLE pipeline mixture over the recorded token ids:
  #
  # - the per-layer token identity embeds the ids through the per-layer table, scaled by the per-layer hidden size root
  # - the combined rows sum the projected context and the identity under
  #   the half factor
  block:
    let ids: seq[int64] = @[9259, 993, 338, 4453, 2082, 531]
    let inputsEmbeds = st.getTensorOwned("ple.inputs_embeds", dev)
    let tokenIdentity = pleEmbedTable.index_select(
      0, ids.toTensor().to(dev)).reshape(
      [1, seqLen, config.num_hidden_layers,
       config.hidden_size_per_layer_input]) * pleTokScale
    assertStats(tokenIdentity, StatsPath, "ple.token_identity_output",
      kElementwise, msg = "per-layer token identity rows")
    let pleProjRaw = perLayerModelProjection.forward(inputsEmbeds) *
      Scalar(pow(config.hidden_size.float64, -0.5))
    let pleProj = perLayerProjectionNorm.forward(
      pleProjRaw.reshape([1, seqLen, config.num_hidden_layers,
        config.hidden_size_per_layer_input]))
    let combined = (pleProj + tokenIdentity) * Scalar(pow(2.0, -0.5))
    assertStats(combined, StatsPath, "ple.combined_output", kReduction,
      msg = "combined per-layer input rows")

  # Shared kv store, the post-rope k and post-norm v the storing layers
  # leave for the kv-sharing layers of the same kind.
  #
  # - layer 0 replays its own mixture, the chain mixture starts at its recorded layer-13 input and feeds layer outputs through 14/15/19
  # - the attention op surface runs over one normalized layer input
  # - the rotation computes f32 over the bf16-grid rows, one bf16
  #   rounding at the rotation output
  var sharedKvK: array[2, Tensor]   # 0 = sliding, 1 = full
  var sharedKvV: array[2, Tensor]
  var x = st.getTensorOwned("layer0.input", dev)

  for li, layerIdx in [0, 13, 14, 15, 19]:
    if layerIdx == 13:
      x = st.getTensorOwned("chain.input", dev)
    let tag = if layerIdx == 0: "layer0" else: "chain.layer" & $layerIdx
    let sliding = config.layerKinds[layerIdx] == alkSlidingAttention
    let headDim = if sliding: config.head_dim else: config.global_head_dim
    let shared = layerIdx >= firstShared
    let kindIdx = if sliding: 0 else: 1
    let pleInput =
      if layerIdx == 0: st.getTensorOwned("layer0.ple_input", dev)
      else: st.getTensorOwned("chain.ple" & $layerIdx, dev)

    ctx.setRopeForPositions(if sliding: rotarySliding else: rotaryFull)
    assertStats(ctx.cos, StatsPath, tag & ".cos", kElementwise,
      msg = tag & " rope cos rows")
    assertStats(ctx.sin, StatsPath, tag & ".sin", kElementwise,
      msg = tag & " rope sin rows")
    let cosRows = ctx.cos.to(F.kFloat32)
    let sinRows = ctx.sin.to(F.kFloat32)
    let hNorm = inputLNs[li].forward(x)
    assertStats(hNorm, StatsPath,
      tag & ".layer.input_layernorm_output", kElementwise, depth = 2,
      msg = tag & " input layernorm output")
    let heads = config.num_attention_heads
    let kvHeads = config.num_key_value_heads
    let q = attns[li].q_proj.forward(hNorm).reshape(
      [1, seqLen, heads, headDim])
    let qNormed = attns[li].q_norm.forward(q)
    var kRotBf: Tensor
    var vNormed: Tensor
    if not shared:
      let k = attns[li].k_proj.forward(hNorm).reshape(
        [1, seqLen, kvHeads, headDim])
      let kNormed = attns[li].k_norm.forward(k)
      let v = attns[li].v_proj.forward(hNorm).reshape(
        [1, seqLen, kvHeads, headDim])
      vNormed = attns[li].vNorm.forward(v)
      let (qRot, kRot) = attns[li].rotary.applyRope(qNormed, kNormed,
        cosRows, sinRows)
      kRotBf = kRot.to(F.kBFloat16)
      sharedKvK[kindIdx] = kRotBf
      sharedKvV[kindIdx] = vNormed
    else:
      kRotBf = sharedKvK[kindIdx]
      vNormed = sharedKvV[kindIdx]
    let qRotPair = attns[li].rotary.applyRope(qNormed, qNormed,
      cosRows, sinRows)
    let qRotBf = qRotPair[0].to(F.kBFloat16)
    assertStats(qRotBf, StatsPath, tag & ".q_rot", kReduction,
      msg = tag & " rotated query rows")
    assertStats(kRotBf, StatsPath, tag & ".k_rot", kReduction,
      msg = tag & " rotated key rows")
    assertStats(vNormed, StatsPath, tag & ".v", kReduction,
      msg = tag & " value norm output")
    let kExpanded = kRotBf.repeat_interleave(heads div kvHeads, 2).contiguous()
    let vExpanded = vNormed.repeat_interleave(heads div kvHeads, 2).contiguous()
    assertStats(kExpanded, StatsPath, tag & ".k_expanded", kReduction,
      depth = 2, msg = tag & " expanded key rows")
    assertStats(vExpanded, StatsPath, tag & ".v_expanded", kReduction,
      depth = 2, msg = tag & " expanded value rows")
    let sdpaOut = attns[li].gqa_attn.forward(qRotBf, kExpanded, vExpanded,
      is_causal = true, enable_gqa = false)
    assertStats(sdpaOut, StatsPath, tag & ".sdpa_output", kReduction,
      depth = 2, msg = tag & " sdpa output")
    let attnOut = attns[li].o_proj.forward(sdpaOut)
    assertStats(attnOut, StatsPath, tag & ".attn_output", kReduction,
      depth = 2, msg = tag & " o_proj output")

    let postAttn = postAttnLNs[li].forward(attnOut)
    let h1 = x + postAttn
    assertStats(h1, StatsPath,
      tag & ".layer.post_attention_layernorm_output", kReduction,
      msg = tag & " post-norm residual sum")
    let h2 = preFfnLNs[li].forward(h1)
    assertStats(h2, StatsPath,
      tag & ".layer.pre_feedforward_layernorm_output", kElementwise,
      depth = 2, msg = tag & " pre-feedforward norm output")
    let mlpOut = mlps[li].forward(h2)
    assertStats(mlpOut, StatsPath, tag & ".layer.mlp_output", kReduction,
      msg = tag & " FFN block output")
    let h3 = h1 + postFfnLNs[li].forward(mlpOut)
    assertStats(h3, StatsPath,
      tag & ".layer.post_feedforward_layernorm_output", kReduction,
      msg = tag & " post-norm residual sum after the FFN")

    let gated = pleGates[li].forward(h3)
    assertStats(gated, StatsPath, tag & ".ple.gate_output", kReduction,
      msg = tag & " per-layer input gate output")
    let product = gelu_tanh(gated) * pleInput
    assertStats(product, StatsPath, tag & ".ple.gated_product", kElementwise,
      depth = 3, msg = tag & " gated product rows")
    let projected = pleProjs[li].forward(product)
    assertStats(projected, StatsPath, tag & ".ple.projection_output",
      kReduction, depth = 2, msg = tag & " per-layer projection output")
    let normalized = pleNorms[li].forward(projected)
    assertStats(normalized, StatsPath, tag & ".ple.post_norm_output",
      kElementwise, depth = 2, msg = tag & " per-layer post-norm output")
    let layerOut = (h3 + normalized) * layerScalars[li]
    assertStats(layerOut, StatsPath, tag & ".layer.layer_output",
      kReduction, msg = tag & " scaled layer output")
    x = layerOut
  result = true

when isMainModule:
  runCppTest("gemma-4-E2B layer 0/13/14/15/19 internals", main)
