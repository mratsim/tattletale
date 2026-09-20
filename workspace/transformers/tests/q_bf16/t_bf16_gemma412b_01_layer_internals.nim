# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer 0/5 unit replay of the gemma-4-12B-it checkpoint, one sliding
## layer and one full layer per the 5:1 sliding/full pattern.
##
## The sandwich block runs two norms around attention and FFN, the layer scalar scales the layer output:
##
## - the full layers are KV-tied (attention_k_eq_v), no v_proj weight exists,
##   the value rows are the unscaled value norm over the same k projection
##   the keys consume, the raw projection source row is the recorded k_proj_output
## - the reference cache keeps K and V as separate entries on both layer kinds,
##   the paged write/gather round-trip over the shared pool replays the recorded
##   cache_k/cache_v boundary rows
##
## Replay runs on testDevice(), Metal here, the fixture recording torch-side mps.
##
## Requires the local model at tests/hf_models/gemma-4-12B-it (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_gemma412b_01_layer_internals.nim

import
  std/math,
  std/options,
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
  workspace/transformers/src/instrumentation,
  workspace/transformers/src/models/gemma4_12b {.all.},
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(RopeGQAttention[FusedRmsNorm])
privateAccess(Gemma4Text12BConfig)

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "gemma-4-12B-it"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "gemma-4-12B-it-layer-0-5" /
    "layer0-5-gemma-4-12B-it-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"

proc main(): bool =
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let config = loadGemma4Text12BConfig(ModelDir / "config.json")
  let cfgJson = (ModelDir / "config.json").parseFile()
  let actDtype = getDeployDtype(cfgJson)
  let view = SafetensorsCollection.open(ModelDir)
  var st = Safetensor.open(FixturePath)
  # The replay seq length reads off the recorded input tensor shape.
  let seqLen = st.getTensorOwned("layer0.input", dev).size(1)

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

  # Layer surfaces loading exactly as the gemma-4-12B model file wires them:
  #
  # - the attention mixer carries the per-head q/k norms, the value norm is
  #   the ones-weight unscaled form on both layer kinds
  # - the KV-tied full layer skips its dead v_proj load, the value rows
  #   derive from the shared k projection at forward
  # - the FFN block loads the tanh-approximate GELU activation
  var inputLNs: array[2, FusedRmsNorm]
  var postAttnLNs: array[2, FusedRmsNorm]
  var preFfnLNs: array[2, FusedRmsNorm]
  var postFfnLNs: array[2, FusedRmsNorm]
  var attns: array[2, RopeGQAttention[FusedRmsNorm]]
  var mlps: array[2, GatedDenseFFN]
  var layerScalars: array[2, Tensor]
  for li, layerIdx in [0, 5]:
    let lp = "model.language_model.layers." & $layerIdx & "."
    let sliding = config.layerKinds[layerIdx] == alkSlidingAttention
    let headDim = if sliding: config.head_dim else: config.global_head_dim
    let kvHeads = if sliding: config.num_key_value_heads
                  else: config.numGlobalKvHeads
    inputLNs[li] = FusedRmsNorm.load(view, cfgJson, lp & "input_layernorm", dev)
    postAttnLNs[li] = FusedRmsNorm.load(view, cfgJson,
      lp & "post_attention_layernorm", dev)
    preFfnLNs[li] = FusedRmsNorm.load(view, cfgJson,
      lp & "pre_feedforward_layernorm", dev)
    postFfnLNs[li] = FusedRmsNorm.load(view, cfgJson,
      lp & "post_feedforward_layernorm", dev)
    let vNorm = FusedRmsNorm.init(
      F.ones(headDim, F.tensorOptions(F.kBFloat16, dev)),
      eps = config.rms_norm_eps)
    attns[li] = RopeGQAttention[FusedRmsNorm].load(view, cfgJson,
      lp & "self_attn", layerIdx,
      config.num_attention_heads, kvHeads, headDim,
      if sliding: rotarySliding else: rotaryFull, dev,
      window = if sliding: config.sliding_window else: FullVisibilityWindow,
      softmaxScale = 1.0,
      vNorm = vNorm,
      kEqV = not sliding)
    mlps[li] = GatedDenseFFN.load(view, cfgJson, lp & "mlp", dev,
      activation = kGeluTanh)
    layerScalars[li] = view.getTensorOwned(lp & "layer_scalar", dev)

  # The pool slots carry the widest kv geometry, the full_attention layers
  # run head_dim 512 over 1 kv head, the sliding layers 256 over 8 kv heads.
  # Both layers narrow the shared slot to their own leading planes.
  # The op-surface replay writes the replayed layers at their checkpoint
  # slot indices, the pool carries six layer slots.
  var (ctx, pool) = newKVContext(
    numLayers = 6, kvHeads = config.num_key_value_heads,
    headDim = config.global_head_dim,
    maxSeq = seqLen, device = dev)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, dev)).unsqueeze(0)

  # Per-layer replay, each layer consumes its recorded input:
  #
  # - the attention op surface runs over one normalized layer input.
  #   The KV-tied layer derives its value rows from the shared k projection
  # - the paged cache round-trip feeds the sdpa path, the stored rows
  #   equal the attention rows verbatim
  # - the rotation computes f32 over the bf16-grid rows, one bf16
  #   rounding at the rotation output
  for li, layerIdx in [0, 5]:
    let tag = "layer" & $layerIdx
    let x = st.getTensorOwned(tag & ".input", dev)
    let sliding = config.layerKinds[layerIdx] == alkSlidingAttention
    let headDim = if sliding: config.head_dim else: config.global_head_dim
    let kvHeads = if sliding: config.num_key_value_heads
                  else: config.numGlobalKvHeads

    ctx.setRopeForPositions(if sliding: rotarySliding else: rotaryFull)
    assertStats(ctx.cos, StatsPath, tag & ".cos", kElementwise,
      msg = tag & " rope cos rows")
    assertStats(ctx.sin, StatsPath, tag & ".sin", kElementwise,
      msg = tag & " rope sin rows")
    # The rotation consumes the bf16 grid rows the reference consumed,
    # the same table rows the op surface reads.
    let hNorm = inputLNs[li].forward(x)
    assertStats(hNorm, StatsPath,
      tag & ".layer.input_layernorm_output", kElementwise, depth = 2,
      msg = tag & " input layernorm output")
    let heads = config.num_attention_heads
    let q = attns[li].q_proj.forward(hNorm).reshape(
      [1, seqLen, heads, headDim])
    let qNormed = attns[li].q_norm.forward(q)
    # The k projection output feeds both the key path and, on the KV-tied
    # full layer, the value path. The raw reshaped rows are the recorded
    # k_proj_output the value derivation consumes.
    let kSource = attns[li].k_proj.forward(hNorm).reshape(
      [1, seqLen, kvHeads, headDim])
    let kNormed = attns[li].k_norm.forward(kSource)
    let vNormed =
      if not sliding:
        assertStats(kSource, StatsPath, tag & ".k_proj_output",
          kElementwise, msg = tag & " k projection source rows")
        attns[li].vNorm.forward(kSource)
      else:
        let v = attns[li].v_proj.forward(hNorm).reshape(
          [1, seqLen, kvHeads, headDim])
        attns[li].vNorm.forward(v)
    let (qRot, kRot) = attns[li].rotary.applyRope(qNormed, kNormed,
      ctx.cos, ctx.sin)
    let qRotBf = qRot.to(F.kBFloat16)
    let kRotBf = kRot.to(F.kBFloat16)
    assertStats(qRotBf, StatsPath, tag & ".q_rot", kReduction,
      msg = tag & " rotated query rows")
    assertStats(kRotBf, StatsPath, tag & ".k_rot", kReduction,
      msg = tag & " rotated key rows")
    assertStats(vNormed, StatsPath, tag & ".v", kReduction,
      msg = tag & " value norm output")

    # Cache boundary:
    # the reference cache stores the attention rows verbatim on the first step.
    #
    # The recorded cache_k/cache_v equal the rotated key rows and the value rows.
    if not sliding:
      assertStats(kRotBf, StatsPath, tag & ".cache_k", kReduction,
        msg = tag & " cache key rows")
      assertStats(vNormed, StatsPath, tag & ".cache_v", kReduction,
        msg = tag & " cache value rows")

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
    let layerOut = h3 * layerScalars[li]
    assertStats(layerOut, StatsPath, tag & ".layer.layer_output",
      kReduction, msg = tag & " scaled layer output")

    # The sandwich block op surface over the paged cache asserts through
    # the same recorded rows the manual replay used, the post-attention
    # residual sum verified against the attention half, the scaled
    # layer output against the FFN half through the block return pair.
    ctx.kv_position = 0
    let sLayer = SandwichDecoderLayer.init(inputLNs[li], attns[li],
      postAttnLNs[li], preFfnLNs[li], mlps[li], postFfnLNs[li])
    let (postMod, h1Mod) = sLayer.forward(ctx, x, none(Tensor))
    assertStats(h1Mod, StatsPath,
      tag & ".layer.post_attention_layernorm_output", kReduction,
      msg = tag & " op-surface post-attention residual sum")
    assertStats((h1Mod + postMod) * layerScalars[li], StatsPath,
      tag & ".layer.layer_output", kReduction,
      msg = tag & " op-surface scaled layer output")
  result = true

when isMainModule:
  runCppTest("gemma-4-12B layer 0/5 internals", main)
