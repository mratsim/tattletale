# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-4/5 unit replay of the gemma-3-270m-it checkpoint, one assertion
## block per fixture mixture.
##
## - the sliding (layer 4) and full (layer 5) mixtures, op surface plus chain
## - the boundary mixture, the layer-4 then layer-5 pair on one seeded input
## - dual-theta rope, the local 1e4 base on sliding, the global 1e6 on full
##
## Replay runs on testDevice(), Metal here, the fixture recording torch-side mps.
##
## Requires the local model at tests/hf_models/gemma-3-270m-it (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_gemma3270m_01_layer_internals.nim

import
  std/math,
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(RopeGQAttention[RmsNormOne])

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "gemma-3-270m-it"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "gemma-3-270m-it-layer-4-5" / "layer4-5-gemma-3-270m-it-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"

proc main(): bool =
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let numHeads = cfgJson{"num_attention_heads"}.getInt()
  let kvHeads = cfgJson{"num_key_value_heads"}.getInt()
  let headDim = cfgJson{"head_dim"}.getInt()
  let window = cfgJson{"sliding_window"}.getInt()
  let softmaxScale = pow(cfgJson{"query_pre_attn_scalar"}.getInt().float64, -0.5)
  let actDtype = getDeployDtype(cfgJson)
  let view = SafetensorsCollection.open(ModelDir)

  # The fixture replays the last sliding layer before the first full
  # layer of the checkpoint's 5:1 pattern, the kinds read off the config.
  var fullIdx = 0
  if cfgJson.hasKey("layer_types"):
    for tj in items(cfgJson{"layer_types"}):
      if tj.getStr() == "full_attention":
        break
      inc fullIdx
  else:
    fullIdx = cfgJson{"sliding_window_pattern"}.getInt() - 1
  let slidingIdx = fullIdx - 1

  # Dual-theta rope tables, one per layer kind, sized to the replay rows.
  let rotarySliding = RotaryPositionEmbedding.new(headDim, window,
    cfgJson{"rope_local_base_freq"}.getFloat(), actDtype, dev)
  let rotaryFull = RotaryPositionEmbedding.new(headDim, window,
    cfgJson{"rope_theta"}.getFloat(), actDtype, dev)

  let (attnS, inputLNS, postLNS, preFFS, ffnS, postFFS) = setupGemma3LayerFixture(
    view, cfgJson, slidingIdx, rotarySliding, window, softmaxScale, dev)
  let (attnF, inputLNF, postLNF, preFFF, ffnF, postFFF) = setupGemma3LayerFixture(
    view, cfgJson, fullIdx, rotaryFull, window, softmaxScale, dev)

  var st = Safetensor.open(FixturePath)
  # The replay seq length reads off the recorded input tensor shape.
  let xSliding = st.getTensorOwned("sliding.input", dev)
  let seqLen = xSliding.size(1)
  let posIds = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, dev)).unsqueeze(0)

  type
    ChainStep = tuple
      rotary: RotaryPositionEmbedding
      attn: RopeGQAttention[RmsNormOne]
      inputLN, postLN, preFF, postFF: RmsNormOne
      ffn: GatedDenseFFN
      inputName: string
      chainPrefix: string
      attnPrefix: string
      finalRow: string

  # The four replay chains below, the sliding and full mixtures first,
  # then the boundary pair over one seeded input, the layer-5 chain
  # consuming the layer-4 output, each kind roping with its own theta.
  let mixtures: array[4, ChainStep] = [
    (
      rotary: rotarySliding, attn: attnS, inputLN: inputLNS, postLN: postLNS,
      preFF: preFFS, postFF: postFFS, ffn: ffnS, inputName: "sliding.input",
      chainPrefix: "sliding.layer.", attnPrefix: "sliding.", finalRow: ""),
    (
      rotary: rotaryFull, attn: attnF, inputLN: inputLNF, postLN: postLNF,
      preFF: preFFF, postFF: postFFF, ffn: ffnF, inputName: "full.input",
      chainPrefix: "full.layer.", attnPrefix: "full.", finalRow: ""),
    (
      rotary: rotarySliding, attn: attnS, inputLN: inputLNS, postLN: postLNS,
      preFF: preFFS, postFF: postFFS, ffn: ffnS, inputName: "boundary.input",
      chainPrefix: "", attnPrefix: "", finalRow: "boundary.layer4_output"),
    (
      rotary: rotaryFull, attn: attnF, inputLN: inputLNF, postLN: postLNF,
      preFF: preFFF, postFF: postFFF, ffn: ffnF, inputName: "",
      chainPrefix: "boundary.layer5_", attnPrefix: "",
      finalRow: "boundary.layer5_output"),
  ]

  var ctx = InferenceContext.init(
    num_layers = 1, batch_size = 1, kv_heads = kvHeads,
    max_seq = window, head_dim = headDim)
  ctx.position_ids = posIds
  var prevOut: F.Tensor
  for step in mixtures:
    ctx.setRopeForPositions(step.rotary)
    let x = if step.inputName.len > 0:
      st.getTensorOwned(step.inputName, dev)
    else:
      prevOut
    let hNorm = step.inputLN.forward(x)
    if step.chainPrefix.len > 0:
      assertStats(hNorm, StatsPath,
        step.chainPrefix & "input_layernorm_output", kElementwise,
        msg = step.chainPrefix & "input layernorm output")
    # The attention op surface over the post-layernorm input, the kv head
    # repeat materialized through repeat_interleave over the head axis,
    # the multiset the recorded repeat_kv produced, sdpa running is_causal.
    let q = step.attn.q_proj.forward(hNorm).reshape(
      [1, seqLen, numHeads, headDim])
    let k = step.attn.k_proj.forward(hNorm).reshape(
      [1, seqLen, kvHeads, headDim])
    let v = step.attn.v_proj.forward(hNorm).reshape(
      [1, seqLen, kvHeads, headDim])
    let qNormed = step.attn.q_norm.forward(q)
    let kNormed = step.attn.k_norm.forward(k)
    if step.attnPrefix.len > 0:
      assertStats(ctx.cos, StatsPath, step.attnPrefix & "cos", kElementwise,
        msg = step.attnPrefix & "rope cos rows")
      assertStats(ctx.sin, StatsPath, step.attnPrefix & "sin", kElementwise,
        msg = step.attnPrefix & "rope sin rows")
      assertStats(qNormed, StatsPath, step.attnPrefix & "q_normed", kReduction,
        msg = step.attnPrefix & "query norm output")
      assertStats(kNormed, StatsPath, step.attnPrefix & "k_normed", kReduction,
        msg = step.attnPrefix & "key norm output")
    let (qRot, kRot) = step.attn.rotary.applyRope(qNormed, kNormed,
      ctx.cos, ctx.sin)
    if step.attnPrefix.len > 0:
      assertStats(qRot, StatsPath, step.attnPrefix & "q_rot", kReduction,
        msg = step.attnPrefix & "rotated query rows")
      assertStats(kRot, StatsPath, step.attnPrefix & "k_rot", kReduction,
        msg = step.attnPrefix & "rotated key rows")
      assertStats(v, StatsPath, step.attnPrefix & "v", kReduction,
        msg = step.attnPrefix & "value projection output")
    let kExpanded = kRot.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    let vExpanded = v.repeat_interleave(numHeads div kvHeads, 2).contiguous()
    if step.attnPrefix.len > 0:
      assertStats(kExpanded, StatsPath, step.attnPrefix & "k_expanded",
        kReduction, msg = step.attnPrefix & "expanded key rows")
      assertStats(vExpanded, StatsPath, step.attnPrefix & "v_expanded",
        kReduction, msg = step.attnPrefix & "expanded value rows")
    let sdpaOut = step.attn.gqa_attn.forward(qRot, kExpanded, vExpanded,
      is_causal = true, enable_gqa = false)
    if step.attnPrefix.len > 0:
      assertStats(sdpaOut, StatsPath, step.attnPrefix & "sdpa_output",
        kReduction, depth = 2, msg = step.attnPrefix & "sdpa output")
    let attnOut = step.attn.o_proj.forward(sdpaOut)
    if step.attnPrefix.len > 0:
      assertStats(attnOut, StatsPath, step.attnPrefix & "attn_output",
        kReduction, depth = 2, msg = step.attnPrefix & "o_proj output")

    # The sandwich decoder-layer chain over the mixture input, the recorded
    # stage rows asserted per chainPrefix, the output row per finalRow.
    let postAttnNormed = step.postLN.forward(attnOut)
    if step.chainPrefix.len > 0:
      assertStats(postAttnNormed, StatsPath,
        step.chainPrefix & "post_attention_layernorm_output", kReduction,
        msg = step.chainPrefix & "post-attention layernorm output")
    let h1 = x + postAttnNormed
    let h2 = step.preFF.forward(h1)
    if step.chainPrefix.len > 0:
      assertStats(h2, StatsPath,
        step.chainPrefix & "pre_feedforward_layernorm_output", kReduction,
        msg = step.chainPrefix & "pre-feedforward layernorm output")
    let mlpOut = step.ffn.forward(h2)
    if step.chainPrefix.len > 0:
      assertStats(mlpOut, StatsPath, step.chainPrefix & "mlp_output",
        kReduction, msg = step.chainPrefix & "dense block output")
    let postFFNormed = step.postFF.forward(mlpOut)
    if step.chainPrefix.len > 0:
      assertStats(postFFNormed, StatsPath,
        step.chainPrefix & "post_feedforward_layernorm_output", kReduction,
        msg = step.chainPrefix & "post-feedforward layernorm output")
    let layerOut = h1 + postFFNormed
    if step.chainPrefix.len > 0 or step.finalRow.len > 0:
      var rowName = step.chainPrefix & "layer_output"
      if step.finalRow.len > 0:
        rowName = step.finalRow
      assertStats(layerOut, StatsPath, rowName, kReduction,
        msg = step.chainPrefix & "layer output")
    prevOut = layerOut
  result = true

when isMainModule:
  runCppTest("gemma3 270m layer-4/5 internals", main)
