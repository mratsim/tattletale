# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-0 unit replay of the Mistral-7B-v0.1 checkpoint, the all-sliding
## decoder block on the single fixture mixture.
##
## - the sliding mixture, attention op surface plus sandwich chain
## - the NEOX rotation over the single 1e4 theta table
##
## Seq 6 stays inside the 4096 window, the window behavior is recorded at tier 04.
##
## Replay runs on testDevice(), Metal here, the fixture recording torch-side mps.
##
## Requires the local model at tests/hf_models/Mistral-7B-v0.1 (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_mistral_01_layer_internals.nim

import
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
  workspace/transformers/tests/layer_utils

{.experimental: "callOperator".}

privateAccess(RopeGQAttention[void])

const
  ModelDir =
    currentSourcePath().parentDir() / ".." / "hf_models" / "Mistral-7B-v0.1"
  FixturePath =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" /
    "Mistral-7B-v0.1-layer-0" / "layer0-Mistral-7B-v0.1-00.safetensor"
  StatsPath = FixturePath & ".stats.json.zst"

proc main(): bool =
  let dev = testDevice()
  echo "    devices: ", deviceName(dev)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let numHeads = cfgJson{"num_attention_heads"}.getInt()
  let kvHeads = cfgJson{"num_key_value_heads"}.getInt()
  # The checkpoint config spells no head_dim key, the port derives it
  # hidden_size div num_attention_heads.
  let headDim = cfgJson{"hidden_size"}.getInt() div numHeads
  let window = cfgJson{"sliding_window"}.getInt()
  let actDtype = getDeployDtype(cfgJson)
  let view = SafetensorsCollection.open(ModelDir)

  # The mixer loads exactly as the Mistral model file wires it.
  # The void variant carries no qk norms, every layer windows.
  let rotary = RotaryPositionEmbedding.new(headDim, window,
    cfgJson{"rope_theta"}.getFloat(), actDtype, dev)
  let attn = RopeGQAttention[void].load(view, cfgJson,
    "model.layers.0.self_attn", 0, numHeads, kvHeads, headDim, rotary, dev,
    window = window)
  let inputLN = RmsNorm.load(view, cfgJson, "model.layers.0.input_layernorm", dev)
  let postLN = RmsNorm.load(view, cfgJson,
    "model.layers.0.post_attention_layernorm", dev)
  let ffn = GatedDenseFFN.load(view, cfgJson, "model.layers.0.mlp", dev)

  var st = Safetensor.open(FixturePath)
  # The replay seq length reads off the recorded input tensor shape.
  let x = st.getTensorOwned("sliding.input", dev)   # (1, 6, 4096) bf16
  let seqLen = x.size(1)
  var ctx = InferenceContext.init(
    num_layers = 1, batch_size = 1, kv_heads = kvHeads,
    max_seq = seqLen, head_dim = headDim)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, dev)).unsqueeze(0)
  ctx.setRopeForPositions(rotary)

  # The decoder-layer chain over the mixture input, input layernorm
  # through the SwiGLU block and both residuals.
  let hNorm = inputLN.forward(x)
  assertStats(hNorm, StatsPath, "sliding.layer.input_layernorm_output",
    kElementwise, msg = "layer-0 input layernorm output")
  assertStats(ctx.cos, StatsPath, "sliding.cos", kElementwise,
    msg = "prefill rope cos rows")
  assertStats(ctx.sin, StatsPath, "sliding.sin", kElementwise,
    msg = "prefill rope sin rows")

  # The attention op surface, the kv head repeat materialized through
  # repeat_interleave over the head axis, the multiset the recorded
  # repeat_kv produced, sdpa running is_causal.
  let q = attn.q_proj.forward(hNorm).reshape(
    [1, seqLen, numHeads, headDim])
  let k = attn.k_proj.forward(hNorm).reshape(
    [1, seqLen, kvHeads, headDim])
  let v = attn.v_proj.forward(hNorm).reshape(
    [1, seqLen, kvHeads, headDim])
  let (qRot, kRot) = attn.rotary.applyRope(q, k, ctx.cos, ctx.sin)
  assertStats(qRot, StatsPath, "sliding.q_rot", kReduction,
    msg = "rotated query rows")
  assertStats(kRot, StatsPath, "sliding.k_rot", kReduction,
    msg = "rotated key rows")
  assertStats(v, StatsPath, "sliding.v", kReduction,
    msg = "value projection output")
  let kExpanded = kRot.repeat_interleave(numHeads div kvHeads, 2).contiguous()
  let vExpanded = v.repeat_interleave(numHeads div kvHeads, 2).contiguous()
  assertStats(kExpanded, StatsPath, "sliding.k_expanded", kReduction,
    msg = "expanded key rows")
  assertStats(vExpanded, StatsPath, "sliding.v_expanded", kReduction,
    msg = "expanded value rows")
  let sdpaOut = attn.gqa_attn.forward(qRot, kExpanded, vExpanded,
    is_causal = true, enable_gqa = false)
  assertStats(sdpaOut, StatsPath, "sliding.sdpa_output", kReduction,
    depth = 2, msg = "sdpa output")
  let attnOut = attn.o_proj.forward(sdpaOut)
  assertStats(attnOut, StatsPath, "sliding.attn_output", kReduction,
    depth = 2, msg = "o_proj output")

  let h1 = x + attnOut
  let h2 = postLN.forward(h1)
  assertStats(h2, StatsPath, "sliding.layer.post_attention_layernorm_output",
    kReduction, msg = "layer-0 post-attention layernorm output")
  let mlpOut = ffn.forward(h2)
  assertStats(mlpOut, StatsPath, "sliding.layer.mlp_output", kReduction,
    msg = "layer-0 dense block output")
  let layerOut = h1 + mlpOut
  assertStats(layerOut, StatsPath, "sliding.layer.layer_output", kReduction,
    msg = "layer-0 post-residual output")
  result = true

when isMainModule:
  runCppTest("mistral layer-0 internals", main)
