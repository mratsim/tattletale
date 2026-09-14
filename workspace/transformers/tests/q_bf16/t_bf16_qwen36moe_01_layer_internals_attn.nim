# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Gated full-attention layer 3 of the Qwen3.6-35B-A3B checkpoint,
## replayed on cpu against the layer-3 fixture payloads.
## Requires the local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_01_layer_internals_attn.nim

import
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/layers/attn_ssm/grouped_query_attention {.all.},
  workspace/transformers/src/layers/attn_ssm/gated_attention {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(RopeElementWiseGatedAttention[RmsNormOne])
privateAccess(GroupedQueryAttention)

const
  FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-3"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  WeightsFile3 = ModelDir / "model-00003-of-00026.safetensors"

proc main() =
  # The real layer-3 self-attention weights, the attention geometry arriving
  # over the checkpoint text config. The `q_proj` packs `[q | sigmoid-gate
  # logits]` per head.
  let cfgJson = (ModelDir / "config.json").parseFile()
  let tc = cfgJson{"text_config"}
  let weights = SafetensorsCollection.open(WeightsFile3)
  let rotary = cfgJson.setup(RotaryPositionEmbedding,
      tc{"max_position_embeddings"}.getInt())
  let attn = cfgJson.setup(RopeElementWiseGatedAttention[RmsNormOne],
      weights, "model.language_model.layers.", 3, rotary)

  # Gated full attention prefill (seq 8) vs fixture
  block:
    echo "    devices: ", deviceName(F.kCPU)
    var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = 2, headDim = 256)
    var st = Safetensor.open(FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor")

    let x = st.getTensorOwned("hidden_states")       # (1, 8, 2048) bf16
    let hfPosIds = st.getTensorOwned("position_ids") # (1, 8) int64
    ctx.position_ids = hfPosIds[0]
    ctx.setRopeForPositions(attn.rotary)

    let seqLen = x.size(1)
    let output = attn(ctx, x)
    assertStats(output, FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor" & ".stats", "output", kReduction, msg = "gated attention prefill output")

    # Intermediates recomputed through the layer's own components, compared against the fixture's captured values.
    let gqa = attn.gqa_attn
    # q_proj packs [q | sigmoid-gate logits] per head, the head axis carries 2 * head_dim.
    let qg = attn.q_proj.forward(x)
    let qgR = qg.reshape([1, seqLen, gqa.num_qo_head, 2 * gqa.head_dim])
    let queryR = qgR.narrow(3, 0, gqa.head_dim)
    let gateR = qgR.narrow(3, gqa.head_dim, gqa.head_dim)
    let gate = gateR.reshape([1, seqLen, gqa.num_qo_head * gqa.head_dim])
    let qNormed = attn.q_norm.forward(queryR)
    let kReshaped = attn.k_proj.forward(x).reshape(
      [1, seqLen, gqa.num_kv_head, gqa.head_dim])
    let kNormed = attn.k_norm.forward(kReshaped)
    let (qRot, kRot) = attn.rotary.applyRope(qNormed, kNormed, ctx.cos, ctx.sin)

    # The fixture rope tensors sit in (batch, heads, seq, dim) storage order, so the transpose
    # aligns the strided-sample index order.
    assertStats(qNormed, FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor" & ".stats", "q_normed", kElementwise, msg = "q norm replay")
    assertStats(kNormed, FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor" & ".stats", "k_normed", kElementwise, msg = "k norm replay")
    assertStats(gate, FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor" & ".stats", "gate", kElementwise, msg = "sigmoid gate replay")
    assertStats(qRot.transpose(1, 2), FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor" & ".stats", "q_rot", kElementwise, msg = "q rope replay")
    assertStats(kRot.transpose(1, 2), FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor" & ".stats", "k_rot", kElementwise, msg = "k rope replay")

    # Pre-o_proj output, SDPA then the sigmoid gate, so the case covers the sigmoid-gate application itself.
    let vReshaped = attn.v_proj.forward(x).reshape(
      [1, seqLen, gqa.num_kv_head, gqa.head_dim])
    let kvGroups = gqa.num_qo_head div gqa.num_kv_head
    let kExp = kRot.repeat_interleave(kvGroups, 2)
    let vExp = vReshaped.repeat_interleave(kvGroups, 2)
    let attnOut = attn.gqa_attn.forward(qRot, kExp, vExp,
      is_causal = true, enable_gqa = false)
    let attnGated = attnOut * F.sigmoid(gate)
    assertStats(attnGated, FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor" & ".stats", "attn_output_gated", kElementwise, msg = "gated attention pre o_proj output")

  # Gated full attention decode (single token, position 5) vs fixture
  block:
    var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = 2, headDim = 256)
    var st = Safetensor.open(FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor")

    let x = st.getTensorOwned("hidden_states")       # (1, 1, 2048) bf16
    let hfPosIds = st.getTensorOwned("position_ids") # (1, 1) int64
    ctx.position_ids = hfPosIds[0]
    ctx.setRopeForPositions(attn.rotary)

    let output = attn(ctx, x)
    assertStats(output, FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor" & ".stats", "output", kReduction, msg = "gated attention decode output")

    let seqLen = x.size(1)
    let gqa = attn.gqa_attn
    # q_proj packs [q | sigmoid-gate logits] per head, the head axis carries 2 * head_dim.
    let qg = attn.q_proj.forward(x)
    let qgR = qg.reshape([1, seqLen, gqa.num_qo_head, 2 * gqa.head_dim])
    let queryR = qgR.narrow(3, 0, gqa.head_dim)
    let gateR = qgR.narrow(3, gqa.head_dim, gqa.head_dim)
    let gate = gateR.reshape([1, seqLen, gqa.num_qo_head * gqa.head_dim])
    let qNormed = attn.q_norm.forward(queryR)
    let kReshaped = attn.k_proj.forward(x).reshape(
      [1, seqLen, gqa.num_kv_head, gqa.head_dim])
    let kNormed = attn.k_norm.forward(kReshaped)
    let (qRot, kRot) = attn.rotary.applyRope(qNormed, kNormed, ctx.cos, ctx.sin)

    # Seq 1 keeps one index order across the (heads, seq) layouts.
    assertStats(qNormed, FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor" & ".stats", "q_normed", kElementwise, msg = "q norm replay")
    assertStats(kNormed, FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor" & ".stats", "k_normed", kElementwise, msg = "k norm replay")
    assertStats(gate, FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor" & ".stats", "gate", kElementwise, msg = "sigmoid gate replay")
    assertStats(qRot, FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor" & ".stats", "q_rot", kElementwise, msg = "q rope replay")
    assertStats(kRot, FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor" & ".stats", "k_rot", kElementwise, msg = "k rope replay")

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
