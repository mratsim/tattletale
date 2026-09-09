# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## EXL3 layer-02 stage trace for Qwen3-0.6B-EXL3-5bpw: the per-stage fixtures
## of the exl3-01-block-02-trace family, the first-divergence bisector of the
## decoder layer. Chained stages accumulate drift linearly, so the per-stage
## bound grows with stage index; a Metal EXL3 reimplementation later adds a
## measured Metal calibration row without touching fixtures or suites.
## The norm stages measured bit-exact across devices act as drift anchors.
##
## Run:
##   TTT_TEST_ON=cpu nim test_tf_exl3_qwen3_01_block_02_trace

import
  std/memfiles, std/strformat, std/tables, std/os, std/options, std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])
privateAccess(GatedDenseFFN)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-01-block-02-trace"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  BaseUlps = 4
    ## Base per-stage bound in fp16 ulps (the bf16 per-op row, the ulp unit
    ## taken in fp16 because EXL3 dequantizes to fp16), the accumulated
    ## bound grows one base step per chained stage since the last anchor.

proc main() =
  let dev = testDevice()
  echo compareLine(FixtureDir, dev)
  let model = loadQwen3ModelRaw(ModelPath, dev)
  let layer = model.layers[2]

  # Load fixture tensors
  var st = Safetensor.open(FixtureDir / "layer02_trace.safetensor")

  template load(name: string): Tensor =
    st.getTensorOwned(name, kCPU).to(dev)

  let x = load("input_hidden_states")
  let e_ln = load("after_input_layernorm")
  let e_q = load("q_proj_out")
  let e_k = load("k_proj_out")
  let e_v = load("v_proj_out")
  let e_qn = load("after_q_norm")
  let e_kn = load("after_k_norm")
  let e_rq = load("after_rope_q")
  let e_rk = load("after_rope_k")
  let e_attn = load("attn_output")
  let e_o = load("after_o_proj")
  let e_res = load("after_residual")
  let e_pln = load("after_post_layernorm")
  let e_gate = load("mlp_gate_out")
  let e_up = load("mlp_up_out")
  let e_act = load("mlp_activation")
  let e_down = load("mlp_down_out")
  let e_out = load("output")

  let batch = x.size(0)
  let S = x.size(1)
  let hd = 128

  let statsFile = loadFingerprintStats(
    FixtureDir / "layer02_trace.safetensor.stats")

  var chained = 0
    ## Stages chained since the last bit-exact anchor; the accumulated bound
    ## of the next stage is (chained + 1) base steps, the linear-in-stages
    ## law of the evaluation-order bound.

  var ctx = InferenceContext.init(
      num_layers = 1, batch_size = 1, kv_heads = 8,
      max_seq = 4096, head_dim = hd)
  ctx.clearState()
  ctx.position_ids = arange(S).unsqueeze(0).to(kInt64).to(dev)
  ctx.setRopeForPositions(layer.sequence_mixer.rotary)

  # Stage 1: input_layernorm (bit-exact anchor, the fp32 elementwise norm)
  let h = layer.input_layernorm(x)
  assertAllClose(h, e_ln, rtol = 0.0, abstol = 0.0, msg = "input_layernorm")
  assertStatsChainBand(h, statsFile.statsTensor("after_input_layernorm"), chained,
    rtol = ChainCheckpointRtolF16, msg = "input_layernorm stats")
  chained = 0

  # Stage 2: QKV projections
  let q = layer.sequence_mixer.q_proj(h)
  let k = layer.sequence_mixer.k_proj(h)
  let v = layer.sequence_mixer.v_proj(h)
  chained = 1
  discard assertChainCheckpoint(q, e_q, chained, reductionLen = 1024,
    msg = "q_proj", rtol = ChainCheckpointRtolF16)
  discard assertChainCheckpoint(k, e_k, chained, reductionLen = 1024,
    msg = "k_proj", rtol = ChainCheckpointRtolF16)
  discard assertChainCheckpoint(v, e_v, chained, reductionLen = 1024,
    msg = "v_proj", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(q, statsFile.statsTensor("q_proj_out"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "q_proj stats")
  assertStatsChainBand(k, statsFile.statsTensor("k_proj_out"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "k_proj stats")
  assertStatsChainBand(v, statsFile.statsTensor("v_proj_out"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "v_proj stats")

  # Stage 3: QK norms (per-head). The norms are pure fp32 elementwise math,
  # but their inputs carry the projection drift, so the stage chains on: the
  # bound grows, no anchor reset.
  let q_mh = q.reshape(batch, S, 16, hd)
  let k_mh = k.reshape(batch, S, 8, hd)
  let v_mh = v.reshape(batch, S, 8, hd)
  let q_normed = layer.sequence_mixer.q_norm.forward(q_mh).reshape(q.shape)
  let k_normed = layer.sequence_mixer.k_norm.forward(k_mh).reshape(k.shape)
  chained = 2
  discard assertChainCheckpoint(q_normed, e_qn, chained, reductionLen = 128,
    msg = "q_norm", rtol = ChainCheckpointRtolF16)
  discard assertChainCheckpoint(k_normed, e_kn, chained, reductionLen = 128,
    msg = "k_norm", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(q_normed, statsFile.statsTensor("after_q_norm"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "q_norm stats")
  assertStatsChainBand(k_normed, statsFile.statsTensor("after_k_norm"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "k_norm stats")

  # Stage 4: RoPE
  let qn_mh = q_normed.reshape(batch, S, 16, hd)
  let kn_mh = k_normed.reshape(batch, S, 8, hd)
  let (q_rot, k_rot) = layer.sequence_mixer.rotary.applyRope(qn_mh, kn_mh, ctx.cos, ctx.sin)
  chained = 1
  discard assertChainCheckpoint(q_rot.reshape(batch, S, -1), e_rq, chained,
    reductionLen = 1024, msg = "rope_q", rtol = ChainCheckpointRtolF16)
  discard assertChainCheckpoint(k_rot.reshape(batch, S, -1), e_rk, chained,
    reductionLen = 1024, msg = "rope_k", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(q_rot.reshape(batch, S, -1),
    statsFile.statsTensor("after_rope_q"), chained, msg = "rope_q stats")
  assertStatsChainBand(k_rot.reshape(batch, S, -1),
    statsFile.statsTensor("after_rope_k"), chained, msg = "rope_k stats")

  # Stage 5: SDPA
  let qs = q_rot.transpose(1, 2)
  let ks = k_rot.transpose(1, 2)
  let vs = v_mh.transpose(1, 2)
  let attn = F.scaled_dot_product_attention(
    qs, ks, vs, attn_mask = none(Tensor), dropout_p = 0.0'f64,
    is_causal = true, scale = some(0.088388347648'f64), enable_gqa = true)
  let attn_out = attn.transpose(1, 2).reshape(batch, S, -1)
  chained = 2
  discard assertChainCheckpoint(attn_out, e_attn, chained, reductionLen = 1024,
    msg = "attn (sdpa)", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(attn_out, statsFile.statsTensor("attn_output"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "attn stats")

  # Stage 6: O projection
  let o_out = layer.sequence_mixer.o_proj(attn_out)
  chained = 3
  discard assertChainCheckpoint(o_out, e_o, chained, reductionLen = 2048,
    msg = "o_proj", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(o_out, statsFile.statsTensor("after_o_proj"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "o_proj stats")

  # Stage 7: Residual
  let res = x + o_out
  chained = 4
  discard assertChainCheckpoint(res, e_res, chained, reductionLen = 1024,
    msg = "residual", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(res, statsFile.statsTensor("after_residual"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "residual stats")

  # Stage 8: Post-attention RMSNorm. Chained on: the norm input carries the
  # attention-side drift, so no anchor reset.
  let h2 = layer.post_attention_layernorm(res)
  chained = 7
  discard assertChainCheckpoint(h2, e_pln, chained, reductionLen = 1024,
    msg = "post_layernorm", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(h2, statsFile.statsTensor("after_post_layernorm"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "post_layernorm stats")

  # Stage 9: MLP
  let gate = layer.hidden_mixer.gate_proj(h2)
  let up = layer.hidden_mixer.up_proj(h2)
  let act = F.silu(gate) * up
  let down = layer.hidden_mixer.down_proj(act)
  chained = 8
  discard assertChainCheckpoint(gate, e_gate, chained, reductionLen = 1024,
    msg = "mlp_gate", rtol = ChainCheckpointRtolF16)
  discard assertChainCheckpoint(up, e_up, chained, reductionLen = 1024,
    msg = "mlp_up", rtol = ChainCheckpointRtolF16)
  discard assertChainCheckpoint(act, e_act, chained, reductionLen = 1024,
    msg = "mlp_activation", rtol = ChainCheckpointRtolF16)
  discard assertChainCheckpoint(down, e_down, chained, reductionLen = 3072,
    msg = "mlp_down", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(gate, statsFile.statsTensor("mlp_gate_out"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "mlp_gate stats")
  assertStatsChainBand(up, statsFile.statsTensor("mlp_up_out"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "mlp_up stats")
  assertStatsChainBand(act, statsFile.statsTensor("mlp_activation"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "mlp_activation stats")
  assertStatsChainBand(down, statsFile.statsTensor("mlp_down_out"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "mlp_down stats")

  # Stage 10: Output
  chained = 9
  discard assertChainCheckpoint(down, e_out, chained, reductionLen = 3072,
    msg = "output", rtol = ChainCheckpointRtolF16)
  assertStatsChainBand(down, statsFile.statsTensor("output"), chained,
    rtol = ChainCheckpointRtolF16,
    msg = "output stats")

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Layer-02 stage trace passed under the accumulated bounds"

when isMainModule:
  main()
