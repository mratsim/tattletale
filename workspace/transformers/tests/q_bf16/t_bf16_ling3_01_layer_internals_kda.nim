# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-1 Kimi Delta Attention composition of the Ling-3.0-tiny checkpoint,
## replayed on cpu against the recorded comp and onorm fixture payloads.
## Requires the local model at tests/hf_models/Ling-3.0-tiny (gitignored).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_ling3_01_layer_internals_kda.nim

import
  std/os,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/quantizations/datatypes,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

{.experimental: "callOperator".}

privateAccess(GatedDeltaNet)
privateAccess(FusedRmsNormGatedSigmoid)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Ling-3.0-tiny-layer-1"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Ling-3.0-tiny"
  LayerPrefix = "model.layers.1.attention"

proc main() =
  # The mixer loads exactly as the Ling model file wires it:
  # per-channel decay, the full-rank lower-bound-sigmoid gate, the fused
  # output norm at the checkpoint eps.
  let cfgJson = (ModelDir / "config.json").parseFile()
  let view = SafetensorsCollection.open(ModelDir)
  let numHeads = cfgJson{"num_attention_heads"}.getInt()
  let headDim = cfgJson{"head_dim"}.getInt()
  let vHeadDim = cfgJson{"v_head_dim"}.getInt()
  let convKernel = cfgJson{"short_conv_kernel_size"}.getInt()
  let lowerBound = cfgJson{"kda_lower_bound"}.getFloat(-5.0)
  let mixer = GatedDeltaNet[perChannel, FullRankGateIn, lowerBoundSigmoid].load(
    view, cfgJson, LayerPrefix, 1,
    numHeads, numHeads, headDim, headDim, convKernel, F.kCPU,
    kdaLowerBound = lowerBound)

  # Fused output-norm one-shot payloads over the checkpoint weight.
  for caseNum in 0 ..< 3:
    let payload = FixtureDir /
      ("onorm-Ling-3.0-tiny-0" & $caseNum & ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (onorm case ", caseNum, ")"
    assertStats(mixer.o_norm.forward(st.getTensorOwned("o"),
      st.getTensorOwned("g")), statsPath, "output", kElementwise,
      msg = "fused o_norm replay, case " & $caseNum)
    assertStats(mixer.o_norm.weight.to(F.kCPU), statsPath, "weight",
      kElementwise, msg = "fused o_norm weight, case " & $caseNum)

  # comp-00, the 70-token prefill recorded on the chunked kernel, the ragged
  # second chunk 70 mod 64 = 6. The chunked kernel has no Nim implementation,
  # the output row compares the composed mixer forward.
  block:
    var st = Safetensor.open(FixtureDir / "comp-Ling-3.0-tiny-00.safetensor")
    let statsPath = FixtureDir /
      "comp-Ling-3.0-tiny-00.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill 70)"
    let x = st.getTensorOwned("x")
    let seqLen = 70
    assertStats(mixer.q_proj.forward(x), statsPath, "q_proj_out", kReduction,
      msg = "comp-00 q projection")
    assertStats(mixer.k_proj.forward(x), statsPath, "k_proj_out", kReduction,
      msg = "comp-00 k projection")
    assertStats(mixer.v_proj.forward(x), statsPath, "v_proj_out", kReduction,
      msg = "comp-00 v projection")

    let qFlat = mixer.q_proj.forward(x).transpose(1, 2)
    let kFlat = mixer.k_proj.forward(x).transpose(1, 2)
    let vFlat = mixer.v_proj.forward(x).transpose(1, 2)
    let (outQ, _) = shortConvSequence(qFlat, mixer.conv_q, nil)
    let (outK, _) = shortConvSequence(kFlat, mixer.conv_k, nil)
    let (outV, _) = shortConvSequence(vFlat, mixer.conv_v, nil)
    assertStats(outQ.transpose(1, 2).reshape(
      [1, seqLen, numHeads, headDim]), statsPath, "conv_q", kElementwise,
      msg = "comp-00 conv q branch")
    assertStats(outK.transpose(1, 2).reshape(
      [1, seqLen, numHeads, headDim]), statsPath, "conv_k", kElementwise,
      msg = "comp-00 conv k branch")
    assertStats(outV.transpose(1, 2).reshape(
      [1, seqLen, numHeads, vHeadDim]), statsPath, "conv_v", kElementwise,
      msg = "comp-00 conv v branch")

    assertStats(computeG(perChannel, lowerBoundSigmoid, mixer.decay_gate,
      mixer.a_log, mixer.dt_bias, x, lowerBound), statsPath, "g_safe",
      kElementwise, msg = "comp-00 safe-decay derivation")
    assertStats(F.sigmoid(mixer.in_proj_b.forward(x).to(F.kFloat32)),
      statsPath, "beta", kElementwise, msg = "comp-00 beta derivation")
    assertStats(mixer.norm_gate.gateInput(x).reshape(
      [1, seqLen, numHeads, vHeadDim]), statsPath, "g_out", kElementwise,
      msg = "comp-00 norm input row")

    assertStats(mixer.o_norm.forward(st.getTensorOwned("o"),
      st.getTensorOwned("g_out")), statsPath, "o_normed", kElementwise,
      msg = "comp-00 fused o_norm row")
    var ctx = InferenceContext.init(num_layers = 1, batch_size = 1,
      kv_heads = 1, max_seq = 128, head_dim = headDim)
    assertStats(mixer(ctx, x), statsPath, "output", kReduction,
      msg = "comp-00 mixer output")

  # comp-01, the 8-token prefill on the recurrent kernel, the dispatch
  # boundary q_len <= 64. The kernel row re-runs the recurrence over the recorded slabs.
  block:
    var st = Safetensor.open(FixtureDir / "comp-Ling-3.0-tiny-01.safetensor")
    let statsPath = FixtureDir /
      "comp-Ling-3.0-tiny-01.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill 8)"
    let x = st.getTensorOwned("x")
    let seqLen = 8
    assertStats(mixer.q_proj.forward(x), statsPath, "q_proj_out", kReduction,
      msg = "comp-01 q projection")
    assertStats(mixer.k_proj.forward(x), statsPath, "k_proj_out", kReduction,
      msg = "comp-01 k projection")
    assertStats(mixer.v_proj.forward(x), statsPath, "v_proj_out", kReduction,
      msg = "comp-01 v projection")

    let qFlat = mixer.q_proj.forward(x).transpose(1, 2)
    let kFlat = mixer.k_proj.forward(x).transpose(1, 2)
    let vFlat = mixer.v_proj.forward(x).transpose(1, 2)
    let (outQ, _) = shortConvSequence(qFlat, mixer.conv_q, nil)
    let (outK, _) = shortConvSequence(kFlat, mixer.conv_k, nil)
    let (outV, _) = shortConvSequence(vFlat, mixer.conv_v, nil)
    assertStats(outQ.transpose(1, 2).reshape(
      [1, seqLen, numHeads, headDim]), statsPath, "conv_q", kElementwise,
      msg = "comp-01 conv q branch")
    assertStats(outK.transpose(1, 2).reshape(
      [1, seqLen, numHeads, headDim]), statsPath, "conv_k", kElementwise,
      msg = "comp-01 conv k branch")
    assertStats(outV.transpose(1, 2).reshape(
      [1, seqLen, numHeads, vHeadDim]), statsPath, "conv_v", kElementwise,
      msg = "comp-01 conv v branch")

    assertStats(computeG(perChannel, lowerBoundSigmoid, mixer.decay_gate,
      mixer.a_log, mixer.dt_bias, x, lowerBound), statsPath, "g_safe",
      kElementwise, msg = "comp-01 safe-decay derivation")
    assertStats(F.sigmoid(mixer.in_proj_b.forward(x).to(F.kFloat32)),
      statsPath, "beta", kElementwise, msg = "comp-01 beta derivation")
    assertStats(mixer.norm_gate.gateInput(x).reshape(
      [1, seqLen, numHeads, vHeadDim]), statsPath, "g_out", kElementwise,
      msg = "comp-01 norm input row")

    assertStats(mixer.o_norm.forward(st.getTensorOwned("o"),
      st.getTensorOwned("g_out")), statsPath, "o_normed", kElementwise,
      msg = "comp-01 fused o_norm row")
    let (outKrn, _) = gatedDeltaRuleRecurrence(perChannel,
      st.getTensorOwned("conv_q"), st.getTensorOwned("conv_k"),
      st.getTensorOwned("conv_v"), st.getTensorOwned("g_safe"),
      st.getTensorOwned("beta"), nil)
    assertStats(outKrn, statsPath, "o", kReduction,
      msg = "comp-01 recurrent kernel output")
    var ctx = InferenceContext.init(num_layers = 1, batch_size = 1,
      kv_heads = 1, max_seq = 128, head_dim = headDim)
    assertStats(mixer(ctx, x), statsPath, "output", kReduction,
      msg = "comp-01 mixer output")

  # comp-02, the real mixer through InferenceContext. Prefill 8 runs
  # end to end, then a 3-step decode sequence at positions 8, 9, 10,
  # the stored conv and SSM state feed the per-step replays.
  block:
    var st = Safetensor.open(FixtureDir / "comp-Ling-3.0-tiny-02.safetensor")
    let statsPath = FixtureDir /
      "comp-Ling-3.0-tiny-02.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill 8 + 3 decode steps)"
    var ctx = InferenceContext.init(num_layers = 1, batch_size = 1,
      kv_heads = 1, max_seq = 128, head_dim = headDim)

    let outPre = mixer(ctx, st.getTensorOwned("x"))
    assertStats(outPre, statsPath, "output", kReduction,
      msg = "comp-02 prefill mixer output")

    assertStats(ctx.kdaConvState[1][0], statsPath, "conv_state_q",
      kElementwise, msg = "comp-02 conv state q after prefill")
    assertStats(ctx.kdaConvState[1][1], statsPath, "conv_state_k",
      kElementwise, msg = "comp-02 conv state k after prefill")
    assertStats(ctx.kdaConvState[1][2], statsPath, "conv_state_v",
      kElementwise, msg = "comp-02 conv state v after prefill")
    assertStats(ctx.kdaSsmState[1], statsPath, "rec_state", kReduction,
      msg = "comp-02 recurrent state after prefill")

    for i in 0 ..< 3:
      let msg = "comp-02 decode step " & $i
      let xStep = st.getTensorOwned("x_step" & $i)
      let stateBefore = ctx.kdaConvState[1]
      let ssmBefore = ctx.kdaSsmState[1]

      let outStep = mixer(ctx, xStep)
      assertStats(outStep, statsPath, "output_step" & $i, kReduction,
        msg = msg & " mixer output")

      # Conv branch replay from the stored history, the q branch meets
      # the recorded step row, the k and v branches feed the kernel replay.
      let qFlat = mixer.q_proj.forward(xStep).transpose(1, 2)
      let kFlat = mixer.k_proj.forward(xStep).transpose(1, 2)
      let vFlat = mixer.v_proj.forward(xStep).transpose(1, 2)
      let (convQ, _) = shortConvStep(qFlat, mixer.conv_q, stateBefore[0])
      let (convK, _) = shortConvStep(kFlat, mixer.conv_k, stateBefore[1])
      let (convV, _) = shortConvStep(vFlat, mixer.conv_v, stateBefore[2])
      assertStats(convQ.transpose(1, 2).reshape(
        [1, 1, numHeads, headDim]), statsPath, "conv_q_step" & $i,
        kElementwise, msg = msg & " conv q branch")

      let (outKrn, _) = gatedDeltaRuleRecurrence(perChannel,
        convQ.transpose(1, 2).reshape([1, 1, numHeads, headDim]),
        convK.transpose(1, 2).reshape([1, 1, numHeads, headDim]),
        convV.transpose(1, 2).reshape([1, 1, numHeads, vHeadDim]),
        st.getTensorOwned("g_safe_step" & $i),
        st.getTensorOwned("beta_step" & $i), ssmBefore.unsqueeze(0))
      assertStats(outKrn, statsPath, "o_step" & $i, kReduction,
        msg = msg & " recurrent kernel output")

      assertStats(mixer.o_norm.forward(outKrn,
        mixer.norm_gate.gateInput(xStep).reshape(
          [1, 1, numHeads, vHeadDim])), statsPath, "o_normed_step" & $i,
        kElementwise, msg = msg & " fused o_norm row")

    assertStats(ctx.kdaSsmState[1], statsPath, "rec_state_final", kReduction,
      msg = "comp-02 recurrent state after decode sequence")

  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
