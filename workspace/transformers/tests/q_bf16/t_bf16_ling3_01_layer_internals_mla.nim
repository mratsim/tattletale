# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-3 gated MLA attention of the Ling-3.0-tiny checkpoint, replayed
## on cpu against the recorded layer-3 fixture payloads.
## Requires the local model at tests/hf_models/Ling-3.0-tiny (gitignored).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_ling3_01_layer_internals_mla.nim

import
  std/os,
  std/options,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/quantizations/datatypes,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/layers/attn_ssm/multi_head_latent_attention,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils,
  workspace/transformers/src/stateful/orchestrator

{.experimental: "callOperator".}

privateAccess(HeadwiseGatedMLAttention[RmsNorm, FullRoPe])
privateAccess(MLAttention[RmsNorm, FullRoPe])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Ling-3.0-tiny-layer-3"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Ling-3.0-tiny"
  LayerPrefix = "model.layers.3.attention"

proc main() =
  # The mixer loads exactly as the Ling model file wires it:
  # compressed-Q bottleneck, both latent norms at the bottleneck eps,
  # the head-wise sigmoid gate before the dense projection.
  let attn = setupMlaGated(ModelDir, LayerPrefix, 3, 64)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let kvLoraRank = cfgJson{"kv_lora_rank"}.getInt()
  let qkRopeHeadDim = cfgJson{"qk_rope_head_dim"}.getInt()
  let qkNopeHeadDim = cfgJson{"qk_nope_head_dim"}.getInt()
  let vHeadDim = cfgJson{"v_head_dim"}.getInt()
  let numHeads = cfgJson{"num_attention_heads"}.getInt()
  let scale = mlaSoftmaxScale(qkNopeHeadDim, qkRopeHeadDim)
  # Frequency table rows for positions 0..15, the plain-theta tables
  # against the recorded pre-cast rows.
  let rotary = MlaRotary.new(qkRopeHeadDim, 16,
    cfgJson{"rope_theta"}.getFloat(), F.kCPU,
    yarnFactor = 0.0)
  block:
    echo "    devices: ", deviceName(F.kCPU), " (tables)"
    let tablesStats = FixtureDir /
      "tables-Ling-3.0-tiny-00.safetensor.stats.json.zst"
    assertStats(rotary.cosCache, tablesStats, "cos_rows", kElementwise,
      depth = 1, msg = "plain-theta cos table rows")
    # TODO sin_rows (kElementwise, depth 1) drifts 3 ulps on Metal against the 2-ulp allowance, sin max 0.99997407 in binade [0.5, 1)
    assertStats(rotary.sinCache, tablesStats, "sin_rows", kElementwise,
      depth = 1, msg = "plain-theta sin table rows")

  # Rotation replay over the recorded planes. The recorded side spells
  # the rotation in bf16, the Nim rotation computes in f32 and casts,
  # the two forms differ by a grid step on the rotating channels.
  # The permutation block verified against the interleaved layout
  # relation on the recorded rows themselves, pure data movement.
  for caseNum in 0 ..< 3:
    let payload = FixtureDir /
      ("rope-Ling-3.0-tiny-0" & $caseNum & ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (rope case ", caseNum, ")"
    let cos = st.getTensorOwned("cos")
    let sin = st.getTensorOwned("sin")
    let qRot = rotateInterleaved(
      st.getTensorOwned("q_pe").transpose(1, 2), cos, sin)
    let kRot = rotateInterleaved(
      st.getTensorOwned("k_pe").transpose(1, 2), cos, sin)
    assertStats(qRot, statsPath, "q_rot_interleaved", kElementwise, depth = 1,
      msg = "q rotation, case " & $caseNum)
    assertStats(kRot, statsPath, "k_rot_interleaved", kElementwise, depth = 1,
      msg = "k rotation, case " & $caseNum)
    assertStats(mlaInterleaveLayout(st.getTensorOwned("q_rot")), statsPath,
      "q_rot_interleaved", kElementwise,
      msg = "recorded q rotation permutation")
    assertStats(mlaInterleaveLayout(st.getTensorOwned("k_rot")), statsPath,
      "k_rot_interleaved", kElementwise,
      msg = "recorded k rotation permutation")

  # Latent norm payloads over the recorded weight and inputs.
  for caseNum in 0 ..< 3:
    let payload = FixtureDir /
      ("norm-Ling-3.0-tiny-0" & $caseNum & ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (norm case ", caseNum, ")"
    assertStats(attn.attn.kv_a_layernorm.forward(st.getTensorOwned("input")),
      statsPath, "output", kElementwise,
      msg = "latent norm replay, case " & $caseNum)
    assertStats(attn.attn.kv_a_layernorm.weight.to(F.kCPU), statsPath,
      "weight", kElementwise, msg = "latent norm weight, case " & $caseNum)

  # Compressed-Q bottleneck chain replays the q_a GEMM, the bottleneck
  # norm and the q_b GEMM over the recorded qchain payloads.
  for caseNum in 0 ..< 3:
    let payload = FixtureDir /
      ("qchain-Ling-3.0-tiny-0" & $caseNum & ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (qchain case ", caseNum, ")"
    let qAOut = attn.attn.q_a_proj.forward(st.getTensorOwned("input"))
    assertStats(qAOut, statsPath, "q_a_out", kReduction,
      msg = "q_a projection, case " & $caseNum)
    let qNormed = attn.attn.q_a_norm.forward(qAOut)
    assertStats(qNormed, statsPath, "q_normed", kElementwise,
      msg = "q bottleneck norm, case " & $caseNum)
    assertStats(attn.attn.q_b_proj.forward(qNormed), statsPath, "q_out",
      kReduction, msg = "q_b projection, case " & $caseNum)

  # Prefill seq4 per-op replay plus the mixer output, every recomputed
  # surface runs through the layer's own components over the recorded inputs,
  # the recorded side spelling the rotation in bf16.
  block:
    var (orc, ctx) = newMlaCacheCtx(64, kvLoraRank, qkRopeHeadDim,
      @[1'u32, 2, 3, 4], numLayers = 4)
    let cache = MlaLatentCache.init(kvLoraRank, qkRopeHeadDim, 64,
      kBFloat16, F.kCPU)
    var st = Safetensor.open(FixtureDir /
      "attn-Ling-3.0-tiny-00.safetensor")
    let statsPath = FixtureDir /
      "attn-Ling-3.0-tiny-00.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill seq4)"
    let seqLen = 4
    let x = st.getTensorOwned("hidden_states")

    let q = attn.attn.q_b_proj.forward(
      attn.attn.q_a_norm.forward(attn.attn.q_a_proj.forward(x))).reshape(
      [1, seqLen, numHeads, qkNopeHeadDim + qkRopeHeadDim])
    assertStats(q.narrow(3, qkNopeHeadDim, qkRopeHeadDim), statsPath,
      "q_rot_raw", kElementwise, msg = "prefill raw q plane")
    let qPeRot = rotateInterleaved(q.narrow(3, qkNopeHeadDim, qkRopeHeadDim),
      st.getTensorOwned("cos"), st.getTensorOwned("sin"))
    assertStats(F.cat([q.narrow(3, 0, qkNopeHeadDim), qPeRot], 3), statsPath,
      "query_states", kElementwise, depth = 1,
      msg = "prefill query states")

    let compressed = attn.attn.kv_a_proj_with_mqa.forward(x)
    let latentNormed = attn.attn.kv_a_layernorm.forward(
      compressed.narrow(2, 0, kvLoraRank)).unsqueeze(2)
    let kPeRot = rotateInterleaved(
      compressed.narrow(2, kvLoraRank, qkRopeHeadDim).unsqueeze(2),
      st.getTensorOwned("cos"), st.getTensorOwned("sin"))
    cache.write(ctx, 0, latentNormed, kPeRot, ctx.kv_position, seqLen)
    let (gotLatent, gotKpe) = cache.gather(ctx, 0, 0, ctx.kv_position + seqLen)
    let expanded = attn.attn.kv_b_proj.forward(gotLatent.reshape(
      [1, seqLen, kvLoraRank])).reshape(
      [1, seqLen, numHeads, qkNopeHeadDim + vHeadDim])
    let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
      gotKpe.expand([1, seqLen, numHeads, qkRopeHeadDim])], 3)
    assertStats(keyStates, statsPath, "key_states", kElementwise, depth = 1,
      msg = "prefill key states")
    assertStats(expanded.narrow(3, qkNopeHeadDim, vHeadDim), statsPath,
      "value_states", kReduction, msg = "prefill value states")

    # Head-wise sigmoid gate replay plus its placement on the recorded
    # attention output, the recorded eager output scales by the gate weight
    # before the dense projection.
    let gate = F.sigmoid(attn.g_proj.forward(x).to(F.kFloat32))
      .to(F.kBFloat16)
    assertStats(gate, statsPath, "gate", kElementwise,
      msg = "prefill gate weight")
    assertStats(st.getTensorOwned("attn_output") * gate.unsqueeze(3),
      statsPath, "gated", kElementwise,
      msg = "prefill gate placement")

    # Isolated sdpa over the recorded same-form inputs.
    let attnOut = F.scaled_dot_product_attention(
      st.getTensorOwned("query_states"),
      st.getTensorOwned("key_states"),
      st.getTensorOwned("value_states"),
      is_causal = true, scale = some(scale))
    assertStats(attnOut, statsPath, "sdpa_twin", kReduction,
      msg = "prefill sdpa output")

    ctx.cos = st.getTensorOwned("cos")
    ctx.sin = st.getTensorOwned("sin")
    # TODO mixer output in attn-Ling-3.0-tiny-00 (kReduction, depth 1) drifts 0.2109375 on CPU, 13.5 ulps against the 4-ulp band 0.0625
    assertStats(attn(ctx, x), statsPath, "output", kReduction, depth = 1,
      msg = "prefill seq4 mixer output")
    # The discard holds the orchestrator alive to the end of the block,
    # the context borrows its pool pages for the gathered replays.
    discard orc

  # Prefill2 plus decode1. The mixer forwards run end to end over
  # the recorded rope rows, the decompress step reading the mixer's
  # own cache contents.
  block:
    var orc = newMlaOrchestrator(64, kvLoraRank, qkRopeHeadDim, numLayers = 4)
    orc.startSequence(@[1'u32, 2])
    var ctx = orc.getInferenceContextMut()
    var st = Safetensor.open(FixtureDir /
      "attn-Ling-3.0-tiny-01.safetensor")
    let statsPath = FixtureDir /
      "attn-Ling-3.0-tiny-01.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill2 + decode1)"

    ctx.cos = st.getTensorOwned("cos")
    ctx.sin = st.getTensorOwned("sin")
    let outPrefill = attn(ctx, st.getTensorOwned("hidden_states"))
    assertStats(outPrefill, statsPath, "output", kReduction, depth = 1,
      msg = "prefill2 mixer output")

    orc.setKvPosition(2)
    orc.appendToken(position = 2, token_id = 1000'u32, device = F.kCPU)
    ctx.cos = st.getTensorOwned("cos_step")
    ctx.sin = st.getTensorOwned("sin_step")
    let outStep = attn(ctx, st.getTensorOwned("x_step"))
    assertStats(outStep, statsPath, "output_step", kReduction, depth = 1,
      msg = "decode1 mixer output")

    # Decode per-op rows over the mixer's own cache. The gathered cache
    # decompresses into the recorded step slabs, the q chain replays
    # over the recorded step input, the gate weight re-runs per step.
    let (gotLatent, gotKpe) = attn.attn.cache.gather(
      orc.getInferenceContextMut(), attn.attn.layer_idx, 0, 3)
    let expanded = attn.attn.kv_b_proj.forward(gotLatent.reshape(
      [1, 3, kvLoraRank])).reshape(
      [1, 3, numHeads, qkNopeHeadDim + vHeadDim])
    let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
      gotKpe.expand([1, 3, numHeads, qkRopeHeadDim])], 3)
    assertStats(keyStates, statsPath, "key_states_step", kElementwise,
      depth = 1, msg = "decode key states")
    assertStats(expanded.narrow(3, qkNopeHeadDim, vHeadDim), statsPath,
      "value_states_step", kReduction, msg = "decode value states")

    let qStep = attn.attn.q_b_proj.forward(
      attn.attn.q_a_norm.forward(attn.attn.q_a_proj.forward(
        st.getTensorOwned("x_step")))).reshape(
      [1, 1, numHeads, qkNopeHeadDim + qkRopeHeadDim])
    let qPeRotStep = rotateInterleaved(
      qStep.narrow(3, qkNopeHeadDim, qkRopeHeadDim),
      st.getTensorOwned("cos_step"), st.getTensorOwned("sin_step"))
    assertStats(F.cat([qStep.narrow(3, 0, qkNopeHeadDim), qPeRotStep], 3),
      statsPath, "query_states_step", kElementwise, depth = 1,
      msg = "decode query states")
    assertStats(
      F.sigmoid(attn.g_proj.forward(st.getTensorOwned("x_step"))
        .to(F.kFloat32)).to(F.kBFloat16),
      statsPath, "gate_step", kElementwise,
      msg = "decode gate weight")
    discard orc

  # Prefill3 plus a 3-step decode sequence. The mixer forwards run
  # end to end, the final cache contents decompress into the recorded
  # final key and value slabs.
  block:
    var orc = newMlaOrchestrator(64, kvLoraRank, qkRopeHeadDim, numLayers = 4)
    orc.startSequence(@[1'u32, 2, 3])
    var ctx = orc.getInferenceContextMut()
    var st = Safetensor.open(FixtureDir /
      "attn-Ling-3.0-tiny-02.safetensor")
    let statsPath = FixtureDir /
      "attn-Ling-3.0-tiny-02.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill3 + 3 decode steps)"

    ctx.cos = st.getTensorOwned("cos")
    ctx.sin = st.getTensorOwned("sin")
    let outPrefill = attn(ctx, st.getTensorOwned("hidden_states"))
    assertStats(outPrefill, statsPath, "output", kReduction, depth = 1,
      msg = "prefill3 mixer output")

    for i in 0 ..< 3:
      let position = 3 + i
      orc.setKvPosition(position)
      orc.appendToken(position = position, token_id = uint32(1000 + i),
        device = F.kCPU)
      ctx.cos = st.getTensorOwned("cos_step" & $i)
      ctx.sin = st.getTensorOwned("sin_step" & $i)
      let xStep = st.getTensorOwned("x_step" & $i)
      let outStep = attn(ctx, xStep)
      assertStats(outStep, statsPath, "output_step" & $i, kReduction,
        depth = 1, msg = "decode step " & $i & " mixer output")

      # Per-step rows over the mixer's own cache, the q chain replay
      # against the recorded step query, the gathered cache decompress
      # against the recorded step key and value slabs, the gate weight
      # re-run per step.
      let qStep = attn.attn.q_b_proj.forward(
        attn.attn.q_a_norm.forward(attn.attn.q_a_proj.forward(xStep))).reshape(
        [1, 1, numHeads, qkNopeHeadDim + qkRopeHeadDim])
      let qPeRotStep = rotateInterleaved(
        qStep.narrow(3, qkNopeHeadDim, qkRopeHeadDim),
        st.getTensorOwned("cos_step" & $i), st.getTensorOwned("sin_step" & $i))
      assertStats(F.cat([qStep.narrow(3, 0, qkNopeHeadDim), qPeRotStep], 3),
        statsPath, "query_states_step" & $i, kElementwise, depth = 1,
        msg = "decode step " & $i & " query states")
      assertStats(
        F.sigmoid(attn.g_proj.forward(xStep).to(F.kFloat32))
          .to(F.kBFloat16),
        statsPath, "gate_step" & $i, kElementwise,
        msg = "decode step " & $i & " gate weight")

      let (gotLatent, gotKpe) = attn.attn.cache.gather(
        orc.getInferenceContextMut(), attn.attn.layer_idx, 0, position + 1)
      let expanded = attn.attn.kv_b_proj.forward(gotLatent.reshape(
        [1, position + 1, kvLoraRank])).reshape(
        [1, position + 1, numHeads, qkNopeHeadDim + vHeadDim])
      let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
        gotKpe.expand([1, position + 1, numHeads, qkRopeHeadDim])], 3)
      assertStats(keyStates, statsPath, "key_states_step" & $i, kElementwise,
        depth = 1, msg = "decode step " & $i & " key states")
      assertStats(expanded.narrow(3, qkNopeHeadDim, vHeadDim), statsPath,
        "value_states_step" & $i, kReduction,
        msg = "decode step " & $i & " value states")

    let (finalLatent, finalKpe) = attn.attn.cache.gather(
      orc.getInferenceContextMut(), attn.attn.layer_idx, 0, 6)
    let expandedFinal = attn.attn.kv_b_proj.forward(finalLatent.reshape(
      [1, 6, kvLoraRank])).reshape(
      [1, 6, numHeads, qkNopeHeadDim + vHeadDim])
    let keyStatesFinal = F.cat([expandedFinal.narrow(3, 0, qkNopeHeadDim),
      finalKpe.expand([1, 6, numHeads, qkRopeHeadDim])], 3)
    assertStats(keyStatesFinal, statsPath, "key_states_final", kElementwise,
      depth = 1, msg = "final key states")
    assertStats(expandedFinal.narrow(3, qkNopeHeadDim, vHeadDim), statsPath,
      "value_states_final", kReduction, msg = "final value states")
    discard orc

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
