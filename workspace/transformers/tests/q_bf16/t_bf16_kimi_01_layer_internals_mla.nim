# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-3 NoPE MLA attention of the Kimi-Linear-48B-A3B-Instruct checkpoint,
## replayed on cpu against the recorded layer-3 fixture payloads.
## Requires the local model at tests/hf_models/Kimi-Linear-48B-A3B-Instruct (gitignored).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_kimi_01_layer_internals_mla.nim

import
  std/math,
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

privateAccess(MLAttention[void, NoPe])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Kimi-Linear-48B-A3B-Instruct-layer-3"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" /
    "Kimi-Linear-48B-A3B-Instruct"
  LayerPrefix = "model.layers.3.self_attn"

proc main() =
  # The mixer loads exactly as the Kimi model file wires it:
  # direct-Q projections, the latent norm at the bottleneck eps,
  # softmax scale 1/sqrt(qk_head_dim), no rotation on any plane.
  let attn = setupMlaDirect[NoPe](ModelDir, LayerPrefix, 0, 64)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let kvLoraRank = cfgJson{"kv_lora_rank"}.getInt()
  let qkRopeHeadDim = cfgJson{"qk_rope_head_dim"}.getInt()
  let qkNopeHeadDim = cfgJson{"qk_nope_head_dim"}.getInt()
  let vHeadDim = cfgJson{"v_head_dim"}.getInt()
  let numHeads = cfgJson{"num_attention_heads"}.getInt()
  let scale = mlaSoftmaxScale(qkNopeHeadDim, qkRopeHeadDim)

  # Latent norm payloads over the recorded weight and inputs.
  for caseNum in 0 ..< 3:
    let payload = FixtureDir /
      ("norm-Kimi-Linear-48B-A3B-Instruct-0" & $caseNum & ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (norm case ", caseNum, ")"
    assertStats(attn.kv_a_layernorm.forward(st.getTensorOwned("input")),
      statsPath, "output", kElementwise,
      msg = "latent norm replay, case " & $caseNum)
    assertStats(attn.kv_a_layernorm.weight.to(F.kCPU), statsPath,
      "weight", kElementwise, msg = "latent norm weight, case " & $caseNum)

  # Prefill seq8 per-op replay. The NoPE spelling caches the raw
  # unrotated plane, so every cache and rope-channel row compares raw bytes.
  block:
    var (orc, ctx) = newMlaCacheCtx(64, kvLoraRank, qkRopeHeadDim,
      @[1'u32, 2, 3, 4, 5, 6, 7, 8])
    let cache = MlaLatentCache.init(kvLoraRank, qkRopeHeadDim, 64,
      kBFloat16, F.kCPU)
    var st = Safetensor.open(FixtureDir /
      "attn-Kimi-Linear-48B-A3B-Instruct-00.safetensor")
    let statsPath = FixtureDir /
      "attn-Kimi-Linear-48B-A3B-Instruct-00.safetensor.stats.json.zst"
    var outSt = Safetensor.open(FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-00.safetensor")
    let outStatsPath = FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-00.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill seq8)"
    let seqLen = 8
    let x = st.getTensorOwned("hidden_states")

    # Direct-Q projection, the recorded two-head slabs lock the GEMM rows.
    let q = attn.q_proj.forward(x).reshape(
      [1, seqLen, numHeads, qkNopeHeadDim + qkRopeHeadDim]).transpose(1, 2)
    for h in 0 .. 1:
      let suffix = if h == 0: "h00" else: "h01"
      assertStats(q.narrow(1, h, 1), statsPath,
        "query_states_" & suffix, kReduction,
        msg = "prefill q projection head " & $h)

    let compressed = attn.kv_a_proj_with_mqa.forward(x)
    let latentNormed = attn.kv_a_layernorm.forward(
      compressed.narrow(2, 0, kvLoraRank)).unsqueeze(2)
    assertStats(latentNormed, statsPath, "latent_normed", kElementwise,
      msg = "prefill latent norm")

    let kPeRaw = compressed.narrow(2, kvLoraRank, qkRopeHeadDim).unsqueeze(2)
    assertStats(kPeRaw, statsPath, "kpe_cached", kElementwise,
      msg = "prefill raw plane identity")
    cache.write(ctx, 0, latentNormed, kPeRaw, ctx.kv_position, seqLen)
    let (gotLatent, gotKpe) = cache.gather(ctx, 0, 0, ctx.kv_position + seqLen)
    assertStats(gotLatent, statsPath, "latent_cached", kElementwise,
      msg = "prefill latent cache")
    assertStats(gotKpe, statsPath, "kpe_cached", kElementwise,
      msg = "prefill plane cache")

    let expanded = attn.kv_b_proj.forward(gotLatent.reshape(
      [1, seqLen, kvLoraRank])).reshape(
      [1, seqLen, numHeads, qkNopeHeadDim + vHeadDim])
    let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
      gotKpe.expand([1, seqLen, numHeads, qkRopeHeadDim])], 3)
    let keyNative = keyStates.permute(0, 2, 1, 3)
    let vNative = expanded.narrow(3, qkNopeHeadDim, vHeadDim).permute(0, 2, 1, 3)
    for h in 0 .. 1:
      let suffix = if h == 0: "h00" else: "h01"
      assertStats(keyNative.narrow(1, h, 1), statsPath,
        "key_states_" & suffix, kReduction,
        msg = "prefill key states head " & $h)
      assertStats(vNative.narrow(1, h, 1), statsPath,
        "value_states_" & suffix, kReduction,
        msg = "prefill value states head " & $h)

    # Eager scores recomputed off the port states, the recorded rows lock
    # the bf16 GEMM times the folded scaling.
    let scoresNim = F.matmul(q, keyNative.transpose(2, 3)) * scale
    assertStats(scoresNim, statsPath, "scores", kReduction,
      msg = "prefill eager scores")

    # Counterfactual scores, the NoPE-ignoring spelling a wrong port
    # would run. Plain-theta rotate_half on the q rope channels
    # and the k plane. The f64 angle tables round once to bf16, the same
    # single rounding the reference rope spellings apply.
    var cosFlat, sinFlat: seq[float64]
    let half = qkRopeHeadDim div 2
    for pos in 0 ..< seqLen:
      for i in 0 ..< half:
        let ang = pos.float64 * pow(10000.0,
          -2.0 * i.float64 / qkRopeHeadDim.float64)
        cosFlat.add cos(ang)
        sinFlat.add sin(ang)
      for i in 0 ..< half:
        let ang = pos.float64 * pow(10000.0,
          -2.0 * i.float64 / qkRopeHeadDim.float64)
        cosFlat.add cos(ang)
        sinFlat.add sin(ang)
    let cosCf = F.toTensor(cosFlat).reshape(
      [seqLen, qkRopeHeadDim]).to(kBFloat16)
    let sinCf = F.toTensor(sinFlat).reshape(
      [seqLen, qkRopeHeadDim]).to(kBFloat16)
    let qPe = q.narrow(3, qkNopeHeadDim, qkRopeHeadDim)
    let kPlane = kPeRaw.permute(0, 2, 1, 3)
    let qPeCf = F.cat([qPe.narrow(3, half, half).neg(),
      qPe.narrow(3, 0, half)], 3) * cosCf +
      F.cat([qPe.narrow(3, 0, half), qPe.narrow(3, half, half)], 3) * sinCf
    let kPlaneCf = F.cat([kPlane.narrow(3, half, half).neg(),
      kPlane.narrow(3, 0, half)], 3) * cosCf +
      F.cat([kPlane.narrow(3, 0, half), kPlane.narrow(3, half, half)], 3) * sinCf
    let qCf = F.cat([q.narrow(3, 0, qkNopeHeadDim), qPeCf], 3)
    let kCf = F.cat([keyNative.narrow(3, 0, qkNopeHeadDim),
      kPlaneCf.expand([1, numHeads, seqLen, qkRopeHeadDim])], 3)
    let scoresCf = F.matmul(qCf, kCf.transpose(2, 3)) * scale
    assertStats(scoresCf, statsPath, "scores_rotated_cf", kReduction,
      msg = "rotated counterfactual scores")

    # Isolated sdpa over the recorded same-form two-head slabs.
    let attnOut = F.scaled_dot_product_attention(
      F.cat([st.getTensorOwned("query_states_h00"),
        st.getTensorOwned("query_states_h01")], 1),
      F.cat([st.getTensorOwned("key_states_h00"),
        st.getTensorOwned("key_states_h01")], 1),
      F.cat([st.getTensorOwned("value_states_h00"),
        st.getTensorOwned("value_states_h01")], 1),
      is_causal = true, scale = some(scale))
    for h in 0 .. 1:
      let suffix = if h == 0: "h00" else: "h01"
      assertStats(attnOut.narrow(1, h, 1), outStatsPath,
        "sdpa_twin_" & suffix, kReduction,
        msg = "prefill sdpa output head " & $h)
    # The discard holds the orchestrator alive to the end of the block,
    # the context borrows its pool pages for the gathered replays.
    discard orc

  # Prefill seq8 end to end over the recorded inputs.
  block:
    var orc = newMlaOrchestrator(64, kvLoraRank, qkRopeHeadDim)
    orc.startSequence(@[1'u32, 2, 3, 4, 5, 6, 7, 8])
    var ctx = orc.getInferenceContextMut()
    var st = Safetensor.open(FixtureDir /
      "attn-Kimi-Linear-48B-A3B-Instruct-00.safetensor")
    var outSt = Safetensor.open(FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-00.safetensor")
    let outStatsPath = FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-00.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill seq8 mixer)"
    assertStats(attn(ctx, st.getTensorOwned("hidden_states")), outStatsPath,
      "output", kReduction, depth = 1, msg = "prefill seq8 mixer output")
    discard orc

  # Prefill3 plus decode1. The mixer forwards run end to end, the per-op
  # decode replay using a cache preloaded with the recorded prefill rows.
  block:
    var orc = newMlaOrchestrator(64, kvLoraRank, qkRopeHeadDim)
    orc.startSequence(@[1'u32, 2, 3])
    var ctx = orc.getInferenceContextMut()
    var st = Safetensor.open(FixtureDir /
      "attn-Kimi-Linear-48B-A3B-Instruct-01.safetensor")
    let statsPath = FixtureDir /
      "attn-Kimi-Linear-48B-A3B-Instruct-01.safetensor.stats.json.zst"
    var outSt = Safetensor.open(FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-01.safetensor")
    let outStatsPath = FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-01.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill3 + decode1)"

    let outPrefill = attn(ctx, st.getTensorOwned("hidden_states"))
    assertStats(outPrefill, outStatsPath, "output", kReduction, depth = 1,
      msg = "prefill3 mixer output")

    orc.setKvPosition(3)
    orc.appendToken(position = 3, token_id = 1000'u32, device = F.kCPU)
    let outStep = attn(ctx, st.getTensorOwned("x_step"))
    assertStats(outStep, outStatsPath, "output_step", kReduction, depth = 1,
      msg = "decode1 mixer output")

    let (gotLatent, gotKpe) = attn.cache.gather(
      orc.getInferenceContextMut(), 0, 0, 4)
    assertStats(gotLatent, statsPath, "latent_cached", kElementwise,
      msg = "decode1 latent cache")
    assertStats(gotKpe, statsPath, "kpe_cached", kElementwise,
      msg = "decode1 plane cache")

    # Decode per-op rows over the preloaded cache. The step write
    # reproduces the recorded cache tensors, the decompress and score
    # rows run over the gathered slabs.
    var (orc2, ctx2) = newMlaCacheCtx(64, kvLoraRank, qkRopeHeadDim,
      @[1'u32, 2, 3, 4, 5, 6, 7, 8])
    let cache = MlaLatentCache.init(kvLoraRank, qkRopeHeadDim, 64,
      kBFloat16, F.kCPU)
    cache.write(ctx2, 0,
      st.getTensorOwned("latent_normed").reshape([1, 3, 1, kvLoraRank]),
      st.getTensorOwned("kpe_cached").reshape(
        [1, 4, 1, qkRopeHeadDim]).narrow(1, 0, 3),
      0, 3)
    orc2.setKvPosition(3)

    let xStep = st.getTensorOwned("x_step")
    let q = attn.q_proj.forward(xStep).reshape(
      [1, 1, numHeads, qkNopeHeadDim + qkRopeHeadDim]).transpose(1, 2)
    for h in 0 .. 1:
      let suffix = if h == 0: "h00" else: "h01"
      assertStats(q.narrow(1, h, 1), statsPath,
        "query_states_step_" & suffix, kReduction,
        msg = "decode q projection head " & $h)

    let compressed = attn.kv_a_proj_with_mqa.forward(xStep)
    let latentNormed = attn.kv_a_layernorm.forward(
      compressed.narrow(2, 0, kvLoraRank)).unsqueeze(2)
    assertStats(latentNormed, statsPath, "latent_normed_step", kElementwise,
      msg = "decode latent norm")

    let kPeRaw = compressed.narrow(2, kvLoraRank, qkRopeHeadDim).unsqueeze(2)
    cache.write(ctx2, 0, latentNormed, kPeRaw, ctx2.kv_position, 1)
    let (gotLatent2, gotKpe2) = cache.gather(ctx2, 0, 0, ctx2.kv_position + 1)
    assertStats(gotLatent2, statsPath, "latent_cached", kElementwise,
      msg = "decode latent cache")
    assertStats(gotKpe2, statsPath, "kpe_cached", kElementwise,
      msg = "decode plane cache")
    assertStats(gotKpe2.narrow(1, 3, 1), statsPath, "k_rot_raw_step",
      kElementwise, msg = "decode raw plane identity")

    let expanded = attn.kv_b_proj.forward(gotLatent2.reshape(
      [1, 4, kvLoraRank])).reshape(
      [1, 4, numHeads, qkNopeHeadDim + vHeadDim])
    let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
      gotKpe2.expand([1, 4, numHeads, qkRopeHeadDim])], 3)
    let keyNative = keyStates.permute(0, 2, 1, 3)
    let vNative = expanded.narrow(3, qkNopeHeadDim, vHeadDim).permute(0, 2, 1, 3)
    for h in 0 .. 1:
      let suffix = if h == 0: "h00" else: "h01"
      assertStats(keyNative.narrow(1, h, 1), statsPath,
        "key_states_step_" & suffix, kReduction,
        msg = "decode key states head " & $h)
      assertStats(vNative.narrow(1, h, 1), statsPath,
        "value_states_step_" & suffix, kReduction,
        msg = "decode value states head " & $h)

    let scoresNim = F.matmul(q, keyNative.transpose(2, 3)) * scale
    assertStats(scoresNim, statsPath, "scores_step", kReduction,
      msg = "decode eager scores")

    let attnOut = F.scaled_dot_product_attention(
      F.cat([st.getTensorOwned("query_states_step_h00"),
        st.getTensorOwned("query_states_step_h01")], 1),
      F.cat([st.getTensorOwned("key_states_step_h00"),
        st.getTensorOwned("key_states_step_h01")], 1),
      F.cat([st.getTensorOwned("value_states_step_h00"),
        st.getTensorOwned("value_states_step_h01")], 1),
      is_causal = false, scale = some(scale))
    for h in 0 .. 1:
      let suffix = if h == 0: "h00" else: "h01"
      assertStats(attnOut.narrow(1, h, 1), outStatsPath,
        "sdpa_twin_step_" & suffix, kReduction,
        msg = "decode sdpa output head " & $h)
    discard orc

  # Prefill3 plus a 3-step decode sequence. The mixer forwards run
  # end to end, then the final cache contents compare.
  block:
    var orc = newMlaOrchestrator(64, kvLoraRank, qkRopeHeadDim)
    orc.startSequence(@[1'u32, 2, 3])
    var ctx = orc.getInferenceContextMut()
    var st = Safetensor.open(FixtureDir /
      "attn-Kimi-Linear-48B-A3B-Instruct-02.safetensor")
    let statsPath = FixtureDir /
      "attn-Kimi-Linear-48B-A3B-Instruct-02.safetensor.stats.json.zst"
    var outSt = Safetensor.open(FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-02.safetensor")
    let outStatsPath = FixtureDir /
      "attnout-Kimi-Linear-48B-A3B-Instruct-02.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill3 + 3 decode steps)"

    let outPrefill = attn(ctx, st.getTensorOwned("hidden_states"))
    assertStats(outPrefill, outStatsPath, "output", kReduction, depth = 1,
      msg = "prefill3 mixer output")

    # Per-op decode replay over a cache preloaded with the prefill rows,
    # the step writes append on top, the mixer cache state left
    # for the final gather compare.
    var (orc2, ctx2) = newMlaCacheCtx(64, kvLoraRank, qkRopeHeadDim,
      @[1'u32, 2, 3, 4, 5, 6, 7, 8])
    let opCache = MlaLatentCache.init(kvLoraRank, qkRopeHeadDim, 64,
      kBFloat16, F.kCPU)
    opCache.write(ctx2, 0,
      st.getTensorOwned("latent_normed").reshape([1, 3, 1, kvLoraRank]),
      st.getTensorOwned("kpe_cached").reshape(
        [1, 6, 1, qkRopeHeadDim]).narrow(1, 0, 3),
      0, 3)
    orc2.setKvPosition(3)

    for i in 0 ..< 3:
      let position = 3 + i
      orc.setKvPosition(position)
      orc.appendToken(position = position, token_id = uint32(1000 + i),
        device = F.kCPU)
      let outStep = attn(ctx, st.getTensorOwned("x_step" & $i))
      assertStats(outStep, outStatsPath, "output_step" & $i, kReduction,
        depth = 1, msg = "decode step " & $i & " mixer output")

      let xStep = st.getTensorOwned("x_step" & $i)
      let q = attn.q_proj.forward(xStep).reshape(
        [1, 1, numHeads, qkNopeHeadDim + qkRopeHeadDim]).transpose(1, 2)
      for h in 0 .. 1:
        let suffix = if h == 0: "h00" else: "h01"
        assertStats(q.narrow(1, h, 1), statsPath,
          "query_states_step" & $i & "_" & suffix, kReduction,
          msg = "decode step " & $i & " q projection head " & $h)

      let compressed = attn.kv_a_proj_with_mqa.forward(xStep)
      let latentNormed = attn.kv_a_layernorm.forward(
        compressed.narrow(2, 0, kvLoraRank)).unsqueeze(2)
      assertStats(latentNormed, statsPath, "latent_normed_step" & $i,
        kElementwise, msg = "decode step " & $i & " latent norm")

      let kPeRaw = compressed.narrow(2, kvLoraRank, qkRopeHeadDim).unsqueeze(2)
      opCache.write(ctx2, 0, latentNormed, kPeRaw, ctx2.kv_position, 1)
      orc2.setKvPosition(position + 1)
      let (gotLatent, gotKpe) = opCache.gather(ctx2, 0, 0, position + 1)
      assertStats(gotKpe.narrow(1, position, 1), statsPath,
        "k_rot_raw_step" & $i, kElementwise,
        msg = "decode step " & $i & " raw plane identity")

      let expanded = attn.kv_b_proj.forward(gotLatent.reshape(
        [1, position + 1, kvLoraRank])).reshape(
        [1, position + 1, numHeads, qkNopeHeadDim + vHeadDim])
      let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
        gotKpe.expand([1, position + 1, numHeads, qkRopeHeadDim])], 3)
      let keyNative = keyStates.permute(0, 2, 1, 3)
      let vNative = expanded.narrow(3, qkNopeHeadDim, vHeadDim).permute(0, 2, 1, 3)
      for h in 0 .. 1:
        let suffix = if h == 0: "h00" else: "h01"
        assertStats(keyNative.narrow(1, h, 1), statsPath,
          "key_states_step" & $i & "_" & suffix, kReduction,
          msg = "decode step " & $i & " key states head " & $h)
        assertStats(vNative.narrow(1, h, 1), statsPath,
          "value_states_step" & $i & "_" & suffix, kReduction,
          msg = "decode step " & $i & " value states head " & $h)
      assertStats(F.matmul(q, keyNative.transpose(2, 3)) * scale, statsPath,
        "scores_step" & $i, kReduction,
        msg = "decode step " & $i & " eager scores")

      let attnOut = F.scaled_dot_product_attention(
        F.cat([st.getTensorOwned("query_states_step" & $i & "_h00"),
          st.getTensorOwned("query_states_step" & $i & "_h01")], 1),
        F.cat([st.getTensorOwned("key_states_step" & $i & "_h00"),
          st.getTensorOwned("key_states_step" & $i & "_h01")], 1),
        F.cat([st.getTensorOwned("value_states_step" & $i & "_h00"),
          st.getTensorOwned("value_states_step" & $i & "_h01")], 1),
        is_causal = false, scale = some(scale))
      for h in 0 .. 1:
        let suffix = if h == 0: "h00" else: "h01"
        assertStats(attnOut.narrow(1, h, 1), outStatsPath,
          "sdpa_twin_step" & $i & "_" & suffix, kReduction,
          msg = "decode step " & $i & " sdpa output head " & $h)

    let (finalLatent, finalKpe) = attn.cache.gather(
      orc.getInferenceContextMut(), 0, 0, 6)
    assertStats(finalLatent, statsPath, "latent_cached", kElementwise,
      msg = "final latent cache")
    assertStats(finalKpe, statsPath, "kpe_cached", kElementwise,
      msg = "final plane cache")
    discard orc
    discard orc2

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
