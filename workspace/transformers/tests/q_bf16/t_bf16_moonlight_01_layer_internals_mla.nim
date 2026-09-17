# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-0 MLA attention of the Moonlight-16B-A3B checkpoint, replayed
## on cpu against the recorded layer-0 fixture payloads.
## Requires the local model at tests/hf_models/Moonlight-16B-A3B (gitignored).
##
## Run:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_moonlight_01_layer_internals_mla.nim

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

privateAccess(MLAttention[void, FullRoPe])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-01-layer-internals" / "Moonlight-16B-A3B-layer-0"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Moonlight-16B-A3B"
  LayerPrefix = "model.layers.0.self_attn"

proc main() =
  # The mixer loads exactly as the Moonlight model file wires it:
  # direct-Q projections, the latent norm at the bottleneck eps,
  # softmax scale 1/sqrt(qk_head_dim).
  let attn = setupMlaDirect[FullRoPe](ModelDir, LayerPrefix, 0, 64)
  let cfgJson = (ModelDir / "config.json").parseFile()
  let kvLoraRank = cfgJson{"kv_lora_rank"}.getInt()
  let qkRopeHeadDim = cfgJson{"qk_rope_head_dim"}.getInt()
  let qkNopeHeadDim = cfgJson{"qk_nope_head_dim"}.getInt()
  let vHeadDim = cfgJson{"v_head_dim"}.getInt()
  let numHeads = cfgJson{"num_attention_heads"}.getInt()
  # Frequency table rows for positions 0..15, the plain-theta tables
  # against the recorded pre-cast rows.
  let rotary = MlaRotary.new(qkRopeHeadDim, 16,
    cfgJson{"rope_theta"}.getFloat(), F.kCPU,
    yarnFactor = 0.0)
  block:
    echo "    devices: ", deviceName(F.kCPU), " (tables)"
    let tablesStats = FixtureDir /
      "tables-Moonlight-16B-A3B-00.safetensor.stats.json.zst"
    assertStats(rotary.cosCache, tablesStats, "cos_rows", kElementwise,
      msg = "plain-theta cos table rows")
    assertStats(rotary.sinCache, tablesStats, "sin_rows", kElementwise,
      msg = "plain-theta sin table rows")

  # Rotation replay over the recorded planes. The recorded side spells
  # the rotation in bf16, the Nim rotation computes in f32 and casts,
  # the two forms differ by a grid step on the rotating channels.
  # The permutation block verified against the interleaved layout
  # relation on the recorded rows themselves, pure data movement.
  for caseNum in 0 ..< 3:
    let payload = FixtureDir /
      ("rope-Moonlight-16B-A3B-0" & $caseNum & ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (rope case ", caseNum, ")"
    let cos = st.getTensorOwned("cos")
    let sin = st.getTensorOwned("sin")
    let qRot = rotateInterleaved(
      st.getTensorOwned("q_pe").transpose(1, 2), cos, sin)
    let kRot = rotateInterleaved(
      st.getTensorOwned("k_pe").transpose(1, 2), cos, sin)
    assertStats(qRot, statsPath, "q_rot_interleaved", kElementwise, depth = 2,
      msg = "q rotation, case " & $caseNum)
    assertStats(kRot, statsPath, "k_rot_interleaved", kElementwise, depth = 2,
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
      ("norm-Moonlight-16B-A3B-0" & $caseNum & ".safetensor")
    var st = Safetensor.open(payload)
    let statsPath = payload & ".stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (norm case ", caseNum, ")"
    assertStats(attn.kv_a_layernorm.forward(st.getTensorOwned("input")),
      statsPath, "output", kElementwise,
      msg = "latent norm replay, case " & $caseNum)
    assertStats(attn.kv_a_layernorm.weight.to(F.kCPU), statsPath,
      "weight", kElementwise, msg = "latent norm weight, case " & $caseNum)

  # Prefill seq8 per-op replay. Every recomputed surface runs through
  # the layer's own components over the recorded inputs, the recorded
  # side carrying the bf16-spelled rotation against the f32 Nim form.
  block:
    var (orc, ctx) = newMlaCacheCtx(64, kvLoraRank, qkRopeHeadDim,
      @[1'u32, 2, 3, 4, 5, 6, 7, 8])
    let cache = MlaLatentCache.init(kvLoraRank, qkRopeHeadDim, 64,
      kBFloat16, F.kCPU)
    var st = Safetensor.open(FixtureDir /
      "attn-Moonlight-16B-A3B-00.safetensor")
    let statsPath = FixtureDir /
      "attn-Moonlight-16B-A3B-00.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill seq8)"
    let seqLen = 8
    let x = st.getTensorOwned("hidden_states")

    let q = attn.q_proj.forward(x).reshape(
      [1, seqLen, numHeads, qkNopeHeadDim + qkRopeHeadDim])
    let qPeRot = rotateInterleaved(q.narrow(3, qkNopeHeadDim, qkRopeHeadDim),
      st.getTensorOwned("cos"), st.getTensorOwned("sin"))
    assertStats(F.cat([q.narrow(3, 0, qkNopeHeadDim), qPeRot], 3), statsPath,
      "query_states", kElementwise, depth = 2,
      msg = "prefill query states")

    let compressed = attn.kv_a_proj_with_mqa.forward(x)
    let latentNormed = attn.kv_a_layernorm.forward(
      compressed.narrow(2, 0, kvLoraRank)).unsqueeze(2)
    assertStats(latentNormed, statsPath, "latent_normed", kElementwise,
      msg = "prefill latent norm")

    let kPeRot = rotateInterleaved(
      compressed.narrow(2, kvLoraRank, qkRopeHeadDim).unsqueeze(2),
      st.getTensorOwned("cos"), st.getTensorOwned("sin"))
    cache.write(ctx, 0, latentNormed, kPeRot, ctx.kv_position, seqLen)
    let (gotLatent, gotKpe) = cache.gather(ctx, 0, 0, ctx.kv_position + seqLen)
    assertStats(gotLatent, statsPath, "latent_cached", kElementwise,
      msg = "prefill latent cache")
    assertStats(gotKpe, statsPath, "kpe_cached_interleaved", kElementwise,
      depth = 2, msg = "prefill kpe cache")

    let expanded = attn.kv_b_proj.forward(gotLatent.reshape(
      [1, seqLen, kvLoraRank])).reshape(
      [1, seqLen, numHeads, qkNopeHeadDim + vHeadDim])
    let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
      gotKpe.expand([1, seqLen, numHeads, qkRopeHeadDim])], 3)
    assertStats(keyStates, statsPath, "key_states", kElementwise, depth = 2,
      msg = "prefill key states")
    assertStats(expanded.narrow(3, qkNopeHeadDim, vHeadDim), statsPath,
      "value_states", kReduction, msg = "prefill value states")

    # Isolated sdpa over the recorded same-form inputs.
    let attnOut = F.scaled_dot_product_attention(
      st.getTensorOwned("query_states"),
      st.getTensorOwned("key_states"),
      st.getTensorOwned("value_states"),
      is_causal = true, scale = some(mlaSoftmaxScale(qkNopeHeadDim,
        qkRopeHeadDim)))
    assertStats(attnOut, statsPath, "attn_output", kReduction,
      msg = "prefill sdpa output")
    # The discard holds the orchestrator alive to the end of the block,
    # the context borrows its pool pages for the gathered replays.
    discard orc

  # Prefill3 plus decode1. The mixer forward runs end to end over
  # the recorded rope rows, the per-op decode replay using a cache
  # preloaded with the recorded prefill rows.
  block:
    let attnDev = setupMlaDirect[FullRoPe](ModelDir, LayerPrefix, 0, 64)
    var orc = newMlaOrchestrator(64, kvLoraRank, qkRopeHeadDim)
    orc.startSequence(@[1'u32, 2, 3])
    var ctx = orc.getInferenceContextMut()
    var st = Safetensor.open(FixtureDir /
      "attn-Moonlight-16B-A3B-01.safetensor")
    let statsPath = FixtureDir /
      "attn-Moonlight-16B-A3B-01.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill3 + decode1)"

    ctx.cos = st.getTensorOwned("cos")
    ctx.sin = st.getTensorOwned("sin")
    let outPrefill = attnDev(ctx, st.getTensorOwned("hidden_states"))
    assertStats(outPrefill, statsPath, "output", kReduction, depth = 3,
      msg = "prefill3 mixer output")

    orc.setKvPosition(3)
    orc.appendToken(position = 3, token_id = 1000'u32, device = F.kCPU)
    ctx.cos = st.getTensorOwned("cos_step")
    ctx.sin = st.getTensorOwned("sin_step")
    let outStep = attnDev(ctx, st.getTensorOwned("x_step"))
    assertStats(outStep, statsPath, "output_step", kReduction, depth = 3,
      msg = "decode1 mixer output")

    # Decode per-op rows over the preloaded cache. The step write
    # reproduces the recorded cache tensors, then the decompress plus
    # sdpa run over the gathered slabs.
    var (orc2, ctx2) = newMlaCacheCtx(64, kvLoraRank, qkRopeHeadDim,
      @[1'u32, 2, 3, 4, 5, 6, 7, 8])
    let cache = MlaLatentCache.init(kvLoraRank, qkRopeHeadDim, 64,
      kBFloat16, F.kCPU)
    cache.write(ctx2, 0,
      st.getTensorOwned("latent_cached").reshape(
        [1, 4, 1, kvLoraRank]).narrow(1, 0, 3),
      st.getTensorOwned("kpe_cached_interleaved").reshape(
        [1, 4, 1, qkRopeHeadDim]).narrow(1, 0, 3),
      0, 3)
    orc2.setKvPosition(3)

    let xStep = st.getTensorOwned("x_step")
    let q = attnDev.q_proj.forward(xStep).reshape(
      [1, 1, numHeads, qkNopeHeadDim + qkRopeHeadDim])
    let qPeRot = rotateInterleaved(q.narrow(3, qkNopeHeadDim, qkRopeHeadDim),
      st.getTensorOwned("cos_step"), st.getTensorOwned("sin_step"))
    assertStats(F.cat([q.narrow(3, 0, qkNopeHeadDim), qPeRot], 3), statsPath,
      "query_states_step", kElementwise, depth = 2,
      msg = "decode query states")

    let compressed = attnDev.kv_a_proj_with_mqa.forward(xStep)
    let latentNormed = attnDev.kv_a_layernorm.forward(
      compressed.narrow(2, 0, kvLoraRank)).unsqueeze(2)
    assertStats(latentNormed, statsPath, "latent_normed_step", kElementwise,
      msg = "decode latent norm")

    let kPeRot = rotateInterleaved(
      compressed.narrow(2, kvLoraRank, qkRopeHeadDim).unsqueeze(2),
      st.getTensorOwned("cos_step"), st.getTensorOwned("sin_step"))
    cache.write(ctx2, 0, latentNormed, kPeRot, ctx2.kv_position, 1)
    let (gotLatent, gotKpe) = cache.gather(ctx2, 0, 0, ctx2.kv_position + 1)
    assertStats(gotLatent, statsPath, "latent_cached", kElementwise,
      msg = "decode latent cache")
    assertStats(gotKpe, statsPath, "kpe_cached_interleaved", kElementwise,
      depth = 2, msg = "decode kpe cache")

    let expanded = attnDev.kv_b_proj.forward(gotLatent.reshape(
      [1, 4, kvLoraRank])).reshape(
      [1, 4, numHeads, qkNopeHeadDim + vHeadDim])
    let keyStates = F.cat([expanded.narrow(3, 0, qkNopeHeadDim),
      gotKpe.expand([1, 4, numHeads, qkRopeHeadDim])], 3)
    assertStats(keyStates, statsPath, "key_states_step", kElementwise,
      depth = 2, msg = "decode key states")
    assertStats(expanded.narrow(3, qkNopeHeadDim, vHeadDim), statsPath,
      "value_states_step", kReduction, msg = "decode value states")
    discard orc

  # Prefill3 plus a 3-step decode sequence. The mixer forward runs
  # end to end, then the final cache contents compare.
  block:
    let attnDev = setupMlaDirect[FullRoPe](ModelDir, LayerPrefix, 0, 64)
    var orc = newMlaOrchestrator(64, kvLoraRank, qkRopeHeadDim)
    orc.startSequence(@[1'u32, 2, 3])
    var ctx = orc.getInferenceContextMut()
    var st = Safetensor.open(FixtureDir /
      "attn-Moonlight-16B-A3B-02.safetensor")
    let statsPath = FixtureDir /
      "attn-Moonlight-16B-A3B-02.safetensor.stats.json.zst"
    echo "    devices: ", deviceName(F.kCPU), " (prefill3 + 3 decode steps)"

    ctx.cos = st.getTensorOwned("cos")
    ctx.sin = st.getTensorOwned("sin")
    let outPrefill = attnDev(ctx, st.getTensorOwned("hidden_states"))
    assertStats(outPrefill, statsPath, "output", kReduction, depth = 3,
      msg = "prefill3 mixer output")

    for i in 0 ..< 3:
      let position = 3 + i
      orc.setKvPosition(position)
      orc.appendToken(position = position, token_id = uint32(1000 + i),
        device = F.kCPU)
      ctx.cos = st.getTensorOwned("cos_step" & $i)
      ctx.sin = st.getTensorOwned("sin_step" & $i)
      let outStep = attnDev(ctx, st.getTensorOwned("x_step" & $i))
      assertStats(outStep, statsPath, "output_step" & $i, kReduction,
        depth = 3, msg = "decode step " & $i & " mixer output")

    let (finalLatent, finalKpe) = attnDev.cache.gather(
      orc.getInferenceContextMut(), 0, 0, 6)
    assertStats(finalLatent, statsPath, "latent_cached", kElementwise,
      msg = "final latent cache")
    assertStats(finalKpe, statsPath, "kpe_cached_interleaved", kElementwise,
      depth = 2, msg = "final kpe cache")

  # Cross-device variant (device-pair report)
  block:
    let runDev = testDevice()
    echo "    devices: ", deviceName(runDev)
    if runDev != F.kCPU:
      echo "    cross-device replay is out of scope for this suite, skipping on ", deviceName(runDev)

when isMainModule:
  main()
