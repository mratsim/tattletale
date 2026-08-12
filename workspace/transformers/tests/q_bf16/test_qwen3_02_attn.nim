# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/memfiles,
  std/os,
  std/options,
  std/strformat,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/layers/attn {.all.},
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

# =============================================================================
# Attention Invariants
# =============================================================================
#
# INVARIANT: KV cache stores ROTATED keys (post-RoPE), not raw keys.
#
# Grounding: RoPE (Su et al., 2021) defines attention as operating on R_θ(q) and
# R_θ(k). The rotation must be applied consistently whether the token arrives
# via prefill (fresh projection + rotation) or decode (read from cache).
# Storing unrotated keys would mean the cache serves different keys than
# what was used during training → broken positional consistency.
#
# Convention: HF transformers, vLLM, SGLang, TGI, llama.cpp — all cache
# post-RoPE keys. This is the universal convention, not an arbitrary choice.
#
# Blast radius: If cache were to store unrotated keys, every downstream
# computation (attention output, MLP, full model) would diverge from HF
# starting at position 1. Position 0 would still match (self-attention is
# identity for the first token regardless of key rotation).
# =============================================================================

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3-0.6B-layer-8"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B" / "model.safetensors"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"
  ModelName = "Qwen3-0.6B"

privateAccess(Qwen3Model)
privateAccess(RopeGQAttention)
privateAccess(GroupedQueryAttention)


proc setupAttn(): RopeGQAttention =
  var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
  defer: close(weightsMemFile)

  let cfgJson = (ModelDir / "config.json").parseFile()
  var weightsSt = safetensors.load(weightsMemFile)

  let qProj = Linear.load(weightsSt, cfgJson, "model.layers.8.self_attn.q_proj")
  let kProj = Linear.load(weightsSt, cfgJson, "model.layers.8.self_attn.k_proj")
  let vProj = Linear.load(weightsSt, cfgJson, "model.layers.8.self_attn.v_proj")
  let oProj = Linear.load(weightsSt, cfgJson, "model.layers.8.self_attn.o_proj")
  let qNorm = RmsNorm.load(weightsSt, cfgJson, "model.layers.8.self_attn.q_norm")
  let kNorm = RmsNorm.load(weightsSt, cfgJson, "model.layers.8.self_attn.k_norm")

  let model = loadQwen3ModelRaw(ModelDir, kCPU)
  let rotary = model.rotary

  return RopeGQAttention.init(
    8, "model.layers.8.self_attn",
    qProj, kProj, vProj, oProj,
    qNorm, kNorm,
    model.config.num_attention_heads, model.config.num_key_value_heads,
    model.config.head_dim, rotary
  )


proc main() =

  # ──────────────────────────────────────────────────────────────────────────
  # Invariant: KV cache stores rotated keys
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "KV cache stores rotated keys":
    proc(): bool =
      let attn = setupAttn()
      let rotary = attn.rotary
      var ctx = InferenceContext.init(
        num_layers = 28, batch_size = 1,
        kv_heads = 8, max_seq = 4096, head_dim = 128)
      # Pool and pages setup BEFORE reset,
      # reset no longer clears them (it only resets non-KV state).

      var fixtureMemFile = memFiles.open(
        FixtureDir / &"attn-{ModelName}-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)

      var st = safetensors.load(fixtureMemFile)
      let hiddenStates = st.getTensorOwned("hidden_states")
      let hfPosIds = st.getTensorOwned("position_ids")

      let pool = PagePool.init(
        64, num_layers = 28, kv_heads = 8, head_dim = 128,
        dtype = F.kBFloat16, device = F.kCPU)
      let numPages = ceilDiv(4096, TokensPerPage)
      for i in 0 ..< numPages:
        ctx.pages.add(pool.borrow())

      ctx.position_ids = hfPosIds[0]
      ctx.setRopeForPositions(rotary)
      let x = hiddenStates[0].unsqueeze(0)
      let _ = attn(ctx, x)

      let totalSeqLen = ctx.position_ids.size(0)
      let page = ctx.pages[0]
      let cachedK = page.k_view[attn.layer_idx, 0 ..< totalSeqLen].permute([1, 0, 2]).unsqueeze(0)

      let k = attn.k_proj.forward(x)
      let k_reshaped = k.reshape([x.size(0), x.size(1), attn.gqa_attn.num_kv_head, attn.gqa_attn.head_dim])
      let k_normed = if attn.k_norm.isSome: attn.k_norm.get()(k_reshaped) else: k_reshaped
      let (_, k_rot) = rotary.applyRope(k_normed, k_normed, ctx.cos, ctx.sin)
      let k_rot_expected = k_rot.permute([0, 2, 1, 3])

      let k_raw = k_reshaped.permute([0, 2, 1, 3])
      let diffRotated = (cachedK - k_rot_expected).abs().max().item(float64)
      let diffRaw = (cachedK - k_raw).abs().max().item(float64)

      echo "  Cached vs rotated keys: max diff = ", diffRotated
      echo "  Cached vs raw keys:     max diff = ", diffRaw

      if diffRotated > 1e-4:
        raise newException(AssertionDefect, "KV cache must store rotated keys")
      if diffRaw < 0.01:
        raise newException(AssertionDefect, "KV cache must NOT store raw keys")

      echo "  ✅ KV cache stores rotated keys"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Invariant: KV cache stores unrotated values
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "KV cache stores unrotated values":
    proc(): bool =
      let attn = setupAttn()
      var ctx = InferenceContext.init(
        num_layers = 28, batch_size = 1,
        kv_heads = 8, max_seq = 4096, head_dim = 128)

      let pool = PagePool.init(
        64, num_layers = 28, kv_heads = 8, head_dim = 128,
        dtype = F.kBFloat16, device = F.kCPU)
      let numPages = ceilDiv(4096, TokensPerPage)
      for i in 0 ..< numPages:
        ctx.pages.add(pool.borrow())

      var fixtureMemFile = memFiles.open(
        FixtureDir / &"attn-{ModelName}-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)

      var st = safetensors.load(fixtureMemFile)
      let hiddenStates = st.getTensorOwned("hidden_states")
      let hfPosIds = st.getTensorOwned("position_ids")

      # No ctx.clearState() — context is freshly created above
      ctx.position_ids = hfPosIds[0]
      ctx.setRopeForPositions(attn.rotary)
      let x = hiddenStates[0].unsqueeze(0)
      let _ = attn(ctx, x)

      let totalSeqLen = ctx.position_ids.size(0)
      let page = ctx.pages[0]
      let cachedV = page.v_view[attn.layer_idx, 0 ..< totalSeqLen].permute([1, 0, 2]).unsqueeze(0)

      let v = attn.v_proj.forward(x)
      let v_expected = v.reshape([x.size(0), x.size(1), attn.gqa_attn.num_kv_head, attn.gqa_attn.head_dim]).permute([0, 2, 1, 3])

      let diff = (cachedV - v_expected).abs().max().item(float64)
      echo "  Cached vs raw values: max diff = ", diff

      if diff > 1e-4:
        raise newException(AssertionDefect, "KV cache must store unrotated values")

      echo "  ✅ KV cache stores unrotated values"
      true


  # ────────────────────────────────────────────────────────────────────────
  # Multi-page attention write loop + gather
  # ────────────────────────────────────────────────────────────────────────
  runCppTest "Multi-page write loop + gather (300 tokens = 1 full + 44 partial)":
    proc(): bool =
      let attn = setupAttn()
      var ctx = InferenceContext.init(
        num_layers = 28, batch_size = 1,
        kv_heads = 8, max_seq = 4096, head_dim = 128)

      let pool = PagePool.init(
        64, num_layers = 28, kv_heads = 8, head_dim = 128,
        dtype = F.kBFloat16, device = F.kCPU)
      let numPages = ceilDiv(4096, TokensPerPage)
      for i in 0 ..< numPages:
        ctx.pages.add(pool.borrow())

      let seqLen = 300
      let x = F.randn(1, seqLen, 1024, F.tensorOptions(F.kBFloat16, F.kCPU))

      ctx.position_ids = F.arange(seqLen, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(attn.rotary)

      let output = attn(ctx, x)
      doAssert output.size(0) == 1, "batch dim mismatch"
      doAssert output.size(1) == seqLen, "seq len mismatch"
      doAssert output.size(2) == 1024, "hidden dim mismatch"
      doAssert ctx.pages.len >= 2,
        "Expected >= 2 pages for 300 tokens, got " & $ctx.pages.len

      # Compute expected k_rot and v_reshaped
      let k = attn.k_proj.forward(x)
      let k_reshaped = k.reshape([x.size(0), x.size(1),
        attn.gqa_attn.num_kv_head, attn.gqa_attn.head_dim])
      let k_normed = if attn.k_norm.isSome: attn.k_norm.get()(k_reshaped)
                     else: k_reshaped
      let (_, k_rot) = attn.rotary.applyRope(k_normed, k_normed,
        ctx.cos, ctx.sin)

      let v = attn.v_proj.forward(x)
      let v_reshaped = v.reshape([x.size(0), x.size(1),
        attn.gqa_attn.num_kv_head, attn.gqa_attn.head_dim])

      let partialCount = seqLen - TokensPerPage

      # Page 0: tokens 0-255 (full page)
      let p0k = ctx.pages[0].k_view[attn.layer_idx, 0 ..< TokensPerPage]
      let p0kExp = k_rot[0, 0 ..< TokensPerPage]
      assertAllClose(p0k, p0kExp, rtol = 1e-4, abstol = 1e-4,
        msg = "Page 0 K mismatch")

      let p0v = ctx.pages[0].v_view[attn.layer_idx, 0 ..< TokensPerPage]
      let p0vExp = v_reshaped[0, 0 ..< TokensPerPage]
      assertAllClose(p0v, p0vExp, rtol = 1e-4, abstol = 1e-4,
        msg = "Page 0 V mismatch")

      # Page 1: tokens 256-299 (partial page, 44 tokens)
      let p1k = ctx.pages[1].k_view[attn.layer_idx, 0 ..< partialCount]
      let p1kExp = k_rot[0, TokensPerPage ..< seqLen]
      assertAllClose(p1k, p1kExp, rtol = 1e-4, abstol = 1e-4,
        msg = "Page 1 K mismatch (partial page)")

      let p1v = ctx.pages[1].v_view[attn.layer_idx, 0 ..< partialCount]
      let p1vExp = v_reshaped[0, TokensPerPage ..< seqLen]
      assertAllClose(p1v, p1vExp, rtol = 1e-4, abstol = 1e-4,
        msg = "Page 1 V mismatch (partial page)")

      # Verify unwritten page 1 positions (>44) are still zero
      let zeroSlots = TokensPerPage - partialCount
      let zeroK = ctx.pages[1].k_view[attn.layer_idx, partialCount ..< TokensPerPage]
      let zeroV = ctx.pages[1].v_view[attn.layer_idx, partialCount ..< TokensPerPage]
      let zeroExp = F.zeros(zeroSlots, attn.gqa_attn.num_kv_head,
        attn.gqa_attn.head_dim, F.tensorOptions(F.kBFloat16, F.kCPU))
      assertAllClose(zeroK, zeroExp, rtol = 1e-6, abstol = 1e-6,
        msg = "Page 1 unwritten slots should be zero (K)")
      assertAllClose(zeroV, zeroExp, rtol = 1e-6, abstol = 1e-6,
        msg = "Page 1 unwritten slots should be zero (V)")

      true

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All attention tests completed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
