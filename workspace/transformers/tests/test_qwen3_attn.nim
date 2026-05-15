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
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
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
  FixtureDir = currentSourcePath().parentDir() / "fixtures" / "layers" / "Qwen3-0.6B-layer-8"
  ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B" / "model.safetensors"
  ModelDir = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B"
  ModelName = "Qwen3-0.6B"

privateAccess(Qwen3Model)
privateAccess(RopeGQAttention)
privateAccess(GroupedQueryAttention)


proc setupAttn(): RopeGQAttention =
  var weightsMemFile = memFiles.open(ModelPath, mode = fmRead)
  defer: close(weightsMemFile)

  var weightsSt = safetensors.load(weightsMemFile)
  let qWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_proj.weight")
  let kWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_proj.weight")
  let vWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.v_proj.weight")
  let oWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.o_proj.weight")
  let qNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.q_norm.weight")
  let kNormWeight = weightsSt.getTensorOwned("model.layers.8.self_attn.k_norm.weight")

  let model = loadQwen3ModelRaw(ModelDir, kCPU)
  let rotary = model.rotary

  return RopeGQAttention.init(
    8, "model.layers.8.self_attn", qWeight, kWeight, vWeight, oWeight,
    qNormWeight, kNormWeight,
    model.config.num_attention_heads, model.config.num_key_value_heads,
    model.config.head_dim, rotary, rms_norm_eps = 1e-6
  )


proc main() =

  # ──────────────────────────────────────────────────────────────────────────
  # Invariant: KV cache stores rotated keys
  # ──────────────────────────────────────────────────────────────────────────
  runTest "KV cache stores rotated keys":
    proc(): bool =
      let attn = setupAttn()
      let rotary = attn.rotary
      var ctx = InferenceContext.init(
        num_layers = 28, batch_size = 1,
        kv_heads = 8, max_seq = 4096, head_dim = 128,
        dtype = F.kBFloat16, device = F.kCPU)

      var fixtureMemFile = memFiles.open(
        FixtureDir / &"attn-{ModelName}-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)

      var st = safetensors.load(fixtureMemFile)
      let hiddenStates = st.getTensorOwned("hidden_states")
      let hfPosIds = st.getTensorOwned("position_ids")

      ctx.reset()
      ctx.position_ids = hfPosIds[0]
      let (cos, sin) = rotary.compute(ctx.position_ids)
      let x = hiddenStates[0].unsqueeze(0)
      let _ = attn(ctx, cos, sin, x)

      let cache = ctx.kv_caches[attn.layer_idx]
      let (cachedK, _) = cache.read(8)

      let k = attn.k_proj.forward(x)
      let k_reshaped = k.reshape([x.size(0), x.size(1), attn.gqa_attn.num_kv_head, attn.gqa_attn.head_dim])
      let k_normed = if attn.k_norm.isSome: attn.k_norm.get()(k_reshaped) else: k_reshaped
      let (_, k_rot) = rotary.applyRope(k_normed, k_normed, cos, sin)
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
  runTest "KV cache stores unrotated values":
    proc(): bool =
      let attn = setupAttn()
      var ctx = InferenceContext.init(
        num_layers = 28, batch_size = 1,
        kv_heads = 8, max_seq = 4096, head_dim = 128,
        dtype = F.kBFloat16, device = F.kCPU)
      var fixtureMemFile = memFiles.open(
        FixtureDir / &"attn-{ModelName}-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)

      var st = safetensors.load(fixtureMemFile)
      let hiddenStates = st.getTensorOwned("hidden_states")
      let hfPosIds = st.getTensorOwned("position_ids")

      ctx.reset()
      ctx.position_ids = hfPosIds[0]
      let (cos, sin) = attn.rotary.compute(ctx.position_ids)
      let x = hiddenStates[0].unsqueeze(0)
      let _ = attn(ctx, cos, sin, x)

      let cache = ctx.kv_caches[attn.layer_idx]
      let (_, cachedV) = cache.read(8)

      let v = attn.v_proj.forward(x)
      let v_expected = v.reshape([x.size(0), x.size(1), attn.gqa_attn.num_kv_head, attn.gqa_attn.head_dim]).permute([0, 2, 1, 3])

      let diff = (cachedV - v_expected).abs().max().item(float64)
      echo "  Cached vs raw values: max diff = ", diff

      if diff > 1e-4:
        raise newException(AssertionDefect, "KV cache must store unrotated values")

      echo "  ✅ KV cache stores unrotated values"
      true

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All attention invariant tests passed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
