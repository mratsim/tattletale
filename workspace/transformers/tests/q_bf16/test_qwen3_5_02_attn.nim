# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-attn \
##   --nimcache:nimcache/tests/qwen35-attn \
##   workspace/transformers/tests/q_bf16/test_qwen3_5_02_attn.nim

import
  std/memfiles,
  std/os,
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
  workspace/transformers/src/layers/gated_attn {.all.},
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3.5-0.8B-layer-3"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"
  LayerPrefix = "model.language_model.layers.3.self_attn"

privateAccess(GatedAttention)
privateAccess(GroupedQueryAttention)

proc setupAttn(): GatedAttention =
  ## Load the real layer-3 gated full-attention weights into a GatedAttention.
  var weightsMemFile = memFiles.open(ModelDir / "model.safetensors-00001-of-00001.safetensors", mode = fmRead)
  defer: close(weightsMemFile)
  let cfgJson = (ModelDir / "config.json").parseFile()
  var weightsSt = safetensors.load(weightsMemFile)

  let qProj = Linear.load(weightsSt, cfgJson, LayerPrefix & ".q_proj")
  let kProj = Linear.load(weightsSt, cfgJson, LayerPrefix & ".k_proj")
  let vProj = Linear.load(weightsSt, cfgJson, LayerPrefix & ".v_proj")
  let oProj = Linear.load(weightsSt, cfgJson, LayerPrefix & ".o_proj")
  let qNorm = GemmaRmsNorm.load(weightsSt, cfgJson, LayerPrefix & ".q_norm")
  let kNorm = GemmaRmsNorm.load(weightsSt, cfgJson, LayerPrefix & ".k_norm")

  let rotary = RotaryPositionEmbeddingRef.new(
    256, 8192, 1e7, F.kBFloat16, F.kCPU, rotary_dim = 64)
  result = GatedAttention.init(
    3, LayerPrefix,
    qProj, kProj, vProj, oProj,
    qNorm, kNorm,
    8, 2, 256, rotary
  )

proc openFixture(name: string): (MemFile, Safetensor) =
  ## Open a fixture and load its safetensor. The memfile must stay open while
  ## the Safetensor is in use (zero-copy views into the file).
  let memFile = memFiles.open(FixtureDir / name, mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc newCtx(maxSeq = 512): (InferenceContext, PagePool) =
  ## Fresh InferenceContext with a page pool. The pool ref is returned with
  ## the context so the borrowed pages stay alive for the test's duration.
  var ctx = InferenceContext.init(
    num_layers = 24, batch_size = 1,
    kv_heads = 2, max_seq = maxSeq, head_dim = 256)
  let pool = PagePool.init(
    64, num_layers = 24, kv_heads = 2, head_dim = 256,
    dtype = F.kBFloat16, device = F.kCPU)
  let numPages = ceilDiv(maxSeq, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())
  (ctx, pool)

proc normFixtureTest(caseNum: int, msg: string): bool =
  ## GemmaRMSNorm (1 + w) in isolation, 0.00 vs vendored ground truth.
  var (memFile, st) = openFixture("norm-Qwen3.5-0.8B-0" & $caseNum & ".safetensor")
  defer: close(memFile)
  let x = st.getTensorOwned("input")
  let expected = st.getTensorOwned("output")
  let w = st.getTensorOwned("weight")
  let norm = GemmaRmsNorm.init(w, eps = 1e-6)
  let got = norm.forward(x)
  assertAllClose(got, expected, rtol = 0.0, abstol = 0.0, msg = msg)
  true

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # Gated full attention, prefill seq 8, vs fixture (real layer-3 weights)
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Gated full attention prefill (seq 8) vs fixture":
    proc(): bool =
      let attn = setupAttn()
      var (ctx, pool) = newCtx()
      var (memFile, st) = openFixture("attn-Qwen3.5-0.8B-00.safetensor")
      defer: close(memFile)

      let x = st.getTensorOwned("hidden_states")       # (1, 8, 1024) bf16
      let hfPosIds = st.getTensorOwned("position_ids") # (1, 8) int64
      ctx.position_ids = hfPosIds[0]
      ctx.setRopeForPositions(attn.rotary)

      let output = attn(ctx, x)
      doAssert output.size(0) == 1 and output.size(1) == 8 and output.size(2) == 1024

      # Final layer output: block-level tolerance.
      let expected = st.getTensorOwned("output")
      assertAllClose(output, expected, rtol = 5e-3, abstol = 5e-3,
        msg = "gated attention output mismatch")

      # Intermediates, recomputed through the layer's own components and
      # compared against the fixture's captured values (deterministic ops).
      let seqLen = x.size(1)
      let gqa = attn.gqa_attn
      # q_proj packs [q | gate] per head, so the head axis is 2 * head_dim.
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

      assertAllClose(qNormed, st.getTensorOwned("q_normed"),
        rtol = 0.0, abstol = 0.0, msg = "q_norm mismatch")
      assertAllClose(kNormed, st.getTensorOwned("k_normed"),
        rtol = 0.0, abstol = 0.0, msg = "k_norm mismatch")
      assertAllClose(gate, st.getTensorOwned("gate"),
        rtol = 0.0, abstol = 0.0, msg = "gate mismatch")
      # Fixture q_rot/k_rot are (batch, heads, seq, dim). Nim keeps
      # (batch, seq, heads, dim).
      assertAllClose(qRot, st.getTensorOwned("q_rot").transpose(1, 2),
        rtol = 0.0, abstol = 0.0, msg = "q_rot mismatch")
      assertAllClose(kRot, st.getTensorOwned("k_rot").transpose(1, 2),
        rtol = 0.0, abstol = 0.0, msg = "k_rot mismatch")

      # Pre-o_proj output: SDPA + sigmoid gate. The layer's forward output
      # was already compared above. This asserts the gate application itself.
      let vReshaped = attn.v_proj.forward(x).reshape(
        [1, seqLen, gqa.num_kv_head, gqa.head_dim])
      let kExp = kRot.repeat_interleave(gqa.num_qo_head div gqa.num_kv_head, 2)
      let vExp = vReshaped.repeat_interleave(gqa.num_qo_head div gqa.num_kv_head, 2)
      let attnOut = attn.gqa_attn.forward(qRot, kExp, vExp,
        is_causal = true, enable_gqa = false)
      let attnGated = attnOut * F.sigmoid(gate)
      assertAllClose(attnGated, st.getTensorOwned("attn_output_gated"),
        rtol = 0.0, abstol = 0.0, msg = "attn_output_gated mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Gated full attention, decode single token at position 5
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Gated full attention decode (single token, position 5) vs fixture":
    proc(): bool =
      let attn = setupAttn()
      var (ctx, pool) = newCtx()
      var (memFile, st) = openFixture("attn-Qwen3.5-0.8B-01.safetensor")
      defer: close(memFile)

      let x = st.getTensorOwned("hidden_states")       # (1, 1, 1024)
      let hfPosIds = st.getTensorOwned("position_ids") # (1, 1)
      ctx.position_ids = hfPosIds[0]
      ctx.setRopeForPositions(attn.rotary)

      let output = attn(ctx, x)
      let expected = st.getTensorOwned("output")
      assertAllClose(output, expected, rtol = 5e-3, abstol = 5e-3,
        msg = "gated attention decode output mismatch")

      let seqLen = x.size(1)
      let gqa = attn.gqa_attn
      # q_proj packs [q | gate] per head, so the head axis is 2 * head_dim.
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

      assertAllClose(qNormed, st.getTensorOwned("q_normed"),
        rtol = 0.0, abstol = 0.0, msg = "q_norm mismatch")
      assertAllClose(kNormed, st.getTensorOwned("k_normed"),
        rtol = 0.0, abstol = 0.0, msg = "k_norm mismatch")
      assertAllClose(gate, st.getTensorOwned("gate"),
        rtol = 0.0, abstol = 0.0, msg = "gate mismatch")
      assertAllClose(qRot, st.getTensorOwned("q_rot").transpose(1, 2),
        rtol = 0.0, abstol = 0.0, msg = "q_rot mismatch")
      assertAllClose(kRot, st.getTensorOwned("k_rot").transpose(1, 2),
        rtol = 0.0, abstol = 0.0, msg = "k_rot mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # KV cache stores rotated keys (post-RoPE), matching the fixture's k_rot
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "KV cache stores rotated keys (vs fixture k_rot)":
    proc(): bool =
      let attn = setupAttn()
      var (ctx, pool) = newCtx()
      var (memFile, st) = openFixture("attn-Qwen3.5-0.8B-00.safetensor")
      defer: close(memFile)

      let x = st.getTensorOwned("hidden_states")
      let hfPosIds = st.getTensorOwned("position_ids")
      ctx.position_ids = hfPosIds[0]
      ctx.setRopeForPositions(attn.rotary)
      let _ = attn(ctx, x)

      let totalSeqLen = ctx.position_ids.size(0)
      let page = ctx.pages[0]
      # page.k_view[layer_idx] is (PAGE_SIZE, kv_heads, head_dim) ->
      # (kv_heads, seq, head_dim) matches the fixture k_rot layout.
      let cachedK = page.k_view[attn.layer_idx, 0 ..< totalSeqLen].permute([1, 0, 2]).unsqueeze(0)
      let kRotExpected = st.getTensorOwned("k_rot")   # (1, 2, 8, 256)
      assertAllClose(cachedK, kRotExpected, rtol = 0.0, abstol = 0.0,
        msg = "KV cache must store the rotated keys")

      # Values must NOT be the raw projected keys (rotation must be visible).
      let kRaw = attn.k_proj.forward(x).reshape([1, 8, 2, 256]).permute([0, 2, 1, 3])
      let diffRaw = (cachedK.to(kFloat32) - kRaw.to(kFloat32)).abs().max().item(float64)
      doAssert diffRaw > 0.01, "KV cache must NOT store raw keys"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Multi-page write loop + gather (300 tokens = 1 full + 44 partial)
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Multi-page write loop + gather (300 tokens = 1 full + 44 partial)":
    proc(): bool =
      let attn = setupAttn()
      var (ctx, pool) = newCtx(maxSeq = 4096)

      let seqLen = 300
      let x = F.randn(1, seqLen, 1024, F.tensorOptions(F.kBFloat16, F.kCPU))
      ctx.position_ids = F.arange(seqLen, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(attn.rotary)

      let output = attn(ctx, x)
      doAssert output.size(0) == 1, "batch dim mismatch"
      doAssert output.size(1) == seqLen, "seq len mismatch"
      doAssert output.size(2) == 1024, "hidden dim mismatch"
      doAssert ctx.pages.len >= 2, "Expected >= 2 pages for 300 tokens"

      let k = attn.k_proj.forward(x)
      let kReshaped = k.reshape([1, seqLen, 2, 256])
      let kNormed = attn.k_norm.forward(kReshaped)
      let (_, kRot) = attn.rotary.applyRope(kNormed, kNormed, ctx.cos, ctx.sin)
      let vReshaped = attn.v_proj.forward(x).reshape([1, seqLen, 2, 256])

      let partialCount = seqLen - TokensPerPage

      # Page 0: tokens 0-255 (full page)
      let p0k = ctx.pages[0].k_view[attn.layer_idx, 0 ..< TokensPerPage]
      let p0kExp = kRot[0, 0 ..< TokensPerPage]
      assertAllClose(p0k, p0kExp, rtol = 1e-4, abstol = 1e-4, msg = "Page 0 K mismatch")
      let p0v = ctx.pages[0].v_view[attn.layer_idx, 0 ..< TokensPerPage]
      let p0vExp = vReshaped[0, 0 ..< TokensPerPage]
      assertAllClose(p0v, p0vExp, rtol = 1e-4, abstol = 1e-4, msg = "Page 0 V mismatch")

      # Page 1: tokens 256-299 (partial page, 44 tokens)
      let p1k = ctx.pages[1].k_view[attn.layer_idx, 0 ..< partialCount]
      let p1kExp = kRot[0, TokensPerPage ..< seqLen]
      assertAllClose(p1k, p1kExp, rtol = 1e-4, abstol = 1e-4,
        msg = "Page 1 K mismatch (partial page)")
      let p1v = ctx.pages[1].v_view[attn.layer_idx, 0 ..< partialCount]
      let p1vExp = vReshaped[0, TokensPerPage ..< seqLen]
      assertAllClose(p1v, p1vExp, rtol = 1e-4, abstol = 1e-4,
        msg = "Page 1 V mismatch (partial page)")

      # Verify unwritten page 1 positions (>44) are still zero
      let zeroSlots = TokensPerPage - partialCount
      let zeroK = ctx.pages[1].k_view[attn.layer_idx, partialCount ..< TokensPerPage]
      let zeroV = ctx.pages[1].v_view[attn.layer_idx, partialCount ..< TokensPerPage]
      let zeroExp = F.zeros(zeroSlots, 2, 256, F.tensorOptions(F.kBFloat16, F.kCPU))
      assertAllClose(zeroK, zeroExp, rtol = 1e-6, abstol = 1e-6,
        msg = "Page 1 unwritten slots should be zero (K)")
      assertAllClose(zeroV, zeroExp, rtol = 1e-6, abstol = 1e-6,
        msg = "Page 1 unwritten slots should be zero (V)")
      true

  runCppTest "GemmaRmsNorm 1+w head_dim forward vs fixture":
    proc(): bool = normFixtureTest(0, "GemmaRmsNorm head_dim_forward mismatch")
  runCppTest "GemmaRmsNorm 1+w single token vs fixture":
    proc(): bool = normFixtureTest(1, "GemmaRmsNorm single_token mismatch")
  runCppTest "GemmaRmsNorm 1+w zeros input (eps guard) vs fixture":
    proc(): bool = normFixtureTest(2, "GemmaRmsNorm zeros_input mismatch")

  echo "\nAll Qwen3.5 gated-attention tests passed!"

when isMainModule:
  main()
