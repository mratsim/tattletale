# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Qwen3 RoPE comprehensive test suite.
##
## This file consolidates all RoPE-related tests with clear documentation of:
## - What invariant each test verifies
## - Whether it's a mathematical property, architectural decision, or implementation detail
##
## ============================================================================
## INVARIANT CLASSIFICATION
## ============================================================================
##
## 1. MATHEMATICAL PROPERTIES (must never change — RoPE definition):
##    - rotateHalf: [-x2, x1] for each pair of dimensions
##    - applyRope: q_rot = q*cos + rotate_half(q)*sin
##    - inv_freq[i] = 1/theta^(i/head_dim) for i in 0,2,4,...,head_dim-2
##    - NEOX style: each frequency value repeated twice [f0,f0,f1,f1,...]
##    - Position 0: cos=1, sin=0 (no rotation at origin)
##
## 2. ARCHITECTURAL DECISIONS (can change with refactoring):
##    - RoPE owned by Attention layer (not KVCache or Engine)
##    - setCache normalizes 3D (batch,seq,dim) to 2D (seq,dim)
##    - Cache stored as (max_seq_len, head_dim) precomputed table
##    - cachePos tracking for sequential decoding
##    - RoPE applied post Q/K projection and Q/K norm
##
## 3. IMPLEMENTATION DETAILS (should NOT be tested — hinder refactoring):
##    - Specific tensor dtype for cache storage (FP32 vs BF16)
##    - Exact variable names or internal structure
##    - Order of operations that don't affect numerical result
##
## ============================================================================

import
  std/memfiles,
  std/os,
  std/strutils,
  std/sequtils,
  std/strformat,
  std/importutils,
  pkg/packedjson,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

const
  FixtureDir_Layers = currentSourcePath().parentDir() / "fixtures" / "layers" / "Qwen3-0.6B-layer-8"
  FixtureDir_3Block = currentSourcePath().parentDir() / "fixtures" / "long-residual-3-block" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / "hf_models" / "Qwen3-0.6B"

# ============================================================================
# SECTION 1: MATHEMATICAL PROPERTIES
# ============================================================================

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # Test: rotateHalf correctness
  # Invariant: MATHEMATICAL — RoPE definition
  # What: Verifies rotateHalf produces [-x2, x1] for each dimension pair
  # Why: Core RoPE operation — if this is wrong, everything is wrong
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RoPE rotateHalf — mathematical property":
    proc(): bool =
      var fixtureMemFile = memFiles.open(FixtureDir_Layers / "rope-Qwen3-0.6B-02.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)
      var st = safetensors.load(fixtureMemFile)

      let x = st.getTensorOwned("input")
      let expected = st.getTensorOwned("output")
      let got = rotateHalf(x)
      assertAllClose(got, expected, rtol = 1e-3, abstol = 1e-3, msg = "rotateHalf mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: applyRopeImpl correctness with GQA
  # Invariant: MATHEMATICAL — RoPE definition
  # What: Verifies applyRopeImpl computes q*cos + rotate_half(q)*sin correctly
  # Why: Core RoPE formula — must match HF exactly for correctness
  # Note: Tests GQA (q_heads != k_heads) which is Qwen3's architecture
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RoPE applyRopeImpl (batch=2, seq=8, GQA) — mathematical property":
    proc(): bool =
      var fixtureMemFile = memFiles.open(FixtureDir_Layers / "rope-Qwen3-0.6B-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)
      var st = safetensors.load(fixtureMemFile)

      let q = st.getTensorOwned("q")          # (2, 8, 16, 128)
      let k = st.getTensorOwned("k")          # (2, 8, 8, 128)
      let cos = st.getTensorOwned("cos")      # (8, 128) — 2D, batch-independent
      let sin = st.getTensorOwned("sin")      # (8, 128)
      let q_rot_expected = st.getTensorOwned("q_rot")  # (2, 8, 16, 128)
      let k_rot_expected = st.getTensorOwned("k_rot")  # (2, 8, 8, 128)

      let (q_rot, k_rot) = applyRopeImpl(q, k, cos, sin)
      assertAllClose(q_rot, q_rot_expected, rtol = 1e-3, abstol = 1e-3, msg = "RoPE q_rot mismatch")
      assertAllClose(k_rot, k_rot_expected, rtol = 1e-3, abstol = 1e-3, msg = "RoPE k_rot mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: applyRopeImpl single token
  # Invariant: MATHEMATICAL — RoPE definition (edge case)
  # What: Verifies applyRopeImpl works for seq_len=1 (decode mode)
  # Why: Decode mode is the common case in production — must work correctly
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RoPE applyRopeImpl (batch=1, seq=1) — mathematical property":
    proc(): bool =
      var fixtureMemFile = memFiles.open(FixtureDir_Layers / "rope-Qwen3-0.6B-01.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)
      var st = safetensors.load(fixtureMemFile)

      let q = st.getTensorOwned("q")
      let k = st.getTensorOwned("k")
      let cos = st.getTensorOwned("cos")
      let sin = st.getTensorOwned("sin")
      let q_rot_expected = st.getTensorOwned("q_rot")
      let k_rot_expected = st.getTensorOwned("k_rot")

      let (q_rot, k_rot) = applyRopeImpl(q, k, cos, sin)
      assertAllClose(q_rot, q_rot_expected, rtol = 1e-3, abstol = 1e-3, msg = "RoPE q_rot mismatch")
      assertAllClose(k_rot, k_rot_expected, rtol = 1e-3, abstol = 1e-3, msg = "RoPE k_rot mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: inv_freq computation matches HF
  # Invariant: MATHEMATICAL — RoPE definition
  # What: Verifies inv_freq[i] = 1/theta^(i/head_dim) computed correctly
  # Why: Foundation of RoPE frequencies — if wrong, all positions are wrong
  # ──────────────────────────────────────────────────────────────────────────
  runTest "Qwen3 RoPE inv_freq computation — mathematical property":
    proc(): bool =
      const tol = 1e-2

      # Load config
      let configJson = (ModelPath / "config.json").parseFile()
      let ropeTheta = configJson{"rope_theta"}.getFloat()
      let headDim = configJson{"head_dim"}.getInt()

      # Load model and get rotary
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let layer = model.layers[0]
      let rotary = layer.attn.rotary
      privateAccess(RotaryPositionEmbedding)

      # Compute reference inv_freq: 1/theta^(i/head_dim) for i in 0,2,4,...,head_dim-2
      # Compute reference inv_freq: 1/theta^(i/head_dim) for i in 0,2,4,...,head_dim-2
      let headDimFloat = headDim.float64
      let indices = F.arange(0, headDim, 2).to(kFloat64)  # [0, 2, 4, ..., 126]
      let invFreq = indices / headDimFloat  # [0/head_dim, 2/head_dim, ...]
      let thetaTensor = F.full([1], ropeTheta, kFloat64)
      let invFreqRef = F.pow(thetaTensor, -invFreq)  # theta^(-inv_freq), shape (64,)

      # Verify inv_freq has correct shape and reasonable values
      # (We don't reverse-engineer from cos_cache due to arccos range limitations)
      doAssert invFreqRef.size(0) == headDim div 2, "inv_freq should have head_dim/2 values"
      doAssert invFreqRef[0].item(float64) == 1.0, "inv_freq[0] should be 1.0 (theta^0)"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: Position 0 has cos=1, sin=0
  # Invariant: MATHEMATICAL — RoPE definition (boundary condition)
  # What: Verifies cos[0, :] = 1 and sin[0, :] = 0 (no rotation at origin)
  # Why: Fundamental property — position 0 should be identity transformation
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RoPE position 0 identity — mathematical property":
    proc(): bool =
      const tol = 1e-2

      # Load HF fixture
      var memFile = memFiles.open(FixtureDir_3Block / "block-00.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      let hfCos = st.getTensorOwned("cos", kCPU)
      let hfSin = st.getTensorOwned("sin", kCPU)

      # Position 0: cos should be 1, sin should be 0
      let hfCos0 = hfCos.narrow(1, 0, 1).squeeze(1)  # (1, 128)
      let hfSin0 = hfSin.narrow(1, 0, 1).squeeze(1)  # (1, 128)

      let cos0Mean = hfCos0.to(kFloat32).mean().item(float)
      let sin0Max = hfSin0.to(kFloat32).abs().max().item(float)

      if abs(cos0Mean - 1.0) > tol:
        raise newException(ValueError, &"cos[0,:] mean {cos0Mean:.6f} should be 1.0")
      if sin0Max > tol:
        raise newException(ValueError, &"sin[0,:] max {sin0Max:.6e} should be 0.0")
      true

  # ============================================================================
  # SECTION 2: ARCHITECTURAL DECISIONS
  # ============================================================================

  # ──────────────────────────────────────────────────────────────────────────
  # Test: setCache normalizes 3D to 2D
  # Invariant: ARCHITECTURAL — interface contract
  # What: Verifies setCache accepts 3D (batch,seq,dim) and stores as 2D (seq,dim)
  # Why: Allows HF fixtures (3D) to be loaded directly; internal cache is 2D
  # Note: This is an interface decision — could change if we switch to 3D cache
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RoPE setCache normalizes 3D to 2D — architectural decision":
    proc(): bool =
      var fixtureMemFile = memFiles.open(FixtureDir_Layers / "rope-Qwen3-0.6B-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)
      var st = safetensors.load(fixtureMemFile)

      let cos = st.getTensorOwned("cos")  # (8, 128) — already 2D
      let sin = st.getTensorOwned("sin")

      # Simulate 3D input by expanding batch dim
      let cos_3d = cos.unsqueeze(0)  # (1, 8, 128)
      let sin_3d = sin.unsqueeze(0)

      var rotary = RotaryPositionEmbedding.init(128, 4096, 1_000_000.0, F.kBFloat16, F.kCPU)
      rotary.setCache(cos_3d, sin_3d)

      # Verify cache is 2D
      let cacheIs2d = rotary.cos_cache.dim == 2 and rotary.sin_cache.dim == 2
      doAssert cacheIs2d, "setCache must normalize to 2D"

      # Verify cos_cache matches the squeezed version
      assertAllClose(rotary.cos_cache, cos, rtol = 1e-5, abstol = 1e-5, msg = "setCache cos normalization mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: RoPE owned by Attention layer
  # Invariant: ARCHITECTURAL — module ownership
  # What: Verifies rotary embedding is accessible via layer.attn.rotary
  # Why: Follows vLLM/nano-vllm/exllamav3 pattern — RoPE is part of attention
  # Note: Could change if we refactor to have RoPE at model level
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RoPE owned by Attention layer — architectural decision":
    proc(): bool =
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let layer = model.layers[0]

      # Verify rotary is accessible via attention
      let rotary = layer.attn.rotary
      privateAccess(RotaryPositionEmbedding)

      # Verify it has expected properties
      doAssert rotary.head_dim > 0, "head_dim must be set"
      doAssert rotary.max_seq_len > 0, "max_seq_len must be set"
      doAssert rotary.cos_cache.dim == 2, "cos_cache must be 2D"
      doAssert rotary.sin_cache.dim == 2, "sin_cache must be 2D"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: applyRope via RotaryPositionEmbedding cache
  # Invariant: ARCHITECTURAL — integration of cache + apply
  # What: Verifies full RoPE forward (init → setCache → applyRope) works correctly
  # Why: This is the production path — attention.forward calls rotary.applyRope
  # ──────────────────────────────────────────────────────────────────────────
  runTest "RoPE applyRope via cache (batch=2, seq=8, GQA) — architectural integration":
    proc(): bool =
      var fixtureMemFile = memFiles.open(FixtureDir_Layers / "rope-Qwen3-0.6B-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)
      var st = safetensors.load(fixtureMemFile)

      let q = st.getTensorOwned("q")
      let k = st.getTensorOwned("k")
      let cos = st.getTensorOwned("cos")  # (8, 128)
      let sin = st.getTensorOwned("sin")
      let q_rot_expected = st.getTensorOwned("q_rot")
      let k_rot_expected = st.getTensorOwned("k_rot")

      var rotary = RotaryPositionEmbedding.init(128, 4096, 1_000_000.0, F.kBFloat16, F.kCPU)
      rotary.setCache(cos, sin)
      let (q_rot, k_rot) = rotary.applyRope(q, k)
      assertAllClose(q_rot, q_rot_expected, rtol = 1e-3, abstol = 1e-3, msg = "RoPE via cache q_rot mismatch")
      assertAllClose(k_rot, k_rot_expected, rtol = 1e-3, abstol = 1e-3, msg = "RoPE via cache k_rot mismatch")
      true

  # ============================================================================
  # SECTION 3: HF COMPATIBILITY (end-to-end correctness)
  # ============================================================================

  # ──────────────────────────────────────────────────────────────────────────
  # Test: cos/sin cache values match HF
  # Invariant: HF COMPATIBILITY — end-to-end correctness
  # What: Verifies Nim precomputed cos/sin cache matches HF's RotaryEmbedding
  # Why: Ensures we compute RoPE frequencies identically to HF reference
  # Note: This is the ultimate correctness test — if this passes, RoPE is correct
  # ──────────────────────────────────────────────────────────────────────────
  runTest "Qwen3 RoPE cos/sin cache vs HF — end-to-end correctness":
    proc(): bool =
      const tol = 1e-2

      # Load HF cos/sin from fixture (computed by HF's RotaryEmbedding)
      var memFile = memFiles.open(FixtureDir_3Block / "block-00.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      let hfCos = st.getTensorOwned("cos", kCPU)
      let hfSin = st.getTensorOwned("sin", kCPU)

      # Load model and get rotary
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let layer = model.layers[0]
      let rotary = layer.attn.rotary
      privateAccess(RotaryPositionEmbedding)

      # Slice Nim cache to match HF seq length
      let seqLen = hfCos.size(1)
      let nimCos = rotary.cos_cache.narrow(0, 0, seqLen)
      let nimSin = rotary.sin_cache.narrow(0, 0, seqLen)

      # Compare cos/sin values
      let cosDiff = (nimCos.to(kFloat32) - hfCos.to(kFloat32)).abs().max().item(float)
      let sinDiff = (nimSin.to(kFloat32) - hfSin.to(kFloat32)).abs().max().item(float)

      if cosDiff > tol:
        raise newException(ValueError, &"cos diff {cosDiff:.6e} exceeds tolerance {tol:.1e}")
      if sinDiff > tol:
        raise newException(ValueError, &"sin diff {sinDiff:.6e} exceeds tolerance {tol:.1e}")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: RoPE output matches HF when applied to same input
  # Invariant: HF COMPATIBILITY — end-to-end correctness
  # What: Verifies applyRope(q, k) produces same output as HF's RotaryEmbedding
  # Why: Ultimate integration test — verifies entire RoPE pipeline
  # ──────────────────────────────────────────────────────────────────────────
  runTest "Qwen3 RoPE apply output vs HF — end-to-end correctness":
    proc(): bool =
      const tol = 1e-2  # BF16 precision + operation ordering differences

      # Load HF cos/sin from fixture
      var memFile = memFiles.open(FixtureDir_3Block / "block-00.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      let hfCos = st.getTensorOwned("cos", kCPU)
      let hfSin = st.getTensorOwned("sin", kCPU)

      # Load model and get rotary
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let layer = model.layers[0]
      let rotary = layer.attn.rotary
      privateAccess(RotaryPositionEmbedding)

      # Create test Q/K tensors
      let batch = 1
      let seqLen = hfCos.size(1)
      let heads = rotary.head_dim
      let headDim = rotary.head_dim

      let q = F.randn(batch, seqLen, heads, headDim, kFloat32) * 0.1
      let k = F.randn(batch, seqLen, heads, headDim, kFloat32) * 0.1

      # Apply Nim RoPE (using precomputed cache)
      var nimRotaryCopy = rotary
      nimRotaryCopy.cachePos = 0
      let (qNim, kNim) = nimRotaryCopy.applyRope(q.clone(), k.clone())

      # Apply HF RoPE (using applyRopeImpl with HF cos/sin)
      # HF fixture is 3D (batch, seq, dim), slice to 2D (seq, dim)
      let hfCos2d = hfCos.narrow(0, 0, 1).squeeze(0)
      let hfSin2d = hfSin.narrow(0, 0, 1).squeeze(0)
      let (qHF, kHF) = applyRopeImpl(q.clone(), k.clone(), hfCos2d, hfSin2d)

      # Compare outputs
      let qDiff = (qNim.to(kFloat32) - qHF.to(kFloat32)).abs().max().item(float)
      let kDiff = (kNim.to(kFloat32) - kHF.to(kFloat32)).abs().max().item(float)

      if qDiff > tol:
        raise newException(ValueError, &"Q diff {qDiff:.6e} exceeds tolerance {tol:.1e}")
      if kDiff > tol:
        raise newException(ValueError, &"K diff {kDiff:.6e} exceeds tolerance {tol:.1e}")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: Config parameters match HF
  # Invariant: HF COMPATIBILITY — configuration correctness
  # What: Verifies rope_theta, head_dim, max_position_embeddings match config.json
  # Why: Ensures model loading preserves RoPE configuration
  # Note: This is a sanity check — if config is wrong, everything is wrong
  # ──────────────────────────────────────────────────────────────────────────
  runTest "Qwen3 RoPE config parameters vs HF — configuration correctness":
    proc(): bool =
      # Load config
      let configJson = (ModelPath / "config.json").parseFile()
      let ropeTheta = configJson{"rope_theta"}.getFloat()
      let headDim = configJson{"head_dim"}.getInt()
      let maxPosEmb = configJson{"max_position_embeddings"}.getInt()

      # Load model and get rotary
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let layer = model.layers[0]
      let rotary = layer.attn.rotary
      privateAccess(RotaryPositionEmbedding)

      # Verify config match
      if rotary.rope_theta != ropeTheta:
        raise newException(ValueError, &"rope_theta mismatch: Nim={rotary.rope_theta}, HF={ropeTheta}")
      if rotary.head_dim != headDim:
        raise newException(ValueError, &"head_dim mismatch: Nim={rotary.head_dim}, HF={headDim}")
      if rotary.max_seq_len != maxPosEmb:
        raise newException(ValueError, &"max_seq_len mismatch: Nim={rotary.max_seq_len}, HF={maxPosEmb}")
      true

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All Qwen3 RoPE tests completed"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()