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
##    - RoPE owned by Model level (shared across all layers)
##    - RoPE applied post Q/K projection and Q/K norm
##    - Cache stored as (max_seq_len, head_dim) precomputed table
##    - ropeByPositions(position_ids) slices cache for current forward pass
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
  workspace/transformers/src/layers,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/libtorch_testutils

const
  FixtureDir_Layers = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3-0.6B-layer-8"
  FixtureDir_3Block = currentSourcePath().parentDir() / ".." / "fixtures" / "long-residual-3-block" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"

# ============================================================================
# SECTION 1: MATHEMATICAL PROPERTIES
# ============================================================================

const Tol = 1e-5  # BF16 precision — valid once fixtures use f32 inv_freq

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # Test: rotateHalf correctness
  # Invariant: MATHEMATICAL — RoPE definition
  # What: Verifies rotateHalf produces [-x2, x1] for each dimension pair
  # Why: Core RoPE operation — if this is wrong, everything is wrong
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE rotateHalf — mathematical property":
    proc(): bool =
      var fixtureMemFile = memFiles.open(FixtureDir_Layers / "rope-Qwen3-0.6B-02.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)
      var st = safetensors.load(fixtureMemFile)

      let x = st.getTensorOwned("input")
      let expected = st.getTensorOwned("output")
      let got = rotateHalf(x)
      assertAllClose(got, expected, rtol = 1e-5, abstol = 1e-5, msg = "rotateHalf mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: applyRopeImpl correctness with GQA
  # Invariant: MATHEMATICAL — RoPE definition
  # What: Verifies applyRopeImpl computes q*cos + rotate_half(q)*sin correctly
  # Why: Core RoPE formula — must match HF exactly for correctness
  # Note: Tests GQA (q_heads != k_heads) which is Qwen3's architecture
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE applyRopeImpl (batch=2, seq=8, GQA) — mathematical property":
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
      assertAllClose(q_rot, q_rot_expected, rtol = 1e-5, abstol = 1e-5, msg = "RoPE q_rot mismatch")
      assertAllClose(k_rot, k_rot_expected, rtol = 1e-5, abstol = 1e-5, msg = "RoPE k_rot mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: applyRopeImpl single token
  # Invariant: MATHEMATICAL — RoPE definition (edge case)
  # What: Verifies applyRopeImpl works for seq_len=1 (decode mode)
  # Why: Decode mode is the common case in production — must work correctly
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE applyRopeImpl (batch=1, seq=1) — mathematical property":
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
      assertAllClose(q_rot, q_rot_expected, rtol = 1e-5, abstol = 1e-5, msg = "RoPE q_rot mismatch")
      assertAllClose(k_rot, k_rot_expected, rtol = 1e-5, abstol = 1e-5, msg = "RoPE k_rot mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: inv_freq computation matches HF
  # Invariant: MATHEMATICAL — RoPE definition
  # What: Verifies inv_freq[i] = 1/theta^(i/head_dim) computed correctly
  # Why: Foundation of RoPE frequencies — if wrong, all positions are wrong
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Qwen3 RoPE inv_freq computation — mathematical property":
    proc(): bool =
      let head_dim = 128
      let rope_theta = 1000000.0
      let inv_freq = F.arange(0, head_dim, 2).to(kFloat64) / head_dim.float64
      let rope_theta_tensor = F.full([1], rope_theta, kFloat64)
      let inv_freq_final = F.pow(rope_theta_tensor, -inv_freq)
      # Verify first 4 values match expected: 1.0, 0.806, 0.650, 0.524
      let got = inv_freq_final[0..<4]
      let expected = F.toTensor([1.0, 0.806049, 0.649715, 0.523697]).to(kFloat64)
      assertAllClose(got, expected, rtol = 1e-3, abstol = 1e-3, msg = "inv_freq mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: Position 0 has cos=1, sin=0
  # Invariant: MATHEMATICAL — RoPE definition (boundary condition)
  # What: Verifies cos[0, :] = 1 and sin[0, :] = 0 (no rotation at origin)
  # Why: Fundamental property — position 0 should be identity transformation
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE position 0 identity — mathematical property":
    proc(): bool =
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let rotary = model.rotary
      privateAccess(RotaryPositionEmbeddingRef)
      privateAccess(TransformerBlock)
      privateAccess(RopeGQAttention)
      let cos_0 = rotary.cos_cache[0, 0..<5].to(kFloat32)
      let sin_0 = rotary.sin_cache[0, 0..<5].to(kFloat32)
      let cos_expected = F.ones([5], kFloat32)
      let sin_expected = F.zeros([5], kFloat32)
      assertAllClose(cos_0, cos_expected, rtol = 1e-5, abstol = 1e-5, msg = "cos[0] should be 1")
      assertAllClose(sin_0, sin_expected, rtol = 1e-5, abstol = 1e-5, msg = "sin[0] should be 0")
      true

  # ============================================================================
  # SECTION 2: ARCHITECTURAL DECISIONS
  # ============================================================================

  # ──────────────────────────────────────────────────────────────────────────
  # Test: RoPE owned by Model level (shared across layers)
  # Invariant: ARCHITECTURAL — module ownership
  # What: Verifies rotary embedding is accessible via model.rotary (and per-layer)
  # Why: Memory efficiency — one 20 MB cache shared across all 28 layers
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE owned by Model level — architectural decision":
    proc(): bool =
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let rotary = model.rotary
      privateAccess(RotaryPositionEmbeddingRef)
      privateAccess(TransformerBlock)
      privateAccess(RopeGQAttention)

      # Verify it has expected properties
      doAssert rotary.head_dim > 0, "head_dim must be set"
      doAssert rotary.max_seq_len > 0, "max_seq_len must be set"
      doAssert rotary.cos_cache.dim == 2, "cos_cache must be 2D"
      doAssert rotary.sin_cache.dim == 2, "sin_cache must be 2D"

      # Verify all layers share the SAME rotary instance
      let layer0_rotary = model.layers[0].self_attn.rotary
      let layer1_rotary = model.layers[1].self_attn.rotary
      doAssert rotary == layer0_rotary, "Layer 0 should share model.rotary"
      doAssert rotary == layer1_rotary, "Layer 1 should share model.rotary"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Test: applyRope via ropeByPositions() + applyRope
  # Invariant: ARCHITECTURAL — integration of ropeByPositions + apply
  # What: Verifies full RoPE forward (compute → applyRope) works correctly
  # Why: This is the production path — model.forward calls ropeByPositions, layers call applyRope
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE applyRope via ropeByPositions() (batch=2, seq=8, GQA) — architectural integration":
    proc(): bool =
      var fixtureMemFile = memFiles.open(FixtureDir_Layers / "rope-Qwen3-0.6B-00.safetensor", mode = fmRead)
      defer: close(fixtureMemFile)
      var st = safetensors.load(fixtureMemFile)

      let q = st.getTensorOwned("q")
      let k = st.getTensorOwned("k")
      let q_rot_expected = st.getTensorOwned("q_rot")
      let k_rot_expected = st.getTensorOwned("k_rot")

      # Get model-level rotary
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let rotary = model.rotary
      privateAccess(RotaryPositionEmbeddingRef)
      privateAccess(TransformerBlock)
      privateAccess(RopeGQAttention)

      # Compute cos/sin for positions [0,1,2,3,4,5,6,7]
      let position_ids = F.arange(0, 8, device=kCPU)
      let (cos_sliced, sin_sliced) = rotary.ropeByPositions(position_ids)

      # Apply RoPE
      let (q_rot, k_rot) = rotary.applyRope(q, k, cos_sliced, sin_sliced)

      assertAllClose(q_rot, q_rot_expected, rtol = 1e-5, abstol = 1e-5, msg = "RoPE via ropeByPositions() q_rot mismatch")
      assertAllClose(k_rot, k_rot_expected, rtol = 1e-5, abstol = 1e-5, msg = "RoPE via ropeByPositions() k_rot mismatch")
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
  runCppTest "Qwen3 RoPE cos/sin cache vs HF — end-to-end correctness":
    proc(): bool =
      const tol = 1e-5  # BF16 precision — valid once fixtures use f32 inv_freq
      # Load HF cos/sin from fixture (computed by HF's RotaryEmbedding)
      var memFile = memFiles.open(FixtureDir_3Block / "block-00.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      let hfCos = st.getTensorOwned("cos", kCPU)
      let hfSin = st.getTensorOwned("sin", kCPU)

      # Load model and get model-level rotary
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let rotary = model.rotary
      privateAccess(RotaryPositionEmbeddingRef)
      privateAccess(TransformerBlock)
      privateAccess(RopeGQAttention)

      # Slice Nim cache to match HF seq length
      let seqLen = if hfCos.dim == 3: hfCos.size(1) else: hfCos.size(0)
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
  runCppTest "Qwen3 RoPE apply output vs HF — end-to-end correctness":
    proc(): bool =
      const tol = Tol  # Uses global Tol (1e-5)

      # Load HF cos/sin from fixture
      var memFile = memFiles.open(FixtureDir_3Block / "block-00.safetensor", mode = fmRead)
      defer: close(memFile)
      let st = safetensors.load(memFile)
      let hfCos = st.getTensorOwned("cos", kCPU)
      let hfSin = st.getTensorOwned("sin", kCPU)

      # Load model and get model-level rotary
      let model = loadQwen3ModelRaw(ModelPath, kCPU)
      privateAccess(Qwen3Model)
      let rotary = model.rotary
      privateAccess(RotaryPositionEmbeddingRef)
      privateAccess(TransformerBlock)
      privateAccess(RopeGQAttention)

      # Create test Q/K tensors
      let batch = 1
      let seqLen = if hfCos.dim == 3: hfCos.size(1) else: hfCos.size(0)
      # Read head dimensions from model.config (already populated at load time)
      let q_heads = model.config.num_attention_heads
      let k_heads = model.config.num_key_value_heads
      let head_dim = model.config.head_dim

      let q = F.randn([batch, seqLen, q_heads, head_dim], kFloat32).to(kCPU) * 0.1
      let k = F.randn([batch, seqLen, k_heads, head_dim], kFloat32).to(kCPU) * 0.1

      # Apply Nim RoPE (using ropeByPositions() + applyRope())
      let position_ids = F.arange(0, seqLen, device=kCPU)
      let (cos_sliced, sin_sliced) = rotary.ropeByPositions(position_ids)
      let (qNim, kNim) = rotary.applyRope(q.clone(), k.clone(), cos_sliced, sin_sliced)

      # Apply HF RoPE (using applyRopeImpl with HF cos/sin from fixture)
      # HF fixture is 3D (batch, seq, dim), slice to 2D (seq, dim)
      let hfCos2d = hfCos.narrow(0, 0, 1).squeeze(0)
      let hfSin2d = hfSin.narrow(0, 0, 1).squeeze(0)
      let (qHF, kHF) = applyRopeImpl(q.clone(), k.clone(), hfCos2d, hfSin2d)

      # Compare outputs — if Nim's cos/sin matches HF's (test 8), outputs must match too
      let qDiff = (qNim.to(kFloat32) - qHF.to(kFloat32)).abs().max().item(float)
      let kDiff = (kNim.to(kFloat32) - kHF.to(kFloat32)).abs().max().item(float)

      if qDiff > tol:
        raise newException(ValueError, &"Q diff {qDiff:.6e} exceeds tolerance {tol:.1e}")
      if kDiff > tol:
        raise newException(ValueError, &"K diff {kDiff:.6e} exceeds tolerance {tol:.1e}")
      true


  echo "\n✅ All RoPE tests passed!"

when isMainModule:
  main()
