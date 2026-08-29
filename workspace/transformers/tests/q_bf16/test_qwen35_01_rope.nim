# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-rope \
##   --nimcache:nimcache/tests/qwen35-rope \
##   workspace/transformers/tests/q_bf16/test_qwen35_01_rope.nim

import
  std/memfiles,
  std/os,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/tests/transformers_testutils,
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "layers" / "Qwen3.5-0.8B-layer-3"

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # rotateHalf over the 64-wide partial slice
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE rotateHalf 64-wide - mathematical property":
    proc(): bool =
      var (memFile03, st) = openSafetensor(FixtureDir, "rope-Qwen3.5-0.8B-03.safetensor")
      defer: close(memFile03)
      let x = st.getTensorOwned("input")       # (2, 8, 8, 64)
      let expected = st.getTensorOwned("output")
      let got = rotateHalf(x)
      assertAllClose(got, expected, rtol = 0.0, abstol = 0.0, msg = "rotateHalf mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Partial applyRopeImpl: only the first 64 of 256 dims rotate
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE partial applyRopeImpl prefill (batch=2, seq=8)":
    proc(): bool =
      var (memFile00, st) = openSafetensor(FixtureDir, "rope-Qwen3.5-0.8B-00.safetensor")
      defer: close(memFile00)
      let q = st.getTensorOwned("q")            # (2, 8, 8, 256)
      let k = st.getTensorOwned("k")            # (2, 8, 2, 256)
      let cos = st.getTensorOwned("cos")        # (8, 64)
      let sin = st.getTensorOwned("sin")        # (8, 64)
      let qRotExpected = st.getTensorOwned("q_rot")  # (2, 8, 8, 256)
      let kRotExpected = st.getTensorOwned("k_rot")  # (2, 8, 2, 256)

      let (qRot, kRot) = applyRopeImpl(q, k, cos, sin)
      assertAllClose(qRot, qRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE q_rot mismatch")
      assertAllClose(kRot, kRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE k_rot mismatch")

      # Pass-through: columns 64..255 must be bit-identical to the input.
      let qPass = qRot.narrow(3, 64, 192)
      let qInPass = q.narrow(3, 64, 192)
      let kPass = kRot.narrow(3, 64, 192)
      let kInPass = k.narrow(3, 64, 192)
      assertAllClose(qPass, qInPass, rtol = 0.0, abstol = 0.0, msg = "q pass-through changed")
      assertAllClose(kPass, kInPass, rtol = 0.0, abstol = 0.0, msg = "k pass-through changed")
      true

  runCppTest "RoPE partial applyRopeImpl decode (single token, position 5)":
    proc(): bool =
      var (memFile01, st) = openSafetensor(FixtureDir, "rope-Qwen3.5-0.8B-01.safetensor")
      defer: close(memFile01)
      let q = st.getTensorOwned("q")            # (1, 1, 8, 256)
      let k = st.getTensorOwned("k")            # (1, 1, 2, 256)
      let cos = st.getTensorOwned("cos")        # (1, 64)
      let sin = st.getTensorOwned("sin")
      let qRotExpected = st.getTensorOwned("q_rot")
      let kRotExpected = st.getTensorOwned("k_rot")

      let (qRot, kRot) = applyRopeImpl(q, k, cos, sin)
      assertAllClose(qRot, qRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE q_rot mismatch")
      assertAllClose(kRot, kRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE k_rot mismatch")
      true

  runCppTest "RoPE partial applyRopeImpl scattered positions (index_select path)":
    proc(): bool =
      var (memFile02, st) = openSafetensor(FixtureDir, "rope-Qwen3.5-0.8B-02.safetensor")
      defer: close(memFile02)
      let q = st.getTensorOwned("q")            # (1, 4, 8, 256)
      let k = st.getTensorOwned("k")            # (1, 4, 2, 256)
      let cos = st.getTensorOwned("cos")        # (4, 64)
      let sin = st.getTensorOwned("sin")
      let qRotExpected = st.getTensorOwned("q_rot")
      let kRotExpected = st.getTensorOwned("k_rot")

      let (qRot, kRot) = applyRopeImpl(q, k, cos, sin)
      assertAllClose(qRot, qRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE q_rot mismatch")
      assertAllClose(kRot, kRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE k_rot mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # inv_freq over rotary_dim (theta 1e7, 32 frequencies, 64-wide repetition)
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Qwen3.5 partial rotary inv_freq - mathematical property":
    proc(): bool =
      let dim = 64
      let rope_theta = 10000000.0
      let inv_freq = F.arange(0, dim, 2).to(kFloat64) / dim.float64
      let inv_freq_final = F.pow(F.full([1], rope_theta, kFloat64), -inv_freq)
      let got = inv_freq_final[0..<6]
      # Reference values: 1/theta^(d/64) for d in {0,2,4,6,8,10}, theta 1e7.
      let expected = F.toTensor([1.0, 0.604296, 0.365174, 0.220673, 0.133352, 0.080584]).to(kFloat64)
      assertAllClose(got, expected, rtol = 1e-5, abstol = 1e-5, msg = "partial inv_freq mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Cache: sized to rotary_dim (64), matches the vendored f32 cos/sin
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Qwen3.5 partial rotary cache (rotary_dim 64) vs HF cos/sin":
    proc(): bool =
      let rotary = RotaryPositionEmbeddingRef.new(
        256, 8192, 1e7, F.kBFloat16, F.kCPU, rotary_dim = 64)
      doAssert rotary.rotary_dim == 64
      doAssert rotary.cos_cache.dim == 2
      doAssert rotary.cos_cache.size(0) == 8192
      doAssert rotary.cos_cache.size(1) == 64

      # Positions 0..7 (rope case 00): cache rows must match the fixture.
      var (memFile00, st) = openSafetensor(FixtureDir, "rope-Qwen3.5-0.8B-00.safetensor")
      defer: close(memFile00)
      let hfCos = st.getTensorOwned("cos")      # (8, 64) bf16
      let hfSin = st.getTensorOwned("sin")
      let nimCos = rotary.cos_cache.narrow(0, 0, 8)
      let nimSin = rotary.sin_cache.narrow(0, 0, 8)
      assertAllClose(nimCos, hfCos, rtol = 0.0, abstol = 0.0, msg = "cos cache mismatch")
      assertAllClose(nimSin, hfSin, rtol = 0.0, abstol = 0.0, msg = "sin cache mismatch")

      # Scattered positions [3, 17, 255, 4096] (rope case 02): index_select.
      var (memFile02b, st2) = openSafetensor(FixtureDir, "rope-Qwen3.5-0.8B-02.safetensor")
      defer: close(memFile02b)
      let pos2d = st2.getTensorOwned("position_ids")   # (1, 4) int64
      let hfCos2 = st2.getTensorOwned("cos")           # (4, 64)
      let hfSin2 = st2.getTensorOwned("sin")
      let (cos2d, sin2d) = rotary.ropeByPositions(pos2d)       # 2D position_ids
      let (cos1d, sin1d) = rotary.ropeByPositions(pos2d[0])    # 1D position_ids
      assertAllClose(cos2d, hfCos2, rtol = 0.0, abstol = 0.0, msg = "cos scattered mismatch (2D)")
      assertAllClose(sin2d, hfSin2, rtol = 0.0, abstol = 0.0, msg = "sin scattered mismatch (2D)")
      assertAllClose(cos1d, hfCos2, rtol = 0.0, abstol = 0.0, msg = "cos scattered mismatch (1D)")
      assertAllClose(sin1d, hfSin2, rtol = 0.0, abstol = 0.0, msg = "sin scattered mismatch (1D)")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Position 0 identity
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE position 0 identity - mathematical property":
    proc(): bool =
      let rotary = RotaryPositionEmbeddingRef.new(
        256, 8192, 1e7, F.kBFloat16, F.kCPU, rotary_dim = 64)
      let cos0 = rotary.cos_cache[0, 0..<5].to(kFloat32)
      let sin0 = rotary.sin_cache[0, 0..<5].to(kFloat32)
      let cosExpected = F.ones([5], kFloat32)
      let sinExpected = F.zeros([5], kFloat32)
      assertAllClose(cos0, cosExpected, rtol = 0.0, abstol = 0.0, msg = "cos[0] should be 1")
      assertAllClose(sin0, sinExpected, rtol = 0.0, abstol = 0.0, msg = "sin[0] should be 0")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Backward compatibility: full-head_dim rotation when cos width == head_dim
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE full-head_dim rotation when cos width == head_dim":
    proc(): bool =
      # A 128-wide rotary over a 128-wide head must still rotate everything
      # (qwen3 path). applyRopeImpl derives the width from cos.size(-1).
      let rotary = RotaryPositionEmbeddingRef.new(
        128, 256, 1e6, F.kBFloat16, F.kCPU)
      doAssert rotary.rotary_dim == 128
      doAssert rotary.cos_cache.size(1) == 128
      let pos = F.arange(0, 4, device = kCPU)
      let (cos, sin) = rotary.ropeByPositions(pos)
      let q = F.randn([2, 4, 8, 128], F.tensorOptions(F.kBFloat16, F.kCPU))
      let k = F.randn([2, 4, 8, 128], F.tensorOptions(F.kBFloat16, F.kCPU))
      let (qRot, kRot) = applyRopeImpl(q, k, cos, sin)
      # Rotation must change the values (not identity) and keep the shape.
      doAssert qRot.size(3) == 128
      let diff = (qRot.to(kFloat32) - q.to(kFloat32)).abs().max().item(float64)
      doAssert diff > 0.0, "full rotation must not be identity"
      true

  echo "\nAll Qwen3.5 partial-rope tests passed!"

when isMainModule:
  main()
