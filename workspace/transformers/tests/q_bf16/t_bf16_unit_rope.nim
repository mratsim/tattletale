# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Per-op RoPE unit suite, shared across the dense ports. One recorded
## family per op: the Qwen3.5-0.8B layer-3 rope family. Its partial-rotary
## shape of 64 of 256 dims exercises every code path that the full-width
## 0.6B rotation also runs, and the recorded family carries the stats
## sidecar. The full-width theta variants stay covered by the property
## tests below, and by the chain suites end to end.
##
## Build:
##   nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on
##     --debugger:native --hints:off --warnings:off --passC:"-std=c++20"
##     --outdir:build/tests/rel --nimcache:nimcache/tests/rel
##     workspace/transformers/tests/q_bf16/t_bf16_unit_rope.nim

import
  std/os,
  workspace/safetensors,
  workspace/libtorch as F,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.5-0.8B-layer-3"

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # rotateHalf over the 64-wide partial slice
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "RoPE rotateHalf 64-wide - mathematical property":
    proc(): bool =
      var st = Safetensor.open(FixtureDir / "rope-Qwen3.5-0.8B-03.safetensor")
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
      var st = Safetensor.open(FixtureDir / "rope-Qwen3.5-0.8B-00.safetensor")
      let q = st.getTensorOwned("q")            # (2, 8, 8, 256)
      let k = st.getTensorOwned("k")            # (2, 8, 2, 256)
      let cos = st.getTensorOwned("cos")        # (8, 64)
      let sin = st.getTensorOwned("sin")        # (8, 64)
      let qRotExpected = st.getTensorOwned("q_rot")  # (2, 8, 8, 256)
      let kRotExpected = st.getTensorOwned("k_rot")  # (2, 8, 2, 256)

      let (qRot, kRot) = applyRopeImpl(q, k, cos, sin)
      assertAllClose(qRot, qRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE q_rot mismatch")
      assertAllClose(kRot, kRotExpected, rtol = 0.0, abstol = 0.0, msg = "partial RoPE k_rot mismatch")

      # Signature checks: element-wise match rate plus recorded order
      # statistics and histogram against the stats-file bootstrap.
      let ropeBudget = defaultBudget(obRope, F.kCPU)
      let ropeStatsFile = loadFingerprintStats(
        FixtureDir / "rope-Qwen3.5-0.8B-00.safetensor.stats")
      assertMatchRate(qRot, qRotExpected, ropeBudget, msg = "q_rot match rate")
      assertStats(qRot, ropeStatsFile.statsTensor("q_rot"), ropeBudget,
        msg = "q_rot stats")
      assertMatchRate(kRot, kRotExpected, ropeBudget, msg = "k_rot match rate")
      assertStats(kRot, ropeStatsFile.statsTensor("k_rot"), ropeBudget,
        msg = "k_rot stats")

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
      var st = Safetensor.open(FixtureDir / "rope-Qwen3.5-0.8B-01.safetensor")
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
      var st = Safetensor.open(FixtureDir / "rope-Qwen3.5-0.8B-02.safetensor")
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
  # inv_freq over rotary_dim for both dense thetas: the partial theta
  # 1e7 variant with a 64-wide slice, and the full-width 0.6B variant
  # with theta 1e6.
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

  runCppTest "Qwen3 full-width rotary inv_freq - mathematical property":
    proc(): bool =
      # The 0.6B configuration: full-width rotation, theta 1e6, head 128.
      let head_dim = 128
      let rope_theta = 1000000.0
      let inv_freq = F.arange(0, head_dim, 2).to(kFloat64) / head_dim.float64
      let inv_freq_final = F.pow(F.full([1], rope_theta, kFloat64), -inv_freq)
      let got = inv_freq_final[0..<4]
      # Reference values: 1/theta^(d/128) for d in {0,2,4,6}.
      let expected = F.toTensor([1.0, 0.80584219, 0.64938163, 0.52329911]).to(kFloat64)
      assertAllClose(got, expected, rtol = 1e-5, abstol = 1e-5, msg = "full-width inv_freq mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # Cache: sized to rotary_dim (64), matches the vendored f32 cos/sin
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Qwen3.5 partial rotary cache (rotary_dim 64) vs HF cos/sin":
    proc(): bool =
      let rotary = RotaryPositionEmbedding.new(
        256, 8192, 1e7, F.kBFloat16, F.kCPU, rotary_dim = 64)
      doAssert rotary.rotary_dim == 64
      doAssert rotary.cos_cache.dim == 2
      doAssert rotary.cos_cache.size(0) == 8192
      doAssert rotary.cos_cache.size(1) == 64

      # Positions 0..7 (rope case 00): cache rows must match the fixture.
      var st = Safetensor.open(FixtureDir / "rope-Qwen3.5-0.8B-00.safetensor")
      let hfCos = st.getTensorOwned("cos")      # (8, 64) bf16
      let hfSin = st.getTensorOwned("sin")
      let nimCos = rotary.cos_cache.narrow(0, 0, 8)
      let nimSin = rotary.sin_cache.narrow(0, 0, 8)
      assertAllClose(nimCos, hfCos, rtol = 0.0, abstol = 0.0, msg = "cos cache mismatch")
      assertAllClose(nimSin, hfSin, rtol = 0.0, abstol = 0.0, msg = "sin cache mismatch")

      # Scattered positions [3, 17, 255, 4096] (rope case 02): index_select.
      var st2 = Safetensor.open(FixtureDir / "rope-Qwen3.5-0.8B-02.safetensor")
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
      let rotary = RotaryPositionEmbedding.new(
        256, 8192, 1e7, F.kBFloat16, F.kCPU, rotary_dim = 64)
      let cos0 = rotary.cos_cache[0, 0..<5].to(kFloat32)
      let sin0 = rotary.sin_cache[0, 0..<5].to(kFloat32)
      let cosExpected = F.ones([5], kFloat32)
      let sinExpected = F.zeros([5], kFloat32)
      assertAllClose(cos0, cosExpected, rtol = 0.0, abstol = 0.0, msg = "cos[0] should be 1")
      assertAllClose(sin0, sinExpected, rtol = 0.0, abstol = 0.0, msg = "sin[0] should be 0")
      true

  echo "\nAll per-op RoPE tests passed!"

when isMainModule:
  main()
