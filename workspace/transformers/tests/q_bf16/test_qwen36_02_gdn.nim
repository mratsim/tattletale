# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off \
##   --outdir:build/tests/test_qwen36_02_gdn --nimcache:nimcache/tests/test_qwen36_02_gdn \
##   workspace/transformers/tests/q_bf16/test_qwen36_02_gdn.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/math,
  std/memfiles,
  std/os,
  std/strutils,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/transformers_testutils,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(GatedDeltaNet)

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  Shard1 = ModelDir / "model-00001-of-00026.safetensors"
  Shard2 = ModelDir / "model-00002-of-00026.safetensors"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "qwen36-gdn"
  GdnPrefix = "model.language_model.layers.0.linear_attn"

  # GDN geometry of the checkpoint (text_config linear_* keys)
  Hidden = 2048
  NumKHeads = 16
  NumVHeads = 32
  HeadKDim = 128
  HeadVDim = 128
  ConvKernelSize = 4
  KeyDim = NumKHeads * HeadKDim       # 2048: one key half of the conv output
  ValueDim = NumVHeads * HeadVDim     # 4096: the value half
  ConvDim = 2 * KeyDim + ValueDim     # 8192: in_proj_qkv out_features

  # Fixture case lengths: prefill T=5, multichunk T=70 (FLA chunks 64+6)
  PrefillSeq = 5
  MultichunkSeq = 70

  # Recurrent-vs-chunked caps the fixture generator asserts on its replays
  FixtureOutputBar = 1e-3
  FixtureCoreBar = 1e-5
  # Block-output bar for the Nim forward against the vendored chunked form
  ChunkedBar = 5e-3
  # Recurrent-vs-chunked SSM cap: four fp32 ulps at the state max magnitude
  SsmUlpMargin = 4.0

proc ulpFp32(m: float64): float64 =
  ## One fp32 ulp at magnitude m: fp32 has 23 significand bits, giving
  ## ulp 2^(e-23) with e = floor(log2(m)). Zero maps to 0.
  if m <= 0.0:
    return 0.0
  result = pow(2.0, floor(log2(m)) - 23.0)

proc ssmCap(ssmState: Tensor): float64 =
  ## Recurrent-vs-chunked SSM cap: four fp32 ulps at the state max
  ## magnitude. The two rules diverge by about one ulp at the largest
  ## divergent element, whose magnitude is bounded by the state max.
  SsmUlpMargin * ulpFp32(ssmState.abs().max().item(float64))

proc buildGdn(): GatedDeltaNet =
  ## Load the real layer-0 GDN weights from the two shards that hold them
  ## and build the layer.
  ##
  ## Shard-by-name access below is a suite-only convenience: the upcoming
  ## loader work replaces it with the index-backed weight_map view.
  var (memFile1, st1) = openSafetensor(Shard1)
  var (memFile2, st2) = openSafetensor(Shard2)
  defer:
    close(memFile1)
    close(memFile2)

  let cfgJson = (ModelDir / "config.json").parseFile()
  let normEps = cfgJson{"text_config"}{"rms_norm_eps"}.getFloat(1e-6)

  # in_proj_qkv/in_proj_z/out_proj live in shard 1. A_log, conv1d, dt_bias,
  # in_proj_a/in_proj_b and the norm weight live in shard 2 (checkpoint index)
  let qkvProj = Linear.init(st1.getTensorOwned(GdnPrefix & ".in_proj_qkv.weight"))
  let zProj = Linear.init(st1.getTensorOwned(GdnPrefix & ".in_proj_z.weight"))
  let oProj = Linear.init(st1.getTensorOwned(GdnPrefix & ".out_proj.weight"))
  let convW = st2.getTensorOwned(GdnPrefix & ".conv1d.weight")
  let aLog = st2.getTensorOwned(GdnPrefix & ".A_log")
  let dtBias = st2.getTensorOwned(GdnPrefix & ".dt_bias")
  let aProj = Linear.init(st2.getTensorOwned(GdnPrefix & ".in_proj_a.weight"))
  let bProj = Linear.init(st2.getTensorOwned(GdnPrefix & ".in_proj_b.weight"))
  let norm = RmsNormGated.init(
    st2.getTensorOwned(GdnPrefix & ".norm.weight"), eps = normEps)
  GatedDeltaNet.init(
    0, GdnPrefix,
    qkvProj, zProj, aProj, bProj,
    convW, aLog, dtBias,
    norm, oProj,
    NumKHeads, NumVHeads, HeadKDim, HeadVDim, ConvKernelSize)

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Real layer-0 GDN tensors: conv-split geometry vs fixture q/k/v":
    proc(): bool =
      let gdn = buildGdn()
      var (memFile, st) = openSafetensor(FixtureDir, "gdn-Qwen3.6-35B-A3B-00.safetensor")
      defer: close(memFile)

      let convOutput = st.getTensorOwned("conv_output")   # (1, conv_dim, 5) bf16
      let qFix = st.getTensorOwned("q")                   # (1, 5, 16, 128) bf16
      let kFix = st.getTensorOwned("k")                   # (1, 5, 16, 128) bf16
      let vFix = st.getTensorOwned("v")                   # (1, 5, 32, 128) bf16

      # The conv output is [key | key | value]. The reference
      # torch.split widths sum to the conv width, not conv_dim / 3
      doAssert convOutput.dim() == 3
      doAssert convOutput.size(0) == 1
      doAssert convOutput.size(1) == ConvDim
      doAssert convOutput.size(2) == PrefillSeq
      doAssert qFix.dim() == 4
      doAssert qFix.size(0) == 1
      doAssert qFix.size(1) == PrefillSeq
      doAssert qFix.size(2) == NumKHeads
      doAssert qFix.size(3) == HeadKDim
      doAssert kFix.dim() == 4
      doAssert kFix.size(0) == 1
      doAssert kFix.size(1) == PrefillSeq
      doAssert kFix.size(2) == NumKHeads
      doAssert kFix.size(3) == HeadKDim
      doAssert vFix.dim() == 4
      doAssert vFix.size(0) == 1
      doAssert vFix.size(1) == PrefillSeq
      doAssert vFix.size(2) == NumVHeads
      doAssert vFix.size(3) == HeadVDim

      # The fixture q/k/v shapes carry the reference split widths:
      # key halves of key_dim and a value half of value_dim
      doAssert qFix.size(2) * qFix.size(3) == KeyDim
      doAssert kFix.size(2) * kFix.size(3) == KeyDim
      doAssert vFix.size(2) * vFix.size(3) == ValueDim
      doAssert convOutput.size(1) == 2 * KeyDim + ValueDim

      # Layer geometry agrees with the reference: conv_dim sizes the fused
      # qkv projection (the real weight shape) and the conv weight channels
      doAssert gdn.in_proj_qkv.out_features == ConvDim
      doAssert gdn.conv_dim == ConvDim
      doAssert gdn.num_k_heads == NumKHeads
      doAssert gdn.num_v_heads == NumVHeads
      doAssert gdn.head_k_dim == HeadKDim
      doAssert gdn.head_v_dim == HeadVDim
      doAssert gdn.conv1d_weight.size(0) == ConvDim

      # The Nim split arithmetic (three narrows on the last dim) reproduces
      # the reference sized split bitwise from the fixture conv output
      let convT = convOutput.transpose(1, 2)              # (1, 5, conv_dim)
      let qNim = convT.narrow(2, 0, KeyDim).reshape(
        [1, PrefillSeq, NumKHeads, HeadKDim])
      let kNim = convT.narrow(2, KeyDim, KeyDim).reshape(
        [1, PrefillSeq, NumKHeads, HeadKDim])
      let vNim = convT.narrow(2, 2 * KeyDim, ValueDim).reshape(
        [1, PrefillSeq, NumVHeads, HeadVDim])
      assertAllClose(qNim, qFix, rtol = 0.0, abstol = 0.0, msg = "split q mismatch")
      assertAllClose(kNim, kFix, rtol = 0.0, abstol = 0.0, msg = "split k mismatch")
      assertAllClose(vNim, vFix, rtol = 0.0, abstol = 0.0, msg = "split v mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN layer forward (recurrent) vs prefill fixture":
    proc(): bool =
      let gdn = buildGdn()
      var ctx = InferenceContext.init(
        num_layers = 1, batch_size = 1, kv_heads = 2, max_seq = 512,
        head_dim = HeadKDim)
      var (memFile, st) = openSafetensor(FixtureDir, "gdn-Qwen3.6-35B-A3B-00.safetensor")
      defer: close(memFile)

      let hidden = st.getTensorOwned("input")             # (1, 5, 2048) bf16
      let layerOut = gdn(ctx, hidden)

      # The Nim recurrence is bit-identical to torch_recurrent_gated_delta_rule:
      # the block output and the final SSM state match the recurrent replay
      let seqOut = st.getTensorOwned("output_seq")
      assertAllClose(layerOut, seqOut, rtol = 0.0, abstol = 0.0,
        msg = "recurrent block output mismatch")
      let nimSsm = ctx.gdnSsmState[0]                     # [num_v_heads, Dk, Dv] f32
      let seqSsm = st.getTensorOwned("ssm_state_seq")     # final state, batch dim dropped
      assertAllClose(nimSsm, seqSsm, rtol = 0.0, abstol = 0.0,
        msg = "recurrent final SSM state mismatch")

      # The vendored chunked form is the reference forward: the block output
      # sits under the block bar against it, and the recurrent-vs-chunked
      # divergence is the documented fp32 floor
      let chunkedOut = st.getTensorOwned("output_chunked")
      assertAllClose(layerOut, chunkedOut, rtol = ChunkedBar, abstol = ChunkedBar,
        msg = "chunked block output mismatch")

      # The fixture's own bands, recomputed from its tensors
      let chunkedSsm = st.getTensorOwned("ssm_state_chunked")
      let outputBandDiff = maxAbsDiff(seqOut, chunkedOut)
      let coreBandDiff = maxAbsDiff(st.getTensorOwned("core_attn_out_seq"),
        st.getTensorOwned("core_attn_out_chunked"))
      let ssmBandDiff = maxAbsDiff(seqSsm, chunkedSsm)
      doAssert outputBandDiff < FixtureOutputBar,
        "recurrent-vs-chunked output diff outside the documented bar"
      doAssert coreBandDiff < FixtureCoreBar,
        "recurrent-vs-chunked core diff outside the documented bar"
      doAssert ssmBandDiff <= ssmCap(seqSsm),
        "recurrent-vs-chunked SSM diff outside the fp32 floor"

      # Nim SSM vs the chunked final state stays inside the same floor
      doAssert maxAbsDiff(nimSsm, chunkedSsm) <= ssmCap(seqSsm),
        "Nim final SSM state outside the fp32 floor vs the chunked state"
      true

  # ──────────────────────────────────────────────────────────────────────────
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "GDN layer forward (recurrent) vs multichunk fixture":
    proc(): bool =
      let gdn = buildGdn()
      var ctx = InferenceContext.init(
        num_layers = 1, batch_size = 1, kv_heads = 2, max_seq = 512,
        head_dim = HeadKDim)
      var (memFile, st) = openSafetensor(FixtureDir, "gdn-Qwen3.6-35B-A3B-01.safetensor")
      defer: close(memFile)

      let hidden = st.getTensorOwned("input")             # (1, 70, 2048) bf16
      doAssert hidden.size(0) == 1 and hidden.size(1) == MultichunkSeq
      doAssert hidden.size(2) == Hidden
      let layerOut = gdn(ctx, hidden)

      let seqOut = st.getTensorOwned("output_seq")
      assertAllClose(layerOut, seqOut, rtol = 0.0, abstol = 0.0,
        msg = "recurrent multichunk output mismatch")
      let nimSsm = ctx.gdnSsmState[0]
      let seqSsm = st.getTensorOwned("ssm_state_seq")
      assertAllClose(nimSsm, seqSsm, rtol = 0.0, abstol = 0.0,
        msg = "recurrent multichunk final SSM state mismatch")

      let chunkedOut = st.getTensorOwned("output_chunked")
      assertAllClose(layerOut, chunkedOut, rtol = ChunkedBar, abstol = ChunkedBar,
        msg = "chunked multichunk output mismatch")

      # Two FLA chunks (64 + 6) exercise the cross-chunk state handoff.
      # The recurrent-vs-chunked divergence is positive here and stays
      # inside the documented fp32 floor
      let chunkedSsm = st.getTensorOwned("ssm_state_chunked")
      let outputBandDiff = maxAbsDiff(seqOut, chunkedOut)
      let coreBandDiff = maxAbsDiff(st.getTensorOwned("core_attn_out_seq"),
        st.getTensorOwned("core_attn_out_chunked"))
      let ssmBandDiff = maxAbsDiff(seqSsm, chunkedSsm)
      doAssert 0.0 < outputBandDiff and outputBandDiff < FixtureOutputBar,
        "multichunk recurrent-vs-chunked output diff outside (0, 1e-3)"
      doAssert coreBandDiff < FixtureCoreBar,
        "multichunk recurrent-vs-chunked core diff outside the documented bar"
      doAssert 0.0 < ssmBandDiff and ssmBandDiff <= ssmCap(seqSsm),
        "multichunk recurrent-vs-chunked SSM diff outside the fp32 floor"
      doAssert maxAbsDiff(nimSsm, chunkedSsm) <= ssmCap(seqSsm),
        "multichunk Nim final SSM state outside the fp32 floor vs the chunked state"
      true


  runCppTest "GatedDeltaNet init refuses non-positive and non-multiple head counts":
    proc(): bool =
      # Value heads per key head share one recurrent slot per group. The init
      # refuses a non-multiple pair: integer division truncates it silently,
      # and the raise names the layer path.
      proc gdn(numK, numV: int): GatedDeltaNet =
        GatedDeltaNet.init(
          0, "model.language_model.layers.0.linear_attn",
          # Fused qkv width at the passing pair, and its conv channels:
          # 2 * 16 * 2 + 32 * 2 = 128, the conv-dim law at that geometry.
          Linear.init(F.zeros(128, 1)), Linear.init(F.zeros(8, 1)),
          Linear.init(F.zeros(4, 1)), Linear.init(F.zeros(4, 1)),
          F.zeros(128, 1, 4), F.zeros(4), F.zeros(4),
          RmsNormGated.init(F.zeros(2), eps = 1e-06),
          Linear.init(F.zeros(8, 1)),
          numK, numV, 2, 2, 4)

      proc gdnMsg(numK, numV: int): string =
        result = ""
        try:
          discard gdn(numK, numV)
        except ValueError as e:
          result = e.msg

      let msg = gdnMsg(2, 3)
      doAssert msg.startsWith("[ttt]"), "3 value heads over 2 key heads was accepted"
      doAssert msg.contains("linear_num_value_heads") and msg.contains("linear_num_key_heads")
      doAssert "model.language_model.layers.0.linear_attn" in msg
      doAssert "3" in msg and "2" in msg

      # 30 value heads over 16 key heads, the paired shape the accessor refused.
      doAssert "30" in gdnMsg(16, 30) and "16" in gdnMsg(16, 30)
      doAssert "num_key_heads" in gdnMsg(0, 0)

      # The real checkpoint pair (16 key heads, 32 value heads) passes.
      discard gdn(16, 32)

      true

  echo "\nAll Qwen3.6 GDN tests passed!"

when isMainModule:
  main()
