# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off \
## --outdir:build/tests/t_bf16_qwen36moe_01_layer_internals_gdn --nimcache:nimcache/tests/t_bf16_qwen36moe_01_layer_internals_gdn \
## workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_01_layer_internals_gdn.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/os,
  std/strutils,
  std/importutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils
from workspace/libtorch/src/raw_libtorch import manual_seed

{.experimental: "callOperator".}

privateAccess(GatedDeltaNet)

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  WeightsFile1 = ModelDir / "model-00001-of-00026.safetensors"
  WeightsFile2 = ModelDir / "model-00002-of-00026.safetensors"
  FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-0"
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

  PrefillSeq = 5
  # The multichunk case splits as two flash-linear-attention (FLA) chunks, 64 + 6.
  MultichunkSeq = 70

  # Recurrent-vs-chunked caps the fixture generator asserts on its replays
  FixtureOutputBar = 1e-3
  # Bar for the Nim block output against the chunked reference form
  ChunkedBar = 5e-3

proc buildGdn(dev = F.kCPU): GatedDeltaNet =
  ## Load the real layer-0 GDN weights from the two safetensor files that hold them and build the layer.
  ## Device move: every loaded tensor moves to `dev` at load time, one move
  ## per tensor. GatedDeltaNet carries no module-level device move: only
  ## iface-registered models have one. The per-tensor moves at load
  ## keep the weights device-consistent with the input.
  ## TODO: The two files here are opened by name. The loader should migrate to the index-backed
  ## weight_map view.
  var st1 = Safetensor.open(WeightsFile1)
  var st2 = Safetensor.open(WeightsFile2)

  let cfgJson = (ModelDir / "config.json").parseFile()
  let normEps = cfgJson{"text_config"}{"rms_norm_eps"}.getFloat(1e-6)

  # in_proj_qkv, in_proj_z and out_proj live in file 1. A_log, conv1d, dt_bias, in_proj_a,
  # in_proj_b and the norm weight live in file 2, per the checkpoint index.
  let qkvProj = Linear.init(
    st1.getTensorOwned(GdnPrefix & ".in_proj_qkv.weight").to(dev))
  let zProj = Linear.init(
    st1.getTensorOwned(GdnPrefix & ".in_proj_z.weight").to(dev))
  let oProj = Linear.init(
    st1.getTensorOwned(GdnPrefix & ".out_proj.weight").to(dev))
  let convW = st2.getTensorOwned(GdnPrefix & ".conv1d.weight").to(dev)
  let aLog = st2.getTensorOwned(GdnPrefix & ".A_log").to(dev)
  let dtBias = st2.getTensorOwned(GdnPrefix & ".dt_bias").to(dev)
  let aProj = Linear.init(
    st2.getTensorOwned(GdnPrefix & ".in_proj_a.weight").to(dev))
  let bProj = Linear.init(
    st2.getTensorOwned(GdnPrefix & ".in_proj_b.weight").to(dev))
  let norm = RmsNormGated.init(
    st2.getTensorOwned(GdnPrefix & ".norm.weight").to(dev), eps = normEps)
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
      var st = Safetensor.open(FixtureDir / "gdn-Qwen3.6-35B-A3B-00.safetensor")

      let convOutput = st.getTensorOwned("conv_output")   # (1, conv_dim, 5) bf16
      let qFix = st.getTensorOwned("q")                   # (1, 5, 16, 128) bf16
      let kFix = st.getTensorOwned("k")                   # (1, 5, 16, 128) bf16
      let vFix = st.getTensorOwned("v")                   # (1, 5, 32, 128) bf16

      # The conv output is [key | key | value]. The reference torch.split widths sum to the conv
      # width, not conv_dim / 3
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

      # The fixture q/k/v shapes carry the reference split widths: key halves of key_dim and a value
      # half of value_dim
      doAssert qFix.size(2) * qFix.size(3) == KeyDim
      doAssert kFix.size(2) * kFix.size(3) == KeyDim
      doAssert vFix.size(2) * vFix.size(3) == ValueDim
      doAssert convOutput.size(1) == 2 * KeyDim + ValueDim

      # Layer geometry agrees with the reference: conv_dim sizes the fused qkv projection, the real
      # weight shape, and the conv weight channels
      doAssert gdn.in_proj_qkv.out_features == ConvDim
      doAssert gdn.conv_dim == ConvDim
      doAssert gdn.num_k_heads == NumKHeads
      doAssert gdn.num_v_heads == NumVHeads
      doAssert gdn.head_k_dim == HeadKDim
      doAssert gdn.head_v_dim == HeadVDim
      doAssert gdn.conv1d_weight.size(0) == ConvDim

      # The Nim split arithmetic (three narrows on the last dim) reproduces the reference sized
      # split bitwise from the fixture conv output
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
      var st = Safetensor.open(FixtureDir / "gdn-Qwen3.6-35B-A3B-00.safetensor")

      let hidden = st.getTensorOwned("input")             # (1, 5, 2048) bf16
      let layerOut = gdn(ctx, hidden)

      # The Nim recurrence is bit-identical to torch_recurrent_gated_delta_rule: the block output
      # and the final SSM state match the recurrent replay
      let seqOut = st.getTensorOwned("output_seq")
      assertAllClose(layerOut, seqOut, rtol = 0.0, abstol = 0.0,
        msg = "recurrent block output mismatch")
      # Signature checks on the block output: match rate plus recorded stats. The final SSM
      # state left the payload under the fixture contract: its external surface is the
      # descriptor sidecar, re-expressed as
      # - the fingerprint (assertStats)
      # - the bulk mean band, signed mean band, tail-probability band (assertDescriptors)
      # - the probe subset, elementwise
      # - every tolerance from the derived four fp32 ulp cap
      let gdnBudget = defaultBudget(obAttention, F.kCPU)
      let gdnStatsFile = loadFingerprintStats(
        FixtureDir / "gdn-Qwen3.6-35B-A3B-00.safetensor.stats")
      assertMatchRate(layerOut, seqOut, gdnBudget,
        msg = "recurrent block output match rate")
      assertStats(layerOut, gdnStatsFile.statsTensor("output_seq"), gdnBudget,
        msg = "recurrent block output stats")
      let nimSsm = ctx.gdnSsmState[0]                     # [num_v_heads, Dk, Dv] f32
      let gdnDescFile = loadFingerprintStats(
        FixtureDir / "gdn-Qwen3.6-35B-A3B-00.safetensor.descriptors")
      let ssmEntry = gdnDescFile.statsTensor("ssm_state_seq")
      assertStats(nimSsm, ssmEntry, descriptorStatsBudget(ssmEntry.probeMode),
        msg = "recurrent final SSM state stats")
      assertDescriptors(nimSsm, ssmEntry,
        msg = "recurrent final SSM state descriptors")

      # The chunked form is the reference forward: the block output sits under ChunkedBar against
      # it. The recorded recurrent-vs-chunked divergence facts (output, core, SSM) stay in the fixture
      # metadata, generator-asserted at record time. The self-consistency of the two computation
      # modes is the prefill-versus-decode property check.
      let chunkedOut = st.getTensorOwned("output_chunked")
      assertAllClose(layerOut, chunkedOut, rtol = ChunkedBar, abstol = ChunkedBar,
        msg = "chunked block output mismatch")

      # The fixture's own band, recomputed from its tensors
      let outputBandDiff = maxAbsDiff(seqOut, chunkedOut)
      doAssert outputBandDiff < FixtureOutputBar,
        "recurrent-vs-chunked output diff outside the documented bar"
      true




  runCppTest "GDN evaluation order: one-shot prefill == step-decode, synthetic T=70":
    proc(): bool =
      echo "    devices: ", compareLine(FixtureDir, F.kCPU)
      # Tier-1 property check on a seeded synthetic input. The input splits as two flash-linear-attention
      # chunks, 64 + 6. The one-shot prefill runs the chunked kernel over the whole sequence, the 70-step
      # decode runs the step-wise recurrence. The equivalence of the two evaluation orders on the same
      # inputs must agree. The final f32 state drift accumulates
      # linearly across the decode steps, see assertSsmEvalOrder,
      # the budget-change record in SPEC.md. The bf16 block outputs stay
      # inside the bf16 rounding scale (a few bf16 ulps) of the two orders.
      let gdn = buildGdn()
      Torch.manual_seed(0x5EEDC0DE'u64)
      let x = F.randn(70 * Hidden, F.tensorOptions(F.kBFloat16, F.kCPU))
        .reshape([1, MultichunkSeq, Hidden])

      var ctxOneShot = InferenceContext.init(
        num_layers = 1, batch_size = 1, kv_heads = 2, max_seq = 512,
        head_dim = HeadKDim)
      let outOneShot = gdn(ctxOneShot, x)

      var ctxDecode = InferenceContext.init(
        num_layers = 1, batch_size = 1, kv_heads = 2, max_seq = 512,
        head_dim = HeadKDim)
      var outDrift = 0.0'f64
      for t in 0 ..< MultichunkSeq:
        let outT = gdn(ctxDecode, x.narrow(1, t, 1))
        outDrift = max(outDrift, maxAbsDiff(outT, outOneShot.narrow(1, t, 1)))
      let outMax = outOneShot.to(F.kFloat32).abs().max().item(float64)
      assertEvalOrderOutput(outDrift, outMax,
        msg = "one-shot prefill vs step-decode block output")
      assertConvEvalOrder(ctxOneShot.gdnConvState[0], ctxDecode.gdnConvState[0],
        msg = "one-shot prefill vs step-decode conv state")
      assertSsmEvalOrder(ctxOneShot.gdnSsmState[0], ctxDecode.gdnSsmState[0],
        steps = MultichunkSeq,
        msg = "one-shot prefill vs step-decode final state")
      true

  runCppTest "GDN evaluation order, cross-device variant (device-pair report)":
    proc(): bool =
      let runDev = testDevice()
      echo "    devices: ", compareLine(FixtureDir, runDev)
      # Budget provenance of the cross-device variant:
      # Measured facts: Metal, both orders run on the device.
      # - bf16 block outputs: 0.25 bf16 ulps at the output max 3.375.
      # - conv state: one bf16 step at the differing element, 0.25 ulp-units
      #   at the conv max 7.34375 (the element sits in a lower binade).
      # - f32 state: 1027.4 fp32 ulps at the pair max 1.032 over the T=70
      #   decode, about 15 fp32 ulps per step.
      # CPU reference: the conv states compare bit-equal across the orders.
      # The state drifts one fp32 ulp.
      # Bounds: the conv budget (two bf16 ulp-units at the max) and the
      # linear-in-decode-length state drift bound, both through the
      # evaluation-order checks.
      # Rejection rows: the selftest evaluation-order corpus rejects past both bounds
      # and accepts the sub-bound drifts.
      let gdn = buildGdn(runDev)
      Torch.manual_seed(0x5EEDC0DE'u64)
      let x = F.randn(70 * Hidden, F.tensorOptions(F.kBFloat16, runDev))
        .reshape([1, MultichunkSeq, Hidden])

      var ctxOneShot = InferenceContext.init(
        num_layers = 1, batch_size = 1, kv_heads = 2, max_seq = 512,
        head_dim = HeadKDim)
      let outOneShot = gdn(ctxOneShot, x)

      var ctxDecode = InferenceContext.init(
        num_layers = 1, batch_size = 1, kv_heads = 2, max_seq = 512,
        head_dim = HeadKDim)
      var outDrift = 0.0'f64
      for t in 0 ..< MultichunkSeq:
        let outT = gdn(ctxDecode, x.narrow(1, t, 1))
        outDrift = max(outDrift, maxAbsDiff(outT, outOneShot.narrow(1, t, 1)))
      let outMax = outOneShot.to(F.kFloat32).abs().max().item(float64)
      assertEvalOrderOutput(outDrift, outMax,
        msg = "cross-device block output")
      assertConvEvalOrder(ctxOneShot.gdnConvState[0], ctxDecode.gdnConvState[0],
        msg = "cross-device conv state")
      assertSsmEvalOrder(ctxOneShot.gdnSsmState[0], ctxDecode.gdnSsmState[0],
        steps = MultichunkSeq, msg = "cross-device final state")
      true

  echo "\nAll Qwen3.6 GDN tests passed!"

when isMainModule:
  main()
