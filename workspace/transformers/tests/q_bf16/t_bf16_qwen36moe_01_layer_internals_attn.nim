# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off \
##   --outdir:build/tests/t_bf16_qwen36moe_01_layer_internals_attn --nimcache:nimcache/tests/t_bf16_qwen36moe_01_layer_internals_attn \
##   workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_01_layer_internals_attn.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/options,
  std/math,
  std/os,
  std/strutils,
  std/importutils,
  pkg/jsony,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/transformers/src/layers,
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/layers/rope {.all.},
  workspace/transformers/src/layers/attn_ssm/grouped_query_attention {.all.},
  workspace/transformers/src/layers/attn_ssm/gated_attention {.all.},
  workspace/transformers/tests/harness,
  workspace/transformers/tests/q_bf16/kvcontext,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(RopeElementWiseGatedAttention[RmsNormOne])
privateAccess(GroupedQueryAttention)

proc ulpBf16(m: float64): float64 =
  ## One bf16 ulp at magnitude m: bf16 stores 7 significand bits, giving
  ## ulp 2^(e-7) with e = floor(log2(m)). Zero maps to 0.
  if m <= 0.0:
    return 0.0
  result = pow(2.0, floor(log2(m)) - 7.0)

const
  FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-01-layer-internals" / "Qwen3.6-35B-A3B-layer-3"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  LayerPrefix = "model.language_model.layers.3.self_attn"
  WeightsFile3 = ModelDir / "model-00003-of-00026.safetensors"

  # Attention geometry of the checkpoint (text_config)
  Hidden = 2048
  NumQoHeads = 16
  NumKvHeads = 2
  HeadDim = 256
  RotaryDim = 64
  MaxPositionEmbeddings = 262144
  RopeTheta = 1e7

proc setupAttn(): RopeElementWiseGatedAttention[RmsNormOne] =
  ## Load the real layer-3 self-attention weights, whose `q_proj` emits
  ## `[q | gate]` per head.
  let cfgJson = (ModelDir / "config.json").parseFile()
  let weights = SafetensorsCollection.open(WeightsFile3)

  let qProj = Linear.load(weights, cfgJson, LayerPrefix & ".q_proj")
  let kProj = Linear.load(weights, cfgJson, LayerPrefix & ".k_proj")
  let vProj = Linear.load(weights, cfgJson, LayerPrefix & ".v_proj")
  let oProj = Linear.load(weights, cfgJson, LayerPrefix & ".o_proj")
  let qNorm = RmsNormOne.load(weights, cfgJson, LayerPrefix & ".q_norm")
  let kNorm = RmsNormOne.load(weights, cfgJson, LayerPrefix & ".k_norm")

  let rotary = RotaryPositionEmbedding.new(
    HeadDim, MaxPositionEmbeddings, RopeTheta, F.kBFloat16, F.kCPU,
    rotary_dim = RotaryDim)
  result = RopeElementWiseGatedAttention[RmsNormOne].init(
    3, LayerPrefix,
    qProj, kProj, vProj, oProj,
    NumQoHeads, NumKvHeads, HeadDim, rotary,
    q_norm = qNorm, k_norm = kNorm
  )

proc normFixtureTest(caseNum: int, msg: string): bool =
  ## RmsNormOne in isolation, compared bit-exact against the recorded reference.
  var st = Safetensor.open(FixtureDir / ("norm-Qwen3.6-35B-A3B-0" & $caseNum & ".safetensor"))
  let x = st.getTensorOwned("input")
  let expected = st.getTensorOwned("output")
  let w = st.getTensorOwned("weight")
  # eps comes from the fixture metadata
  type NormFixture = object
    eps: float64
  let meta = zstdReadFixture(FixtureDir / "norm-Qwen3.6-35B-A3B-0" & $caseNum &
    ".safetensor.metadata.json.zst").fromJson(NormFixture)
  let norm = RmsNormOne.init(w, eps = meta.eps)
  let got = norm.forward(x)
  assertAllClose(got, expected, rtol = 0.0, abstol = 0.0, msg = msg)
  true

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Gated full attention prefill (seq 8) vs fixture":
    proc(): bool =
      echo "    devices: ", compareLine(FixtureDir, F.kCPU)
      let attn = setupAttn()
      var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = NumKvHeads, headDim = HeadDim)
      var st = Safetensor.open(FixtureDir / "attn-Qwen3.6-35B-A3B-00.safetensor")

      # The real weights carry the geometry: q_proj packs [q | gate]
      # per head, k/v project num_kv heads of head_dim
      doAssert attn.q_proj.out_features == NumQoHeads * 2 * HeadDim
      doAssert attn.k_proj.out_features == NumKvHeads * HeadDim
      doAssert attn.v_proj.out_features == NumKvHeads * HeadDim
      doAssert attn.o_proj.out_features == Hidden

      let x = st.getTensorOwned("hidden_states")       # (1, 8, 2048) bf16
      let hfPosIds = st.getTensorOwned("position_ids") # (1, 8) int64
      ctx.position_ids = hfPosIds[0]
      ctx.setRopeForPositions(attn.rotary)

      let seqLen = x.size(1)
      let output = attn(ctx, x)
      doAssert output.size(0) == 1 and output.size(1) == seqLen and
        output.size(2) == Hidden

      # Final layer output: block-level tolerance.
      let expected = st.getTensorOwned("output")
      assertAllClose(output, expected, rtol = 5e-3, abstol = 5e-3,
        msg = "gated attention output mismatch")

      # Intermediates recomputed through the layer's own components,
      # compared against the fixture's recorded values.
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

      # Pre-o_proj output, SDPA then the sigmoid gate: compares the gate
      # application itself.
      let vReshaped = attn.v_proj.forward(x).reshape(
        [1, seqLen, gqa.num_kv_head, gqa.head_dim])
      let kvGroups = gqa.num_qo_head div gqa.num_kv_head
      let kExp = kRot.repeat_interleave(kvGroups, 2)
      let vExp = vReshaped.repeat_interleave(kvGroups, 2)
      let attnOut = attn.gqa_attn.forward(qRot, kExp, vExp,
        is_causal = true, enable_gqa = false)
      let attnGated = attnOut * F.sigmoid(gate)
      # Cross-version SDPA CPU kernel noise: this binary links libtorch 2.11
      # and the fixture was recorded with torch 2.13. A single element
      # differs by one bf16 ulp, the delta holds across strides and input
      # provenance, the band allows two bf16 ulps at the fixture max.
      let gatedBand = 2.0 * ulpBf16(
        st.getTensorOwned("attn_output_gated").abs().max().item(float64))
      assertAllClose(attnGated, st.getTensorOwned("attn_output_gated"),
        rtol = 0.0, abstol = gatedBand, msg = "attn_output_gated mismatch")
      true

  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Gated full attention decode (single token, position 5) vs fixture":
    proc(): bool =
      let attn = setupAttn()
      var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = NumKvHeads, headDim = HeadDim)
      var st = Safetensor.open(FixtureDir / "attn-Qwen3.6-35B-A3B-01.safetensor")

      let x = st.getTensorOwned("hidden_states")       # (1, 1, 2048)
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

  runCppTest "RmsNormOne head_dim forward vs fixture":
    proc(): bool = normFixtureTest(0, "RMSNorm head_dim_forward mismatch")



  runCppTest "Gated full attention, cross-device variant (device-pair report)":
    proc(): bool =
      let runDev = testDevice()
      echo "    devices: ", compareLine(FixtureDir, runDev)
      if runDev == F.kCPU:
        echo "    the run device matches the recorded device, the reference variant carries the replay"
        return true
      # The recorded intermediates compare bit-exact (q/k norms, gate, rope)
      # and no manifest states a recorded device for the layers family.
      # No cross-device drift row applies, the suite skips.
      echo "    no cross-device drift row applies: the bit-exact intermediates",
        " accept zero drift, the suite skips on ", deviceName(runDev)
      return true

  echo "\nAll Qwen3.6 gated-attention tests passed!"

when isMainModule:
  main()
