# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35dense-long-residual \
##   --nimcache:nimcache/tests/qwen35dense-long-residual \
##   workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_04_long_residual_3_blocks.nim

import
  std/options,
  std/strformat,
  std/os,
  std/importutils,
  pkg/iface,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen35Model)
privateAccess(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "long-residual-3-block" / "Qwen3.5-0.8B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc openLayer(layerIdx: int): Safetensor =
  ## Read one block fixture. The result owns its memory mapping,
  ## released with the last reference to the reader.
  Safetensor.open(FixtureDir / &"block-{layerIdx:02d}.safetensor")

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # 3-block chain (layers 0-2, all GDN) vs the sequential + chunked fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Qwen3.5 long-residual 3-block chain + local residual invariant":
    proc(): bool =
      let model = loadQwen35ModelRaw(ModelPath, kCPU)
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)

      var st0 = openLayer(0)
      var hidden = st0.getTensorOwned("layer_input_seq")
      var blockInput = hidden
      var stream: Option[Tensor]

      for i in 0 ..< 3:
        var st = openLayer(i)
        # Sequential chain inputs/outputs at 0.00, vendored chunked chain
        # at 5e-3. The T=4 single-chunk chain fixture has chunked ==
        # sequential bit-exact, so the 5e-3-vs-chunked asserts are
        # degenerate. The real contracts are the 0.00-vs-seq asserts and
        # the recomposed local-residual invariant. Multi-chunk divergence
        # is covered by test 03's T=70 band test.
        assertAllClose(hidden, st.getTensorOwned("layer_input_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential input mismatch")
        assertAllClose(hidden, st.getTensorOwned("layer_input"),
          rtol = 5e-3, abstol = 5e-3, msg = "chain layer " & $i & " chunked input mismatch")
        let pair = model.layers[i].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])
        let layerOut = pair[1] + pair[0]
        assertAllClose(layerOut, st.getTensorOwned("layer_output_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential output mismatch")
        assertAllClose(layerOut, st.getTensorOwned("layer_output"),
          rtol = 5e-3, abstol = 5e-3, msg = "chain layer " & $i & " chunked output mismatch")

        # Local residual invariant: block output = input + attn delta + mlp
        # delta. The deltas are recomputed through the layer components and
        # the sum is compared against the sequential fixture (0.00) and the
        # layer forward (0.00, same op order). A fresh context per block
        # keeps the GDN state at the sequence start, matching the fixture.
        let layer = model.layers[i].to(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])
        doAssert layer != nil, "chain layers 0-2 hold the Gated DeltaNet pairing"
        var invCtx = InferenceContext.init(24, 1, 2, 512, 256)
        let hNorm = layer.input_layernorm(hidden)
        let attnOut = layer.sequence_mixer(invCtx, hNorm)
        let h1 = hidden + attnOut
        let hNorm2 = layer.post_attention_layernorm(h1)
        let mlpOut = layer.hidden_mixer.forward(hNorm2)
        let deltaSum = h1 + mlpOut
        assertAllClose(deltaSum, st.getTensorOwned("layer_output_seq"),
          rtol = 0.0, abstol = 0.0,
          msg = "local residual invariant mismatch, layer " & $i)
        assertAllClose(deltaSum, layerOut,
          rtol = 0.0, abstol = 0.0,
          msg = "layer forward != input + attn + mlp deltas, layer " & $i)
        hidden = layerOut
      true

when isMainModule:
  main()
