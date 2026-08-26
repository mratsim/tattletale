# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-long-residual \
##   --nimcache:nimcache/tests/qwen35-long-residual \
##   workspace/transformers/tests/q_bf16/test_qwen3_5_04_long_residual_3_blocks.nim

import
  std/memfiles,
  std/strformat,
  std/os,
  std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/models/qwen3_5 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen3_5Model)
privateAccess(Qwen35DecoderLayer)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "long-residual-3-block" / "Qwen3.5-0.8B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc openBlock(layerIdx: int): (MemFile, Safetensor) =
  ## Open one block fixture. The memfile must stay open while the Safetensor
  ## is in use (zero-copy views into the file).
  let memFile = memFiles.open(FixtureDir / &"block-{layerIdx:02d}.safetensor", mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc main() =
  # ──────────────────────────────────────────────────────────────────────────
  # 3-block chain (layers 0-2, all GDN) vs the sequential + chunked fixtures
  # ──────────────────────────────────────────────────────────────────────────
  runCppTest "Qwen3.5 long-residual 3-block chain + local residual invariant":
    proc(): bool =
      let model = loadQwen3_5ModelRaw(ModelPath, kCPU)
      var ctx = InferenceContext.init(24, 1, 2, 512, 256)

      var (memFile0, st0) = openBlock(0)
      defer: close(memFile0)
      var hidden = st0.getTensorOwned("layer_input_seq")

      for i in 0 ..< 3:
        var (memFile, st) = openBlock(i)
        defer: close(memFile)
        # Sequential chain inputs/outputs at 0.00, vendored chunked chain
        # at 5e-3. The T=4 single-chunk chain fixture has chunked ==
        # sequential bit-exact, so the 5e-3-vs-chunked asserts are
        # degenerate; the real contracts are the 0.00-vs-seq asserts and
        # the recomposed local-residual invariant. Multi-chunk divergence
        # is covered by test 03's T=70 band test.
        assertAllClose(hidden, st.getTensorOwned("layer_input_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential input mismatch")
        assertAllClose(hidden, st.getTensorOwned("layer_input"),
          rtol = 5e-3, abstol = 5e-3, msg = "chain layer " & $i & " chunked input mismatch")
        let layerOut = model.layers[i](ctx, hidden)
        assertAllClose(layerOut, st.getTensorOwned("layer_output_seq"),
          rtol = 0.0, abstol = 0.0, msg = "chain layer " & $i & " sequential output mismatch")
        assertAllClose(layerOut, st.getTensorOwned("layer_output"),
          rtol = 5e-3, abstol = 5e-3, msg = "chain layer " & $i & " chunked output mismatch")

        # Local residual invariant: block output = input + attn delta + mlp
        # delta. The deltas are recomputed through the layer components and
        # the sum is compared against the sequential fixture (0.00) and the
        # layer forward (0.00, same op order). A fresh context per block
        # keeps the GDN state at the sequence start, matching the fixture.
        let layer = model.layers[i]
        doAssert layer.layer_type == "linear_attention",
          "chain layers 0-2 are GDN layers, got " & layer.layer_type
        var invCtx = InferenceContext.init(24, 1, 2, 512, 256)
        let hNorm = layer.input_layernorm(hidden)
        let attnOut = layer.gdn(invCtx, hNorm)
        let h1 = hidden + attnOut
        let hNorm2 = layer.post_attention_layernorm(h1)
        let mlpOut = layer.mlp(hNorm2)
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
