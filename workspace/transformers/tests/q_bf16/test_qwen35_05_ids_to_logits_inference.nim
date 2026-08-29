# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-ids \
##   --nimcache:nimcache/tests/qwen35-ids \
##   workspace/transformers/tests/q_bf16/test_qwen35_05_ids_to_logits_inference.nim

import
  std/memfiles,
  std/strformat,
  std/os,
  std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/tests/transformers_testutils,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen35Model)
privateAccess(Qwen35DecoderLayer)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "ids-inference" / "Qwen3.5-0.8B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc openLayerFixture(layerIdx: int): (MemFile, Safetensor) =
  ## Open one per-layer fixture. The memfile must stay open while the
  ## Safetensor is in use (zero-copy views into the file).
  let memFile = memFiles.open(FixtureDir / &"layer-{layerIdx:02d}.safetensor", mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc openFinalLogits(): (MemFile, Safetensor) =
  ## Open the final logits fixture.
  let memFile = memFiles.open(FixtureDir / "final_logits.safetensor", mode = fmRead)
  result = (memFile, safetensors.load(memFile))

proc main() =
  runCppTest "Qwen3.5-0.8B ids to logits - 24 layers + final logits vs fixtures":
    proc(): bool =
      let model = loadQwen35ModelRaw(ModelPath, kCPU)
      doAssert model.layers.len == 24

      var (ctx, pool) = newKVContext(numLayers = 24, kvHeads = 2, headDim = 256)

      # "Hello, how are you?" as 6 token ids, matching the fixture metadata.
      let inputIds = @[9419'i64, 11, 1204, 513, 488, 30].toTensor().unsqueeze(0)

      ctx.position_ids = F.arange(6, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(model.rotary)

      # Manual layer loop mirroring the model forward, capturing each hidden
      # state so every layer can be compared against its fixture.
      var h = model.embedTokens(inputIds)
      for layerIdx in 0 ..< model.layers.len:
        var (memFile, st) = openLayerFixture(layerIdx)
        defer: close(memFile)

        # The sequential replay is the hard contract: the Nim forward must
        # match the vendored sequential replay bit for bit at every layer
        # boundary (0.00 bar).
        assertAllClose(h, st.getTensorOwned("layer_input_seq"),
          rtol = 0.0, abstol = 0.0, msg = "layer " & $layerIdx & " sequential input mismatch")

        # Tolerances for the chunked comparisons come from the fixture's own
        # sequential-vs-chunked band, locked below at < 0.05 per layer.
        # Independent checks: the 0.00-vs-seq asserts plus the band-regime doAsserts.
        let inputBand = maxAbsDiff(st.getTensorOwned("layer_input_seq"),
                                   st.getTensorOwned("layer_input"))
        assertAllClose(h, st.getTensorOwned("layer_input"),
          rtol = 0.0, abstol = inputBand,
          msg = "layer " & $layerIdx & " chunked input mismatch (fixture band " & $inputBand & ")")

        h = model.layers[layerIdx](ctx, h)

        assertAllClose(h, st.getTensorOwned("layer_output_seq"),
          rtol = 0.0, abstol = 0.0, msg = "layer " & $layerIdx & " sequential output mismatch")
        let outputBand = maxAbsDiff(st.getTensorOwned("layer_output_seq"),
                                    st.getTensorOwned("layer_output"))
        doAssert outputBand < 0.05,
          "layer " & $layerIdx & " fixture band exceeds the documented 0.05 guard"
        if layerIdx >= 2:
          doAssert outputBand > 0.0,
            "layer " & $layerIdx & " sequential and chunked outputs are identical, band not exercised"
        assertAllClose(h, st.getTensorOwned("layer_output"),
          rtol = 0.0, abstol = outputBand,
          msg = "layer " & $layerIdx & " chunked output mismatch (fixture band " & $outputBand & ")")

      let normed = model.norm(h)
      let logits = model.lmHead(normed)

      var (memFileF, stF) = openFinalLogits()
      defer: close(memFileF)

      # Same contract as the layers: 0.00 against the sequential replay, and
      # the fixture's own sequential vs chunked band (locked < 0.25, the
      # generator's documented guard) against the chunked forward.
      assertAllClose(logits, stF.getTensorOwned("logits_seq"),
        rtol = 0.0, abstol = 0.0, msg = "logits vs sequential replay mismatch")
      let logitsBand = maxAbsDiff(stF.getTensorOwned("logits_seq"),
                                  stF.getTensorOwned("logits"))
      doAssert logitsBand < 0.25,
        "logits fixture band exceeds the documented 0.25 guard"
      doAssert logitsBand > 0.0,
        "sequential and chunked logits are identical, band not exercised"
      assertAllClose(logits, stF.getTensorOwned("logits"),
        rtol = 0.0, abstol = logitsBand,
        msg = "logits vs chunked forward mismatch (fixture band " & $logitsBand & ")")
      true

when isMainModule:
  main()
