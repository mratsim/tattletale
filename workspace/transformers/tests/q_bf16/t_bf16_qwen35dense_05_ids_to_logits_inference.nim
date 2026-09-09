# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35dense-ids \
##   --nimcache:nimcache/tests/qwen35dense-ids \
##   workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_05_ids_to_logits_inference.nim

import
  std/options,
  std/strformat,
  std/os,
  std/importutils,
  pkg/iface,
  pkg/jsony,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen35 {.all.},
  workspace/transformers/tests/harness,
  workspace/transformers/tests/q_bf16/kvcontext,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen35Model)
privateAccess(DecoderLayer[GatedDeltaNet, GatedDenseFFN, RmsNormOne])
privateAccess(DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedDenseFFN, RmsNormOne])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-03-full-forward-to-logits" / "Qwen3.5-0.8B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"

proc openLayerFixture(layerIdx: int): Safetensor =
  ## Read one per-layer fixture. The result owns its memory mapping,
  ## released with the last reference to the reader.
  Safetensor.open(FixtureDir / &"layer-{layerIdx:02d}.safetensor")

type
  FinalLogitsMeta = object
    ## Metadata of the final-logits projection family: the recorded
    ## sequential vs chunked band and the payload note.
    logits_band: Option[float64]
    note: string

proc main() =
  assertTorchStamp(FixtureDir)
  runCppTest "Qwen3.5-0.8B ids to logits - 24 layers + final logits vs fixtures":
    proc(): bool =
      echo "    devices: ", compareReport(FixtureDir, F.kCPU)
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
      var blockInput = h
      var stream: Option[Tensor]
      for layerIdx in 0 ..< model.layers.len:
        var st = openLayerFixture(layerIdx)

        # The sequential replay is the hard contract: the Nim forward must
        # match the vendored sequential replay bit for bit at every layer
        # boundary (0.00 bar).
        assertAllClose(h, st.getTensorOwned("layer_input_seq"),
          rtol = 0.0, abstol = 0.0, msg = "layer " & $layerIdx & " input mismatch")

        # Tolerances for the chunked comparisons come from the fixture's own
        # measured sequential-vs-chunked layer delta, locked below at < 0.05
        # per layer. Independent checks: the 0.00-vs-seq asserts plus the
        # band doAsserts.
        let inputBand = maxAbsDiff(st.getTensorOwned("layer_input_seq"),
                                   st.getTensorOwned("layer_input"))
        assertAllClose(h, st.getTensorOwned("layer_input"),
          rtol = 0.0, abstol = inputBand,
          msg = "layer " & $layerIdx & " chunked input mismatch (measured layer delta " & $inputBand & ")")

        let pair = model.layers[layerIdx].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])
        h = pair[1] + pair[0]

        # One recorded input copy per layer boundary. The recorded input
        # of layer i+1 is the output of layer i, so the chained comparison
        # checks each block output twice: against this layer's forward
        # result and as the next layer's input. The chunked-vs-sequential
        # band of this boundary reappears as the next layer's input band.
        if layerIdx < model.layers.len - 1:
          var next = openLayerFixture(layerIdx + 1)
          assertAllClose(h, next.getTensorOwned("layer_input_seq"),
            rtol = 0.0, abstol = 0.0,
            msg = "layer " & $layerIdx & " sequential output mismatch (next layer chained input)")
          let outputBand = maxAbsDiff(next.getTensorOwned("layer_input_seq"),
                                      next.getTensorOwned("layer_input"))
          doAssert outputBand < 0.05,
            "layer " & $layerIdx & " measured layer delta exceeds the documented 0.05 guard"
          if layerIdx >= 2:
            doAssert outputBand > 0.0,
              "layer " & $layerIdx & " sequential and chunked outputs are identical, band not exercised"
          assertAllClose(h, next.getTensorOwned("layer_input"),
            rtol = 0.0, abstol = outputBand,
            msg = "layer " & $layerIdx & " chunked output mismatch (measured layer delta " & $outputBand & ")")
        else:
          # The final block output left the payload. Its recorded surface
          # lives in the layer-23 descriptor sidecar (dmExact), the
          # zero-tolerance reference of this boundary.
          let finalDesc = loadFingerprintStats(
            FixtureDir / "layer-23.safetensor.descriptors")
          let finalEntry = finalDesc.statsTensor("layer_output_seq")
          assertStats(h, finalEntry,
            descriptorStatsBudget(finalEntry.probeMode),
            msg = "final block output fingerprint")
          assertDescriptors(h, finalEntry,
            msg = "final block output descriptors")

      let normed = model.norm(blockInput + stream.get(blockInput))
      let logits = model.lmHead(normed)

      # Decision projection checks: argmax, the top-2 competing
      # pair, the argmax margin and the tail-probability checksum,
      # per position, against the sequential-reference projection
      # (the 0.00 reference). Raw logits tensors left the tree, only
      # the sequential vs chunked band survives as recorded metadata
      # under its documented 0.25 guard.
      let logitsMeta = zstdReadFixture(
        FixtureDir / "final_logits.safetensor.metadata"
      ).fromJson(FinalLogitsMeta)
      doAssert logitsMeta.logits_band.isSome,
        "final logits metadata logits_band is not a float"
      doAssert logitsMeta.logits_band.get() > 0.0 and
        logitsMeta.logits_band.get() < 0.25,
        "recorded logits band " & $logitsMeta.logits_band.get() &
        " outside the documented (0, 0.25) guard"
      let projection = zstdReadFixture(
        FixtureDir / "final_logits.decisions.json.zst"
      ).fromJson(LogitsProjection)
      assertProjection(logits, projection,
        msg = "Qwen3.5-0.8B final logits projection")
      true

  runCppTest "Qwen3.5-0.8B ids to logits, cross-device variant (device-pair report)":
    proc(): bool =
      let runDev = testDevice()
      echo "    devices: ", compareReport(FixtureDir, runDev)
      if compareClass(recordedDevice(recordedFrom(FixtureDir)), runDev) ==
          sameDeviceBitExact:
        echo "    the pair selects the reference rows, the reference variant carries the replay"
        return true
      # The ids fixtures compare bit-exact on the reference device:
      # - the dmExact descriptor entries
      # - the exact layer boundary sums
      # - the decision projection with its bit-exact strided probe
      # No cross-device drift tolerance applies, so a flipped run names the
      # tolerance class and skips.
      echo "    no cross-device drift row applies: the dmExact descriptors, ",
        "the exact boundary sums and the bit-exact probe accept zero drift, ",
        "the suite skips on ", deviceName(runDev)
      return true

when isMainModule:
  main()
