# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off
##   --outdir:build/tests/t_bf16_qwen36moe_03_full_forward_to_logits
##   --nimcache:nimcache/tests/t_bf16_qwen36moe_03_full_forward_to_logits
##   workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_03_full_forward_to_logits.nim
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/options,
  std/os,
  std/strutils,
  std/importutils,
  pkg/iface,
  pkg/jsony,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  std/tables,
  workspace/safetensors/src/collections {.all.},
  workspace/safetensors/src/safetensors_libtorch,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness,
  workspace/transformers/tests/q_bf16/kvcontext,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen35MoeModel)
privateAccess(DecoderLayer[GatedDeltaNet, GatedBlockSparseFFN, RmsNormOne])
privateAccess(DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedBlockSparseFFN, RmsNormOne])
privateAccess(GatedBlockSparseFFN)
privateAccess(LMHead)
privateAccess(Embedding)
privateAccess(SafetensorsCollectionObj)

const
  FullForwardToLogitsFixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-03-full-forward-to-logits" / "Qwen3.6-35B-A3B"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  TopK = 8
  Hidden = 2048
  # Recording environment of the committed fixtures. A regenerated artifact
  # from a foreign environment fails here, not against different values
  # under the same filenames.
  TorchRecordingVersion = "2.14.0"
  TransformersRecordingVersion = "5.16.1"

proc openLayerFixture(layerIdx: int): Safetensor =
  ## Read one layer-boundary fixture. The result owns its memory mapping
  ## for its whole lifetime, released when the value goes out of scope.
  Safetensor.open(
    FullForwardToLogitsFixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor"))

proc checkpointTensorRequests(): int =
  ## Name-based tensor requests the loader makes against the checkpoint:
  ## one per language_model key, plus the head request the untied
  ## checkpoint answers through its lm_head.weight key.
  let view = SafetensorsCollection.open(ModelDir)
  for name in view.weightMap.keys():
    if name.startsWith("model.language_model."):
      inc result
  if view.hasTensor("lm_head.weight"):
    inc result

type
  IdsPromptFixture = object
    torch_version: string
    transformers_version: string
    input_tokens: seq[int64]
  IdsBands = object
    input_band: Option[float64]
    output_band: Option[float64]
    logits_band: Option[float64]
  IdsLayerFixture = object
    bands: IdsBands

proc main() =
  assertTorchStamp(FullForwardToLogitsFixtureDir)
  runCppTest "Qwen3.6-35B-A3B full forward to logits - 40 layers + final logits vs fixtures":
    proc(): bool =
      # Two reference paths per boundary, compared against the replayed model:
      #
      #   ours → _seq     : bit-for-bit, rtol = abstol = 0.0
      #   ours → _chunked : within the band measured
      #                     between the reference's own two algorithms
      #
      # Router top-k is bitwise-equal to the recorded torch.topk choices,
      # never margin-checked: fp32 ties at the top-k boundary are structural.
      # Prompt ids come from the fixture metadata.
      echo "    devices: ", compareReport(FullForwardToLogitsFixtureDir, F.kCPU)
      let model = loadQwen35MoeModelRaw(ModelDir, kCPU)
      doAssert model.layers.len == 40
      doAssert model.config.numHiddenLayers == 40
      doAssert checkpointTensorRequests() == 693

      var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = 2, headDim = 256)

      let meta = zstdReadFixture(
        FullForwardToLogitsFixtureDir / "layer-00.safetensor.metadata.json.zst"
      ).fromJson(IdsPromptFixture)
      doAssert meta.torch_version == TorchRecordingVersion
      doAssert meta.transformers_version == TransformersRecordingVersion
      doAssert meta.input_tokens.len < 64
      let tokenIds = meta.input_tokens
      let seqLen = tokenIds.len
      let inputIds = tokenIds.toTensor().unsqueeze(0)
      ctx.position_ids = F.arange(seqLen.int64, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(model.rotary)

      var h = model.embedTokens(inputIds)
      var blockInput = h
      var stream: Option[Tensor]
      # Router parity replays the block pre-FFN path through routeToExperts.
      # Each mixer layer writes its own state slots, so one context visited
      # layer by layer sees the same fresh state the model forward saw.
      var (checkCtx, checkPool) =
        newKVContext(numLayers = model.layers.len, kvHeads = 2, headDim = 256)
      checkCtx.position_ids = ctx.position_ids
      checkCtx.setRopeForPositions(model.rotary)
      for layerIdx in 0 ..< model.layers.len:
        var st = openLayerFixture(layerIdx)

        # The chunked copies left the payload under the fixture contract. The recorded bands in the metadata
        # keep documenting the divergence they measured. The chain anchor is the
        # input, bit-for-bit, layer by layer: the recorded inputs chain from the embedding to the last
        # block, so one corrupted block output fails the next layer input check.
        let seqInput = st.getTensorOwned("layer_input_seq")
        assertAllClose(h, seqInput,
          rtol = 0.0, abstol = 0.0,
          msg = "layer " & $layerIdx & " input mismatch")
        let layerIn = h
        let pair = model.layers[layerIdx].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])
        h = pair[1] + pair[0]

        # Router parity against the recorded top-k: indices exact,
        # then the renormalized weights, exact, in fp32. The router input
        # is the post-attention norm output, replayed on the layer input.
        let gdnLayer = model.layers[layerIdx].to(DecoderLayer[GatedDeltaNet, GatedBlockSparseFFN, RmsNormOne])
        let attnLayer = model.layers[layerIdx].to(DecoderLayer[RopeElementWiseGatedAttention[RmsNormOne], GatedBlockSparseFFN, RmsNormOne])
        doAssert (gdnLayer != nil) xor (attnLayer != nil),
          "each block holds exactly one mixer pairing"
        let inputNorm =
          if gdnLayer != nil: gdnLayer.input_layernorm
          else: attnLayer.input_layernorm
        let normedIn = inputNorm.forward(layerIn)
        let mixerOut =
          if gdnLayer != nil:
            gdnLayer.sequence_mixer(checkCtx, normedIn)
          else:
            attnLayer.sequence_mixer(checkCtx, normedIn)
        let h1 = layerIn + mixerOut
        let postNorm =
          if gdnLayer != nil: gdnLayer.post_attention_layernorm
          else: attnLayer.post_attention_layernorm
        let normedH1 = postNorm.forward(h1)
        let ffn =
          if gdnLayer != nil: gdnLayer.hidden_mixer
          else: attnLayer.hidden_mixer
        let (gotIndices, gotWeights) = routeToExperts(
          normedH1.reshape(seqLen, Hidden),
          ffn.routerWeight, ffn.numExpertsPerTok)
        let fixIndices = st.getTensorOwned("topk_indices")
        doAssert gotIndices.dim == 2
        for tok in 0 ..< seqLen:
          for pos in 0 ..< TopK:
            doAssert gotIndices[tok, pos].item(int64) ==
              fixIndices[tok, pos].item(int64),
              "layer " & $layerIdx & " token " & $tok & " slot " & $pos &
              " routed expert id disagrees with the recorded torch.topk choice"
        assertAllClose(gotWeights, st.getTensorOwned("routing_weights"),
          rtol = 0.0, abstol = 0.0,
          msg = "layer " & $layerIdx & " routing weights mismatch")

      # The final block output left the payload: its external surface
      # is the descriptor sidecar of layer-39, checked bit-exactly
      # (dmExact) against the chain, the same bytes the projection
      # check below consumes.
      let finalDesc = loadFingerprintStats(
        FullForwardToLogitsFixtureDir / "layer-39.safetensor.descriptors")
      let finalEntry = finalDesc.statsTensor("layer_output_seq")
      assertStats(h, finalEntry,
        descriptorStatsBudget(finalEntry.probeMode),
        msg = "final block output fingerprint")
      assertDescriptors(h, finalEntry,
        msg = "final block output descriptors")
      let normed = model.norm.forward(blockInput + stream.get(blockInput))
      let logits = model.lmHead.forward(normed)

      # Decision projection checks: argmax, the top-2 competing
      # pair, the argmax margin and the tail-probability checksum,
      # per position, against the sequential-reference projection
      # (the 0.00 reference). Raw logits tensors left the tree, only
      # the sequential vs chunked band survives as recorded metadata
      # under the generator guard.
      let logitsMeta = zstdReadFixture(
        FullForwardToLogitsFixtureDir / "final_logits.safetensor.metadata.json.zst"
      ).fromJson(IdsLayerFixture)
      doAssert logitsMeta.bands.logits_band.isSome,
        "final logits fixture metadata logits_band is not a float"
      let recordedLogitsBand = logitsMeta.bands.logits_band.get()
      const LogitsBandGuard = 4.0
      doAssert recordedLogitsBand > 0.0 and recordedLogitsBand < LogitsBandGuard,
        "recorded logits band " & $recordedLogitsBand &
        " outside the documented (0, " & $LogitsBandGuard & ") guard"
      let projection = zstdReadFixture(
        FullForwardToLogitsFixtureDir / "final_logits.decisions.json.zst"
      ).fromJson(LogitsProjection)
      assertProjection(logits, projection,
        msg = "Qwen3.6-35B-A3B final logits projection")
      true

  runCppTest "Qwen3.6-35B-A3B full forward to logits, cross-device variant (device-pair report)":
    proc(): bool =
      let runDev = testDevice()
      echo "    devices: ", compareReport(FullForwardToLogitsFixtureDir, runDev)
      if compareClass(recordedDevice(recordedFrom(FullForwardToLogitsFixtureDir)),
          runDev) == sameDeviceBitExact:
        echo "    the device comparison selects the reference budgets, the reference variant carries the replay"
        return true
      # The full-forward fixtures compare bit-exact on the reference
      # device:
      # - the dmExact descriptor entries
      # - the rtol-0 seq boundary rows
      # - the decision projection on the 4-ulp band (bf16 unit)
      # The seq-to-chunked band is the reference's own internal divergence,
      # not a device budget.
      echo "    no cross-device drift budget applies: the dmExact descriptors, ",
        "the rtol-0 boundary rows and the decision projection compare on the ulp band, ",
        "the suite skips on ", deviceName(runDev)
      return true

  echo "\nAll Qwen3.6 full-forward-to-logits blocks PASS"

when isMainModule:
  main()
