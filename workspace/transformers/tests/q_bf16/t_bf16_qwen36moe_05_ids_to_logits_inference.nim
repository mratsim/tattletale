# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --verbosity:0 --hints:off --warnings:off
##   --outdir:build/tests/t_bf16_qwen36moe_05_ids_to_logits_inference
##   --nimcache:nimcache/tests/t_bf16_qwen36moe_05_ids_to_logits_inference
##   workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_05_ids_to_logits_inference.nim
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
  workspace/transformers/tests/transformers_testutils,
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
  IdsInferenceFixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "ids-inference" / "Qwen3.6-35B-A3B"
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  TopK = 8
  Hidden = 2048
  # Recording environment of the committed fixtures. A regenerated artifact
  # from a foreign environment fails here, not against different values
  # under the same filenames.
  TorchRecordingVersion = "2.11.0"
  TransformersRecordingVersion = "5.16.0.dev0"

proc openLayerFixture(layerIdx: int): Safetensor =
  ## Read one layer-boundary fixture. The result owns its memory mapping
  ## for its whole lifetime, released when the value goes out of scope.
  Safetensor.open(
    IdsInferenceFixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor"))

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
  runCppTest "Qwen3.6-35B-A3B ids to logits - 40 layers + final logits vs fixtures":
    proc(): bool =
      # Two reference rails per boundary, compared against the replayed model:
      #
      #   ours → _seq     : bit-for-bit, rtol = abstol = 0.0
      #   ours → _chunked : within the band measured
      #                     between the reference's own two algorithms
      #
      # Router top-k is bitwise-equal to the recorded torch.topk choices,
      # never margin-checked: fp32 ties at the top-k boundary are structural.
      # Prompt ids come from the fixture metadata.
      let model = loadQwen35MoeModelRaw(ModelDir, kCPU)
      doAssert model.layers.len == 40
      doAssert model.config.numHiddenLayers == 40
      doAssert checkpointTensorRequests() == 693

      var (ctx, pool) = newKVContext(numLayers = 40, kvHeads = 2, headDim = 256)

      let meta = readFixture(
        IdsInferenceFixtureDir / "layer-00.safetensor.metadata.json.zip"
      ).fromJson(IdsPromptFixture)
      doAssert meta.torch_version == TorchRecordingVersion
      doAssert meta.transformers_version == TransformersRecordingVersion
      doAssert meta.input_tokens.len < 64
      let tokenIds = meta.input_tokens
      let seqLen = tokenIds.len
      let inputIds = tokenIds.toTensor().unsqueeze(0)
      ctx.position_ids = F.arange(seqLen.int64, F.tensorOptions(F.kInt64, F.kCPU))
      ctx.setRopeForPositions(model.rotary)

      # Ceiling for the recorded bands, the binding tolerance is each band.
      const IdsBandGuard = 2.0

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
      # An all-zero band everywhere would make the chunked comparisons
      # vacuous, at least one band must be positive.
      var exercisedInputBand = false
      var exercisedOutputBand = false
      for layerIdx in 0 ..< model.layers.len:
        var st = openLayerFixture(layerIdx)

        # Sequential rail: bit-for-bit. Chunked rail: within the recorded
        # band, re-measured from the fixture pair before use.
        let seqInput = st.getTensorOwned("layer_input_seq")
        let chunkedInput = st.getTensorOwned("layer_input")
        assertAllClose(h, seqInput,
          rtol = 0.0, abstol = 0.0,
          msg = "layer " & $layerIdx & " sequential input mismatch")
        let layerMeta = readFixture(
          IdsInferenceFixtureDir / ("layer-" & ($layerIdx).align(2, '0') &
          ".safetensor.metadata.json.zip")
        ).fromJson(IdsLayerFixture)
        doAssert layerMeta.bands.input_band.isSome,
          "layer " & $layerIdx & " fixture metadata input_band is not a float"
        doAssert layerMeta.bands.output_band.isSome,
          "layer " & $layerIdx & " fixture metadata output_band is not a float"
        let recordedInputBand = layerMeta.bands.input_band.get()
        let recordedOutputBand = layerMeta.bands.output_band.get()
        let inputBand = maxAbsDiff(seqInput, chunkedInput)
        doAssert inputBand == recordedInputBand,
          "layer " & $layerIdx & " measured input band " & $inputBand &
          " disagrees with the recorded band " & $recordedInputBand
        if inputBand > 0.0:
          exercisedInputBand = true
        doAssert inputBand < IdsBandGuard,
          "layer " & $layerIdx & " fixture input band " & $inputBand &
          " exceeds the documented " & $IdsBandGuard & " generator guard"
        assertAllClose(h, chunkedInput,
          rtol = 0.0, abstol = recordedInputBand,
          msg = "layer " & $layerIdx & " chunked input mismatch (band " &
            $inputBand & ")")

        let layerIn = h
        let pair = model.layers[layerIdx].forward(ctx, blockInput, stream)
        blockInput = pair[0]
        stream = some(pair[1])
        h = pair[1] + pair[0]

        let seqOutput = st.getTensorOwned("layer_output_seq")
        let chunkedOutput = st.getTensorOwned("layer_output")
        assertAllClose(h, seqOutput,
          rtol = 0.0, abstol = 0.0,
          msg = "layer " & $layerIdx & " sequential output mismatch")
        let outputBand = maxAbsDiff(seqOutput, chunkedOutput)
        doAssert outputBand == recordedOutputBand,
          "layer " & $layerIdx & " measured output band " & $outputBand &
          " disagrees with the recorded band " & $recordedOutputBand
        if outputBand > 0.0:
          exercisedOutputBand = true
        doAssert outputBand < IdsBandGuard,
          "layer " & $layerIdx & " fixture output band " & $outputBand &
          " exceeds the documented " & $IdsBandGuard & " generator guard"
        assertAllClose(h, chunkedOutput,
          rtol = 0.0, abstol = recordedOutputBand,
          msg = "layer " & $layerIdx & " chunked output mismatch (band " &
            $outputBand & ")")

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

      doAssert exercisedInputBand and exercisedOutputBand,
        "the seq-vs-chunked fixture recorded a bitwise-identical band " &
        "everywhere, the chunked comparisons are vacuous"
      let normed = model.norm.forward(blockInput + stream.get(blockInput))
      let logits = model.lmHead.forward(normed)

      var stF = Safetensor.open(
        IdsInferenceFixtureDir / "final_logits.safetensor")

      # Same two rails as the layer boundaries: sequential match
      # bit-for-bit, chunked match within the recorded band.
      let seqLogits = stF.getTensorOwned("logits_seq")
      let chunkedLogits = stF.getTensorOwned("logits")
      assertAllClose(logits, seqLogits,
        rtol = 0.0, abstol = 0.0, msg = "final logits vs sequential replay mismatch")
      let logitsMeta = readFixture(
        IdsInferenceFixtureDir / "final_logits.safetensor.metadata.json.zip"
      ).fromJson(IdsLayerFixture)
      doAssert logitsMeta.bands.logits_band.isSome,
        "final logits fixture metadata logits_band is not a float"
      let recordedLogitsBand = logitsMeta.bands.logits_band.get()
      let logitsBand = maxAbsDiff(seqLogits, chunkedLogits)
      doAssert logitsBand == recordedLogitsBand,
        "measured final logits band " & $logitsBand &
        " disagrees with the recorded band " & $recordedLogitsBand
      const LogitsBandGuard = 4.0
      doAssert logitsBand < LogitsBandGuard,
        "final logits band " & $logitsBand & " exceeds the " & $LogitsBandGuard & " generator guard"
      assertAllClose(logits, chunkedLogits,
        rtol = 0.0, abstol = recordedLogitsBand,
        msg = "final logits vs chunked forward mismatch (band " & $logitsBand & ")")
      true

  echo "\nAll Qwen3.6 ids-to-logits blocks PASS"

when isMainModule:
  main()
