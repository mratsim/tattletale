# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Run:
##   TTT_TEST_ON=cpu nim test_tf_exl3_qwen3_03_full_forward_to_logits

import
  std/strformat,
  std/tables,
  std/os,
  std/options,
  std/importutils,
  pkg/jsony,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/safetensors {.all.},
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness,
  workspace/libtorch_testutils

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(SafetensorObj)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-03-full-forward-to-logits" / "Qwen3-0.6B-EXL3-5bpw"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  LogitsChainDepth = 29
    ## The final logits sit one stage past the last recorded layer: the
    ## decision-band depth of the accumulated chain law.

proc loadLayerFixture(layerIdx: int): Table[string, Tensor] =
  ## Load EXL3 layer intermediates from safetensor fixture.
  let fixturePath = FixtureDir / &"layer-{layerIdx:02d}.safetensor"
  var st = Safetensor.open(fixturePath)
  result = initTable[string, Tensor]()
  for name in st.tensors.keys():
    result[name] = st.getTensorOwned(name, kCPU)

proc replayChain(dev: F.DeviceKind): Tensor =
  ## Full 28-layer replay on `dev`, the layer boundaries checked against the
  ## recorded anchors under the accumulated chain band. Returns the final
  ## logits.
  let model = loadQwen3ModelRaw($ModelPath, dev)

  var ctx = InferenceContext.init(
    num_layers = model.config.num_hidden_layers,
    batch_size = 1, kv_heads = model.config.num_key_value_heads,
    max_seq = 4096, head_dim = model.config.head_dim)

  let pool = PagePool.init(
    64, num_layers = model.config.num_hidden_layers,
    kv_heads = model.config.num_key_value_heads,
    head_dim = model.config.head_dim,
    dtype = F.kFloat16, device = dev)
  let numPages = ceilDiv(4096, TokensPerPage)

  # Borrow pages once — reused across all layers (each writes to own layerIdx slice)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())

  # Input tokens: "Hello, how are you?"
  let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0).to(dev)

  # Embedding pass
  let x = model.embedTokens(inputIds)
  var hidden = x
  var residual: Option[Tensor] = none(Tensor)

  echo "Comparing layer-by-layer EXL3 intermediates..."
  echo "================================================================="

  for layerIdx in 0..<model.layers.len:
    let fixturePath = FixtureDir / &"layer-{layerIdx:02d}.safetensor"
    var st = Safetensor.open(fixturePath)
    let fixtureInput = st.getTensorOwned("layer_input", kCPU).to(dev)
    let fixtureOutput = st.getTensorOwned("layer_output", kCPU).to(dev)
    let statsFile = loadFingerprintStats(fixturePath & ".stats")
    let layer = model.layers[layerIdx]

    let nimInput = if residual.isSome():
      hidden + residual.unsafeGet()
    else:
      hidden
    let depth = layerIdx + 1
    discard assertChainCheckpoint(nimInput, fixtureInput, depth,
      reductionLen = model.config.intermediate_size,
      msg = &"layer {layerIdx} input", rtol = ChainCheckpointRtolF16)

    # Prepare InferenceContext for this layer — reuse pages, reset positional state
    ctx.kv_position = 0
    ctx.position_ids = arange(hidden.size(1)).unsqueeze(0).to(kInt64).to(dev)
    ctx.setRopeForPositions(layer.sequence_mixer.rotary)

    # Forward through layer (long residual stream pattern)
    let (output, newResidual) = layer(ctx, hidden, residual)

    # Compare: Nim (output + residual) vs EXL3 fixture (layer_output) under
    # the accumulated chain band, then the recorded stats sidecar.
    let nimSum = output + newResidual
    let mw = assertChainCheckpoint(nimSum, fixtureOutput, depth,
      reductionLen = model.config.intermediate_size,
      msg = &"layer {layerIdx} output", rtol = ChainCheckpointRtolF16)
    assertStatsChainBand(nimSum, statsFile.statsTensor("layer_output"), depth,
      msg = &"layer {layerIdx} output stats")
    echo &"Layer {layerIdx:02d}: worst band width {mw.worst:.3f}"

    # Update state
    hidden = output
    residual = some(newResidual)

  # No drift-scaling check here: the flat ratio-to-first-bound model assumes
  # a stable per-depth honest drift, and the measured mac-replay widths
  # against the CUDA recording decline with depth (0.82 at depth 3 down to
  # 0.09 at depth 27) because the depth-1 boundary anchor sits below the
  # per-depth drift scale. The per-layer band and mean-drift checks carry
  # the compounding detection.

  # Final logits
  echo "================================================================="
  echo "Final logits:"
  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm(hidden + finalResidual)
  result = model.lmHead(finalNorm)

proc main() =
  let dev = testDevice()

  runCppTest "Qwen3-0.6B-EXL3-5bpw: full forward to logits — long residual stream vs EXL3 fixtures":
    proc(): bool =
      echo "    devices: ", compareLine(FixtureDir, dev)
      let finalLogits = replayChain(dev)

      if dev == F.kCUDA:
        # Reference-device variant: the chain replays bit-exactly, the final
        # logits go through the full decision projection: the recorded
        # strided sample (probe_bits), the tail-probability checksum, and
        # the ulp-banded decision rows with the fp16 unit.
        let projection = zstdReadFixture(
          FixtureDir / "final_logits.decisions.json.zst"
        ).fromJson(LogitsProjection)
        assertProjection(finalLogits, projection,
          msg = "Qwen3-0.6B-EXL3-5bpw final logits projection",
          ulpUnitF16 = true)
      else:
        # Cross-device variant: the raw logits tensor is not in the
        # payload, its recorded distribution compares under the accumulated
        # chain band, plus the discrete argmax agreement per position.
        let statsFile = loadFingerprintStats(
          FixtureDir / "final_logits.safetensor.stats")
        assertStatsChainBand(finalLogits, statsFile.statsTensor("logits"),
          LogitsChainDepth, msg = "final logits stats")
        let projection = zstdReadFixture(
          FixtureDir / "final_logits.decisions.json.zst"
        ).fromJson(LogitsProjection)
        let row = finalLogits.contiguous().to(F.kCPU).to(F.kFloat32)
        for step in projection.steps:
          let flat = row.narrow(1, step.position, 1).squeeze(1).squeeze(0)
          let n = flat.numel()
          let raw = cast[ptr UncheckedArray[float32]](flat.data_ptr(float32))
          var obsArgmax = 0
          var obsTop1 = NegInf
          for i in 0 ..< n:
            if raw[i].float64 > obsTop1:
              obsTop1 = raw[i].float64
              obsArgmax = i
          if obsArgmax != step.argmaxId:
            raise newException(HarnessCheckError,
              "position " & $step.position & " argmax " & $obsArgmax &
              " != recorded " & $step.argmaxId &
              " (recorded margin " & $step.argmaxMargin & ")")

      echo "✓ PASS: All layers and the logits surface checked"
      true

when isMainModule:
  main()
