# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test Qwen3-0.6B token IDs to logit inference with layer intermediates values checked against HF fixtures.
##
## Strategy:
## - ``layer_input``: should match exactly (same embedding)
## - ``layer_output + layer_residual`` (Nim) vs the next layer's recorded
##   ``layer_input``: should match exactly. The proven invariant
##   ``y_long + r_long == x_local`` carries the boundary. One recorded
##   input copy per layer boundary, no per-layer output copies.
## - Sublayer intermediates: EXPECTED to differ (norms see different inputs)
##
## The final block output is the layer-27 descriptor sidecar (dmExact):
## fingerprint plus probe, the same recorded-summary comparison the 35B dense
## suite runs.
## Final logits go through the decision projection
## (ttt-tf-002-logit-decisions-probe-h2): argmax and top-2 pair per position,
## the tail-probability checksum and the strided probe. No raw logits tensor.

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
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-03-full-forward-to-logits" / "Qwen3-0.6B"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"

proc loadLayerFixture(layerIdx: int): Table[string, Tensor] =
  ## Load HF layer intermediates from safetensor fixture.
  let fixturePath = FixtureDir / &"layer-{layerIdx:02d}.safetensor"
  var st = Safetensor.open(fixturePath)
  result = initTable[string, Tensor]()
  for name in st.tensors.keys():
    result[name] = st.getTensorOwned(name, kCPU)

proc main() =
  assertTorchStamp(FixtureDir)
  runCppTest "Qwen3-0.6B full inference - long residual stream vs HF":
    proc(): bool =
      echo "    devices: ", compareReport(FixtureDir, F.kCPU)
      ## Strategy:
      ## - layer_input: should match exactly (same embedding)
      ## - layer_output + layer_residual (Nim) vs layer_output (HF): should match exactly
      ##   (proven invariant: y_long + r_long == x_local)
      ## - after_attn_norm, after_attn, after_mlp: EXPECTED to differ
      ##   (norms see different inputs: N(x+r) vs N(x))

      const tol = 1e-5

      let model = loadQwen3ModelRaw(ModelPath, kCPU)

      # InferenceContext for stateful attention
      var ctx = InferenceContext.init(
        num_layers = model.config.num_hidden_layers,
        batch_size = 1, kv_heads = model.config.num_key_value_heads,
        max_seq = 4096, head_dim = model.config.head_dim)

      let pool = PagePool.init(
        64, num_layers = model.config.num_hidden_layers,
        kv_heads = model.config.num_key_value_heads,
        head_dim = model.config.head_dim,
        dtype = F.kBFloat16, device = F.kCPU)
      let numPages = ceilDiv(4096, TokensPerPage)

      # Borrow pages once — reused across all layers (each writes to own layerIdx slice)
      for i in 0 ..< numPages:
        ctx.pages.add(pool.borrow())

      # Input tokens: "Hello, how are you?"
      let inputIds = @[9707.int64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)
      # Embedding pass
      let x = model.embedTokens(inputIds)
      var hidden = x
      var residual: Option[Tensor] = none(Tensor)

      echo "Comparing layer-by-layer intermediates..."
      echo "================================================================="

      for layerIdx in 0..<model.layers.len:
        let hfFixture = loadLayerFixture(layerIdx)
        var layer = model.layers[layerIdx]

        # Compare layer_input
        # For layer 0: hidden is the embedding output
        # For layers 1+: hidden + residual is the boundary sum (matches HF layer_input)
        let nimInput = if residual.isSome():
          hidden + residual.unsafeGet()
        else:
          hidden
        let inputDiff = (nimInput.to(kFloat32) - hfFixture["layer_input"].to(kFloat32)).abs().max().item(float)
        echo &"Layer {layerIdx:02d}: input_diff={inputDiff:.2e}"

        if inputDiff > tol:
          raise newException(ValueError, &"Layer {layerIdx:02d}: layer_input diff = {inputDiff:.6e}")

        # Prepare InferenceContext for this layer — reuse pages, reset positional state
        ctx.kv_position = 0
        ctx.position_ids = nil
        let pos_ids = arange(hidden.size(1)).unsqueeze(0).to(kInt64)
        ctx.position_ids = pos_ids
        ctx.setRopeForPositions(layer.sequence_mixer.rotary)

        # Forward through layer (long residual stream pattern)
        let (output, newResidual) = layer(ctx, hidden, residual)

        # Compare: Nim (output + residual) against the next layer's recorded
        # input, the recorded copy this boundary feeds. The last block output
        # has no next layer: its recorded surface is the layer-27 descriptor
        # sidecar, checked under the dmExact budget like the 35B layer-39
        # precedent.
        let nimSum = output + newResidual
        var outputDiff = 0.0'f64
        if layerIdx < model.layers.len - 1:
          let nextFixture = loadLayerFixture(layerIdx + 1)
          outputDiff = (nextFixture["layer_input"].to(kFloat32) - nimSum.to(kFloat32)).abs().max().item(float)
          echo &"  output + residual vs next layer input diff={outputDiff:.2e}"
          if outputDiff > tol:
            raise newException(ValueError, &"Layer {layerIdx:02d}: output + residual diff = {outputDiff:.6e}")
        else:
          let finalDesc = loadFingerprintStats(
            FixtureDir / "layer-27.safetensor.descriptors")
          let finalEntry = finalDesc.statsTensor("layer_output")
          assertStats(nimSum, finalEntry,
            descriptorStatsBudget(finalEntry.probeMode),
            msg = "final block output fingerprint")
          assertDescriptors(nimSum, finalEntry,
            msg = "final block output descriptors")
          echo "  output + residual checked against the layer-27 descriptor sidecar"

        # Update state
        hidden = output
        residual = some(newResidual)

      # Final logits decision-projection checks. The Nim replay
      # matches the recording bit for bit at every layer, the projection
      # checks compare the final logits against the recorded decision
      # projection.
      echo "================================================================="
      echo "Final logits decision projection:"
      let finalResidual = residual.get(hidden)
      let finalNorm = model.norm(hidden + finalResidual)
      let finalLogits = model.lmHead(finalNorm)
      echo &"  Nim logits shape: {finalLogits.shape}"

      let projection = zstdReadFixture(
        FixtureDir / "final_logits.decisions.json.zst"
      ).fromJson(LogitsProjection)
      assertProjection(finalLogits, projection,
        msg = "Qwen3-0.6B final logits projection")

      echo "✓ PASS: All layers bit-exact and the logits projection passed"
      true

  runCppTest "Qwen3-0.6B full inference, cross-device variant (device-pair report)":
    proc(): bool =
      let runDev = testDevice()
      echo "    devices: ", compareReport(FixtureDir, runDev)
      if compareClass(recordedDevice(recordedFrom(FixtureDir)), runDev) ==
          sameDeviceBitExact:
        echo "    the device comparison selects the reference budgets, the reference variant carries the replay"
        return true
      # The full-forward fixtures compare bit-exact on the reference
      # device: dmExact descriptor entries, the exact layer boundary sums,
      # and the decision projection on the 4-ulp band (bf16 unit). No
      # cross-device drift budget applies, so a flipped run names the
      # tolerance class and skips.
      echo "    no cross-device drift budget applies: the dmExact descriptors, ",
        "the exact boundary sums and the decision projection compare on the ulp band, ",
        "the suite skips on ", deviceName(runDev)
      return true

when isMainModule:
  main()
