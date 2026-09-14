# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the Qwen3-0.6B-EXL3-5bpw stack, the 28
## decoder blocks replayed layer by layer over the long residual stream.
## Requires the local model at tests/hf_models/Qwen3-0.6B-EXL3-5bpw (gitignored).
##
## Run through the test_tf_exl3_qwen3_03_full_forward_to_logits task in config.nims.

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  std/tables,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/safetensors/src/safetensors {.all.},
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(SafetensorObj)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "exl3-03-full-forward-to-logits" / "Qwen3-0.6B-EXL3-5bpw"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"

proc main() =
  ## Replays the 28-block chain of the EXL3 checkpoint against the committed frames.
  ##
  ## Every layer boundary enforces through assertStats against its 004 stats frame.
  ##
  ## Every recorded logits position enforces through assertArgMax against the 002 decisions frame.
  ##
  ## Depth carries the reordered stages since the recorded embedding reference, 23
  ## reordered stages compose one EXL3 block (the derivation of the 01 layer internals suite).
  # The chain replay runs on the cpu reference device, the recorded class of this family.
  echo "device pair: ", deviceName(testDevice())
  let model = loadQwen3ModelRaw($ModelPath, F.kCPU)

  var ctx = InferenceContext.init(
    num_layers = model.config.num_hidden_layers,
    batch_size = 1, kv_heads = model.config.num_key_value_heads,
    max_seq = 4096, head_dim = model.config.head_dim)

  let pool = PagePool.init(
    64, num_layers = model.config.num_hidden_layers,
    kv_heads = model.config.num_key_value_heads,
    head_dim = model.config.head_dim,
    dtype = F.kFloat16, device = F.kCPU)
  let numPages = ceilDiv(4096, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())

  # Input tokens are the tokenizer ids of "Hello, how are you?"
  let inputIds = @[9707'i64, 11, 1246, 525, 498, 30].toTensor().unsqueeze(0)

  # The committed 002 decisions frame carries one decimal record per position,
  # the argmax id, the top-2 pair with logits, the margin and the tail probability.
  var hidden = model.embedTokens(inputIds)
  var residual: Option[Tensor] = none(Tensor)

  for layerIdx in 0 ..< model.layers.len:
    let fixturePath = FixtureDir / ("layer-" & ($layerIdx).align(2, '0') & ".safetensor")
    let layer = model.layers[layerIdx]

    # The boundary input is the embedding output at layer 0 and the hidden plus residual sum at layers 1+.
    let nimInput =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    # The boundary input of layer 0 is the exact embedding gather, the boundary
    # input of layers 1+ composes the preceding blocks.
    assertStats(nimInput, fixturePath & ".stats", "layer_input", kElementwise, depth = (if layerIdx == 0: 1 else: 23 * layerIdx), msg = "layer " & $layerIdx & " boundary input")

    ctx.kv_position = 0
    ctx.position_ids = F.arange(
      hidden.size(1).int64, F.tensorOptions(F.kInt64, F.kCPU)).unsqueeze(0)
    ctx.setRopeForPositions(layer.sequence_mixer.rotary)

    # Forward through layer (long residual stream pattern)
    let (output, newResidual) = layer(ctx, hidden, residual)

    # The boundary sum feeds the next layer's recorded input.
    let nimSum = output + newResidual
    assertStats(nimSum, fixturePath & ".stats", "layer_output", kReduction, depth = 23 * (layerIdx + 1), msg = "layer " & $layerIdx & " output plus residual, chained input")

    hidden = output
    residual = some(newResidual)

  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm(hidden + finalResidual)
  let finalLogits = model.lmHead(finalNorm)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"
  var flipCount = 0

  for pos in 0 ..< finalLogits.size(1):
    assertArgMax(finalLogits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = model.layers.len,
      msg = "Qwen3-0.6B-EXL3-5bpw final logits position " & $pos)

when isMainModule:
  main()
