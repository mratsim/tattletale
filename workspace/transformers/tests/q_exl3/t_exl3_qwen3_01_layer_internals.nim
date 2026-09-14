# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Layer-0 internals of the Qwen3-0.6B-EXL3-5bpw checkpoint, replayed against
## the recorded linear, attention and block cases.
## Requires the local model at tests/hf_models/Qwen3-0.6B-EXL3-5bpw (gitignored).
##
## Run through the test_tf_exl3_qwen3_01_layer_internals task in config.nims.

import
  std/options,
  std/os,
  std/strutils,
  std/importutils,
  workspace/libtorch as F,
  workspace/safetensors,
  workspace/transformers/src/layers,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/qwen3 {.all.},
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device

{.experimental: "callOperator".}

privateAccess(Qwen3Model)
privateAccess(DecoderLayer[RopeGQAttention[RmsNorm], GatedDenseFFN, RmsNorm])
privateAccess(RopeGQAttention[RmsNorm])
privateAccess(GatedDenseFFN)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "exl3-01-layer-internals" / "Qwen3-0.6B-EXL3-5bpw-layer-0"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  ModelName = "Qwen3-0.6B-EXL3-5bpw"

proc main() =
  ## Replays the recorded layer-0 cases of the EXL3 checkpoint, every computed output
  ## enforced through assertStats against its committed 004 stats frame.
  ##
  ## Recorded provenance is the production CUDA kernel, the suite replay is cross-device.
  ##
  ## Depth carries the reordered stages since each recorded reference.
  let dev = testDevice()
  echo "device: ", deviceName(dev)
  let model = loadQwen3ModelRaw($ModelPath, dev)

  # Linear cases, one EXL3 linear composes the pre-Hadamard FWHT, the fp16 GEMM
  # and the post-Hadamard FWHT, depth 3.
  block:
    let layer = model.layers[0]
    let projNames = @[
      ("self_attn.q_proj", layer.sequence_mixer.q_proj),
      ("self_attn.k_proj", layer.sequence_mixer.k_proj),
      ("self_attn.v_proj", layer.sequence_mixer.v_proj),
      ("self_attn.o_proj", layer.sequence_mixer.o_proj),
      ("mlp.gate_proj", layer.hidden_mixer.gate_proj),
      ("mlp.up_proj", layer.hidden_mixer.up_proj),
      ("mlp.down_proj", layer.hidden_mixer.down_proj),
    ]
    for (projName, linear) in projNames:
      let fixturePrefix = "linear-" & projName & "-" & ModelName
      for caseNum in 0 .. 3:
        let fixturePath =
          FixtureDir / (fixturePrefix & "-" & ($caseNum).align(2, '0') & ".safetensor")
        if not fileExists(fixturePath):
          continue
        var st = Safetensor.open(fixturePath)
        let input = st.getTensorOwned("input", dev)
        let output = linear(input)
        assertStats(output, fixturePath & ".stats", "output", kReduction, depth = 3, msg = "linear " & projName & " case " & $caseNum)
      echo "linear " & projName & " replayed under the recorded stats"

  # Attention cases, the recorded hidden_states reference composes the qkv linear
  # (3 stages), qk-norm, rope, sdpa and the o_proj linear (3 stages), depth 9.
  block:
    let attn = model.layers[0].sequence_mixer
    let rotary = model.rotary
    var ctx = InferenceContext.init(
      num_layers = model.config.num_hidden_layers,
      batch_size = 1, kv_heads = model.config.num_key_value_heads,
      max_seq = 4096, head_dim = model.config.head_dim)
    let pool = PagePool.init(
      64, num_layers = 1,
      kv_heads = model.config.num_key_value_heads,
      head_dim = model.config.head_dim,
      dtype = F.kFloat16, device = dev)
    for i in 0 ..< ceilDiv(4096, TokensPerPage):
      ctx.pages.add(pool.borrow())

    for caseNum in 0 .. 1:
      let fixturePath =
        FixtureDir / ("attn-" & ModelName & "-" & ($caseNum).align(2, '0') & ".safetensor")
      if not fileExists(fixturePath):
        echo "attention case " & $caseNum & " has no recorded fixture, skipped"
        continue
      var st = Safetensor.open(fixturePath)
      let hiddenStates = st.getTensorOwned("hidden_states", dev)
      let hfPosIds = st.getTensorOwned("position_ids", dev)

      let batch = hiddenStates.size(0)
      var outputs: seq[Tensor] = @[]
      var cosTables: seq[Tensor] = @[]
      var sinTables: seq[Tensor] = @[]
      for b in 0 ..< batch:
        ctx.kv_position = 0
        ctx.position_ids = hfPosIds[b]
        ctx.setRopeForPositions(rotary)
        cosTables.add(ctx.cos)
        sinTables.add(ctx.sin)
        let o = attn(ctx, hiddenStates[b].unsqueeze(0))
        outputs.add(o)
      let finalOutput = F.cat(outputs)
      # The batched rope tables flatten to the recorded [batch, seq, head_dim]
      # word order, one case covers the whole batch.
      let batchedCos = F.cat(cosTables)
      let batchedSin = F.cat(sinTables)
      assertStats(finalOutput, fixturePath & ".stats", "output", kReduction, depth = 9, msg = "attention case " & $caseNum)
      echo "attention case " & $caseNum & " replayed under the recorded stats"

  # Decoder-block cases, the recorded references compose the full block, the input norm,
  # the attention composition (9 stages), the residual add, the post norm, the mlp chain
  # (gate_proj 3, up_proj 3, activation 1, down_proj 3) and the output add, depth 23.
  block:
    let layer = model.layers[0]
    let rotary = model.rotary
    var ctx = InferenceContext.init(
      num_layers = model.config.num_hidden_layers,
      batch_size = 1, kv_heads = model.config.num_key_value_heads,
      max_seq = 4096, head_dim = model.config.head_dim)
    let pool = PagePool.init(
      64, num_layers = 1,
      kv_heads = model.config.num_key_value_heads,
      head_dim = model.config.head_dim,
      dtype = F.kFloat16, device = dev)
    for i in 0 ..< ceilDiv(4096, TokensPerPage):
      ctx.pages.add(pool.borrow())

    for caseNum in 0 .. 3:
      let fixturePath = FixtureDir /
        ("transformer-block-" & ModelName & "-" & ($caseNum).align(2, '0') & ".safetensor")
      if not fileExists(fixturePath):
        echo "block case " & $caseNum & " has no recorded fixture, skipped"
        continue
      var st = Safetensor.open(fixturePath)
      let inputHiddenStates = st.getTensorOwned("input_hidden_states", dev)
      let hfPosIds = st.getTensorOwned("position_ids", dev)
      let residualTensor = st.getTensorOwned("residual", dev)

      let batch = inputHiddenStates.size(0)
      var outputs: seq[Tensor] = @[]
      var outputResiduals: seq[Tensor] = @[]
      for b in 0 ..< batch:
        ctx.kv_position = 0
        ctx.position_ids = hfPosIds[b]
        ctx.setRopeForPositions(rotary)
        let x = inputHiddenStates[b].unsqueeze(0)
        let (o, oRes) = layer(ctx, x, some(residualTensor[b].unsqueeze(0)))
        outputs.add(o)
        outputResiduals.add(oRes)
      let finalOutput = F.cat(outputs)
      let finalOutputResidual = F.cat(outputResiduals)
      assertStats(finalOutput, fixturePath & ".stats", "output", kReduction, depth = 23, msg = "block output case " & $caseNum)
      assertStats(finalOutputResidual, fixturePath & ".stats", "output_residual", kReduction, depth = 23, msg = "block output_residual case " & $caseNum)
      echo "block case " & $caseNum & " replayed under the recorded stats"

when isMainModule:
  main()
