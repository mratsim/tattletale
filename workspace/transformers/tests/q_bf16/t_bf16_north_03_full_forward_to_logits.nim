# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full-forward-to-logits suite for the North-Mini-Code-1.0 stack, the 49 parallel blocks replayed layer by layer.
##
## Rope sits on the sliding layers plus the dense prefix, the unrotated full routed layers consume zero-angle rows.
##
## Every routed layer re-routes its composed boundary input, the recorded routing_weights fingerprint the top-k selection.
##
## - tied boundary swaps carry equal score values
## - top-8 order swaps keep the weight multiset
## - the order-invariant stats instrument reads both unchanged
##
## The un-renormalized sigmoid router passes composed drift into the weights undamped, the bands carry the composed stage counts.
##
## Requires the local model at tests/hf_models/North-Mini-Code-1.0 (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_north_03_full_forward_to_logits.nim

import
  std/os,
  std/options,
  std/importutils,
  std/strutils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/safetensors,
  workspace/safetensors/src/collections,
  workspace/zstd/zstd_highlevel,
  workspace/transformers/src/layers,
  workspace/transformers/src/layers/moe_router,
  workspace/transformers/src/models/north {.all.},
  workspace/transformers/src/deserialization,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/harness/select_device,
  workspace/transformers/tests/stateful_utils

{.experimental: "callOperator".}

privateAccess(NorthModel)

const
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-03-full-forward-to-logits" / "North-Mini-Code-1.0"
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "North-Mini-Code-1.0"

proc main(): bool =
  # The fixtures were recorded on mps, the replay resolves GPU over CPU
  # through testDevice(), Metal on this host, the same device class.
  let runDev = testDevice()
  echo "device pair: ", deviceName(runDev)

  let model = loadNorthModelRaw(ModelPath, runDev)
  let cfgJson = (ModelPath / "config.json").parseFile()
  let hiddenSize = cfgJson{"hidden_size"}.getInt()
  let numExperts = cfgJson{"num_experts"}.getInt()
  let numExpertsPerTok = cfgJson{"num_experts_per_tok"}.getInt()
  let normTopkProb = cfgJson{"norm_topk_prob"}.getBool(false)
  # The checkpoint view backs the per-layer router rebuilds below,
  # the gate weights load-only.
  let view = SafetensorsCollection.open(ModelPath)

  let meta = parseJson(readFile(
    FixtureDir / "layer-00.safetensor.metadata.json.zst").zstdDecompress(string))
  var tokenIds: seq[int64] = @[]
  for tj in items(meta{"input_tokens"}):
    tokenIds.add tj.getInt()
  let seqLen = tokenIds.len

  var (ctx, pool) = newKVContext(
    numLayers = model.config.numHiddenLayers,
    kvHeads = model.config.numKeyValueHeads,
    headDim = model.config.headDim, device = runDev)
  ctx.position_ids = F.arange(seqLen.int64,
    F.tensorOptions(F.kInt64, runDev)).unsqueeze(0)

  var hidden = model.embedTokens(tokenIds.toTensor().unsqueeze(0))
  var residual: Option[Tensor] = none(Tensor)

  for layerIdx in 0 ..< model.layers.len:
    let statsPath = FixtureDir /
      ("layer-" & ($layerIdx).align(2, '0') & ".safetensor.stats")
    # The boundary input is the embedding output at layer 0 and the hidden
    # plus residual sum at layers 1+, the composed depth after N blocks
    # counts 5N - 1 stages under the moonlight grammar.
    let nimInput =
      if residual.isSome():
        hidden + residual.unsafeGet()
      else:
        hidden
    if layerIdx == 0:
      assertStats(nimInput, statsPath, "layer_input_seq", kElementwise,
        msg = "layer " & $layerIdx & " boundary input")
    else:
      assertStats(nimInput, statsPath, "layer_input_seq", kReduction,
        depth = 5 * layerIdx - 1,
        msg = "layer " & $layerIdx & " boundary input")

    # Per-layer rope seating, the model file's own decision.
    #
    # - rotating layers upcast the bf16-grid rows to f32
    # - the unrotated full routed layers consume zero-angle rows
    if model.ropeApplied[layerIdx]:
      ctx.setRopeForPositions(model.rotary)
      ctx.cos = ctx.cos.to(F.kFloat32)
      ctx.sin = ctx.sin.to(F.kFloat32)
    else:
      let opts = F.tensorOptions(F.kBFloat16, runDev)
      ctx.cos = F.ones(seqLen, model.config.headDim, opts)
      ctx.sin = F.zeros(seqLen, model.config.headDim, opts)
    ctx.kv_position = 0

    let (output, newResidual) = model.layers[layerIdx].forward(ctx, hidden, residual)

    # The routed layers re-route the composed boundary input through
    # the checkpoint router, the recorded routing_weights the tie-robust
    # fingerprint of the top-k selection.
    if layerIdx > 0:
      let lp = "model.layers." & $layerIdx & "."
      let inputLN = RmsNorm.load(view, cfgJson, lp & "input_layernorm", runDev)
      let router = NoAuxTopCorr.init(
        view.getTensorOwned(lp & "mlp.gate.weight", runDev),
        F.zeros(numExperts, F.tensorOptions(F.kFloat32, runDev)),
        numExpertsPerTok, 1, 1, 1.0'f64, normTopkProb)
      let hNorm = inputLN.forward(nimInput)
      let decision = router.route(hNorm.reshape([seqLen, hiddenSize]))
      # The un-renormalized sigmoid router passes the composed boundary
      # drift through undamped, the band carries the boundary chain depth
      # 5N - 1 plus the norm and router reductions.
      assertStats(decision.weights.to(F.kBFloat16), statsPath,
        "routing_weights", kReduction, depth = 5 * layerIdx + 1,
        msg = "layer " & $layerIdx & " routing weights")

    # The boundary sum feeds the next layer's recorded input, the last
    # layer's sum is the recorded layer_output_seq, both carry the full
    # composed chain depth of the stack.
    let nimSum = output + newResidual
    if layerIdx < model.layers.len - 1:
      assertStats(nimSum, FixtureDir / ("layer-" & ($(layerIdx + 1)).align(2, '0') & ".safetensor.stats"), "layer_input_seq", kReduction, depth = 5 * (layerIdx + 1) - 1, msg = "layer " & $layerIdx & " output plus residual, chained input")
    else:
      assertStats(nimSum, statsPath, "layer_output_seq", kReduction, depth = 5 * model.layers.len - 1, msg = "layer " & $layerIdx & " output plus residual")

    hidden = output
    residual = some(newResidual)

  let finalResidual = residual.get(hidden)
  let finalNorm = model.norm(hidden + finalResidual)
  let finalLogits = model.lmHead(finalNorm) * Scalar(model.config.logitScale)

  let decisionsPath = FixtureDir / "final_logits.decisions.json.zst"

  var flipCount = 0

  # The recorded decisions frame carries one step per prompt position,
  # the argmax rows asserted over the whole prefill. The logits compose
  # the stack boundary chain plus the final norm plus the head projection,
  # two more stages past the composed boundary depth.
  for pos in 0 ..< finalLogits.size(1):
    assertArgMax(finalLogits.narrow(1, pos.int64, 1), decisionsPath, pos,
      kReduction, flipCount, depth = 5 * model.layers.len + 1,
      msg = "North-Mini-Code-1.0 final logits position " & $pos)
  result = true

when isMainModule:
  runCppTest("north full forward to logits", main)
