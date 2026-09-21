# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/MIT).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the Laguna-XS-2.1 stack, the recorded chains
## replayed teacher-forced against the committed decision frames.
##
## - per step the argmax decision via assertArgMax (token, top-32 logits, KL, tail)
## - tie-eligible picks teacher-force the recorded token
## - the Dragon story prefill runs 555 prompt tokens through the production
##   KV path, the only chain crossing the 512-token sliding window
##
## Requires the local model at tests/hf_models/Laguna-XS-2.1 (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/transformers/tests/q_bf16/t_bf16_laguna_04_greedy_text_generation.nim

import
  std/os,
  std/monotimes,
  std/sequtils,
  std/strformat,
  std/strutils,
  std/times,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/libtorch_testutils,
  workspace/transformers/src/models,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils,
  workspace/transformers/tests/harness/select_device

from workspace/transformers/tests/harness import zstdReadFixture

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Laguna-XS-2.1"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-04-greedy-text-generation" / "Laguna-XS-2.1"
  GreedyFixtureFiles = [
    "The_sky_is_blue_today_32_steps.json.zst",
    "To_be_or_not_to_be_that_is_32_steps.json.zst",
    "Dragon_story_crosses_512_window_32_steps.json.zst",
  ]

proc main(): bool =
  echo "Loading model..."
  let model = loadModel($ModelPath, testDevice())
  echo "Model loaded."
  let device = model.getDeviceKind()
  # Composed depth of one full-stack forward, the depth the 03 suite
  # carries for this model's final logits, one rounding stage per
  # layer through the routed block and the normed sums.
  let chainDepth = model.getConfig().num_hidden_layers
  for f in GreedyFixtureFiles:
    echo "Fixture: " & f
    let fixture = parseJson(zstdReadFixture(FixtureDir / f))
    var flipCount = 0
    let name = f.replace(".json.zst", "")
    let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
    let expected = fixture["generated_ids"].mapIt(it.getInt())
    let horizon = expected.len
    let decisionsPath =
      FixtureDir / f.replace(".json.zst", ".decisions.json.zst")
    # The decode-context ceiling covers prompt plus horizon, the Dragon
    # prefill past the 512-token sliding window stays inside it.
    let maxContextLen = promptIds.len + horizon
    var orc = newOrchestrator(model, maxContextLen = maxContextLen)
    defer: orc.endSequence()

    var ids = promptIds
    orc.startSequence(ids.mapIt(it.uint32))
    let chainStart = getMonoTime()
    let inputIds = F.toTensor([ids]).to(device)
    let logits = model.forward(orc.getInferenceContextMut(), inputIds)
    orc.setKvPosition(ids.len)
    var row = nextStepRow(logits, ids.len - 1)

    for step in 0 ..< horizon:
      assertArgMax(row, decisionsPath, step, kReduction, flipCount,
        msg = name & " step " & $step, depth = chainDepth)
      # The recorded token is teacher-forced at every step, so a tie
      # flip stays recorded and every following check must re-converge.
      let chosen = expected[step]
      ids.add chosen
      if step + 1 < horizon:
        orc.appendToken(ids.len - 1, chosen.uint32, device)
        let stepLogits = model.forward(orc.getInferenceContextMut(),
          F.toTensor([[chosen]]).to(device))
        orc.setKvPosition(ids.len)
        row = nextStepRow(stepLogits, 0)

    # The wall clock and tok/s of the replay chain make device and kernel
    # regressions visible per run.
    let chainWall = (getMonoTime() - chainStart).inNanoseconds.float64 * 1e-9
    echo &"    chain wall {chainWall:.3f} s ({horizon.float64 / chainWall:.3f} tok/s)"
  result = true

when isMainModule:
  runCppTest("laguna greedy text generation", main)
