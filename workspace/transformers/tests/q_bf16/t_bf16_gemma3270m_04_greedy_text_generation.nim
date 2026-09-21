# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the gemma-3-270m-it stack against the recorded decision frames.
##
## - per step the argmax decision via assertArgMax (token, top-32 logits, KL, tail)
## - tie-eligible picks teacher-force the recorded token
## - the Fox story chain stays unreplayed, its prefill row exceeds every device tolerance
##
## The excluded Fox row measures top-1 6.4375 bf16-MPS, 4.59 bf16-CPU,
## 5.859 f32, 4.75 Metal, tails 6.45e-4 to 0.0128, the argmax id 107
## identical on every device, the gemma-3-1b tier 01 suite replays the chain.
##
## Requires the local model at tests/hf_models/gemma-3-270m-it (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_gemma3270m_04_greedy_text_generation.nim

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
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "gemma-3-270m-it"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-04-greedy-text-generation" / "gemma-3-270m-it"
  GreedyFixtureFiles = [
    "Hello_how_are_you_32_steps.json.zst",
    "Big_blue_whales_eat_krill_32_steps.json.zst",
  ]

proc main(): bool =
  echo "Loading model..."
  let model = loadModel($ModelPath, testDevice())
  echo "Model loaded."
  let device = model.getDeviceKind()
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
    # The decode-context ceiling covers prompt plus horizon, the Fox
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
      # Step 0 decides off the whole-prompt prefill row, a head decision
      # over the full chain, the allowance carries the layer count the way
      # the tier-03 final-logits rows do, decode steps keep one pass.
      let stepDepth = if step == 0: model.getConfig().num_hidden_layers else: 1
      assertArgMax(row, decisionsPath, step, kReduction, flipCount,
        msg = name & " step " & $step, depth = stepDepth)
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
  runCppTest("gemma3 270m greedy text generation", main)
