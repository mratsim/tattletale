# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the wired Qwen3.6-35B-A3B stack, the recorded
## chains replayed teacher-forced against the committed decision frames.
##
## - per step the argmax decision via assertArgMax (token, top-32 logits, KL, tail)
## - tie-eligible picks teacher-force the recorded token
## - chain wall and tok/s echo per replay, for device and kernel regressions
##
## Requires the local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/par-b --nimcache:nimcache/par-b workspace/transformers/tests/q_bf16/t_bf16_qwen36moe_04_greedy_text_generation.nim

import
  std/os,
  std/monotimes,
  std/sequtils,
  std/strformat,
  std/strutils,
  std/times,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/transformers/src/models,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils,
  workspace/transformers/tests/harness/select_device

from workspace/transformers/tests/harness import zstdReadFixture

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  GreedyFixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "bf16-04-greedy-text-generation" / "Qwen3.6-35B-A3B"
  GreedyFixtureFiles = [
    "Hello_how_are_you_32_steps.json.zst",
    "The_capital_of_France_is_32_steps.json.zst",
    "Big_blue_whales_eat_krill_32_steps.json.zst",
  ]

proc main() =
  # Decode context ceiling, one TokensPerPage page covers every
  # recorded chain comfortably above the prompt + horizon footprint.
  const MaxContextLen = 256

  echo "Loading model..."
  let model = loadModel($(ModelDir), testDevice())
  echo "Model loaded."
  let device = model.getDeviceKind()

  # Recorded-chain replay against the committed decision frames, one
  # chain per fixture, the recorded tokens teacher-forced at every step.
  for f in GreedyFixtureFiles:
    echo "Fixture: " & f
    let fixture = parseJson(zstdReadFixture(GreedyFixtureDir / f))
    var flipCount = 0
    let name = f.replace(".json.zst", "")
    let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
    let expected = fixture["generated_ids"].mapIt(it.getInt())
    let horizon = expected.len
    let decisionsPath =
      GreedyFixtureDir / f.replace(".json.zst", ".decisions.json.zst")
    var orc = newOrchestrator(model, maxContextLen = MaxContextLen)
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
        msg = name & " step " & $step, depth = 1)
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

when isMainModule:
  main()
