# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the Qwen3.5-0.8B dense stack, the recorded
## chains replayed teacher-forced against the committed decision frames.
##
## - per step the argmax decision via assertArgMax (token, top-32 logits, KL, tail)
## - tie-eligible picks teacher-force the recorded token
##
## Requires the local model at tests/hf_models/Qwen3.5-0.8B (gitignored).
##
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/par-b --nimcache:nimcache/par-b workspace/transformers/tests/q_bf16/t_bf16_qwen35dense_04_greedy_text_generation.nim

import
  std/os,
  std/sequtils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/transformers/src/models,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils,
  workspace/transformers/tests/harness/select_device

from workspace/transformers/tests/harness import zstdReadFixture

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.5-0.8B"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-04-greedy-text-generation" / "Qwen3.5-0.8B"

proc main() =
  # The chain replays on the device testDevice() resolves, Metal
  # is the auto default on macOS with a PyTorch fallback where
  # kernels are missing. TTT_TEST_ON=metal|cpu|cuda flips the device.

  # Recorded-chain replay against the committed decision frames, one
  # chain per fixture, the recorded tokens teacher-forced at every step.
  echo "Loading model..."
  let model = loadModel($ModelPath, testDevice())
  echo "Model loaded."
  let device = model.getDeviceKind()
  for stem in ["Hello_how_are_you",
               "The_resume_is_ready",
               "What_is_the_capital_of_France"]:
    echo "Fixture: " & stem
    let fixture = parseJson(zstdReadFixture(FixtureDir / stem & ".json.zst"))
    var flipCount = 0
    let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
    let expected = fixture["generated_ids"].mapIt(it.getInt())
    let decisionsPath = FixtureDir / stem & ".decisions.json.zst"
    var orc = newOrchestrator(model)
    defer: orc.endSequence()

    var ids = promptIds
    orc.startSequence(ids.mapIt(it.uint32))
    let inputIds = F.toTensor([ids]).to(device)
    let logits = model.forward(orc.getInferenceContextMut(), inputIds)
    orc.setKvPosition(ids.len)
    var row = nextStepRow(logits, ids.len - 1)

    for step in 0 ..< expected.len:
      assertArgMax(row, decisionsPath, step, kReduction, flipCount,
        msg = stem & " step " & $step, depth = 1)
      # The recorded token is teacher-forced at every step, so a tie
      # flip stays recorded and every following check must re-converge.
      let chosen = expected[step]
      ids.add chosen
      if step + 1 < expected.len:
        orc.appendToken(ids.len - 1, chosen.uint32, device)
        let stepLogits = model.forward(orc.getInferenceContextMut(),
          F.toTensor([[chosen]]).to(device))
        orc.setKvPosition(ids.len)
        row = nextStepRow(stepLogits, 0)

when isMainModule:
  main()
