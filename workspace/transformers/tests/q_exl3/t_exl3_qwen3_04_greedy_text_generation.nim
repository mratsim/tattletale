# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp 0) decoding of the Qwen3-0.6B-EXL3-5bpw stack against the committed 005 decision frames.
##
## Requires the local model at tests/hf_models/Qwen3-0.6B-EXL3-5bpw (gitignored).
##
## Run through the test_tf_exl3_qwen3_04_greedy_text_generation task in config.nims.

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
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B-EXL3-5bpw"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "exl3-04-greedy-text-generation" / "Qwen3-0.6B-EXL3-5bpw"

proc main() =
  ## Replays the greedy chains teacher-forced against the committed 005 decision frames, every step through assertArgMax.
  ##
  ## - teacher forcing at every step keeps the chain aligned with the recording
  # The Metal backend falls back to the cpu kernels where the device kernels
  # are missing, the chains replay on whatever testDevice() resolves.
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  # Recorded-chain replay against the committed decision frames, one
  # chain per fixture, the recorded tokens teacher-forced at every step.
  echo "Loading model..."
  let model = loadModel($ModelPath, testDevice())
  echo "Model loaded."
  let device = model.getDeviceKind()
  for stem in ["Hello_how_are_you",
               "Do_you_know_the_story_of_this_proverb_磨刀"]:
    echo "Fixture: " & stem
    let fixture = parseJson(zstdReadFixture(FixtureDir / stem & ".json.zst"))
    var flipCount = 0
    let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
    let expected = fixture["generated_ids"].mapIt(it.getInt())
    let decisionsPath = FixtureDir / stem & ".decisions.json.zst"
    var orc = newOrchestrator(model, dtype = F.kFloat16)
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
