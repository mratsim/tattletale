# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the Moonlight-16B-A3B MLA + routed stack,
## replaying recorded chains teacher-forced against the committed decision frames.
##
## - per step the argmax decision via assertArgMax (token, top-32 logits, KL, tail)
## - tie-eligible picks teacher-force the recorded token
## - chain wall and tok/s echo per replay, for device and kernel regressions
##
## Requires:
## - the local model at tests/hf_models/Moonlight-16B-A3B (gitignored)
## - the bf16-04 greedy-text-generation fixture dir.
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_moonlight_04_greedy_text_generation.nim

import
  std/os,
  std/monotimes,
  std/sequtils,
  std/strformat,
  std/times,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/transformers/src/models,
  workspace/transformers/src/models/moonlight {.all.},
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils,
  workspace/transformers/tests/harness/select_device

from workspace/transformers/tests/harness import zstdReadFixture

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" /
    "Moonlight-16B-A3B"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-04-greedy-text-generation" / "Moonlight-16B-A3B"
  # Horizon-32 fixture stems, matching prompt_fixture_name in the generator
  # (alphanumeric prompt, 32-steps suffix).
  FixtureStems = ["Hello_how_are_you_32_steps",
                  "The_capital_of_France_is_32_steps",
                  "Big_blue_whales_eat_krill_32_steps"]

proc main() =

  echo "Loading model..."
  let model = loadModel($ModelPath, testDevice())
  echo "Model loaded."
  let device = model.getDeviceKind()
  echo "Replay device: " & deviceName(device)
  let cfg = model.getConfig()

  # Composed depth of one full-stack forward, used as the assertArgMax depth.
  # Moonlight is a pure MLA stack, every block runs one MLA mixer and one
  # hidden mixer, and the stage spelling follows the model's own 03 suites:
  # Stage ledger per block class:
  # - each MLA mixer composes 3 accumulation stages
  # - each routed block output composes 2 stages, past the grouped_mm record
  #
  # Remaining stages:
  # - each leading dense block output composes 1 stage, the dense count is
  #   first_k_dense_replace in the checkpoint config
  # - the final norm and the head projection add 2 stages
  # This model's getConfig leaves layerKinds empty, the schedule carries no
  # per-layer entry to loop, so the depth derives from the layer count.
  # At first_k_dense_replace = 1 the sum is 3N + 2(N - 1) + 1 + 2 = 5N + 1,
  # the depth the 03 full-forward suite carries for this model's final logits.
  let denseBlocks =
    parseFile($ModelPath / "config.json"){"first_k_dense_replace"}.getInt()
  var composedDepth = 0
  for i in 0 ..< cfg.num_hidden_layers:
    composedDepth += 3
    if i < denseBlocks:
      composedDepth += 1
    else:
      composedDepth += 2
  let chainDepth = composedDepth + 2
  echo "Composed chain depth: " & $chainDepth

  # MLA per-buffer-width pool, mirroring generate() in models.nim.
  # K and V use kvLoraRank and qkRopeHeadDim, both single head.
  # The generic newOrchestrator sizes K and V to the same head_dim,
  # which would mismatch the V plane.
  # No recurrent states live outside the page pool on this stack.
  let maxContextLen = 64
  let numPoolPages = computeNumPages(maxContextLen, concurrentRequests = 1)
  let dtype = F.kBFloat16

  for stem in FixtureStems:
    echo "Fixture: " & stem
    let fixture = parseJson(zstdReadFixture(FixtureDir / stem & ".json.zst"))
    var flipCount = 0
    let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
    let expected = fixture["generated_ids"].mapIt(it.getInt())
    let decisionsPath = FixtureDir / stem & ".decisions.json.zst"

    var orc = Orchestrator.init(
      cfg.num_hidden_layers, 1,
      1, cfg.mlaKvLoraRank,
      1, cfg.mlaKpeWidth,
      maxContextLen, numPoolPages, dtype, device)
    defer: orc.endSequence()

    var ids = promptIds
    orc.startSequence(ids.mapIt(it.uint32))
    let chainStart = getMonoTime()
    let inputIds = F.toTensor([ids]).to(device)
    let logits = model.forward(orc.getInferenceContextMut(), inputIds)
    orc.setKvPosition(ids.len)
    var row = nextStepRow(logits, ids.len - 1)

    for step in 0 ..< expected.len:
      assertArgMax(row, decisionsPath, step, kReduction, flipCount,
        msg = stem & " step " & $step, depth = chainDepth)
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

    # The wall clock and tok/s of the replay chain make device and kernel
    # regressions visible per run.
    let chainWall = (getMonoTime() - chainStart).inNanoseconds.float64 * 1e-9
    echo &"    chain wall {chainWall:.3f} s ({expected.len.float64 / chainWall:.3f} tok/s)"

when isMainModule:
  main()
