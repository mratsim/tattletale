# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the Kimi-Linear-48B-A3B-Instruct hybrid stack,
## replaying recorded chains teacher-forced against the committed decision frames.
##
## Requires:
## - the local model at tests/hf_models/Kimi-Linear-48B-A3B-Instruct (gitignored)
## - the bf16-04 greedy-text-generation fixture dir.
## Run:
##   nim cpp -r --hints:off --warnings:off --passC:"-std=c++20" \
##     --outdir:build/wip --nimcache:nimcache/wip workspace/transformers/tests/q_bf16/t_bf16_kimi_04_greedy_text_generation.nim

import
  std/os,
  std/sequtils,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/models,
  workspace/transformers/src/models/kimi_linear {.all.},
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/tests/harness/harness,
  workspace/transformers/tests/layer_utils,
  workspace/transformers/tests/stateful_utils,
  workspace/transformers/tests/harness/select_device

from workspace/transformers/tests/harness import zstdReadFixture

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" /
    "Kimi-Linear-48B-A3B-Instruct"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" /
    "bf16-04-greedy-text-generation" / "Kimi-Linear-48B-A3B-Instruct"
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
  let cfg = model.getConfig()

  # Composed depth of one full-stack forward, used as the assertArgMax depth.
  # - An MLA block composes 4 stages, three through its mixer and one through
  #   its routed output.
  # - A KDA/dense block composes 2 stages, one through its mixer and one through
  #   its routed output.
  # Every deciding row comes from a full stack forward. The final norm and head
  # projection add 2 stages, so every step carries the SAME chain depth.
  var composedDepth = 0
  for kind in cfg.layerKinds:
    if kind == alkMla:
      composedDepth += 4
    else:
      composedDepth += 2
  let chainDepth = composedDepth + 2
  echo "Composed chain depth: " & $chainDepth

  # MLA per-buffer-width pool, mirroring generate() in models.nim.
  # K and V use kvLoraRank and qkRopeHeadDim, both single head.
  # The generic newOrchestrator sizes K and V to the same head_dim,
  # which would mismatch the V plane.
  # The KDA recurrent states live in InferenceContext (ensureKdaStates),
  # outside the page pool.
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

when isMainModule:
  main()
