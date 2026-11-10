# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy (temp=0) decoding of the Qwen3-0.6B stack, the recorded chains
## replayed teacher-forced against the committed decision frames.
##
## Test that both recorded chains re-converge on Metal with per-step
## assertArgMax (token, top-32 logits, KL, tail), tie-eligible picks teacher-force
## the recorded token, tie flips stay within the cap.
##
## Recorded frames (Python testgen, consumed in place):
##   workspace/transformers/tests/fixtures/bf16-04-greedy-text-generation/Qwen3-0.6B
## Local model: /Users/pi/Documents/Programming/MODELS/Qwen3-0.6B
## (TTT_QWEN3_PATH env override).
##
## Readback protocol (harness header): model.forward evaluates its logits
## root and the assertArgMax readback runs on that evaluated path.
## Each decode step reads its own step forward's root.
##
## Run from the worktree root:
##   nim c -r -d:release --stackTrace:on --lineTrace:on --lineDir:on \
##     --hints:off --warnings:off --path:"$PWD" \
##     --nimcache:nimcache/wipM09suite04 --outdir:build/wipM09suite04 \
##     workspace/tattleformers/tests/q_bf16/t_bf16_qwen3_04_greedy_text_generation.nim

import
  std/os,
  std/json,
  std/sequtils,
  workspace/tensorcery/src/tensorcery,
  workspace/zstd/zstd_highlevel,
  workspace/tattleformers/src/layers/embedding,
  workspace/tattleformers/src/layers/linear,
  workspace/tattleformers/src/layers/norm,
  workspace/tattleformers/src/layers/rope,
  workspace/tattleformers/src/layers/ffn,
  workspace/tattleformers/src/layers/lmhead,
  workspace/tattleformers/src/layers/attention,
  workspace/tattleformers/src/layers/decoder_layers,
  workspace/tattleformers/src/stateful/orchestrator,
  workspace/tattleformers/src/models,
  workspace/tattleformers/tests/harness/harness

const
  FixtureDir = currentSourcePath().parentDir().parentDir().parentDir().parentDir() /
    "transformers" / "tests" / "fixtures" /
    "bf16-04-greedy-text-generation" / "Qwen3-0.6B"
  ModelPathDefault = "/Users/pi/Documents/Programming/MODELS/Qwen3-0.6B"
  MaxCtx = 256
    ## Pool and RoPE sizing override, the recorded chains span at most 48 tokens:
    ## the config's max_position_embeddings would eager-allocate
    ## a multi-gigabyte pool (the models.nim generate override rationale).

proc main(modelPath: string): bool =
  # Recorded-chain replay against the committed decision frames, one chain per fixture with the recorded tokens teacher-forced at each step.
  echo "Loading model..."
  let model = loadModel(modelPath)
  let cfg = model.config
  echo "Model loaded."

  let numPoolPages = computeNumPages(MaxCtx, concurrentRequests = 1)
  for stem in ["Hello_how_are_you",
               "Do_you_know_the_story_of_this_proverb_\u78e8\u5200"]:
    echo "Fixture: " & stem
    let fixture = parseJson(
      readFile(FixtureDir / stem & ".json.zst").zstdDecompress(string))
    var flipCount = 0
    let promptIds = fixture["prompt_ids"].mapIt(it.getInt())
    let expected = fixture["generated_ids"].mapIt(it.getInt())
    let decisionsPath = FixtureDir / stem & ".decisions.json.zst"
    var orc = Orchestrator.init(cfg.num_hidden_layers,
      cfg.num_key_value_heads, MaxCtx, cfg.head_dim, numPoolPages)
    defer: orc.endSequence()

    var ids = promptIds
    orc.startSequence(ids.mapIt(uint32(it)))
    let logits = model.forward(orc.getInferenceContextMut(), ids)
    orc.setKvPosition(ids.len)
    var row = logits.narrow(1, ids.len - 1, 1)

    for step in 0 ..< expected.len:
      assertArgMax(row, decisionsPath, step, kReduction, flipCount,
        msg = stem & " step " & $step, depth = 1)
      # Recorded token teacher-forced at each step, so a tie flip
      # stays recorded and every later check must re-converge.
      let chosen = expected[step]
      ids.add chosen
      if step + 1 < expected.len:
        orc.appendToken(ids.len - 1, chosen.uint32)
        let stepLogits = model.forward(orc.getInferenceContextMut(), @[chosen])
        orc.setKvPosition(ids.len)
        row = stepLogits
    stderr.writeLine stem & ": tie flips " & $flipCount & " / " & $expected.len & " steps"
  result = true

when isMainModule:
  let modelPath = getEnv("TTT_QWEN3_PATH", ModelPathDefault)
  if main(modelPath):
    stderr.writeLine "qwen3 greedy text generation (Metal): all asserts green"
