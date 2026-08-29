# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -r --hints:off --warnings:off --outdir:build/tests/lfm2-greedy \
##   --nimcache:nimcache/tests/lfm2-greedy \
##   workspace/transformers/tests/q_bf16/test_lfm2_25_05_greedy_decoding.nim

import
  std/json,
  std/os,
  workspace/libtorch as F,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/page_pool,
  workspace/transformers/src/models/lfm2 {.all.},
  workspace/libtorch_testutils

{.experimental: "callOperator".}

const
  FixtureDir =
    currentSourcePath().parentDir() / ".." / "fixtures" / "greedy-decoding" / "LFM2.5-230M"
  # Weights load from the real checkpoint through the git-ignored
  # hf_models/LFM2.5-230M symlink, the layout the Qwen3.5 suites use.
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "LFM2.5-230M"
  NumLayers = 14
  NumKvHeads = 8
  HeadDim = 64
  # Two bf16 steps at the |logit| <= 16.75 range of LFM2.5-230M, the band
  # test_lfm2_25_04_model.nim accepts for last-position logits (LogitsAbsTol there).
  # A reference top-1/top-2 pair closer than this cannot be resolved
  # by a bf16 reimplementation with a different reduction order,
  # so those steps accept the fixture's runner-up as an equally admissible argmax.
  # Worst-case global band applied per step, so at low-magnitude logits
  # it spans many steps and forgives a resolvable pick.
  LogitNoiseBand = 0.25'f64
  # In the committed chat fixture, at most 3 of 16 steps have a margin inside the band
  # (0.125, 0.125, 0.25), so wrong == 0 alone implies at least 13 strict matches.
  # This floor triggers only after a fixture regeneration widens the tie set.
  # wrong == 0, not a top-2-membership check, is what keeps the test token-exact.
  MinStrictChat = 12

type
  Verdict = enum
    Strict,       ## Nim argmax == reference top-1
    NearTie,      ## reference margin <= LogitNoiseBand and Nim argmax == runner-up
    Wrong         ## anything else

proc readIds(node: JsonNode): seq[int64] =
  result.newSeq(node.len)
  for i in 0 ..< node.len:
    result[i] = node[i].getInt().int64

proc lastArgmax(logits: Tensor): int =
  ## Vocabulary index of the last row of a (batch, seq, vocab) logit tensor.
  logits.narrow(1, logits.size(1) - 1, 1).argmax().item(int)

proc verdict(nimPick: int, margin: float64, top1, runnerUp: int): Verdict =
  if nimPick == top1:
    result = Strict
  elif margin <= LogitNoiseBand and nimPick == runnerUp:
    result = NearTie
  else:
    result = Wrong

proc replay(model: Lfm2Model, fixture: JsonNode, teacherForced: bool): (int, int, int) =
  ## Replay of one greedy fixture, returning the (strict, nearTie, wrong)
  ## counts.
  ##
  ## Free-running (teacherForced = false) feeds Nim's own argmax back, so it
  ## also exercises the compounding of conv state and KV pages across the 16
  ## decode steps.
  ##
  ## Teacher forcing feeds the fixture's own token each step,
  ## which keeps every step's state history equal to the reference's.
  ## That is required once a step is a near-tie:
  ## from there the two trajectories continue from different tokens,
  ## and are not comparable.
  let promptIds = fixture["prompt_ids"].readIds
  let expected = fixture["generated_ids"].readIds
  let total = promptIds.len + expected.len

  var ctx = InferenceContext.init(NumLayers, 1, NumKvHeads, total, HeadDim)
  let pool = PagePool.init(
    64, num_layers = NumLayers, kv_heads = NumKvHeads, head_dim = HeadDim,
    dtype = F.kBFloat16, device = F.kCPU)
  for i in 0 ..< ceilDiv(total, TokensPerPage):
    ctx.pages.add(pool.borrow())
  ctx.position_ids =
    F.arange(promptIds.len, F.tensorOptions(F.kInt64, F.kCPU)).unsqueeze(0)
  ctx.kv_position = 0

  var logits = model.forward(ctx, F.toTensor(promptIds).unsqueeze(0))
  var next = logits.lastArgmax
  var pos = promptIds.len
  var strict = 0
  var nearTie = 0
  var wrong = 0
  for step in 0 ..< expected.len:
    let margin = fixture["argmax_margins"][step].getFloat()
    let top1 = fixture["argmax_top2_ids"][step][0].getInt()
    let runnerUp = fixture["argmax_top2_ids"][step][1].getInt()
    let nimPick = next
    case verdict(nimPick, margin, top1, runnerUp)
    of Strict:
      inc strict
    of NearTie:
      inc nearTie
      echo "    step ", step, ": near-tie margin ", margin,
        ", reference top-2 (", top1, ", ", runnerUp, "), Nim picked ", nimPick
    of Wrong:
      inc wrong
      echo "    step ", step, ": WRONG Nim picked ", nimPick,
        ", reference top-2 (", top1, ", ", runnerUp, ") at margin ", margin
    let fed = if teacherForced: expected[step].int else: nimPick
    ctx.position_ids =
      F.arange(pos, pos + 1, F.tensorOptions(F.kInt64, F.kCPU)).unsqueeze(0)
    ctx.kv_position = pos
    logits = model.forward(ctx, F.toTensor([fed.int64]).unsqueeze(0))
    next = logits.lastArgmax
    inc pos
  result = (strict, nearTie, wrong)

proc checkFixtures(): bool =
  let model = loadLfm2ModelRaw(ModelDir, kCPU)
  result = true

  for name in ["Long_prose_prompt.json", "Chat_template_prompt.json"]:
    let fixture = parseJson(readFile(FixtureDir / name))
    let steps = fixture["generated_ids"].len
    echo "  ", name, ": ", fixture["prompt_ids"].len, " prompt tokens, ",
      steps, " greedy steps"
    let (strict, nearTie, wrong) = replay(model, fixture,
      teacherForced = name == "Chat_template_prompt.json")
    echo "    ", strict, " strict, ", nearTie, " near-tie, ", wrong, " wrong"
    if wrong != 0:
      result = false
    if name == "Chat_template_prompt.json" and strict < MinStrictChat:
      echo "    only ", strict, " of ", steps, " steps strict: the gate degraded"
      result = false

when isMainModule:
  runCppTest("LFM2.5-230M greedy decoding vs fixtures", checkFixtures)
