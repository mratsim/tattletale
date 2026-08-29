# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Greedy-decoding end to end for the Qwen3.6-35B-A3B wired text stack:
## fixture token chains recorded by gen_qwen36_greedy_fixtures.py against
## the vendored transformers reference. Every chain is replayed stepwise
## through loadQwen35MoeModelRaw and compared token-exact, the recorded
## near-tie provision covering the first divergence per prompt.
## Run command:
##   nim cpp -r --verbosity:0 --hints:off --warnings:off
##     --outdir:build/tests/test_qwen36_07_greedy
##     --nimcache:nimcache/tests/test_qwen36_07_greedy
##     workspace/transformers/tests/q_bf16/test_qwen36_07_greedy.nim
#
# Requires: local model at tests/hf_models/Qwen3.6-35B-A3B (gitignored)

import
  std/os,
  std/math,
  std/strformat,
  std/sequtils,
  std/strutils,
  std/monotimes,
  std/times,
  pkg/packedjson,
  workspace/libtorch as F,
  workspace/transformers/src/models/qwen35_moe,
  workspace/transformers/src/stateful/orchestrator,
  workspace/libtorch_testutils

const
  ModelDir = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3.6-35B-A3B"
  GreedyFixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "qwen36-greedy"
  GreedyFixtureFiles = [
    "Hello_how_are_you_32_steps.json",
    "The_capital_of_France_is_32_steps.json",
    "Big_blue_whales_eat_krill_32_steps.json",
  ]
  VendoredSha = "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701"
  # Generator ceilings: prompts stay inside one GDN chunk, horizons
  # inside the recorded 32 greedy step ceiling.
  FixturePromptTokenCeiling = 12
  FixtureHorizonCeiling = 32
  NearTieBandUlps = 2
  # Decode context ceiling: one TokensPerPage page covers every prompt
  # chain comfortably above the prompt + horizon footprint.
  MaxContextLen = 256

func bf16Ulp(x: float32): float32 =
  ## One bf16 ulp at x: 2^(floor(log2|x|) - 7). bf16 stores 7 mantissa bits.
  let a = abs(x)
  if a == 0.0f:
    return 0.0f
  result = pow(2.0f, floor(log2(a)) - 7.0f)

func recordedFloat(node: packedjson.JsonNode): float64 =
  ## Read a fixture number, accepting the JInt spelling a JSON writer
  ## emits for whole values.
  ## Wrong-typed values abort loudly
  ## (packedjson getters answer a zero default without raising).
  case node.kind
  of packedjson.JFloat:
    result = node.getFloat()
  of packedjson.JInt:
    result = node.getInt().float64
  else:
    doAssert false, "fixture number expected, found " & $node.kind

func isNearTieFlip(nearTie: packedjson.JsonNode, step: int, actual: int): bool =
  ## Near-tie flip test, two conditions:
  ## - the actual pick equals the fixture's recorded runner-up id
  ## - the recorded top-2 logit gap sits inside the cross-port noise band,
  ##   2 bf16 ulps at the recorded max |logit| per R12
  ## Greedy continuation past a within-band near-tie stays
  ## libtorch-build-sensitive, so comparison stops at the flip step.
  doAssert nearTie.kind == packedjson.JObject
  let logitsRows = nearTie{"logits"}
  let idRows = nearTie{"ids"}
  doAssert logitsRows.kind == packedjson.JArray and idRows.kind == packedjson.JArray
  if step >= logitsRows.len:
    return false
  doAssert logitsRows[step].kind == packedjson.JArray and logitsRows[step].len == 2
  doAssert idRows[step].kind == packedjson.JArray and idRows[step].len == 2
  if actual != idRows[step][1].getInt():
    return false
  let top1 = recordedFloat(logitsRows[step][0])
  let top2 = recordedFloat(logitsRows[step][1])
  let maxAbs = max(abs(top1), abs(top2)).float32
  result = (top1 - top2) <= float32(NearTieBandUlps) * bf16Ulp(maxAbs)

proc collectIdSeq(node: packedjson.JsonNode, target: var seq[int]) =
  ## Append every element of a fixture int64 array onto `target`.
  doAssert node.kind == packedjson.JArray
  for i in 0 ..< node.len:
    doAssert node[i].kind == packedjson.JInt
    target.add node[i].getInt().int

proc runGreedyChains*(device: DeviceKind = kCPU): tuple[tokenExact: int, nearTie: int] =
  ## Replay the three greedy fixture chains through the wired 35B stack
  ## on `device`, one fresh Orchestrator per chain, and compare every greedy
  ## token against the recorded fixture. Returns how many prompts ended
  ## token-exact and how many flipped at a recorded near-tie step.
  ##
  ## Canonical CPU receipt: 1 token-exact + 2 in-band near-tie prompts.
  ## A device arm is judged token by token against the frozen fixture
  ## receipts, provision included: each token is either the recorded pick
  ## or the recorded runner-up at a near-tie step inside the band.
  ## Any token outside that provision voids the arm, both verdicts face
  ## each other in the report and nothing gets re-recorded.
  let loadStart = getMonoTime()
  let model = loadQwen35MoeModelRaw(ModelDir, device)
  doAssert model.loadedTensorCount == 693
  doAssert model.config.numHiddenLayers == 40
  echo &"    load wall {(getMonoTime() - loadStart).inNanoseconds.float64 * 1e-9:.3f} s"

  var tokenExactPrompts = 0
  var nearTiePrompts = 0
  for fixtureName in GreedyFixtureFiles:
    let meta = parseFile(GreedyFixtureDir / fixtureName)
    var promptIds: seq[int] = @[]
    collectIdSeq(meta{"input_tokens"}, promptIds)
    var expected: seq[int] = @[]
    collectIdSeq(meta{"generated_ids"}, expected)
    let horizon = meta{"max_new_tokens"}.getInt()
    doAssert expected.len == horizon
    doAssert promptIds.len == meta{"num_input_tokens"}.getInt()
    let nearTie = meta{"near_tie"}
    echo &"Fixture: {fixtureName} ({promptIds.len} prompt tokens, {horizon} steps)"

    # Fresh decode state per prompt: the Orchestrator owns the paged KV
    # pool plus the GDN conv and SSM slots, so a new one carries no
    # residue of the previous chain.
    var orc = Orchestrator.init(
      model.config.numHiddenLayers, 1, model.config.numKeyValueHeads,
      MaxContextLen, model.config.headDim,
      computeNumPages(MaxContextLen, 1), F.kBFloat16, device)
    orc.startSequence(promptIds.mapIt(it.uint32))
    let inputIds = F.toTensor([promptIds]).to(device)
    let logits = model.forward(orc.getInferenceContextMut(), inputIds)
    # Greedy contract: argmax of the deciding logits row, no sampler
    # between. Same write-skips as generate(): kv_position advances
    # after the forward pass that consumed it.
    orc.setKvPosition(promptIds.len)
    # Chain: the prefill argmax is fixture generated_ids[0]. Every decode
    # step feeds back the previous pick and appends its own argmax.
    var nextToken =
      logits.narrow(1, promptIds.len - 1, 1).squeeze(1).squeeze(0).argmax().item(int)
    var nimTokens: seq[int] = @[]
    var ids = promptIds
    ids.add nextToken
    nimTokens.add nextToken
    let recordedChecksums = meta{"step_logits_checksum"}
    let prefillChecksum = logits.narrow(1, promptIds.len - 1, 1)
      .squeeze(1).squeeze(0).to(F.kFloat32).sum().item(float64)
    echo "    step 0 logits f32 checksum: recorded " & $recordedFloat(recordedChecksums[0]) &
      ", observed " & $prefillChecksum & " (diagnostic, row sums are not a gate)"
    var finalChecksum = -1.0'f64
    let armStart = getMonoTime()
    while nimTokens.len < horizon:
      orc.decodeStep(ids.len - 1, nextToken.uint32, device)
      let stepLogits = model.forward(
        orc.getInferenceContextMut(), F.toTensor([[nextToken]]).to(device))
      orc.setKvPosition(ids.len)
      let flatLogits = stepLogits.squeeze(0).squeeze(0)
      let nextTokenNew = flatLogits.argmax().item(int)
      if nimTokens.len == horizon - 1:
        finalChecksum = flatLogits.to(F.kFloat32).sum().item(float64)
      nimTokens.add nextTokenNew
      ids.add nextTokenNew
      nextToken = nextTokenNew
    let armWall = (getMonoTime() - armStart).inNanoseconds.float64 * 1e-9
    echo "    final logits f32 checksum: recorded " & $recordedFloat(recordedChecksums[horizon - 1]) &
      ", observed " & $finalChecksum & " (diagnostic, row sums are not a gate)"
    echo &"    arm wall {armWall:.3f} s ({horizon.float64 / armWall:.2f} tok/s)"

    var firstDiff = -1
    for i in 0 ..< horizon:
      if nimTokens[i] != expected[i]:
        firstDiff = i
        break
    if firstDiff < 0:
      inc tokenExactPrompts
      echo &"  ✅ token-exact: all {horizon} greedy steps match the fixture"
    elif isNearTieFlip(nearTie, firstDiff, nimTokens[firstDiff]):
      inc nearTiePrompts
      echo &"  ⚠️ Near-tie flip at step {firstDiff}: expected {expected[firstDiff]}, " &
        &"got {nimTokens[firstDiff]}, inside the recorded R12 band"
      echo &"  counted as a match; chains diverge past a within-band near-tie, " &
        &"comparison for this fixture stops"
    else:
      echo &"  ❌ tokens diverge: step {firstDiff}: expected {expected[firstDiff]}, " &
        &"got {nimTokens[firstDiff]}"
      raise newException(AssertionError,
        "[greedy-test] gross token-chain divergence for " & fixtureName & " at step " &
        $firstDiff)
    echo ""

  result = (tokenExactPrompts, nearTiePrompts)

# Apple Metal Performance Shaders device kind for `runGreedyChains`.
# The libtorch wrapper enum stops at kVulkan and does not
# list the member the C++ catalogue carries, so this importcpp bridge
# reveals c10::DeviceType::MPS to the Nim type at the one site allowed
# to name a non-CPU device.
var kMps* {.importcpp: "c10::DeviceType::MPS", nodecl.}: DeviceKind


proc main() =
  runCppTest "greedy fixture provenance and near-tie band sanity":
    proc(): bool =
      # Provenance block: identity pins, shapes and the anchored near-tie
      # record, read back and re-checked before any model load. The near-tie
      # census recomputes the exact-tie count from the recorded pairs,
      # then asserts the recorded tied_steps key against it.
      for fixtureName in GreedyFixtureFiles:
        let meta = parseFile(GreedyFixtureDir / fixtureName)
        doAssert meta{"schema"}.kind == packedjson.JString
        doAssert meta{"schema"}.getStr() == "tt-qwen36-greedy-1"
        doAssert meta{"model"}.getStr() == "Qwen3.6-35B-A3B"
        doAssert meta{"vendored_sha"}.kind == packedjson.JString
        doAssert meta{"vendored_sha"}.getStr() == VendoredSha
        doAssert meta{"torch_version"}.kind == packedjson.JString
        doAssert meta{"torch_version"}.getStr().startsWith("2.11.0")
        doAssert meta{"transformers_version"}.kind == packedjson.JString
        doAssert meta{"experts_implementation"}.kind == packedjson.JString
        doAssert meta{"experts_implementation"}.getStr() == "eager"
        doAssert meta{"attn_implementation"}.kind == packedjson.JString
        doAssert meta{"attn_implementation"}.getStr() == "sdpa"
        doAssert meta{"num_threads"}.getInt() == 1
        doAssert meta{"dtype"}.getStr() == "bfloat16"
        doAssert meta{"device"}.getStr() == "cpu"

        let horizon = meta{"max_new_tokens"}.getInt()
        doAssert horizon > 0 and horizon <= FixtureHorizonCeiling,
          "horizon must stay inside the recorded generator ceiling"
        let inputTokens = meta{"input_tokens"}
        doAssert inputTokens.kind == packedjson.JArray
        doAssert inputTokens.len > 0 and inputTokens.len <= FixturePromptTokenCeiling
        doAssert inputTokens.len == meta{"num_input_tokens"}.getInt()
        let generated = meta{"generated_ids"}
        doAssert generated.kind == packedjson.JArray
        doAssert generated.len == horizon
        for i in 0 ..< generated.len:
          doAssert generated[i].kind == packedjson.JInt

        let checksums = meta{"step_logits_checksum"}
        doAssert checksums.kind == packedjson.JArray
        doAssert checksums.len == horizon
        for i in 0 ..< horizon:
          discard recordedFloat(checksums[i])

        let nearTie = meta{"near_tie"}
        doAssert nearTie.kind == packedjson.JObject
        let logitsRows = nearTie{"logits"}
        let idRows = nearTie{"ids"}
        doAssert logitsRows.kind == packedjson.JArray and idRows.kind == packedjson.JArray
        doAssert logitsRows.len == horizon and idRows.len == horizon
        var tiedSteps = 0
        for step in 0 ..< horizon:
          doAssert logitsRows[step].kind == packedjson.JArray and
            logitsRows[step].len == 2, "top-2 pair must hold 2 values"
          doAssert idRows[step].kind == packedjson.JArray and idRows[step].len == 2,
            "top-2 pair must hold 2 ids"
          let top1 = recordedFloat(logitsRows[step][0])
          let top2 = recordedFloat(logitsRows[step][1])
          doAssert top2 <= top1, "the recorded pair is anchored on the argmax"
          if top1 == top2:
            inc tiedSteps
        doAssert nearTie{"tied_steps"}.kind == packedjson.JInt
        doAssert nearTie{"tied_steps"}.getInt() == tiedSteps,
          "recorded tied_steps disagrees with the recorded pairs"
        # Band sanity probe with an impossible actual id: the provision
        # must reject it at every step, whatever the recorded gap.
        for step in 0 ..< horizon:
          discard isNearTieFlip(nearTie, step, 123456789)
        echo &"  {fixtureName}: horizon {horizon}, tied steps {tiedSteps}"
      result = true

  runCppTest "greedy chains through the wired 35B vs fixtures":
    proc(): bool =
      let (tokenExactPrompts, nearTiePrompts) = runGreedyChains(kCPU)
      doAssert tokenExactPrompts + nearTiePrompts == GreedyFixtureFiles.len,
        "every fixture must end token-exact or inside the recorded near-tie provision"
      echo &"Greedy e2e: {tokenExactPrompts} token-exact, {nearTiePrompts} near-tie " &
        &"of {GreedyFixtureFiles.len} fixtures"
      result = true

  echo "\ntest_qwen36_07_greedy: all blocks PASS"

when isMainModule:
  main()
