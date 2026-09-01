# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Verify greedy (temp=0) decoding matches HF Transformers output.
## Token-exact except near-tie argmax flips inside the cross-kernel
## noise band (R12 verdict: 2 bf16 ulps at the fixture max).

import
  std/os,
  std/json,
  std/math,
  std/strformat,
  std/sequtils,
  workspace/libtorch,
  workspace/toktoktok,
  workspace/transformers/src/models

const
  ModelPath = currentSourcePath().parentDir() / ".." / "hf_models" / "Qwen3-0.6B"
  FixtureDir = currentSourcePath().parentDir() / ".." / "fixtures" / "greedy-decoding" / "Qwen3-0.6B"

const TieFlipBandUlps = 2

func bf16Ulp(x: float32): float32 =
  ## One bf16 ulp at x: 2^(floor(log2|x|) - 7). bf16 stores 7 mantissa bits.
  let a = abs(x)
  if a == 0.0f:
    return 0.0f
  result = pow(2.0f, floor(log2(a)) - 7.0f)

func isTieFlip(data: JsonNode; step: int; actual: int): bool =
  ## True when the argmax divergence at `step` is a near-tie flip:
  ## the actual token is the fixture's recorded runner-up and the recorded
  ## top-2 logit gap sits inside the cross-kernel noise band
  ## (R12: 2 bf16 ulps at the fixture max). Greedy continuation
  ## past a within-band near-tie is kernel-build-sensitive, so comparison
  ## stops at the flip step.
  if not data.hasKey("steps") or step >= data["steps"].len:
    return false
  let s = data["steps"][step]
  if not s.hasKey("top10_tokens") or not s.hasKey("top10_logits"):
    return false
  if s["top10_tokens"].len < 2 or s["top10_logits"].len < 2:
    return false
  var inTop2 = false
  for i in 0 ..< 2:
    if s["top10_tokens"][i].getInt() == actual:
      inTop2 = true
  if not inTop2:
    return false
  var maxAbs = 0.0f
  for v in s["top10_logits"]:
    maxAbs = max(maxAbs, abs(v.getFloat()))
  let gap = s["top10_logits"][0].getFloat() - s["top10_logits"][1].getFloat()
  result = gap <= float32(TieFlipBandUlps) * bf16Ulp(maxAbs)

proc checkFixture(model: Model, jsonPath: string) =
  let data = parseJson(readFile(jsonPath))
  let prompt = data["prompt"].getStr()

  # Parse generated_ids from JSON array
  var expectedIds: seq[int] = @[]
  for el in data["generated_ids"]:
    expectedIds.add(el.getInt())

  let expectedText = data["generated_text"].getStr()
  let numPrompt = data["num_prompt_tokens"].getInt()
  let numGen = data["num_generated_tokens"].getInt()

  echo &"  Prompt ({numPrompt} tokens): {prompt}"

  # Generate with temp=0 (greedy)
  let output = model.generate(prompt, temp = 0.0f, maxTokens = numGen)

  # Compare token IDs. Re-encode the output to get the generated portion
  let actualIds = model.getTokenizer().encode(output)
  let actualGenerated = if actualIds.len >= numPrompt:
    actualIds[numPrompt ..< actualIds.len]
  else:
    actualIds

  echo &"  Expected ({expectedIds.len} tokens): {expectedText}"
  echo &"  Actual   ({actualGenerated.len} tokens): {model.getTokenizer().decodeToString(actualGenerated)}"

  let idMatch = actualGenerated == expectedIds
  if idMatch:
    echo "  ✅ Token IDs match perfectly"
  else:
    var firstDiff = "N/A"
    var firstDiffIdx = -1
    for i in 0 ..< min(expectedIds.len, actualGenerated.len):
      if expectedIds[i] != actualGenerated[i]:
        firstDiff = &"token {i}: expected {expectedIds[i]}, got {actualGenerated[i]}"
        firstDiffIdx = i
        break
    if expectedIds.len != actualGenerated.len:
      firstDiff = &"length: expected {expectedIds.len}, got {actualGenerated.len}"
    echo &"  ❌ Token IDs diverge: {firstDiff}"
    if firstDiffIdx >= 0 and isTieFlip(data, firstDiffIdx, actualGenerated[firstDiffIdx]):
      echo "  ⚠️ Near-tie flip inside the cross-kernel noise band; counted"
      echo "  as a match. Chains diverge past a within-band near-tie."
      return
    raise newException(AssertionError, &"[greedy-test] Token mismatch for {jsonPath}")

proc main*() =
  echo "Loading model..."
  let model = loadModel($ModelPath, kCPU)
  echo "Model loaded.\n"

  var passed = 0
  var total = 0

  for fixture in walkPattern($FixtureDir & "/*.json"):
    inc total
    try:
      checkFixture(model, fixture)
      inc passed
    except AssertionError:
      discard
    echo ""

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if passed == total:
    echo &"✅ PASS | Greedy decoding: {passed}/{total} fixtures match"
  else:
    echo &"❌ FAIL | Greedy decoding: {passed}/{total} fixtures match"
    raise newException(AssertionError, &"{passed}/{total} greedy fixtures passed")
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

when isMainModule:
  main()
