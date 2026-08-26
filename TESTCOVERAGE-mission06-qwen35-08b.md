# Test-Coverage Checker Report — Mission 06 (Q35-A, Qwen3.5-0.8B text stack, bf16)

Commit under review: `e61ce975` (diff `0dab210f..e61ce975`)
Worktree: `tattletale-worktrees/20260827-1910-qwen35-08b-text`
Deliverables: `workspace/transformers/tests/q_bf16/test_qwen3_5_06_t2t_inference.nim` (new), `workspace/transformers/tests/q_bf16/test_qwen3_5_07_greedy_decoding.nim` (new)

## VERDICT: PASS-WITH-FINDINGS

Both deliverables are genuine, non-vacuous, match the series conventions, and pass when run.
All regressions (qwen3.5 00–07 + shard, qwen3 00–02) pass. Three MINOR findings: two
pre-known prose semicolons, the (documented) never-exercised eos-break path (GOAL-003),
and a partial-strength \p{M} round-trip lock.

## Summary

| ID | Severity | Confidence | File | Issue |
|----|----------|------------|------|-------|
| TEST-001 | MINOR | 1.0 | test_qwen3_5_06_t2t_inference.nim:28,61 | Prose semicolons (the 2 known ones) |
| TEST-002 | MINOR | 1.0 | test_qwen3_5_07_greedy_decoding.nim (whole file) | eos-break path never exercised end-to-end (GOAL-003), honestly documented but untested |
| TEST-003 | MINOR | 0.8 | test_qwen3_5_06_t2t_inference.nim:57–66 | Combining-mark round-trip is byte-concatenation; does not fully "prove" the \p{M} regex path |

---

## Per-deliverable verification

### Deliverable 1: `test_qwen3_5_06_t2t_inference.nim` — PASS

Runs: `nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-t2t --nimcache:nimcache/tests/qwen35-t2t workspace/transformers/tests/q_bf16/test_qwen3_5_06_t2t_inference.nim` → exit 0, `✅ PASS` (1m2s).

1. **Tokenize → forward → untokenize exercised?** Yes. The `generate()` call (line 72, temp=1.0, maxTokens=16) runs tokenize → prefill forward → 16 decode forwards → final `decodeToString`. A crash, a broken decode entry, or immediate-eos would fail it. It is a smoke assert (startsWith + longer), so wrong *weights* producing plausible text would not fail it — the token-exact weight check lives in test 07. Non-vacuous at the smoke level, correctly scoped.
2. **Combining-mark round-trip uses DECOMPOSED form?** Yes. Line 63: `"The re\u0301sume\u0301 is ready"` (e + U+0301, decomposed; verified the fixture file also stores the decomposed form). Exercises the `\p{M}` pre-tokenizer branch on the decomposed text. See TEST-003 for the strength caveat.
3. **Decode-entry assertion meaningful?** Yes. Lines 46–53: `encode(prompt) == prompt_ids` for the two NFC-clean fixtures locks the no-bos convention against the vendored generator. Passed. Note `encode` = `encodeWithSpecialTokens` in toktoktok; tokenizer.json has `bos_token: None` so nothing is prepended — the assert proves it empirically. The same convention is additionally exercised end-to-end through `generate()` by test 07's token-exact comparisons.
4. **generate() smoke asserts?** `output.len > resumePrompt.len` and `output.startsWith(resumePrompt)` (lines 75–76) — real asserts, passed.
5. **Config eos lock?** Line 42 asserts `getConfig().eosTokenId == 248044` (config.json nests it under `text_config.eos_token_id: 248044`; the Nim parser surfaces it correctly — the assert passed). This locks the stop-token constant, not the break behavior (see TEST-002).

### Deliverable 2: `test_qwen3_5_07_greedy_decoding.nim` — PASS

Runs: `nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-greedy --nimcache:nimcache/tests/qwen35-greedy workspace/transformers/tests/q_bf16/test_qwen3_5_07_greedy_decoding.nim` → exit 0, `✅ PASS`, "Greedy decoding: 3/3 fixtures match" (2m31s).

1. **Token-exact vs generated_ids for all 3 fixtures?** Yes. `checkFixture` calls `model.generate(prompt, temp=0.0f, maxTokens=numGen, maxContextLen=512)`, re-encodes the output text, slices off `numPrompt` tokens, and compares `actualGenerated == expectedIds` element-wise with a first-diff report. A wrong model/weights, wrong logits, non-argmax sampling, tokenizer divergence, or a slice bug all fail it. All 3 fixtures matched token-for-token (verified in output: "token ids match" ×3, plus the echoed expected/actual text lines are identical).
2. **temp=0 argmax lock real?** Yes. Line 105: `sample(fixedLogits, 0.0f) == fixedLogits.argmax().item(int)` on `[[1.0, 2.5, 0.3, 4.0, 3.2]]` (argmax = idx 3). `sample` (samplers.nim:25–27) short-circuits to argmax at temp 0; if that short-circuit were removed, the Gumbel path would return a random index and this single-call assert would fail ~80% of the time. Not vacuous. (Minor note: both sides call libtorch `argmax`, so a hypothetical broken `argmax` would be self-consistent — same structure as the pre-existing `test_sampler.nim` `testLowTempIsGreedy`; not a finding.)
3. **eos stop?** The fixtures' `generated_ids`/`full_ids` never contain 248044 (verified by inspection of all 3 JSON files) and the test comment (lines 29–32) says so. `generate()` breaks on `ids[^1] == cfg.eosTokenId` (models.nim:133) — verified by code reading, but no test drives the model to emit 248044. The maxTokens bound IS exercised: each fixture stops after exactly `numGen=8` tokens via the `while` condition. The eos break remains a coverage gap (TEST-002).

### Fixture consumption — PASS

- All 3 JSON fixtures exist in `tests/fixtures/greedy-decoding/Qwen3.5-0.8B/`; fields read by the tests (`prompt`, `prompt_ids`, `generated_ids`, `generated_text`, `num_prompt_tokens`, `num_generated_tokens`) are present.
- Consistency: `len(prompt_ids) == num_prompt_tokens` (6/6, 4/4, 7/7) and `len(generated_ids) == num_generated_tokens` (8/8 each); `full_ids == prompt_ids + generated_ids` (checked for Hello).
- `numGen` is used as `maxTokens`; `numPrompt` as the re-encode slice offset — both correct.
- Resume fixture: file stores the DECOMPOSED prompt (verified: `'The re\u0301sume\u0301 is ready'`, contains `e\u0301`, no `\u00e9`). Test 07 comment (lines 41–46) explains toktoktok has no NFC normalizer and passes the precomposed form (the NFC result the vendored tokenizer computes internally); the generated-ids comparison stays token-exact. The comparison is valid — it passed, matching `[310, 381, 3106, 310, 279, 22413, 6436, 13]`.

### Regressions — PASS (all exit 0)

See "Command Results" below: qwen3.5 00–05 + shard + qwen3 00–02 all pass; 06/07 (new) pass.

### House rules — PASS except 2 pre-known semicolons

| Rule | Result |
|------|--------|
| `runCppTest` (not runProbe) | PASS — both files use `runCppTest` |
| Headers run-command-only | PASS — both headers are only the `nim cpp -r` command (matches mission-05 files 01–05) |
| Banned jargon in prose | PASS — none found (scanned marketing-jargon list) |
| No prose semicolons | FAIL (MINOR) — exactly 2, at test 06:28 and 06:61 (the 2 known); pre-existing series pattern (mission 04:58, 05:96) |
| No em-dashes | PASS — none |
| Lines ≤ 128c | PASS — no line exceeds 128 |
| No artifact IDs | PASS — no SLOP/QA/iter/RID/BUG/TEST-/GOAL- strings in either file; section names self-describing |
| No absolute paths | PASS — `ModelPath`/`FixtureDir` use `currentSourcePath().parentDir()` relative resolution |

Cheating detection (C4): diff is purely additive (2 new files, 203 insertions, 0 deletions). No existing test modified, commented out, skipped, deleted, or weakened. No trivial/tautological asserts. Clean.

---

## Command Results

### Test 06 (new deliverable)

```
$ nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-t2t \
    --nimcache:nimcache/tests/qwen35-t2t \
    workspace/transformers/tests/q_bf16/test_qwen3_5_06_t2t_inference.nim
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Section: Qwen3.5-0.8B t2t round-trip + decode entry (no bos, eos 248044)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loading model...
 (source: Wikipedia)
Sublime of League of Forgotten Heroes is aOutput: The résumé is ready. (source: Wikipedia)
Sublime of League of Forgotten Heroes is a
✅ PASS | Qwen3.5-0.8B t2t round-trip + decode entry (no bos, eos 248044)
exit code: 0  (elapsed 1m2s)
```

### Test 07 (new deliverable)

```
$ nim cpp -r --hints:off --warnings:off --outdir:build/tests/qwen35-greedy \
    --nimcache:nimcache/tests/qwen35-greedy \
    workspace/transformers/tests/q_bf16/test_qwen3_5_07_greedy_decoding.nim
Section: Qwen3.5-0.8B greedy decoding - token-exact vs fixtures
Loading model...
Model loaded.

Fixture: Hello_how_are_you.json
  Prompt (6 tokens): Hello, how are you?
Hello! I'm doing well,  Expected (8 tokens): 

Hello! I'm doing well,
  Actual   (8 tokens): 

Hello! I'm doing well,
  token ids match

Fixture: The_resume_is_ready.json
  Prompt (4 tokens): The résumé is ready
 be sent to the hiring manager.  Expected (8 tokens):  to be sent to the hiring manager.
  Actual   (8 tokens):  to be sent to the hiring manager.
  token ids match

Fixture: What_is_the_capital_of_France.json
  Prompt (7 tokens): What is the capital of France?
The capital of France is **Paris  Expected (8 tokens): 

The capital of France is **Paris
  Actual   (8 tokens): 

The capital of France is **Paris
  token ids match

Greedy decoding: 3/3 fixtures match
✅ PASS | Qwen3.5-0.8B greedy decoding - token-exact vs fixtures
exit code: 0  (elapsed 2m31s)
```

### q_bf16 regressions 00–05 + shard (all exit 0)

```
test_qwen3_5_00_config:  exit=0  3/3 [OK] (Parse config, Wrapper JSON, Registry)
test_qwen3_5_01_rope:    exit=0  8/8 ✅ PASS sections, "All Qwen3.5 partial-rope tests passed!"
test_qwen3_5_02_attn:    exit=0  7/7 ✅ PASS sections, "All Qwen3.5 gated-attention tests passed!"
test_qwen3_5_03_layers:  exit=0  8/8 ✅ PASS sections, "All Qwen3.5 GDN / layer / state tests passed!"
test_qwen3_5_04_long_residual_3_blocks: exit=0  ✅ PASS (1 section)
test_qwen3_5_05_ids_to_logits_inference: exit=0  ✅ PASS (1 section)
test_qwen3_5_shard_load: exit=0  2/2 ✅ PASS (single shard + loadQwen3_5ModelRaw + generate plumbing)
```

### qwen3 regressions 00–02 (all exit 0)

```
test_qwen3_00_config:    exit=0  3/3 [OK] (Qwen3-0.6B, Qwen3-4B, Qwen3-4B-AWQ)
test_qwen3_01_rope:      exit=0  9/9 ✅ PASS sections, "✅ All RoPE tests passed!"
test_qwen3_02_attn:      exit=0  3/3 ✅ PASS sections, "All attention tests completed"
```

No test panicked or timed out in any run.

---

## Findings

### [TEST] TEST-001: Prose semicolons at test 06:28 and 06:61 — house style "no prose semicolons"

**Location:** `workspace/transformers/tests/q_bf16/test_qwen3_5_06_t2t_inference.nim:28` and `:61`
**Severity:** MINOR
**Confidence:** 1.0

**Evidence:**
```
28: # directly. generate() tokenizes the raw prompt, prefills, then decodes;
61:       # tokenizes to different ids than the vendored tokenizer; the contract
```

**Issue:** Exactly the 2 known prose semicolons flagged by the mission. House style bans prose semicolons. Pre-existing series pattern (mission 04 file line 58 and mission 05 file line 96 have the same), so this is consistent with what prior gates accepted — not introduced as a new deviation class.

**Suggested Change:** Replace with a period and split, or restructure:
```
# directly. generate() tokenizes the raw prompt, prefills, then decodes.
...
# tokenizes to different ids than the vendored tokenizer. The contract
```

### [TEST] TEST-002: eos-break path (GOAL-003) never exercised end-to-end

**Location:** `workspace/transformers/tests/q_bf16/test_qwen3_5_07_greedy_decoding.nim:29–32` (comment admits it); `workspace/transformers/src/models.nim:133` (the untested `break`)
**Severity:** MINOR
**Confidence:** 1.0

**Evidence:** All 3 fixtures' `generated_ids`/`full_ids` contain no 248044 (verified programmatically for all 3 files). Test 07 comment: "None of the three fixtures reaches eos within num_generated_tokens (8), so the eos stop path is not exercised by them." Test 06 only asserts the config constant (`eosTokenId == 248044`), not the break behavior. `generate()`'s decode loop does `if ids[^1] == cfg.eosTokenId: break` (models.nim:133) — verified present by code reading, but no test drives a generation that emits 248044.

**Issue:** The eos stop condition is a core generation contract (it is what keeps output from running to `maxTokens`/`maxCtx`) and it has zero test coverage, matching the goal checker's GOAL-003. The deliverables are honest about the gap, which is why this is MINOR rather than a blocker, but the gap is real.

**Suggested Change:** Add a 4th fixture where the model emits the config eos 248044 within `num_generated_tokens` (generate with the vendored reference at temp 0 and record the ids), then extend test 07:

```nim
proc checkEosBreak(model: Model, jsonPath: string): bool =
  ## Fixture whose generated_ids END with 248044 (config eos). generate()
  ## must stop there instead of filling maxTokens.
  let data = parseJson(readFile(jsonPath))
  var prompt = data["prompt"].getStr()
  let expectedIds = data["generated_ids"].getElems().mapIt(it.getInt())
  let numPrompt = data["num_prompt_tokens"].getInt()
  # eos decodes to empty bytes, so the re-encoded output drops it.
  # maxTokens is oversized on purpose: only the eos break may stop early.
  let output = model.generate(prompt, temp = 0.0f,
                              maxTokens = expectedIds.len + 8,
                              maxContextLen = 512)
  let actualIds = model.getTokenizer().encode(output)
  doAssert actualIds.len == numPrompt + expectedIds.len - 1,
    "generate must stop at eos 248044 (output re-encodes to " &
    $numPrompt & " prompt + " & $(expectedIds.len - 1) & " generated tokens)"
  let actualGen = actualIds[numPrompt ..< actualIds.len]
  doAssert actualGen == expectedIds[0 ..< ^1],
    "tokens before eos must match the fixture"
```

### [TEST] TEST-003: Combining-mark round-trip only partially proves the \p{M} regex path

**Location:** `workspace/transformers/tests/q_bf16/test_qwen3_5_06_t2t_inference.nim:57–66`
**Severity:** MINOR
**Confidence:** 0.8

**Evidence:** The decomposed prompt `"The re\u0301sume\u0301 is ready"` is tokenized and `decodeToString(markTokens) == resumePrompt` asserted. `decodeToString` is byte-concatenation of per-token byte strings (toktoktok bpe_codec.nim:323). The pre-tokenizer regex has a punctuation fallback branch ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`, so if a regression removed `\p{M}` from both branch 2 (`[\p{L}\p{M}]+`) and the branch-4 exclusion, U+0301 would be tokenized as its own punctuation token and the byte round-trip would still pass.

**Issue:** The mission's "proving the \p{M} regex path" bar is only partially met: the decomposed form IS used (requirement satisfied), but the assert cannot distinguish "\p{M} group tokenization" from "combining mark tokenized separately with byte-identical decode". The comment is honest about the NFC-normalizer limitation, but overstates the contract slightly.

**Suggested Change:** Strengthen the assert so a separately-tokenized combining mark fails — no single token id may decode to U+0301 alone:

```nim
let markTokens = model.getTokenizer().encode(resumePrompt)
doAssert model.getTokenizer().decodeToString(markTokens) == resumePrompt,
  "combining-mark prompt must round-trip through tokenize/untokenize"
for id in markTokens:
  let tok = model.getTokenizer().decodeToString([id])
  doAssert not tok.contains("\u0301") or tok.len > 2,
    "U+0301 must be grouped with its base letter (\\p{M} path), got lone token " & tok
```

---

## Coverage gaps (decode paths NOT exercised)

1. **eos-break end-to-end (GOAL-003)** — see TEST-002. Only the `while` maxTokens bound is exercised (all 3 fixtures stop at exactly 8 generated tokens).
2. **maxContextLen guard** — `generate()` raising `ValueError` when `encode(prompt).len > maxContextLen` (models.nim:94–97) is untested.
3. **Long multi-turn** — all fixtures are single-turn, 4–7 prompt tokens, ≤ 16 generated tokens; no multi-turn or long-context decode.
4. **Batch > 1** — `generate()` is hardcoded to batch 1 (`Orchestrator` init with `1` and `concurrentRequests = 1`, models.nim:76–79); batch > 1 decode is only covered at lower levels (attn/layers tests), never through generate.

## Positive changes

- Test 07's token-exact comparison against 3 vendored fixtures is the strongest possible end-to-end lock for this model: wrong weights, wrong logits, broken argmax, tokenizer divergence, or decode-entry drift all fail it. It passed 3/3, which also cross-validates mission 05's ids-to-logits fixtures (same final-logits path).
- The honest, well-documented comments (decode-entry convention, eos non-reachability, NFC-normalizer limitation, precomposed-form substitution for the resume fixture) match the series' documentation standards.
- No cheating: purely additive diff; no weakened/deleted/skipped tests; no tautological asserts; temp-0 argmax lock is real.
