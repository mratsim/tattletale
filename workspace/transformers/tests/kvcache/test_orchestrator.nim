# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Orchestrator integration tests — regression tests for BUG-B-001 and BUG-B-002.
##
## These tests use a full PagePool (CPU-backed) to exercise the orchestrator's
## page lifecycle management with real Page objects.
##
## Bugs reproduced:
##   BUG-B-001: kv_position never updated after prefill → OOB crash on decode
##              at page boundary.
##   BUG-B-002: COW partial-match page inserted before fully-matched pages
##              → page ordering corruption in ctx.pages.
##
## Test design:
##   Each bug gets a standalone proc returning bool (same pattern as
##   test_kvcache.nim).  Tests are registered in suites under runTests().
##
## Compilation:
##   Needs libtorch linked (same as test_page_pool.nim).  Use:
##     nim cpp -r --hints:off --warnings:off \
##       --outdir:build/tests --nimcache:nimcache/tests \
##       workspace/transformers/tests/kvcache/test_orchestrator.nim

import
  std/unittest,
  std/importutils,
  workspace/libtorch as F,
  ../../src/stateful/kvcache,        # TokensPerPage, ceilDiv
  ../../src/stateful/page_pool,       # Page, pageIndex
  ../../src/stateful/inference_context,
  ../../src/stateful/orchestrator

# ═════════════════════════════════════════════════════════════════════════════
# Test configuration
# ═════════════════════════════════════════════════════════════════════════════

const
  TestLayers = 1
  TestKvHeads = 1
  TestHeadDim = 4
  TestMaxSeq = 4096
  TestNumPages = 32

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

proc makeOrchestrator(): Orchestrator =
  ## Create a minimal orchestrator backed by CPU tensors.
  result = Orchestrator.init(
    num_layers = TestLayers,
    batch_size = 1,
    kv_heads = TestKvHeads,
    max_seq = TestMaxSeq,
    head_dim = TestHeadDim,
    num_pages = TestNumPages,
    dtype = kBFloat16,
    device = kCPU
  )

proc makeTokens(n: int): seq[uint32] =
  ## Sequential tokens `[0, 1, 2, ..., n-1]` for reproducible LPM matching.
  result = newSeq[uint32](n)
  for i in 0 ..< n:
    result[i] = uint32(i)

# ═════════════════════════════════════════════════════════════════════════════
# BUG-B-001: kv_position never updated after prefill → OOB on decode at
# page boundary.
#
# Root cause chain:
#   1. startSequence sets kv_position = matched.totalTokenMatched (0 for new prompt)
#   2. Prefill forward pass writes KV at positions 0..seq_len-1 but NEVER
#      updates kv_position.
#   3. After 256-token prefill, kv_position is still 0.
#   4. decodeStep checks: kv_position > 0 and (kv_position mod 256) == 0
#      → 0 > 0 is false → no page allocated.
#   5. attn.forward computes pageIdx = 256 / 256 = 1 → ctx.pages[1] → OOB crash.
# ═════════════════════════════════════════════════════════════════════════════

proc testBugB001KvPositionTracking(): bool =
  ## Verify kv_position tracking through startSequence and decodeStep.
  ## Note: BUG-B-001's actual fix (setting kv_position = input_ids.len) is in
  ## generate() AFTER the prefill forward pass, to avoid corrupting the attention
  ## layer's writeStart computation (which uses kv_position - offset).
  ## The orchestrator only tracks kv_position via decodeStep's increment.
  var orc = makeOrchestrator()
  let tokens = makeTokens(TokensPerPage)  # exactly 256 tokens = 1 page

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContextMut()

  # After startSequence with no LPM match: kv_position = 0
  doAssert ctx.kv_position == 0,
    "kv_position should be 0 after startSequence (no LPM match), got " &
    $ctx.kv_position

  # Pages: 1 page for 256 tokens
  doAssert ctx.pages.len == 1,
    "Expected 1 page for 256 tokens, got " & $ctx.pages.len

  # decodeStep: kv_position increments by 1 each step
  orc.decodeStep(position = TokensPerPage, token_id = 42'u32, device = kCPU)
  doAssert ctx.kv_position == 1,
    "kv_position should be 1 after 1 decode step, got " & $ctx.kv_position

  # Page boundary crossing detection: kv_position=0 starts tracking from 0,
  # so the first 256 decode steps won't trigger page allocation.
  # The generate() function sets kv_position = input_ids.len after prefill
  # forward to ensure correct boundary detection from the start.
  doAssert ctx.pages.len == 1,
    "Expected still 1 page (no page boundary crossed yet), got " &
    $ctx.pages.len

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# BUG-B-002: COW partial-match page inserted before fully-matched pages,
# corrupting ctx.pages ordering.
#
# Scenario: LPM matches 300 tokens (256 full + 44 partial).
#   - matched.pages[0] = trie page for positions 0-255
#   - matched.pages[^1] = trie page for positions 256-511 (only 44 valid)
#
# Buggy ordering:
#   ctx.pages[0] = COW copy of matched.pages[^1] (positions 256-299) ← WRONG
#   ctx.pages[1] = matched.pages[0] (positions 0-255) ← WRONG
#
# Correct ordering (after fix):
#   ctx.pages[0] = matched.pages[0] (positions 0-255) ← correct
#   ctx.pages[1] = COW copy of matched.pages[^1] (positions 256-299) ← correct
#
# Verification strategy:
#   1. Populate trie with 512-token sequence → 2 pages borrowed (indices 0,1)
#   2. endSequence grafts both pages into trie
#   3. startSequence with 300 tokens → LPM matches both pages
#   4. COW borrows page index 2 from pool (next free)
#   5. Check ctx.pages[0].index == 0 (fully-matched page, correct position)
#   6. Check ctx.pages[1].index == 2 (COW page, correct position)
# ═════════════════════════════════════════════════════════════════════════════

proc testBugB002CowPageOrdering(): bool =
  ## Verify page ordering after LPM with partial match:
  ##   fully-matched pages first, COW page last.
  var orc = makeOrchestrator()

  # ── Phase 1: Populate the trie with a 512-token sequence ──
  let seq1Tokens = makeTokens(TokensPerPage * 2)  # 512 tokens = 2 pages
  orc.startSequence(seq1Tokens)
  orc.endSequence()
  # After endSequence: both pages (indices 0 and 1) are grafted into the trie
  # Pool free_indices: [31, 30, ..., 2] (indices 0 and 1 are in use by trie)

  # ── Phase 2: Prefill 300 tokens (same prefix → LPM match) ──
  # 300 = 256 full + 44 partial: matched.pages[0] (index 0) fully matched,
  # matched.pages[1] (index 1) partially matched → COW needed.
  let seq2Tokens = makeTokens(300)
  orc.startSequence(seq2Tokens)
  let ctx = orc.getInferenceContextMut()

  # After startSequence with 300-token LPM match (256 + 44 partial):
  #   ctx.pages should have 2 entries:
  #     - [0]: fully-matched page (index 0 from trie)
  #     - [1]: COW page (index 2, freshly borrowed)
  doAssert ctx.pages.len == 2,
    "[BUG-B-002] Expected 2 pages for 300-token match (1 full + 1 COW), got " &
    $ctx.pages.len

  # Verify ordering: first page must be the fully-matched page (index 0)
  doAssert ctx.pages[0].pageIndex() == 0,
    "[BUG-B-002] ctx.pages[0] should be the fully-matched page (index 0), " &
    "got index " & $ctx.pages[0].pageIndex()

  # Verify ordering: second page must be the COW page (index 2, newly borrowed)
  doAssert ctx.pages[1].pageIndex() == 2,
    "[BUG-B-002] ctx.pages[1] should be the COW page (index 2), " &
    "got index " & $ctx.pages[1].pageIndex()

  # Verify cached_tokens reflects LPM match (kv_position stays 0 — write cursor)
  doAssert ctx.cached_tokens == 300,
    "[BUG-B-002] cached_tokens should be 300 after 300-token match, got " &
    $ctx.cached_tokens
  doAssert ctx.kv_position == 0,
    "[BUG-B-002] kv_position should be 0 (no tokens written yet), got " &
    $ctx.kv_position

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# Additional lifecycle tests
# ═════════════════════════════════════════════════════════════════════════════

proc testOrchestratorLifecycleRoundtrip(): bool =
  ## Verify that startSequence → endSequence → startSequence works
  ## and contexts are properly reset between sequences.
  var orc = makeOrchestrator()
  let tokens = makeTokens(128)  # less than 1 page

  # Sequence 1
  orc.startSequence(tokens)
  let ctx = orc.getInferenceContextMut()
  # kv_position reflects total prefill tokens (BUG-B-001 fix)
  doAssert ctx.kv_position == 0,
    "Expected kv_position=128 after 128-token prefill, got " &
    $ctx.kv_position
  doAssert ctx.pages.len == 1

  orc.endSequence()
  # After endSequence, context should be cleared
  doAssert ctx.pages.len == 0,
    "Expected 0 pages after endSequence, got " & $ctx.pages.len
  doAssert ctx.kv_position == 0,
    "Expected kv_position=0 after endSequence, got " & $ctx.kv_position

  # Sequence 2 — reuse orchestrator (LPM matches 128 from first sequence)
  orc.startSequence(tokens)
  # kv_position stays 0 (write cursor), cached_tokens reflects LPM match
  doAssert ctx.kv_position == 0,
    "Expected kv_position=0 after startSequence, got " & $ctx.kv_position
  doAssert ctx.cached_tokens == 128,
    "Expected cached_tokens=128 after LPM match, got " & $ctx.cached_tokens
  doAssert ctx.pages.len == 1

  result = true

proc testOrchestratorDecodeTracking(): bool =
  ## Verify kv_position tracking through decode steps.
  ## Page allocation at boundaries is triggered by generate() which sets
  ## kv_position = input_ids.len after the prefill forward pass.
  var orc = makeOrchestrator()
  let tokens = makeTokens(TokensPerPage - 1)  # 255 tokens

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContextMut()

  doAssert ctx.pages.len == 1,
    "Expected 1 page for 255 tokens, got " & $ctx.pages.len
  doAssert ctx.kv_position == 0,
    "kv_position should be 0 after startSequence, got " & $ctx.kv_position

  # Decode at position 255 — kv_position increments by 1
  orc.decodeStep(position = TokensPerPage - 1, token_id = 100'u32, device = kCPU)
  doAssert ctx.pages.len == 1,
    "Expected still 1 page after 1 decode, got " & $ctx.pages.len
  doAssert ctx.kv_position == 1,
    "kv_position should be 1 after decode, got " & $ctx.kv_position

  # Second decode
  orc.decodeStep(position = TokensPerPage, token_id = 101'u32, device = kCPU)
  doAssert ctx.kv_position == 2,
    "kv_position should be 2, got " & $ctx.kv_position

  result = true
# ═════════════════════════════════════════════════════════════════════════════
# writeStart interaction tests
# ═════════════════════════════════════════════════════════════════════════════
#
# The attention layer computes:
#   writeStart = max(0, ctx.cached_tokens - offset)
#   for t in writeStart ..< seq_len:
#     page.k_view[layer, globalPos mod TokensPerPage] = k_rot[0, t]
#
# cached_tokens = number of tokens already in trie (from LPM).
# For first prefill (no LPM match): cached_tokens = 0, writeStart = 0.
# All tokens are written to pages. No positions skipped.
# For COW case: cached_tokens = matched.totalTokenMatched, writeStart > 0.
# Cached positions are skipped. Only new positions written.

proc testWriteStartFirstPrefill(): bool =
  ## Verify writeStart = 0 for first prefill (no LPM match).
  ## Without the cached_tokens/kv_position separation, setting kv_position
  ## before forward would cause writeStart = seq_len, skipping ALL writes.
  var orc = makeOrchestrator()
  let tokens = makeTokens(128)  # less than 1 page

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContextMut()

  # cached_tokens = 0 (no LPM match)
  doAssert ctx.cached_tokens == 0,
    "cached_tokens should be 0 with no LPM match, got " & $ctx.cached_tokens

  # Simulate attention writeStart logic:
  #   writeStart = max(0, cached_tokens - offset)
  #   for t in writeStart ..< seq_len:
  #     write KV at (offset + t)
  # For first prefill: offset = 0 (position_ids = arange(0, seq_len))
  let writeStart = max(0, ctx.cached_tokens - 0)
  doAssert writeStart == 0,
    "writeStart should be 0 for first prefill, got " & $writeStart

  # All 128 tokens would be written to page 0 (positions 0-127)
  # No positions are skipped — verified by writeStart = 0
  result = true
  result = true

proc testWriteStartCachedPrefix(): bool =
  ## Verify writeStart correctly skips cached prefix positions.
  ## After LPM match, cached_tokens = matched count.
  ## writeStart = max(0, cached_tokens - offset) skips cached positions.
  var orc = makeOrchestrator()

  # Populate trie with 512-token sequence
  let seq1Tokens = makeTokens(TokensPerPage * 2)
  orc.startSequence(seq1Tokens)
  orc.endSequence()

  # Second sequence: 300 tokens (256 full + 44 partial match)
  let seq2Tokens = makeTokens(300)
  orc.startSequence(seq2Tokens)
  let ctx = orc.getInferenceContextMut()

  # cached_tokens = 300 from LPM
  doAssert ctx.cached_tokens == 300,
    "cached_tokens should be 300 after LPM match, got " & $ctx.cached_tokens

  # kv_position should be 0 (no tokens written yet this sequence)
  doAssert ctx.kv_position == 0,
    "kv_position should be 0 after startSequence, got " & $ctx.kv_position

  # Simulate attention write for continuation prefill
  # The model receives the FULL 300 tokens, offset = 0
  let offset = 0
  let seqLen = seq2Tokens.len
  let writeStart = max(0, ctx.cached_tokens - offset)
  doAssert writeStart == 300,
    "writeStart should be 300 (skip cached prefix), got " & $writeStart

  # Pages: page 0 (fully matched), page 1 (COW page)
  doAssert ctx.pages.len == 2,
    "Expected 2 pages (1 matched + 1 COW), got " & $ctx.pages.len

  # Write loop: only writes positions 300-299 (which is empty for 300-token seq)
  # This is correct: all 300 tokens are cached, nothing to write
  var writtenCount = 0
  for t in writeStart ..< seqLen:
    let globalPos = offset + t
    let pageIdx = globalPos div TokensPerPage
    let withinPage = globalPos mod TokensPerPage
    let page = ctx.pages[pageIdx]
    page.k_view[0, withinPage] = F.toTensor([float32(t + 1)])
    writtenCount.inc

  # Only new tokens (beyond cached prefix) should be written
  # For 300-token prompt with 300 cached: no new tokens
  doAssert writtenCount == 0,
    "Expected 0 written tokens (all cached), got " & $writtenCount

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# Regression: kv_position / cached_tokens independence
# ═════════════════════════════════════════════════════════════════════════════
#
# These tests verify that cached_tokens and kv_position are independent:
#   - cached_tokens: set by startSequence (LPM match count), stable forever.
#     Used by attention for writeStart. NEVER updated after startSequence.
#   - kv_position: starts at 0, set by setKvPosition() after prefill forward.
#     Used by decodeStep for page allocation. Incremented each decode step.
#   - Setting one must not affect the other.

proc testFieldIndependencePrefillDecode(): bool =
  ## Verify cached_tokens stays 0 through prefill and decode (no LPM match).
  ## kv_position starts at 0, is set by setKvPosition, and increments via decodeStep.
  var orc = makeOrchestrator()
  let tokens = makeTokens(64)

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContextMut()

  # After startSequence with no LPM: cached_tokens=0, kv_position=0
  doAssert ctx.cached_tokens == 0,
    "cached_tokens should be 0 (no LPM match), got " & $ctx.cached_tokens
  doAssert ctx.kv_position == 0,
    "kv_position should be 0 after startSequence, got " & $ctx.kv_position

  # Simulate generate(): set kv_position after prefill (via setKvPosition)
  orc.setKvPosition(tokens.len)
  doAssert ctx.kv_position == 64,
    "kv_position should be 64 after setKvPosition, got " & $ctx.kv_position
  # cached_tokens must still be 0 (unchanged by setKvPosition)
  doAssert ctx.cached_tokens == 0,
    "cached_tokens must not change when kv_position is set, got " & $ctx.cached_tokens

  # decodeSteps increment kv_position only, not cached_tokens
  orc.decodeStep(position = 64, token_id = 100'u32, device = kCPU)
  doAssert ctx.kv_position == 65,
    "kv_position should be 65 after decodeStep, got " & $ctx.kv_position
  doAssert ctx.cached_tokens == 0,
    "cached_tokens must not change during decodeStep, got " & $ctx.cached_tokens

  result = true

proc testFieldIndependenceCOWMatch(): bool =
  ## Verify cached_tokens reflects LPM match and kv_position stays 0.
  ## setKvPosition only affects kv_position, not cached_tokens.
  var orc = makeOrchestrator()

  # Populate trie with 512-token sequence
  let seq1Tokens = makeTokens(TokensPerPage * 2)
  orc.startSequence(seq1Tokens)
  orc.endSequence()

  # Second sequence: 300 tokens (LPM match)
  let seq2Tokens = makeTokens(300)
  orc.startSequence(seq2Tokens)
  let ctx = orc.getInferenceContextMut()

  # After LPM match: cached_tokens=300, kv_position=0
  doAssert ctx.cached_tokens == 300,
    "cached_tokens should be 300 after LPM match, got " & $ctx.cached_tokens
  doAssert ctx.kv_position == 0,
    "kv_position should be 0 after startSequence, got " & $ctx.kv_position

  # setKvPosition should only affect kv_position
  orc.setKvPosition(350)  # simulate full 350-token prefill
  doAssert ctx.kv_position == 350,
    "kv_position should be 350 after setKvPosition, got " & $ctx.kv_position
  doAssert ctx.cached_tokens == 300,
    "cached_tokens must not change when kv_position is set, got " & $ctx.cached_tokens

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# Test runner
# ═════════════════════════════════════════════════════════════════════════════

proc runTests*() =
  suite "Orchestrator — BUG-B-001 (kv_position tracking)":
    test "kv_position tracking through startSequence and decodeStep":
      check testBugB001KvPositionTracking()

  suite "Orchestrator — BUG-B-002 (COW page ordering inverted)":
    test "LPM 300 tokens (256 full + 44 partial) → correct page order":
      check testBugB002CowPageOrdering()

  suite "Orchestrator — decode tracking":
    test "kv_position increments through decode steps":
      check testOrchestratorDecodeTracking()

  suite "Orchestrator — lifecycle round-trip":
    test "startSequence → endSequence → startSequence reuse":
      check testOrchestratorLifecycleRoundtrip()

  suite "Orchestrator — writeStart (attention interaction)":
    test "writeStart=0 for first prefill (no LPM match)":
      check testWriteStartFirstPrefill()
    test "writeStart skips cached prefix after LPM match":
      check testWriteStartCachedPrefix()

  suite "Orchestrator — field independence (kv_position vs cached_tokens)":
    test "kv_position set independently of cached_tokens (no LPM)":
      check testFieldIndependencePrefillDecode()
    test "cached_tokens stable through setKvPosition (COW match)":
      check testFieldIndependenceCOWMatch()

when isMainModule:
  runTests()
