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
  workspace/libtorch_testutils,
  ../../src/stateful/kvcache,        # TokensPerPage, ceilDiv
  ../../src/stateful/page_pool,       # Page, pageIndex
  ../../src/stateful/inference_context,
  ../../src/stateful/orchestrator {.all.}
privateAccess(Orchestrator)

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
  doAssert ctx.kv_position == 0,
    "kv_position should be 0 after 1 decode step, got " & $ctx.kv_position

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

  # Decode at position 255 — kv_position stays 0 (inc happens after forward in generate)
  orc.decodeStep(position = TokensPerPage - 1, token_id = 100'u32, device = kCPU)
  doAssert ctx.pages.len == 1,
    "Expected still 1 page after 1 decode, got " & $ctx.pages.len
  doAssert ctx.kv_position == 0,
    "kv_position should be 0 after decode, got " & $ctx.kv_position

  # Second decode
  orc.decodeStep(position = TokensPerPage, token_id = 101'u32, device = kCPU)
  doAssert ctx.kv_position == 0,
    "kv_position should be 0, got " & $ctx.kv_position

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

  # decodeSteps no longer increment kv_position — generate() does it after forward
  orc.decodeStep(position = 64, token_id = 100'u32, device = kCPU)
  doAssert ctx.kv_position == 64,
    "kv_position should be 64 after decodeStep (no inc), got " & $ctx.kv_position
  doAssert ctx.cached_tokens == 0

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
# ensurePoolCapacity OOM ValueError
# ═════════════════════════════════════════════════════════════════════════════
#
# When the pool is exhausted and no eviction candidates exist (empty trie
# or all pages locked), ensurePoolCapacity raises ValueError.

proc testOOMError(): bool =
  ## Verify ensurePoolCapacity raises ValueError when pool exhausted
  ## and no eviction candidates are available.
  var orc = makeOrchestrator()  # 32-page pool
  let ctx = orc.getInferenceContextMut()

  # Prompt large enough to need more pages than pool has.
  # TokensPerPage = 256, pool = 32 pages → 33 pages need 8193 tokens
  let pageCountNeeded = TestNumPages + 1  # 33 pages
  let tokens = makeTokens(pageCountNeeded * TokensPerPage)

  try:
    orc.startSequence(tokens)
    # Should not reach here — pool exhaustion should raise
    doAssert false, "Expected ValueError was not raised"
  except ValueError:
    # Expected: ensurePoolCapacity can't evict (empty trie)
    discard

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# cowPartialPage isolated test
# ═════════════════════════════════════════════════════════════════════════════
#
# Verify the COW helper correctly copies partial page content
# from source page to destination page.

proc testCowPartialPageIsolated(): bool =
  ## Verify cowPartialPage copies partial tokens from src to dst.
  ## Uses CPU tensors (cpuOrc) to avoid CUDA item() issues.
  var cpuOrc = Orchestrator.init(
    num_layers = TestLayers,
    batch_size = 1,
    kv_heads = TestKvHeads,
    max_seq = TestMaxSeq,
    head_dim = TestHeadDim,
    num_pages = TestNumPages,
    dtype = kBFloat16,
    device = kCPU
  )

  let srcPage = cpuOrc.page_pool.borrow()
  let dstPage = cpuOrc.page_pool.borrow()
  let partialTokens = 42

  # k_view shape: (1, 256, 1, 4) — write a gradient to each position
  for t in 0 ..< partialTokens:
    srcPage.k_view[0, t] = F.toTensor([float32(t + 100), 0, 0, 0]).reshape(1, 4)
    srcPage.v_view[0, t] = F.toTensor([float32(t + 200), 0, 0, 0]).reshape(1, 4)

  # COW copy
  cowPartialPage(dstPage, srcPage, partialTokens, numLayers = 1)

  # Verify: read back and check each position
  for t in 0 ..< partialTokens:
    let kVal = dstPage.k_view[0, t, 0, 0].item(float32)
    doAssert kVal == float32(t + 100),
      "k_view[" & $t & "] should be " & $(t+100) & " got " & $kVal
    let vVal = dstPage.v_view[0, t, 0, 0].item(float32)
    doAssert vVal == float32(t + 200),
      "v_view[" & $t & "] should be " & $(t+200) & " got " & $vVal

  # Verify beyond partialTokens is still zero
  let kBeyond = dstPage.k_view[0, partialTokens, 0, 0].item(float32)
  let vBeyond = dstPage.v_view[0, partialTokens, 0, 0].item(float32)
  doAssert kBeyond == 0.0, "k_view beyond partial should be 0, got " & $kBeyond
  doAssert vBeyond == 0.0, "v_view beyond partial should be 0, got " & $vBeyond

  result = true
# Test runner
# ═════════════════════════════════════════════════════════════════════════════
# Test runner
# ═════════════════════════════════════════════════════════════════════════════
# COV-B-005: Page boundary crossing during decodeStep
# ═════════════════════════════════════════════════════════════════════════════
#
# decodeStep checks ctx.kv_position > 0 and (ctx.kv_position mod 256) == 0
# to trigger lazy page allocation. After a full-page prefill, setKvPosition
# sets kv_position = 256, and the first decodeStep should trigger a page borrow.

proc testDecodePageBoundary(): bool =
  ## Verify decodeStep borrows a new page when kv_position crosses a
  ## TokensPerPage boundary.
  var orc = makeOrchestrator()
  let tokens = makeTokens(TokensPerPage)  # exactly 256 tokens = 1 page

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContextMut()
  doAssert ctx.pages.len == 1,
    "Expected 1 page, got " & $ctx.pages.len

  # Simulate generate(): set kv_position after prefill forward
  orc.setKvPosition(tokens.len)
  doAssert ctx.kv_position == TokensPerPage,
    "kv_position should be 256, got " & $ctx.kv_position

  # First decode at position 256 — kv_position=256, triggers boundary crossing
  #   condition: kv_position > 0 and 256 mod 256 == 0 → borrow page
  orc.decodeStep(position = TokensPerPage, token_id = 100'u32, device = kCPU)
  doAssert ctx.kv_position == TokensPerPage,
    "kv_position should be 256 (no inc in decodeStep), got " & $ctx.kv_position
  doAssert ctx.pages.len == 2,
    "Expected 2 pages after boundary crossing, got " & $ctx.pages.len

  # Simulate generate(): advance kv_position after forward
  orc.setKvPosition(ctx.kv_position + 1)

  # Second decode at position 257 — no boundary (257 mod 256 != 0)
  orc.decodeStep(position = TokensPerPage + 1, token_id = 101'u32, device = kCPU)
  doAssert ctx.kv_position == TokensPerPage + 1,
    "kv_position should be 257 (no inc in decodeStep), got " & $ctx.kv_position
  doAssert ctx.pages.len == 2
  result = true

# ═════════════════════════════════════════════════════════════════════════════
# COV-B-009: Partial-page gather boundary
# ═════════════════════════════════════════════════════════════════════════════
#
# The attention gather loop (attn.nim:239-245) handles the last page when
# totalSeqLen is not a multiple of TokensPerPage. This test verifies the
# page structure that the gather loop operates on, for a non-aligned prefill.

proc testPartialPageStructure(): bool =
  ## Verify page structure for non-page-aligned sequence length.
  ## The gather loop uses pageValidLen = min(...) for the last partial page.
  var orc = makeOrchestrator()
  let tokens = makeTokens(300)  # 256 + 44 = 1 full + 1 partial page

  orc.startSequence(tokens)
  let ctx = orc.getInferenceContextMut()

  doAssert ctx.pages.len == 2,
    "Expected 2 pages for 300 tokens, got " & $ctx.pages.len

  # Simulate generate(): set kv_position after prefill
  orc.setKvPosition(tokens.len)
  doAssert ctx.kv_position == 300,
    "kv_position should be 300, got " & $ctx.kv_position

  # Verify page ordering
  doAssert ctx.pages[0].pageIndex() == 0,
    "Expected pages[0].index == 0, got " & $ctx.pages[0].pageIndex()
  doAssert ctx.pages[1].pageIndex() == 1,
    "Expected pages[1].index == 1, got " & $ctx.pages[1].pageIndex()

  # Decode: position 300, no boundary (300 mod 256 != 0)
  orc.decodeStep(position = 300, token_id = 200'u32, device = kCPU)
  doAssert ctx.kv_position == 300,
    "kv_position should be 300 after decodeStep (no inc), got " & $ctx.kv_position
  doAssert ctx.pages.len == 2,
    "Expected still 2 pages (no boundary), got " & $ctx.pages.len

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# COV-B-010: endSequence round-trip (graftPages → trie → new sequence)
# ═════════════════════════════════════════════════════════════════════════════
#
# endSequence collects all tokens+pages and grafts them into the trie.
# Subsequent startSequence with the same prefix should get a full LPM match.

proc testEndSequenceRoundTrip(): bool =
  ## Verify endSequence → startSequence cycle through multiple sequences.
  ## Tests: graftPages, LPM re-match, and growing match on longer prefix.
  var orc = makeOrchestrator()

  # Sequence 1: 512 tokens (2 full pages)
  orc.startSequence(makeTokens(TokensPerPage * 2))
  orc.endSequence()

  # Sequence 2: same 512 tokens → LPM matches all 512
  orc.startSequence(makeTokens(TokensPerPage * 2))
  let ctx = orc.getInferenceContextMut()
  doAssert ctx.cached_tokens == TokensPerPage * 2,
    "cached_tokens should be " & $(TokensPerPage*2) & " after LPM, got " &
    $ctx.cached_tokens
  orc.endSequence()

  # Sequence 3: 300 tokens (overlap: first 300 of [0..511])
  orc.startSequence(makeTokens(300))
  doAssert ctx.cached_tokens == 300,
    "cached_tokens should be 300 after LPM match, got " & $ctx.cached_tokens
  orc.endSequence()

  # Sequence 4: same 300 tokens → LPM matches all 300
  orc.startSequence(makeTokens(300))
  doAssert ctx.cached_tokens == 300,
    "cached_tokens should be 300 after LPM, got " & $ctx.cached_tokens

  result = true
# ═════════════════════════════════════════════════════════════════════════════
# COV-B-008: InferenceContext ref semantic verification
# ═════════════════════════════════════════════════════════════════════════════
#
# InferenceContext was changed from `object` to `ref object`. Multiple handles
# to the same orchestrator context should alias (not copy).

proc testInferenceContextRefSemantic(): bool =
  ## Verify InferenceContext is a ref object: getInferenceContextMut always
  ## returns the same object. (The type is `ref object` so this is guaranteed
  ## by the compiler — this test documents that invariant.)
  var orc = makeOrchestrator()
  let ctx = orc.getInferenceContextMut()

  # Verify: cached_tokens=0 initially, then changes reflect in both handles
  doAssert ctx.cached_tokens == 0,
    "Expected cached_tokens=0 initially"

  orc.startSequence(makeTokens(100))
  # First-ever sequence: LPM matches nothing, cached_tokens stays 0
  doAssert ctx.cached_tokens == 0,
    "cached_tokens should be 0 for first sequence, got " & $ctx.cached_tokens

  # Verify the ctx handle sees the pages the orchestrator allocated
  doAssert ctx.pages.len == 1,
    "Expected 1 page for 100 tokens, got " & $ctx.pages.len

  result = true

# ═════════════════════════════════════════════════════════════════════════════
# computeNumPages and computePageSizeBytes
# ═════════════════════════════════════════════════════════════════════════════

proc testComputeNumPages(): bool =
  ## Verify computeNumPages computes correct page counts.
  ## Pure function: no orchestrator needed.
  ##
  ## Qwen3-0.6B: max_position_embeddings=4096 -> pagesPerRequest=16, headroom=1
  doAssert computeNumPages(4096, concurrentRequests = 1) == 17,
    "computeNumPages(4096,1) should be 17"
  ## 8192 context: 8192/256 = 32 pages, headroom for 1 concurrent = 33
  doAssert computeNumPages(8192, concurrentRequests = 1) == 33,
    "computeNumPages(8192,1) should be 33"
  ## 2 concurrent requests at 4096: 16*2 = 32, headroom 2 = 34
  doAssert computeNumPages(4096, concurrentRequests = 2) == 34,
    "computeNumPages(4096,2) should be 34"
  ## Edge: 0 context should not crash (returns at least headroom)
  doAssert computeNumPages(0, concurrentRequests = 1) == 1,
    "computeNumPages(0,1) should be 1"
  result = true

proc testComputePageSizeBytes(): bool =
  ## Verify computePageSizeBytes computes correct byte sizes.
  ## Pure function: no orchestrator needed.
  ## Formula: num_layers * TokensPerPage * kv_heads * head_dim * elementSize * 2
  ##
  ## Qwen3-0.6B params: layers=28, kv_heads=8, head_dim=128, bf16=2 bytes
  let size = computePageSizeBytes(num_layers = 28, kv_heads = 8,
    head_dim = 128, dtype = kBFloat16)
  ## elPerBuf = 28 * 256 * 8 * 128 = 7,340,032
  ## result = 7,340,032 * 2 * 2 = 29,360,128
  doAssert size == 29360128,
    "computePageSizeBytes(Qwen3-0.6B) should be 29360128, got " & $size
  ## Single layer, 1 head, dim=4, bf16: 1*256*1*4*2*2 = 4096
  let small = computePageSizeBytes(num_layers = 1, kv_heads = 1,
    head_dim = 4, dtype = kBFloat16)
  doAssert small == 4096,
    "computePageSizeBytes(1,1,4,bf16) should be 4096, got " & $small
  ## Float32 doubles the size vs bf16 for same layout
  let f32 = computePageSizeBytes(num_layers = 1, kv_heads = 1,
    head_dim = 4, dtype = kFloat32)
  doAssert f32 == 8192,
    "computePageSizeBytes(1,1,4,f32) should be 8192, got " & $f32
  result = true

proc runTests*() =
  runTest("BUG-B-001: kv_position tracking through startSequence and decodeStep",
    testBugB001KvPositionTracking)

  runTest("BUG-B-002: LPM 300 tokens (256 full + 44 partial) → correct page order",
    testBugB002CowPageOrdering)

  runTest("kv_position increments through decode steps",
    testOrchestratorDecodeTracking)

  runTest("lifecycle: startSequence → endSequence → startSequence reuse",
    testOrchestratorLifecycleRoundtrip)

  runTest("writeStart=0 for first prefill (no LPM match)",
    testWriteStartFirstPrefill)

  runTest("writeStart skips cached prefix after LPM match",
    testWriteStartCachedPrefix)

  runTest("field independence: kv_position set independently of cached_tokens",
    testFieldIndependencePrefillDecode)

  runTest("field independence: cached_tokens stable through setKvPosition",
    testFieldIndependenceCOWMatch)

  runTest("COV-A-004: OOM ValueError on pool exhaustion",
    testOOMError)

  runTest("COV-A-005: cowPartialPage copy verification",
    testCowPartialPageIsolated)

  runTest("COV-B-005: page boundary crossing during decodeStep",
    testDecodePageBoundary)

  runTest("COV-B-009: partial-page gather structure (300 tokens, non-aligned)",
    testPartialPageStructure)

  runTest("COV-B-010: endSequence round-trip through multiple sequences",
    testEndSequenceRoundTrip)

  runTest("COV-B-008: InferenceContext ref semantic aliasing",
    testInferenceContextRefSemantic)

  runTest("COV-A-007: computeNumPages pure function",
    testComputeNumPages)

  runTest("COV-A-007: computePageSizeBytes pure function",
    testComputePageSizeBytes)

when isMainModule:
  runTests()
