# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## InferenceContext suite: startSequence-shaped flows over the pool and trie.
##
## Test that the context tracks pages, write cursor, cached token counts
## and position ids.
##
## Flows: a startSequence, an endSequence with graftPages, a restart cycle and an eviction.
##
## Baselines: the committed PagePool and the KVCache trie.
## The rope-slice flow (setRopeForPositions) comes with the layers port.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/ttf --outdir:build/ttf --path:"$PWD" \
##     workspace/tattleformers/tests/stateful/test_inference_context.nim
##

import workspace/tensorcery/src/tensorcery
from workspace/tattleformers/src/stateful/kvcache import KVCache, new, lpm,
    graftPages, evict, TokensPerPage
import workspace/tattleformers/src/stateful/page_pool
import workspace/tattleformers/src/stateful/inference_context

const
  NumLayers = 2
  NumPages = 4
  KVHeads = 8
  HeadDim = 128

proc makeTokens(n: int): seq[uint32] =
  ## Deterministic distinct token ids 1 .. n (0 is the trie's "no token").
  result = newSeq[uint32](n)
  for i in 0 ..< n:
    result[i] = uint32(i + 1)

proc main =
  let pool = PagePool.init(NumPages, NumLayers, KVHeads, HeadDim)
  var trie = KVCache[uint32, Page].new()
  let ctx = InferenceContext[uint32, Page].init(
    NumLayers, batch_size = 1, KVHeads, max_seq = 1024, HeadDim)

  # ── startSequence-shaped: LPM miss, borrow fresh pages ──────────────
  block:
    let tokens = makeTokens(TokensPerPage + 44) # 300 tokens, 2 pages
    let matched = trie.lpm(tokens)
    doAssert matched.totalTokenMatched == 0, "no LPM match on a fresh trie"
    doAssert matched.pages.len == 0, "no LPM pages on a fresh trie"

    ctx.cached_tokens = matched.totalTokenMatched
    ctx.pages = @[]
    for p in matched.pages:
      ctx.pages.add p
    let toBorrow = (tokens.len + TokensPerPage - 1) div TokensPerPage
    for i in 0 ..< toBorrow:
      ctx.pages.add pool.borrow()
    ctx.input_tokens = tokens
    # kv_position: the write cursor, 0 after startSequence (no LPM match)
    ctx.kv_position = 0

    doAssert ctx.kv_position == 0, "kv_position should be 0 after startSequence (no LPM match)"
    doAssert ctx.pages.len == 2, "expected 2 pages for 300 tokens, got " & $ctx.pages.len
    doAssert ctx.cached_tokens == 0, "no cached tokens on a fresh trie"
    let idx0 = ctx.pages[0].pageIndex()
    let idx1 = ctx.pages[1].pageIndex()
    doAssert idx0 != idx1, "distinct pool slots for distinct pages"

    # prefill positions [0, 1, 2]. The arange builder covers the decode
    # offset form too
    ctx.setPositionIdsArange(seq_len = 3, offset = 0)
    doAssert ctx.position_ids == @[0, 1, 2], "prefill positions"
    ctx.setPositionIds(@[3])
    doAssert ctx.position_ids == @[3], "decode position"

    # pageSlots: the table the pool write and jagged attention ops consume
    let slots = ctx.pageSlots()
    doAssert slots.len == 2, "one slot per page"
    doAssert slots[0] == idx0.int and slots[1] == idx1.int,
      "pageSlots carry the pages' pool slots"

  # ── endSequence-shaped: graftPages commits pages into the trie ──────
  block:
    var pages: seq[Page]
    for p in ctx.pages:
      pages.add p
    trie.graftPages(ctx.input_tokens, pages)
    # invariants re-checked implicitly by the restart LPM below.
    # Page refs of the context drop with the graft's ownership transfer.
    ctx.pages = default(seq[Page])

    # ── restart: the full 300-token sequence LPM-matches ──────────────
    let tokens = makeTokens(TokensPerPage + 44)
    let matched = trie.lpm(tokens)
    doAssert matched.totalTokenMatched == TokensPerPage + 44,
      "LPM matches the grafted sequence, got " & $matched.totalTokenMatched
    doAssert matched.pages.len == 2, "both grafted pages matched"
    doAssert matched.lastLevelMatched == matched.totalTokenMatched,
      "the whole match sits in one node span"

    ctx.clearState()
    doAssert ctx.kv_position == 0, "kv_position resets"
    doAssert ctx.cached_tokens == 0, "cached_tokens resets"
    doAssert ctx.pages.len == 0, "pages dropped"
    doAssert ctx.input_tokens.len == 0, "input_tokens dropped"
    doAssert ctx.position_ids.len == 0, "position_ids dropped"
    doAssert ctx.num_layers == NumLayers, "metadata kept for reuse"

    # matched trie pages carry the cached sequence, nothing to borrow
    ctx.cached_tokens = matched.totalTokenMatched
    ctx.pages = @[]
    for p in matched.pages:
      ctx.pages.add p
    doAssert ctx.pages.len == 2, "both matched pages ride the context"
    doAssert ctx.pages[0].pageIndex() == matched.pages[0].pageIndex(),
      "the cached prefix rides the trie's page"

  # ── evict + second restart: the pool recycles through the trie ──────
  block:
    let tokens = makeTokens(TokensPerPage + 44)
    let matched = trie.lpm(tokens)
    discard trie.evict()
    let after = trie.lpm(tokens)
    doAssert after.totalTokenMatched <= matched.totalTokenMatched,
      "eviction only shrinks the trie's matchable prefix"

  echo "inference context (pool + trie): all checks passed"

when isMainModule:
  main()
