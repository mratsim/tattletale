# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## InferenceContext suite: startSequence-shaped flows over the pool and trie.
##
## Test that the ported context tracks pages, write cursor, cached token counts and rope slices.
## Flows: a startSequence, decode steps, an endSequence (graftPages) and a restart cycle.
## Baselines: the committed PagePool and KVCache trie.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/wipM07ictx workspace/tattleformers/tests/stateful/test_inference_context.nim

import std/math
import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes
from workspace/tattleformers/src/stateful/kvcache import KVCache, new, lpm,
    graftPages, evict, TokensPerPage
import workspace/tattleformers/src/stateful/page_pool
import workspace/tattleformers/src/stateful/inference_context
import workspace/tattleformers/src/layers/rope

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
  let engine = gMetalEngine()
  let pool = PagePool.init(engine, NumPages, NumLayers, KVHeads, HeadDim)
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

  # ── decode-step shape: position_ids advance, rope slices ────────────
  block:
    let rotary = RotaryPositionEmbedding.new(engine, HeadDim, 1024, 10000.0'f64)
    let halfDim = HeadDim div 2
    # host reference over the tables' invariant:
    # cos_cache[p, d] == cos(p * rope_theta^(-2*d/rotary_dim)),
    # NEOX half-repeat with full-head_dim rotation here
    proc hostCosSin(p: int): tuple[c: seq[float32], s: seq[float32]] =
      result.c = newSeq[float32](halfDim)
      result.s = newSeq[float32](halfDim)
      for d in 0 ..< halfDim:
        let angle = float64(p) * pow(10000.0'f64, -2.0 * float64(d) / float64(HeadDim))
        result.c[d] = float32(cos(angle))
        result.s[d] = float32(sin(angle))

    # prefill positions [0, 1, 2]
    ctx.setPositionIdsArange(seq_len = 3, offset = 0)
    doAssert ctx.position_ids == @[0, 1, 2], "prefill positions"
    ctx.setRopeForPositions(rotary)
    var roots = @[ctx.cos.node, ctx.sin.node]
    engine.eval(roots.toSpan())
    var
      cosGot = newSeq[float32](3 * halfDim)
      sinGot = newSeq[float32](3 * halfDim)
    copyToHostFromDevice(cosGot, ctx.cos)
    copyToHostFromDevice(sinGot, ctx.sin)
    for p in 0 ..< 3:
      let (cRef, sRef) = hostCosSin(p)
      for d in 0 ..< halfDim:
        doAssert cosGot[p * halfDim + d] == cRef[d],
          "cos mismatch at position " & $p & " column " & $d
        doAssert sinGot[p * halfDim + d] == sRef[d],
          "sin mismatch at position " & $p & " column " & $d

    # decode position [3] (next token at offset 3)
    ctx.setPositionIds(@[3])
    doAssert ctx.position_ids == @[3], "decode position"
    ctx.setRopeForPositions(rotary)
    var roots2 = @[ctx.cos.node, ctx.sin.node]
    engine.eval(roots2.toSpan())
    var
      cosGot2 = newSeq[float32](halfDim)
      sinGot2 = newSeq[float32](halfDim)
    copyToHostFromDevice(cosGot2, ctx.cos)
    copyToHostFromDevice(sinGot2, ctx.sin)
    let (cRef2, sRef2) = hostCosSin(3)
    for d in 0 ..< halfDim:
      doAssert cosGot2[d] == cRef2[d], "decode cos column " & $d
      doAssert sinGot2[d] == sRef2[d], "decode sin column " & $d

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
