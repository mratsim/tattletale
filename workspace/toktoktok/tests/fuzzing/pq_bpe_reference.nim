# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Priority-queue BPE merge reference implementation, test-side only,
## no served path:
## - per-piece merge over the double-array vocab trie, ids streamed
##   dst-first into caller buffers.
## - the served ordinary encoder is the lazy-DP merge (src/merge.nim),
##   this core lives in the fuzz tree as the cross-check reference
##   of the cross-check suites and as the tie-break carrier
##   for hypothetical duplicate-rank rank tables (the lazy-DP contract is rank-distinct tables, which every trained checkpoint has).
##
## Reference contract, byte-identical to the tiktoken naive-merge
## algorithm (the tiktoken reference functions bytePairMerge, bytePairEncode and encodeOrdinaryImpl):
## - the naive algorithm rescans the whole parts list every round,
##   merging the leftmost pair of minimum rank (strict less-than scan, ties resolve to the leftmost position).
## - merge sequence identity with the naive algorithm is by construction:
##
##   1. Parts keep their setup slot indices for their whole lifetime
##      (merges delete only the right element of a pair, so slot order is alive order and leftmost position = smallest slot index).
##   2. The heap holds (rank, slot) entries and pops the lexicographic minimum,
##      exactly the naive scan's selection rule.
##   3. Invariant, every live mergeable pair has at least one heap
##      entry carrying its exact current rank. A part's stored rank
##      changes only when a merge fires at the part itself, or when
##      one fires at its alive predecessor (both recomputed exactly like the naive getRank, byte-slice rank lookups over the vocabulary),
##      and each such change pushes a fresh entry. Rank lookups are
##      content-determined (byte slice -> rank), so a popped entry
##      whose stored rank still equals the current rank merges
##      the current pair at exactly the rank the naive algorithm
##      would have used. Stale entries (rank moved on) are skipped, dead
##      slots (next < 0) are skipped, and the heap drains when no
##      mergeable pair remains, matching the naive minRank == MaxInt stop.
##
## Part token ids and ranks:
## - a part's token id is tracked alongside its rank, single bytes
##   get their vocab id at setup, a merged part gets the pair rank it
##   was born with, which is by definition the rank of its byte span.
## - the output walk emits part ids directly, the same ids the naive
##   output loop re-looks-up per part span.
##
## Recorded quirks of the naive contract, replicated byte-identically
## (the recorded expectations are the contract, never fudged):
##   - the naive bytePairEncode lacks the tiktoken fast-path return
##     for 1-byte pieces, so a 1-byte piece emits its id twice
##     (the fast-path add is not exclusive with the merge output loop).
##     Unreachable through encodeOrdinaryImpl whenever every single-byte rank
##     exists (the whole-piece branch fires first), reachable through direct
##     bytePairEncode calls. Replicated here in bytePairEncodePQ and asserted
##     by a unit row.
##   - a 0-byte piece makes the naive algorithm crash (ranks lookup of the empty slice).
##     The pipeline never produces one, this core emits nothing for it.
##
## Consumes only the public src surface (the BpeEngine ref and the merge tables of merge.nim):
## - trie byte-slice lookups.
## - the PairIndex id-pair lookups, denseToRank and the per-byte dense ids.
##
## Hot-path discipline mirrors the served core:
## - no seq[string].
## - no per-call allocation, parts, heap and scratch buffers are
##   pipeline- or call-site-owned and reused across segments.
## - byte spans arrive as openArray[byte] zero-copy views over
##   the buffer the caller holds.

import workspace/toktoktok/src/merge

const
  MaxRank* = high(int32)
    ## Sentinel rank of a non-mergeable pair (the naive algorithm's MaxInt default of ranks.getOrDefault).

type
  ## Raised where the naive algorithm would raise (unknown rank on a direct lookup path that the naive code serves with a Table KeyError),
  ## the lazy-DP merge instead emits its dead-end semantics and never raises.
  BpeError* = object of ValueError

type
  Part = object
    ## One merge state, a byte span [start, nxt's start):
    ## - the token id, the rank of merging with the alive successor
    ##   (MaxRank when none), and the alive-order neighbour links.
    ## - dead parts (absorbed right elements) carry nxt < 0.
    start: int32
    id: int32
    rank: int32
    prev: int32
    nxt: int32

  ## Reusable per-merge scratch (pipeline- or call-site-owned), parts
  ## slots plus the (rank, slot) min-heap, the lifetime contract is
  ## never-reset warm reuse, only shrunk by setLen, allocating nothing.
  MergeBuf* = object
    parts: seq[Part]
    heap: seq[(int32, int32)]

proc init*(_: type MergeBuf): MergeBuf {.inline.} =
  ## Zero-initialized scratch buffer.
  MergeBuf(parts: newSeq[Part](0), heap: newSeq[(int32, int32)](0))

proc heapLess(a, b: (int32, int32)): bool {.inline.} =
  a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

proc heapPush(h: var seq[(int32, int32)], e: (int32, int32)) {.inline.} =
  var i = h.len
  h.add e
  while i > 0:
    let p = (i - 1) shr 1
    if heapLess(h[i], h[p]):
      swap(h[i], h[p])
      i = p
    else:
      break

proc heapPop(h: var seq[(int32, int32)]): (int32, int32) {.inline.} =
  result = h[0]
  let last = h.len - 1
  h[0] = h[last]
  h.setLen(last)
  var i = 0
  while true:
    let l = 2 * i + 1
    if l >= h.len:
      break
    var s = l
    let r = l + 1
    if r < h.len and heapLess(h[r], h[l]):
      s = r
    if heapLess(h[s], h[i]):
      swap(h[i], h[s])
      i = s
    else:
      break

proc pairRank(e: BpeEngine, data: openArray[byte],
    lo, hi: int): int32 {.inline.} =
  ## Rank of the byte span [lo, hi) as a mergeable pair, the naive
  ## getOrDefault(slice, MaxInt), served by the trie byte-slice lookup directly.
  let r = e.trie.lookup(data, lo, hi)
  if r < 0: MaxRank else: int32(r)

proc rankAtLegacy(e: BpeEngine, m: MergeBuf, data: openArray[byte],
    L: int32): int32 {.inline.} =
  ## Rank source shape without PairIndex, direct byte-span trie walk,
  ## serving the paired structure check only.
  ##
  ## The trie is dense-valued, so the returned id is a dense id,
  ## rank-equal only under a dense rank permutation. Mirrors the naive
  ## getRank, the pair span runs from L's start to the next successor start.
  let R = m.parts[L].nxt
  if R < 0:
    return MaxRank
  if m.parts[R].nxt < 0:
    return MaxRank # R is the end marker, no real partner
  let lo = int(m.parts[L].start)
  let hi = int(m.parts[m.parts[R].nxt].start)
  pairRank(e, data, lo, hi)

proc pairMergeTarget(e: BpeEngine, m: MergeBuf,
    data: openArray[byte], L, R: int32): int32 {.inline.} =
  ## Merged dense id of the adjacent parts L and R, -1 when their span
  ## is not a vocab token:
  ## - PairIndex id-pair lookup on the dense id space.
  ## - the byte-span trie fallback serves spans keyed by no id pair,
  ##   a part whose byte is absent from the vocabulary.
  ## The reference contract keeps the span answer.
  let idL = m.parts[L].id
  let idR = m.parts[R].id
  if idL >= 0 and idR >= 0:
    return int32(e.mt.pairIndex.lookup(uint32(idL), uint32(idR)))
  let lo = int(m.parts[L].start)
  let hi = int(m.parts[m.parts[R].nxt].start)
  int32(e.trie.lookup(data, lo, hi))

proc rankAt(e: BpeEngine, m: MergeBuf, data: openArray[byte],
    L: int32): int32 {.inline.} =
  ## Rank of merging the part at slot L with its alive successor.
  ## MaxRank when none exists. PairIndex served, the merged dense id
  ## of the id pair is the rank source, no byte-span walk, no memo.
  let R = m.parts[L].nxt
  if R < 0:
    return MaxRank
  if m.parts[R].nxt < 0:
    return MaxRank # R is the end marker, no real partner
  let merged = e.pairMergeTarget(m, data, L, R)
  if merged < 0:
    return MaxRank
  e.mt.denseToRank[merged]

proc bytePairEncodePQLegacy*(e: BpeEngine, m: var MergeBuf,
    data: openArray[byte], lo, hi: int, dst: var seq[int]) =
  ## PairIndex-free merge core, paired-structure check contract:
  ## - the rank source is the direct byte-span trie walk.
  ## - the trie is dense-valued, so part ids live in the dense id
  ##   space and the merge order carries the rank-ascending dense order
  ##   (identical ordering, no remap needed mid-merge). The output edge
  ##   maps dense ids back to ranks.
  ## - only a dense rank permutation of 0 ..< n (r50k) makes dense id equal rank.
  ## - a tiktoken file with a skipped rank shifts every id above
  ##   the gap by one (p50k drops rank 50256), a test-only core.
  e.ensureBuilt()
  let n = hi - lo
  if n == 0:
    return # the naive algorithm crashes here, unreachable through the pipeline
  if n == 1:
    let id1 = e.trie.lookup(data, lo, hi)
    if id1 < 0:
      raise newException(BpeError,
        "rank not found for single-byte piece (naive algorithm would KeyError)")
    dst.add int(e.mt.denseToRank[id1]) # recorded quirk, fast-path add without return
  m.parts.setLen(0)
  m.heap.setLen(0)
  for i in 0 ..< n:
    m.parts.add Part(start: int32(lo + i), id: e.byteId[data[lo + i]],
                     rank: 0, prev: int32(i - 1), nxt: int32(i + 1))
  m.parts.add Part(start: int32(hi), id: -1, rank: MaxRank,
                   prev: int32(n - 1), nxt: -1) # the end marker line
  for i in 0 ..< n - 1:
    let r = rankAtLegacy(e, m, data, int32(i))
    m.parts[i].rank = r
    if r != MaxRank:
      heapPush(m.heap, (r, int32(i)))
  while m.heap.len > 0:
    let (r, slot) = heapPop(m.heap)
    if m.parts[slot].nxt < 0 or m.parts[slot].rank != r:
      continue # stale entry (rank moved on) or dead slot
    let L = slot
    let R = m.parts[L].nxt
    let pred = m.parts[L].prev
    let newNext = m.parts[R].nxt
    m.parts[L].id = r        # merged part id = pair rank by definition
    m.parts[L].nxt = newNext
    m.parts[newNext].prev = L
    m.parts[R].nxt = -1      # absorbed right element dies
    if pred >= 0:
      let nr = rankAtLegacy(e, m, data, pred)
      m.parts[pred].rank = nr
      if nr != MaxRank:
        heapPush(m.heap, (nr, pred))
    let nr2 = rankAtLegacy(e, m, data, L)
    m.parts[L].rank = nr2
    if nr2 != MaxRank:
      heapPush(m.heap, (nr2, L))
  var cur: int32 = 0 # slot 0 is always alive (never a right element)
  while m.parts[cur].nxt >= 0:
    let R = m.parts[cur].nxt
    var id = int(m.parts[cur].id)
    if id < 0:
      # single byte absent from the vocabulary:
      #   the naive output loop
      # raises KeyError here, mirror it
      raise newException(BpeError,
        "rank not found for part span (naive algorithm would KeyError)")
    dst.add int(e.mt.denseToRank[id])
    cur = R

proc bytePairEncodePQ*(e: BpeEngine, m: var MergeBuf,
    data: openArray[byte], lo, hi: int, dst: var seq[int]) =
  ## Priority-queue merge over the byte slice [lo, hi):
  ## Byte-identical contract to the naive bytePairEncode, merge
  ## sequence identical to the naive full-rescan merge by the construction
  ## argument in the module header:
  ## - the 1-byte double-add quirk, then the merge sequence, then
  ##   the output walk emitting one rank per alive part span.
  ## - merge lookups run through the PairIndex over dense id pairs.
  ## - part ids occupy dense id space, the output walk maps them to ranks.
  e.ensureBuilt()
  let n = hi - lo
  if n == 0:
    return # the naive algorithm crashes here, unreachable through the pipeline
  if n == 1:
    let id1 = e.trie.lookup(data, lo, hi)
    if id1 < 0:
      raise newException(BpeError,
        "rank not found for single-byte piece (naive algorithm would KeyError)")
    dst.add int(e.mt.denseToRank[id1]) # recorded quirk, fast-path add without return
  m.parts.setLen(0)
  m.heap.setLen(0)
  for i in 0 ..< n:
    m.parts.add Part(start: int32(lo + i), id: e.byteId[data[lo + i]],
                     rank: 0, prev: int32(i - 1), nxt: int32(i + 1))
  m.parts.add Part(start: int32(hi), id: -1, rank: MaxRank,
                   prev: int32(n - 1), nxt: -1) # the end marker line
  for i in 0 ..< n - 1:
    let r = rankAt(e, m, data, int32(i))
    m.parts[i].rank = r
    if r != MaxRank:
      heapPush(m.heap, (r, int32(i)))
  while m.heap.len > 0:
    let (r, slot) = heapPop(m.heap)
    if m.parts[slot].nxt < 0 or m.parts[slot].rank != r:
      continue # stale entry (rank moved on) or dead slot
    let L = slot
    let R = m.parts[L].nxt
    let pred = m.parts[L].prev
    let newNext = m.parts[R].nxt
    let merged = e.pairMergeTarget(m, data, L, R)
    doAssert merged >= 0, "live merge lost its pair lookup"
    m.parts[L].id = merged  # merged part id = the pair's merged dense id
    m.parts[L].nxt = newNext
    m.parts[newNext].prev = L
    m.parts[R].nxt = -1      # absorbed right element dies
    if pred >= 0:
      let nr = rankAt(e, m, data, pred)
      m.parts[pred].rank = nr
      if nr != MaxRank:
        heapPush(m.heap, (nr, pred))
    let nr2 = rankAt(e, m, data, L)
    m.parts[L].rank = nr2
    if nr2 != MaxRank:
      heapPush(m.heap, (nr2, L))
  var cur: int32 = 0 # slot 0 is always alive (never a right element)
  while m.parts[cur].nxt >= 0:
    let R = m.parts[cur].nxt
    var id = int(m.parts[cur].id)
    if id < 0:
      # single byte absent from the vocabulary:
      #   the naive output loop
      # raises KeyError here, mirror it
      raise newException(BpeError,
        "rank not found for part span (naive algorithm would KeyError)")
    dst.add int(e.mt.denseToRank[id])
    cur = R

proc encodeSegmentPQ*(e: BpeEngine, m: var MergeBuf,
    data: openArray[byte], lo, hi: int, dst: var seq[int]) =
  ## One pre-tokenized piece, naive encodeOrdinaryImpl semantics:
  ## - the whole-piece encoder hit (a single id) first, else the PQ merge.
  ## - the reference of the ordinary path, the served encoder is
  ##   encodeSegment in src/merge.nim, the two are checked byte-identical
  ##   by the cross-check suite.
  ## Buffer contract, dst may accumulate across pieces, fresh scratch
  ## per call while call sites may reuse an accumulating buffer.
  e.ensureBuilt()
  if hi == lo:
    return # the pipeline never produces empty pieces
  let whole = e.trie.lookup(data, lo, hi)
  if whole >= 0:
    dst.add int(e.mt.denseToRank[whole])
  else:
    bytePairEncodePQ(e, m, data, lo, hi, dst)
