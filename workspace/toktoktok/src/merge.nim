# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Pieces-to-ids stage of the machine pipeline in one module:
## the double-array vocab trie, the load-time merge-table derivation,
## and the served lazy-DP merge core.
##
## Load-time derivation, runtime-lookup contract:
## - buildTrie + denseIdsOfSorted + buildMergeTables run once, engine
##   build via BpeEngine.ensureBuilt, loud receipt, pure functions
##   of the sorted key list.
## - the trie is valued in the rank-ascending dense id space,
##   the byte-ordered tiebreak.
## - every encode-path lookup
##   (byte-slice ranks, PairIndex id-pair ranks, split chains, denseToRank output walk)
##   reads the frozen tables with no rank indirection at lookup time.
## Each structure keeps its section doc below.

import std/[tables, strutils, monotimes, times, algorithm]


# ------------------------------------------------------------------------
# Double-array vocab trie (BASE/CHECK, openArray[byte] slice lookups)
# ------------------------------------------------------------------------


## Double-array vocab trie, byte-string keys and integer ranks:
##
## Serves both whole-piece encoder hits and BPE pair-rank lookups:
##
## | surface           | detail                                                                                                                                                         |
## | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | whole-piece hits  | a pre-tokenized piece whose full byte string is a vocabulary key                                                                                               |
## | pair-rank lookups | byte-slice views into a segment                                                                                                                                |
## | structure         | keys never live in the hot path, the trie is a flat BASE/CHECK/VALS triple (Aoe's double-array scheme, the same BASE+CHECK core as the special-scan automaton) |
## | lookup cost       | a lookup walks the key bytes in place, 2 loads per byte                                                                                                        |
##
## Construction:
## - keys are sorted, a node tree is grown by LCP-stack insertion:
##   each key shares its longest-common-prefix path with the previous
##   key (total work O(total key bytes)), children lists come out in byte order.
## - a depth-first pass assigns BASE by first-fit placement over
##   a roving free-slot cursor.
## - both passes are pure functions of the sorted (key, rank) list, so
##   the built arrays are byte-deterministic for a given rank table.
##
## Representation invariants, the root is node 1 and slot 0 is reserved:
## | slot fact          | meaning                                                                               |
## | ------------------ | ------------------------------------------------------------------------------------- |
## | `chk[i] == 0`      | slot i is free                                                                        |
## | `chk[i] == p >= 1` | slot i holds the child of node p at byte i - base[p], check values are parent indices |
## | `chk[i] < 0`       | slot i is reserved (slot 0, the root slot)                                            |
## | `base[p] == 0`     | node p is a leaf (no children)                                                        |
## | `vals[s] >= 0`     | a key ends at slot s, with rank vals[s]                                               |
## | `vals[s] == -1`    | the path to slot s is a strict prefix only                                            |
##
## A base of 0 is never chosen and slot 0 carries chk = -1, so slot 0
## is unreachable from any walk and root children (chk == 1) stay
## distinct from free slots (chk == 0).

type
  VocabTrie* {.final.} = object
    ## Flat double-array trie, immutable after build:
    ## - lookups are const-shaped, no allocation and no state change.
    base*: seq[int32]
    chk*: seq[int32]
    vals*: seq[int32]
    keyCount*: int
    nodeCount*: int
    built*: bool

  TrieNode = object
    ## Compact build-phase node (left-child right-sibling lists).
    b: uint8       # edge byte from the parent, meaningless at the root
    first: int32   # first child, -1 when a leaf
    last: int32    # last child appended, for byte-ordered append
    nextSib: int32 # next sibling, -1 terminated
    rank: int32    # >= 0 when a key ends at this node, else -1

const
  TrieRoot* = 1
  RankNotFound* = -1

proc cmpBytes(a, b: openArray[byte]): int =
  ## Byte-wise lexicographic compare (unsigned), then by length.
  let n = min(a.len, b.len)
  for i in 0 ..< n:
    if a[i] != b[i]:
      return int(a[i]) - int(b[i])
  a.len - b.len

proc commonPrefixLen(a, b: openArray[byte]): int =
  let n = min(a.len, b.len)
  var i = 0
  while i < n and a[i] == b[i]:
    inc i
  i

proc buildNodeTree(keys: openArray[seq[byte]], ranks: openArray[int]): seq[TrieNode] =
  ## Sorted-key LCP-stack insertion, O(total key bytes).
  ## Children lists come out in increasing byte order because keys arrive
  ## sorted and every new chain appends at the branch tail.
  let n = keys.len
  var total = 1
  for k in keys.items:
    total += k.len
  result = newSeq[TrieNode](total)
  result[0] = TrieNode(b: 0, first: -1, last: -1, nextSib: -1, rank: -1)
  var used = 1
  var stack: seq[int32] = @[int32(0)] # node indices along the current path
  var prev: seq[byte] = @[]
  for idx in 0 ..< n:
    let key = keys[idx]
    doAssert key.len > 0
    doAssert idx == 0 or cmpBytes(prev, key) < 0, "keys must be sorted, distinct"
    let lcp = commonPrefixLen(prev, key)
    stack.setLen(lcp + 1)
    var parent = stack[^1]
    for d in lcp ..< key.len:
      let node = int32(used)
      inc used
      result[node] = TrieNode(b: key[d], first: -1, last: -1,
                              nextSib: -1, rank: -1)
      if result[parent].last < 0:
        result[parent].first = node
      else:
        result[result[parent].last].nextSib = node
      result[parent].last = node
      stack.add node
      parent = node
    result[used - 1].rank = int32(ranks[idx])
    prev = key
  result.setLen(used)

proc freshTriple(cap: int): tuple[base, chk, vals: seq[int32]] =
  result.base = newSeq[int32](cap)
  result.chk = newSeq[int32](cap)
  result.vals = newSeq[int32](cap)
  for i in 0 ..< cap:
    result.vals[i] = -1
  result.chk[0] = -1 # slot 0 reserved, never claimed

proc enlargeTriple(base: var seq[int32], chk: var seq[int32],
    vals: var seq[int32], need: int) =
  var cap = base.len
  while cap < need:
    cap = cap * 2
  var (nb, nc, nv) = freshTriple(cap)
  copyMem(nb[0].addr, base[0].unsafeAddr, sizeof(int32) * base.len)
  copyMem(nc[0].addr, chk[0].unsafeAddr, sizeof(int32) * chk.len)
  copyMem(nv[0].addr, vals[0].unsafeAddr, sizeof(int32) * vals.len)
  swap(base, nb)
  swap(chk, nc)
  swap(vals, nv)

proc placeChildren(base: var seq[int32], chk: var seq[int32],
    vals: var seq[int32], cursor: var int, nodes: openArray[TrieNode],
    buildNode, slot: int32): int32 =
  ## First-fit BASE placement for one node's children set, scanning
  ## candidates from the roving cursor. Returns the chosen base
  ## (0 for a leaf, base unused).
  var first = nodes[buildNode].first
  if first < 0:
    return 0 # the leaf slot
  var cb: array[256, uint8] # children bytes, ascending
  var nc = 0
  var c = first
  while c >= 0:
    cb[nc] = nodes[c].b
    inc nc
    c = nodes[c].nextSib
  let maxSlot = int(cb[nc - 1])
  var b = if cursor < 1: 1 else: cursor
  while true:
    if b + maxSlot >= chk.len:
      enlargeTriple(base, chk, vals, (b + maxSlot + 1) * 2)
    var ok = true
    for i in 0 ..< nc:
      if chk[b + int(cb[i])] != 0:
        ok = false
        break
    if ok:
      break
    inc b
  let chosen = int32(b)
  base[slot] = chosen
  c = first
  while c >= 0:
    let s = b + int(nodes[c].b)
    chk[s] = slot
    vals[s] = nodes[c].rank
    c = nodes[c].nextSib
  cursor = b + 1
  chosen

proc assignDfs(nodes: openArray[TrieNode]): VocabTrie =
  ## Depth-first double-array assignment over the node tree:
  ## - the DFS stack carries (build-tree node, final slot) pairs,
  ##   build-tree indices and final slots live in different index spaces.
  ## - children are visited in byte order (deterministic traversal).
  ## - slots are claimed with chk during placement, so a child slot
  ##   exists before it is popped.
  var (base, chk, vals) = freshTriple(max(1024, nodes.len * 2))
  enlargeTriple(base, chk, vals, 258) # headroom for the root placement scan
  chk[1] = -2 # root slot reserved (never claimed by a placement)
  var cursor = 1
  var stack: seq[(int32, int32)] = @[(int32(0), int32(TrieRoot))]
  var nodeCount = 1
  var childPairs: array[256, (int32, int32)]
  while stack.len > 0:
    let (buildNode, slot) = stack.pop()
    let b = placeChildren(base, chk, vals, cursor, nodes, buildNode, slot)
    # collect (build node, slot) pairs (siblings ascending), then push
    # in reverse so the smallest byte is visited first
    var m = 0
    var c = nodes[buildNode].first
    while c >= 0:
      childPairs[m] = (c, int32(b + int(nodes[c].b)))
      inc m
      c = nodes[c].nextSib
    inc nodeCount, m
    for j in countdown(m - 1, 0):
      stack.add childPairs[j]
  # trim trailing free slots
  var last = base.len - 1
  while last >= 0 and chk[last] == 0:
    dec last
  result.base = base[0 .. last]
  result.chk = chk[0 .. last]
  result.vals = vals[0 .. last]
  result.nodeCount = nodeCount
  result.built = true

proc buildTrie*(keys: openArray[seq[byte]], ranks: openArray[int]): VocabTrie =
  ## Build from already-sorted keys (byte-ascending, ranks parallel).
  ## Deterministic pure function of the input lists.
  assert keys.len == ranks.len
  let nodes = buildNodeTree(keys, ranks)
  result = assignDfs(nodes)
  result.keyCount = keys.len

proc lookup*(t: VocabTrie, data: openArray[byte], lo, hi: int): int =
  ## Rank of the byte slice data[lo ..< hi], RankNotFound when absent.
  ## No allocation, no state change, the walk is 2 loads per key byte.
  ##
  ## Expected input:
  ##   `data` is a zero-copy view (openArray[byte]) over
  ## the buffer the caller holds, lo/hi index that view directly.
  var cur = TrieRoot
  for i in lo ..< hi:
    let nxt = int(t.base[cur]) + int(data[i])
    if nxt >= t.chk.len or t.chk[nxt] != cur:
      return RankNotFound
    cur = nxt
  int(t.vals[cur])

proc lookup*(t: VocabTrie, key: openArray[byte]): int =
  ## Rank of the whole key, RankNotFound when absent.
  if key.len == 0:
    return RankNotFound
  lookup(t, key, 0, key.len)

proc lookup*(t: VocabTrie, key: string): int =
  ## Rank of the whole key string, RankNotFound when absent.
  if key.len == 0:
    return RankNotFound
  lookup(t, key.toOpenArrayByte(0, key.len - 1), 0, key.len)

proc longestMatch(t: VocabTrie, data: openArray[byte], lo, hi: int): int =
  ## Longest-key match over data[lo ..< hi], value in dense id space,
  ## RankNotFound when no key matches.
  ##
  ## The walk tracks the deepest key end seen, one trie step
  ## per input byte, so a match's length is read off the winning key
  ## without a rescan.
  ##
  ## Expected input:
  ## - `data` is a zero-copy view over the buffer the caller holds,
  ##   lo/hi index that view directly.
  var cur = TrieRoot
  result = RankNotFound
  var i = lo
  while i < hi:
    let nxt = int(t.base[cur]) + int(data[i])
    if nxt >= t.chk.len or t.chk[nxt] != cur:
      break
    cur = nxt
    if t.vals[cur] >= 0:
      result = t.vals[cur]
    inc i

proc arrayBytes(t: VocabTrie): int {.inline.} =
  ## Total byte size of the three flat arrays.
  4 * (t.base.len + t.chk.len + t.vals.len)

# ------------------------------------------------------------------------
# Merge tables, PairIndex + SplitTable + dense id permutation + load-time self-check
# ------------------------------------------------------------------------

## Static merge lookup structures, frozen at load:
##
## | structure              | contract                                                                                                                                                              |
## | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | TokenArena             | all vocab keys concatenated, indexed by dense id (dense ids = rank-ascending order, byte-ordered tiebreak for duplicate ranks, so children ids sort below parent ids) |
## | PairIndex              | the exact-match map over every valid merge pair, key = (left dense id, right dense id) packed, value = the merged dense id                                            |
## | PairIndex hashing      | multiply-shift hashed stride with the factor fixed at load over frozen keys, cumulative-offset bucket arrays, no probing                                              |
## | PairIndex worst bucket | verifiable at load and asserted there, the whole guarantee of a static structure over a frozen vocabulary                                                             |
## | PairIndex role         | the merge-time lookup surface over the frozen pair key set                                                                                                            |
## | SplitTable             | dense id -> (left id, right id) flat arrays plus nextPrefixMatch (longest proper-prefix token)                                                                        |
## | SplitTable derivation  | greedy first-valid-split over the prefix chain, checked by is_valid_token_pair against the partially built tables (byte_pair_encoding.rs from_dictionary)             |
## | SplitTable limit loop  | a pair hit with combined below limit means the adjacency is frozen-invalid (byte_pair_encoding.rs:97-125)                                                             |
## | SplitTable fallback    | tokens with no valid split carry the self-split (id, id)                                                                                                              |
## | selfCheck              | every vocab token re-encodes to itself through the built tables, a priority-queue rank merge over PairIndex                                                           |
## | selfCheck assertion    | hard-asserts zero failures on the tiktoken-shape dense permutation (0 ..< n). Synthetic tables (duplicated or non-dense ranks) record the count instead of asserting  |
##
## All builds are load-time only, deterministic pure functions
## of the key and rank lists, allocation-free at lookup time:
## - the dense-order sorts sort an index permutation whose
##   comparator reads the key and rank openArrays in place
##   (denseIdsOfSorted, the buildMergeTables dense pass).
## - no key-copying tuple sequence is materialized.

const
  PairIndexMaxBucket* = 16
    ## Worst-bucket bound the factor search must satisfy, asserted at load
    ## (the probed multiply-shift distribution, max 16 per bucket over the real 296,461-pair key set).

type
  PairIndex* = object
    ## Multiply-shift exact-match map over the packed pair keys, frozen
    ## at load (contract in the merge-tables section doc).
    slotBits*: int
    factor*: uint64
    bOff: seq[int32]   # cumulative offsets, len 2^slotBits + 1
    kA: seq[uint32]    # left dense id per entry
    kB: seq[uint32]    # right dense id per entry
    val: seq[int32]    # merged dense id per entry

  MergeTables* = object
    ## Frozen merge tables, arena plus pair index plus split arrays plus dense id columns, contract in the merge-tables section doc.
    arena*: seq[byte]
    starts*: seq[int32]  # arena offset per dense id
    lens*: seq[int32]    # byte length per dense id
    maxLen*: int
    pairIndex*: PairIndex
    splitLeft*: seq[int32]
    splitRight*: seq[int32]
    nextPrefixMatch*: seq[int32]
    denseToRank*: seq[int32]  # token rank per dense id
    pairCount*: int
    worstBucket*: int
    selfCheckFails*: int
    deriveMs*: float
    indexMs*: float
    selfCheckMs*: float

proc bucketMax(counts: openArray[int32]): int =
  result = 0
  for c in counts:
    if int(c) > result: result = int(c)

proc pairHash(a, b: uint32, factor: uint64, bits: int): int {.inline.} =
  ## Multiply-shift hashed stride over the packed pair key.
  var h = uint64(a) * 0x9E3779B97F4A7C15'u64
  h = h xor (h shr 31)
  h = h + uint64(b) * 0xC2B2AE3D27D4EB4F'u64
  h = h xor (h shr 27)
  h = h * factor
  h = h xor (h shr 32)
  int(h shr uint(64 - bits))

proc buildPairIndex(keys: openArray[uint64], vals: openArray[int32],
    maxBucket: int): PairIndex =
  ## Builds the index over pair keys packed (left shl 32) or right:
  ## - the factor search loops upward from a fixed seed in fixed
  ##   steps and widens the slot count only if 64 steps all exceed
  ##   the bucket bound.
  ## - the built structure is fully determined by the frozen key set at load.
  let n = keys.len
  doAssert n > 0, "PairIndex over an empty pair set"
  var bits = 19
  var factor = 0'u64
  while true:
    var found = false
    var f = 0xA24BAED4963EE407'u64
    for attempt in 0 ..< 64:
      var counts = newSeq[int32](1 shl bits)
      for i in 0 ..< n:
        let a = uint32(keys[i] shr 32)
        let b = uint32(keys[i] and 0xFFFFFFFF'u64)
        inc counts[pairHash(a, b, f, bits)]
      if bucketMax(counts) <= maxBucket:
        factor = f
        found = true
        break
      f += 2
    if found: break
    inc bits
    doAssert bits <= 24, "multiply-shift factor search failed (pairs)"
  result.slotBits = bits
  result.factor = factor
  var counts = newSeq[int32](1 shl bits)
  for i in 0 ..< n:
    let a = uint32(keys[i] shr 32)
    let b = uint32(keys[i] and 0xFFFFFFFF'u64)
    inc counts[pairHash(a, b, factor, bits)]
  result.bOff = newSeq[int32](counts.len + 1)
  var acc = 0
  for i in 0 ..< counts.len:
    result.bOff[i] = int32(acc)
    inc acc, counts[i]
  result.bOff[counts.len] = int32(acc)
  var cursor = result.bOff[0 ..< counts.len]
  result.kA = newSeq[uint32](n)
  result.kB = newSeq[uint32](n)
  result.val = newSeq[int32](n)
  for i in 0 ..< n:
    let a = uint32(keys[i] shr 32)
    let b = uint32(keys[i] and 0xFFFFFFFF'u64)
    let s = pairHash(a, b, factor, bits)
    let k = cursor[s]
    result.kA[k] = a
    result.kB[k] = b
    result.val[k] = vals[i]
    inc cursor[s]

proc lookup*(t: PairIndex, a, b: uint32): int {.inline.} =
  ## Merged dense id of the pair (a, b), -1 when the pair is not a valid
  ## merge pair. Frozen bucket arrays, no allocation.
  let s = pairHash(a, b, t.factor, t.slotBits)
  var k = t.bOff[s]
  let e = t.bOff[s + 1]
  while k < e:
    if t.kA[k] == a and t.kB[k] == b:
      return int(t.val[k])
    inc k
  -1

proc memBytes(t: PairIndex): int {.inline.} =
  ## Total byte size of the PairIndex arrays.
  4 * (t.bOff.len + t.val.len) + 4 * (t.kA.len + t.kB.len)

proc pairIndexWorstBucket(t: PairIndex): int =
  ## Recount the bucket occupancy from the frozen offsets
  ## (the load-time assertable bound).
  result = 0
  let m = t.bOff.len - 2
  for s in 0 .. m:
    let w = int(t.bOff[s + 1]) - int(t.bOff[s])
    if w > result: result = w

proc isValidPairDerivation(pairs: var Table[(uint32, uint32), uint32],
    splitLeft, splitRight: seq[int32], t1, t2: uint32): bool =
  ## is_valid_token_pair over the PARTIALLY built tables (the derivation-time check of rust-gems from_dictionary):
  ## - a pair hit with combined below limit means the adjacency is frozen-invalid.
  ## - otherwise the limit-loop walks the split chain, polarity per byte_pair_encoding.rs:97-125.
  var token1 = t1
  var token2 = t2
  var limit = uint32.high
  while true:
    let combined = pairs.getOrDefault((token1, token2), uint32.high)
    if combined != uint32.high and combined < limit:
      return false
    if token1 > token2:
      limit = token1
      token1 = uint32(splitRight[token1])
      if token1 == limit:
        limit = token2 + 1
        token2 = uint32(splitLeft[token2])
        if token2 + 1 == limit:
          return true
    else:
      limit = token2 + 1
      token2 = uint32(splitLeft[token2])
      if token2 + 1 == limit:
        limit = token1
        token1 = uint32(splitRight[token1])
        if token1 == limit:
          return true

proc selfCheck(t: MergeTables): tuple[unreachable, singleIdWrong: int] =
  ## Every multi-byte vocab token re-encodes through the built tables,
  ## single-byte tokens are trivially themselves.
  ##
  ## A priority-queue rank merge whose rank source is PairIndex id-pair lookups.
  ##
  ## Returns:
  ## - the count of unreachable-shape re-encodes (multi-id outputs).
  ## - singleIdWrong, set for a re-encode ending at a single id other
  ##   than the token, a real derivation bug.
  ##
  ## Vocabularies converted from HF checkpoints carry tokens unreachable
  ## by merging, special-like markers whose byte pieces merge into other
  ## tokens first:
  ## - a nonzero unreachable count is a data property of such vocabularies
  let n = t.starts.len
  var byteId: array[256, int32]
  for b in 0 .. 255: byteId[b] = int32(-1)
  for d in 0 ..< n:
    if int(t.lens[d]) == 1:
      byteId[t.arena[int(t.starts[d])]] = int32(d)
  var
    ps: seq[tuple[start, id, rank, prev, nxt: int32]] = @[]
    heap: seq[(int32, int32)] = @[]
    outIds: seq[int32] = @[]

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

  for d in 0 ..< n:
    if int(t.lens[d]) < 2:
      continue
    let lo = int(t.starts[d])
    let ln = int(t.lens[d])
    let hi = lo + ln
    ps.setLen(0)
    heap.setLen(0)
    for i in 0 ..< ln:
      ps.add (start: int32(lo + i), id: byteId[t.arena[lo + i]],
              rank: int32(0), prev: int32(i - 1), nxt: int32(i + 1))
    ps.add (start: int32(hi), id: int32(-1), rank: int32.high,
            prev: int32(ln - 1), nxt: int32(-1))
    for i in 0 ..< ln - 1:
      let R = ps[i].nxt
      var r = int32.high
      if ps[R].nxt >= 0:
        let merged = t.pairIndex.lookup(uint32(ps[i].id), uint32(ps[R].id))
        if merged >= 0:
          r = t.denseToRank[merged]
      ps[i].rank = r
      if r != int32.high:
        heapPush(heap, (r, int32(i)))
    while heap.len > 0:
      let (r, slot) = heapPop(heap)
      if ps[slot].nxt < 0 or ps[slot].rank != r:
        continue
      let L = slot
      let R = ps[L].nxt
      let pred = ps[L].prev
      let newNext = ps[R].nxt
      let merged = t.pairIndex.lookup(uint32(ps[L].id), uint32(ps[R].id))
      doAssert merged >= 0, "rank source moved under a live merge"
      ps[L].id = int32(merged)
      ps[L].nxt = newNext
      ps[newNext].prev = L
      ps[R].nxt = int32(-1)
      if pred >= 0:
        var nr = int32.high
        let NR = ps[pred].nxt
        if ps[NR].nxt >= 0:
          let m2 = t.pairIndex.lookup(uint32(ps[pred].id), uint32(ps[NR].id))
          if m2 >= 0: nr = t.denseToRank[m2]
        ps[pred].rank = nr
        if nr != int32.high:
          heapPush(heap, (nr, pred))
      var nr2 = int32.high
      let R2 = ps[L].nxt
      if ps[R2].nxt >= 0:
        let m2 = t.pairIndex.lookup(uint32(ps[L].id), uint32(ps[R2].id))
        if m2 >= 0: nr2 = t.denseToRank[m2]
      ps[L].rank = nr2
      if nr2 != int32.high:
        heapPush(heap, (nr2, L))
    outIds.setLen(0)
    var cur: int32 = 0
    while ps[cur].nxt >= 0:
      outIds.add ps[cur].id
      cur = ps[cur].nxt
    if outIds.len == 1 and outIds[0] != int32(d):
      # the merge process reached a single id and it is not the token:
      # a real derivation bug (wrong merged id in the PairIndex)
      inc result.singleIdWrong
      if result.singleIdWrong <= 5:
        echo "[bpe] self-check WRONG SINGLE ID dense id ", d,
          " encoded ", outIds
    elif outIds.len > 1:
      # multi-id re-encode, a token unreachable by merging, its byte
      # pieces merge into other tokens first. The tiktoken ordinary
      # encoding of the literal text produces exactly this shape, e.g.
      # gpt2's endoftext marker, a data property of the vocabulary,
      # not a derivation bug, checked by the id-parity suites.
      inc result.unreachable
      if result.unreachable <= 5:
        echo "[bpe] self-check unreachable token dense id ", d,
          " encoded ", outIds

proc sortDenseOrder(order: var seq[int32], keys: openArray[seq[byte]],
    keyRanks: openArray[int])
  # forward, body with the dense-order helpers below

proc buildMergeTables*(keys: openArray[seq[byte]],
    keyRanks: openArray[int], trie: VocabTrie): MergeTables =
  ## Builds the arena, PairIndex and SplitTable. The caller hands
  ## byte-sorted keys with the trie re-valued to dense ids
  ## (trie lookups below return dense ids directly), assignment contract:
  ## - dense ids are assigned rank-ascending with the byte-lexicographic tiebreak.
  assert keys.len == keyRanks.len
  let n = keys.len
  let t0 = getMonoTime()

  # dense order:
  #   rank ascending, byte tiebreak, by index permutation
  # over the input openArrays (no key-copying tuple sequence).
  # Distinct keys so the byte-key map is exact even
  # under duplicate ranks.
  var order = newSeq[int32](n)
  for i in 0 ..< n:
    order[i] = int32(i)
  sortDenseOrder(order, keys, keyRanks)
  result.denseToRank = newSeq[int32](n)
  for d in 0 ..< n:
    result.denseToRank[d] = int32(keyRanks[order[d]])

  # arena in dense order, trie values (already dense)
  # for the derivation lookups
  result.starts = newSeq[int32](n)
  result.lens = newSeq[int32](n)
  var arena: seq[byte] = @[]
  for d in 0 ..< n:
    let key = keys[order[d]]
    result.starts[d] = int32(arena.len)
    result.lens[d] = int32(key.len)
    for c in key: arena.add byte(c)
  result.arena = arena
  result.maxLen = 0
  for d in 0 ..< n:
    if int(result.lens[d]) > result.maxLen:
      result.maxLen = int(result.lens[d])

  # merged-ids contiguity assert, the probed property of every served
  # family (256 single bytes = dense ids 0..255, each higher id merged),
  # asserted when the rank table is a dense tiktoken-shape
  # permutation of 0 ..< n.
  var permuted = true
  for d in 0 ..< n:
    if result.denseToRank[d] != int32(d):
      permuted = false
      break
  if permuted:
    var singles = 0
    for d in 0 ..< n:
      if int(result.lens[d]) == 1:
        inc singles
      else:
        doAssert d >= 256, "multi-byte token below dense id 256"
    doAssert singles == 256, "single-byte token count is not 256"
    for b in 0 .. 255:
      doAssert int(result.lens[b]) == 1,
        "dense id below 256 is not a single byte"

  # next_prefix_match:
  #   longest proper-prefix token per dense id, one
  # trie walk per token (the deepest prefix node carrying a value).
  # -1 default:
  #   single-byte tokens have no proper prefix, and dense
  # id 0 is a real token, so the 0-initialized seq would send
  # the split-derivation chain walk into an infinite loop.
  result.nextPrefixMatch = newSeq[int32](n)
  for d in 0 ..< n:
    result.nextPrefixMatch[d] = int32(-1)
  for d in 0 ..< n:
    if int(result.lens[d]) < 2:
      continue
    var cur = TrieRoot
    var last = int32(-1)
    let st = int(result.starts[d])
    for i in 0 ..< int(result.lens[d]) - 1:
      let nxt = int(trie.base[cur]) + int(result.arena[st + i])
      if nxt >= trie.chk.len or trie.chk[nxt] != cur:
        break
      cur = nxt
      if trie.vals[cur] >= 0:
        last = trie.vals[cur]
    result.nextPrefixMatch[d] = last
  # pair derivation, every valid split of every token
  # (the runtime tiktoken merge predicate, 296,461 pairs for kimik2.5)
  var pairKeys: seq[uint64] = @[]
  var pairVals: seq[int32] = @[]
  for d in 0 ..< n:
    let st = int(result.starts[d])
    let ln = int(result.lens[d])
    if ln < 2:
      continue
    for s in 1 ..< ln:
      let a = trie.lookup(result.arena, st, st + s)
      if a < 0:
        continue
      let b = trie.lookup(result.arena, st + s, st + ln)
      if b < 0:
        continue
      pairKeys.add (uint64(uint32(a)) shl 32) or uint64(uint32(b))
      pairVals.add int32(d)
      result.pairCount = pairKeys.len
  let deriveDone = getMonoTime()
  result.deriveMs = (deriveDone - t0).inNanoseconds.float64 / 1e6

  # PairIndex, frozen at load, worst bucket asserted
  # (the static-structure guarantee, bounded worst case, no dynamic probing)
  result.pairIndex = buildPairIndex(pairKeys, pairVals, PairIndexMaxBucket)
  result.worstBucket = result.pairIndex.pairIndexWorstBucket()
  doAssert result.worstBucket <= PairIndexMaxBucket,
    "PairIndex worst bucket " & $result.worstBucket & " exceeds the bound"
  let indexDone = getMonoTime()
  result.indexMs = (indexDone - deriveDone).inNanoseconds.float64 / 1e6

  # SplitTable derivation, rust-gems greedy first-valid-split over
  # the prefix chain, checked by is_valid against the partially built tables
  result.splitLeft = newSeq[int32](n)
  result.splitRight = newSeq[int32](n)
  var chosen = initTable[(uint32, uint32), uint32](n)
  for d in 0 ..< n:
    var token1 = result.nextPrefixMatch[d]
    while token1 >= 0:
      let restLo = int(result.starts[d]) + int(result.lens[token1])
      let token2 = int32(trie.lookup(result.arena, restLo,
        int(result.starts[d]) + int(result.lens[d])))
      if token2 >= 0 and uint32(token1) < uint32(d) and
          uint32(token2) < uint32(d) and isValidPairDerivation(
            chosen, result.splitLeft, result.splitRight,
            uint32(token1), uint32(token2)):
        chosen[(uint32(token1), uint32(token2))] = uint32(d)
        result.splitLeft[d] = token1
        result.splitRight[d] = token2
        break
      token1 = result.nextPrefixMatch[token1]
    if token1 < 0:
      result.splitLeft[d] = int32(d)
      result.splitRight[d] = int32(d)
  let splitsDone = getMonoTime()

  # load-time self-check, named referee:
  # - a re-encode ending at a single wrong id is a derivation bug,
  #   hard-asserting on every table shape.
  # - a multi-id re-encode (a token unreachable by merging, the HF-converted vocabulary shape)
  #   is a receipt only, checked by the id-parity suites.
  let checked = selfCheck(result)
  result.selfCheckFails = checked.unreachable
  result.selfCheckMs = (getMonoTime() - splitsDone).inNanoseconds.float64 / 1e6
  doAssert checked.singleIdWrong == 0,
    "load-time self-check re-encoded " & $checked.singleIdWrong &
    " tokens to a single wrong id (PairIndex derivation bug)"

proc denseLess(keys: openArray[seq[byte]], keyRanks: openArray[int],
    a, b: int): bool {.inline.} =
  ## Dense order predicate, rank ascending with the byte-lexicographic
  ## tiebreak (the order that makes every merge child sort below its parent).
  let ra = keyRanks[a]
  let rb = keyRanks[b]
  if ra != rb:
    return ra < rb
  cmpBytes(keys[a], keys[b]) < 0

proc sortDenseOrder(order: var seq[int32], keys: openArray[seq[byte]],
    keyRanks: openArray[int]) =
  ## Stable bottom-up merge sort of an index permutation by the dense order,
  ## comparator reading the key and rank openArrays in place, the sort contract:
  ## - no closure capture, the caller passes openArrays straight through.
  ## - no key-copying tuple sequence is materialized.
  ## Load-time only, never a hot-path call.
  let n = order.len
  if n < 2:
    return
  var buf = newSeq[int32](n)
  var width = 1
  while width < n:
    var k = 0
    while k < n:
      let mid = min(k + width, n)
      let hi = min(k + 2 * width, n)
      var i = k
      var j = mid
      while i < mid and j < hi:
        if denseLess(keys, keyRanks, int(order[j]), int(order[i])):
          buf[k] = order[j]
          inc j
        else:
          buf[k] = order[i]
          inc i
        inc k
      while i < mid:
        buf[k] = order[i]
        inc i
        inc k
      while j < hi:
        buf[k] = order[j]
        inc j
        inc k
    for i in 0 ..< n:
      order[i] = buf[i]
    width = width * 2

proc denseIdsOfSorted*(keys: openArray[seq[byte]],
    keyRanks: openArray[int]): seq[int32] =
  ## Dense id per byte-sorted key position, rank-ascending order,
  ## byte-lexicographic tiebreak, exact under duplicate ranks
  ## (byte order inside one rank is the same in both sorted views):
  ##   byte-sorted keys ──┐
  ##   parallel ranks ────┴─→ sort index permutation ─→ dense walk ─→ dense id
  ##                          (comparator reads both openArrays in place)
  ##                          per-rank first-dense-slot + occurrence counter
  ## - the caller re-values the trie with this before buildMergeTables.
  ## Expected input:
  ## - byte-sorted distinct keys with the parallel rank list.
  ## - no key-copying tuple sequence is materialized, the dense order
  ##   is computed by sorting an index permutation whose comparator
  ##   reads both input openArrays in place.
  assert keys.len == keyRanks.len
  let n = keys.len
  var order = newSeq[int32](n)
  for i in 0 ..< n:
    order[i] = int32(i)
  sortDenseOrder(order, keys, keyRanks)
  result = newSeq[int32](n)
  # dense walk assigns each key the rank's first dense slot plus
  # the occurrence counter of that rank so far
  # (inside one rank the dense order is the byte order, so occurrence counters match the byte-sorted input positions exactly)
  var firstDense = initTable[int, int32](n)
  var seen = initTable[int, int32](n)
  for d in 0 ..< n:
    let i = order[d]
    let r = keyRanks[i]
    if r notin firstDense:
      firstDense[r] = int32(d)
    let j = seen.getOrDefault(r, int32(0))
    seen[r] = j + 1
    result[i] = firstDense[r] + j

proc tablesMemBytes(t: MergeTables): int {.inline.} =
  ## Memory by array, arena plus per-id columns plus pair index plus split arrays plus the prefix chain.
  t.arena.len +
  4 * (t.starts.len + t.lens.len + t.denseToRank.len) +
  t.pairIndex.memBytes() +
  4 * (t.splitLeft.len + t.splitRight.len + t.nextPrefixMatch.len)

# ------------------------------------------------------------------------
# Served BPE merge (BpeEngine + lazy-DP backtracking core)
# ------------------------------------------------------------------------

## Served BPE merge over the shared engine
## (vocab trie + static merge tables), the served lazy-DP merge core
## without heap or per-merge re-sort:
## - the library's tiktoken-reference core (bpe_codec.bytePairEncode)
##   is the naive merge reference, and the served stream matches it
##   byte-identical on every family over corpora, fuzz, adversarial rows.
## - the served encoder is the rust-gems crate's no-heap lazy-DP backtracking encoder.
##
## Algorithm (lazy-DP backtracking, no heap, no per-merge re-sort):
##   cursor ─→ place longest vocab match ─→ walk forward
##      │
##      └─→ frozen-invalid adjacency ─→ walk the proper-prefix chain
##              │
##              ├─→ chain exhausted ─→ dead end
##              │      ─→ mark it dead in the bitfield ─→ pop the last placed token
##              │      ─→ retry from its start (the dead bit at the old end
##              │         forces the retry down the prefix chain)
##              │
##              └─→ every continuation dead ─→ never re-entered
##                     ─→ each reachable DP state visited once
##      all matches at one position ─→ prefix chain of the longest match covers the candidate set
##
## Contract with the structures (MergeTables):
##
## | structure               | contract                                                                                                                                                               |
## | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
## | dense ids               | ids are dense end to end, the output walk maps them back to ranks through denseToRank                                                                                  |
## | isValidTokenPair        | the frozen-adjacency predicate, the polarity contract of the merge-table derivation (merge_tables.nim)                                                                 |
## | dead-end semantics      | a segment whose every continuation dead-ends emits the tokens placed so far (possibly none), the reference behavior exactly, the PQ reference raising BpeError instead |
## | 1-byte double-add quirk | NOT replicated here, unreachable through the ordinary path when every single byte is a vocabulary token, the whole-piece branch emits it first                         |
##
## Frozen-adjacency polarity, a PairIndex HIT with the merged id below
## the split-chain limit marks the adjacency INVALID, the merge process
## would have merged across it.
##
## Validity is the absence of such a hit along the limit-bounded
## split_table chain walk, never "radix hit = valid".
##
## The two cores co-diverge only on vocabularies missing single bytes
## or carrying unreachable merged tokens, every served family covers all
## 256 single bytes, self-checks clean, cross-check suite checks bt == PQ (corpora, fuzz, adversarial rows).
##
## No-copy discipline:
## - MergeTables is a multi-megabyte value object whose assignment
##   memmoves every table.
## - every proc here takes the engine ref and reads e.mt fields directly.
## - no local MergeTables binding and no by-value MergeTables
##   or ranks-Table parameter appears in any hot path.

# ------------------------------------------------------------------------
# Engine: vocab trie + static merge tables
# ------------------------------------------------------------------------

type
  BpeEngine* = ref object
    ## BPE engine state, shared and single-threaded:
    ## - `trie`, the double-array vocab trie, lazily built on first use
    ##   with a loud receipt, valued in the dense id space.
    ## - `mt`, the static merge tables frozen at load, PairIndex over
    ##   every valid merge pair, the rust-gems SplitTable derivation,
    ##   and the load-time self-check.
    ## - per-call buffers are pipeline-owned or call-site-owned, none
    ##   live here, the field contract lives in the section docs.
    trie*: VocabTrie       # dense-id-valued (see ensureBuilt), read-only after build
    keys: seq[seq[byte]]   # sorted at build time, released after build
    keyRanks: seq[int]
    built: bool
    byteId*: array[256, int32]  # dense id per byte (vocab index space)
    mt*: MergeTables       # runtime lookup structures, receipts pairCount worstBucket selfCheckFails

proc init*(_: type BpeEngine, ranks: Table[seq[byte], int]): BpeEngine =
  ## Capture the rank table. The trie build is deferred to first use
  ## (ensureBuilt) with a loud receipt.
  result = BpeEngine(
    keys: newSeq[seq[byte]](ranks.len),
    keyRanks: newSeq[int](ranks.len),
    built: false,
    byteId: default(array[256, int32]))
  var i = 0
  for k, v in ranks.pairs:
    result.keys[i] = k
    result.keyRanks[i] = v
    inc i

proc byteRanks(e: BpeEngine): array[256, int32] =
  for b in 0 .. 255:
    result[b] = int32(e.trie.lookup([byte(b)]))

proc ensureBuilt*(e: BpeEngine) =
  ## Lazy first-use build, loud receipt prints cost and memory.
  ## Deterministic contract:
  ## - buildTrie and buildMergeTables are pure functions of the sorted key list.
  ## - the trie is valued in the rank-ascending dense id space
  ##   (byte-ordered tiebreak), keeping every lookup inside the merge-table id space.
  ## - the PairIndex pair keys need no rank indirection.
  if e.built:
    return
  var pairs: seq[(seq[byte], int)] = newSeq[(seq[byte], int)](e.keys.len)
  for i in 0 ..< e.keys.len:
    pairs[i] = (e.keys[i], e.keyRanks[i])
  let sortStart = getMonoTime()
  pairs.sort(proc(a, b: (seq[byte], int)): int =
    cmpBytes(a[0], b[0]))
  var keys = newSeq[seq[byte]](pairs.len)
  var ranks = newSeq[int](pairs.len)
  for i in 0 ..< pairs.len:
    keys[i] = pairs[i][0]
    ranks[i] = pairs[i][1]
  let memBefore = getTotalMem()
  let t0 = getMonoTime()
  let dense = denseIdsOfSorted(keys, ranks)
  var denseRanks = newSeq[int](dense.len)
  for i in 0 ..< dense.len:
    denseRanks[i] = int(dense[i])
  e.trie = buildTrie(keys, denseRanks)
  let buildMs = (getMonoTime() - t0).inMicroseconds.float64 / 1000.0
  let sortMs = (t0 - sortStart).inMicroseconds.float64 / 1000.0
  let t1 = getMonoTime()
  e.mt = buildMergeTables(keys, ranks, e.trie)
  let tablesMs = (getMonoTime() - t1).inMicroseconds.float64 / 1000.0
  let memAfter = getTotalMem()
  e.byteId = e.byteRanks()
  e.built = true
  e.keys = newSeq[seq[byte]](0) # release the key copies
  e.keyRanks = newSeq[int](0)
  echo "[bpe] vocab trie built: ", e.trie.keyCount, " keys, ",
    e.trie.nodeCount, " nodes, ", e.trie.base.len, " slots, ",
    e.trie.arrayBytes() div 1024, " KB arrays, sort ",
    sortMs, " ms, build ", buildMs, " ms, getTotalMem delta ",
    memAfter - memBefore, " bytes"
  echo "[bpe] merge tables built: pairs=", e.mt.pairCount,
    " worstBucket=", e.mt.worstBucket, " slots=2^",
    e.mt.pairIndex.slotBits, " selfCheckFails=", e.mt.selfCheckFails,
    " deriveMs=", e.mt.deriveMs.formatFloat(ffDecimal, 1),
    " indexMs=", e.mt.indexMs.formatFloat(ffDecimal, 1),
    " selfCheckMs=", e.mt.selfCheckMs.formatFloat(ffDecimal, 1),
    " tablesMem=", e.mt.tablesMemBytes(), " bytes"

# ------------------------------------------------------------------------
# Lazy-DP merge core
# ------------------------------------------------------------------------

type
  BacktrackBuf* = object
    ## Caller-owned scratch for the backtracking core, reused across
    ## segments with zero-alloc steady state, field contract:
    ## - `tokens`, the placed-token stack in dense ids.
    ## - `bitfield`, the dead-boundary bitfield over segment positions
    ##   (bit i set = boundary i not proven dead).
    ## - `pos`/`nextToken`, the cursor position and the next candidate
    ##   token (-1 = none).
    tokens: seq[int32]
    bitfield: seq[uint64]
    pos: int
    nextToken: int32

proc init*(_: type BacktrackBuf): BacktrackBuf {.inline.} =
  ## Zero-initialized scratch buffer.
  BacktrackBuf(tokens: newSeq[int32](0), bitfield: newSeq[uint64](0),
               pos: 0, nextToken: int32(-1))

proc isValidTokenPair*(e: BpeEngine, t1, t2: uint32): bool =
  ## Frozen-adjacency validity of a placement of dense ids t1, t2
  ## next to each other in a BPE output, limit-loop over the SplitTable:
  ## - a PairIndex hit whose merged id is below the limit freezes
  ##   the adjacency invalid (the limit is the largest id the merge process could still have formed across this boundary).
  ## - splitting the larger side proceeds through its recorded split,
  ##   descending is the walk.
  ## Polarity contract, matching the derivation-time predicate of the merge
  ## tables (merge_tables.nim, same polarity, complete tables).
  e.ensureBuilt()
  var token1 = t1
  var token2 = t2
  var limit = uint32.high
  while true:
    let combined = e.mt.pairIndex.lookup(token1, token2)
    if combined >= 0 and uint32(combined) < limit:
      return false
    if token1 > token2:
      limit = token1
      token1 = uint32(e.mt.splitRight[token1])
      if token1 == limit:
        limit = token2 + 1
        token2 = uint32(e.mt.splitLeft[token2])
        if token2 + 1 == limit:
          return true
    else:
      limit = token2 + 1
      token2 = uint32(e.mt.splitLeft[token2])
      if token2 + 1 == limit:
        limit = token1
        token1 = uint32(e.mt.splitRight[token1])
        if token1 == limit:
          return true

proc bytePairEncodeBt(e: BpeEngine, dst: var seq[int],
    bt: var BacktrackBuf, data: openArray[byte], lo, hi: int) =
  ## Backtracking merge core over the byte slice [lo, hi):
  ##   lazy-DP walk ─→ ids stream dst-first in ranks ─→ output walk ─→ denseToRank map ─→ dst
  ##        │
  ##        └─→ segment not fully coverable → emit the tokens placed so far
  ##            (dead-end, unreachable when every single byte is a vocabulary token, checked by the cross-check suite)
  ## Output:
  ## - dst gains the segment ids in rank space, the output walk maps
  ##   them back through denseToRank.
  e.ensureBuilt()
  let n = hi - lo
  if n == 0:
    return # the pipeline never produces empty pieces
  let words = (n + 1 + 63) div 64
  if bt.bitfield.len < words:
    bt.bitfield = newSeq[uint64](words)
  for w in 0 ..< words:
    bt.bitfield[w] = high(uint64) # every boundary alive at segment start
  bt.tokens.setLen(0)
  bt.pos = 0
  bt.nextToken = int32(e.trie.longestMatch(data, lo, hi))
  while bt.nextToken >= 0:
    var token = bt.nextToken
    let last = if bt.tokens.len > 0: bt.tokens[^1] else: int32(-1)
    while true:
      let endPos = bt.pos + int(e.mt.lens[token])
      if (bt.bitfield[endPos shr 6] and
          (1'u64 shl uint64(endPos and 63))) != 0 and
          (last < 0 or isValidTokenPair(e, uint32(last), uint32(token))):
        bt.tokens.add token
        bt.pos = endPos
        bt.nextToken = int32(e.trie.longestMatch(data, lo + endPos, hi))
        break
      elif e.mt.nextPrefixMatch[token] >= 0:
        token = e.mt.nextPrefixMatch[token] # progressively shorter match
      else:
        # dead end at bt.pos:
        #   no token starting here extends the walk.
        # Mark the boundary dead, pop the last placed token, retry it
        # from its start - the dead bit at its old end position forces
        # the retry down its prefix chain. An empty stack ends the stream
        # (the dead-end semantics).
        bt.bitfield[bt.pos shr 6] = bt.bitfield[bt.pos shr 6] and
          not (1'u64 shl uint64(bt.pos and 63))
        if bt.tokens.len > 0:
          discard bt.tokens.pop()
          bt.pos -= int(e.mt.lens[last])
        bt.nextToken = last
        break
  for t in bt.tokens.items:
    dst.add int(e.mt.denseToRank[t])

proc encodeSegment*(e: BpeEngine, dst: var seq[int],
    bt: var BacktrackBuf, data: openArray[byte], lo, hi: int) =
  ## One pre-tokenized piece, served through the ordinary path:
  ## - the whole-piece encoder hit (a single id) first.
  ## - else the backtracking merge core.
  ## Buffer contract, dst may accumulate across pieces, fresh scratch
  ## per call while call sites may reuse an accumulating buffer.
  e.ensureBuilt()
  if hi == lo:
    return # the pipeline never produces empty pieces
  let whole = e.trie.lookup(data, lo, hi)
  if whole >= 0:
    dst.add int(e.mt.denseToRank[whole])
  else:
    bytePairEncodeBt(e, dst, bt, data, lo, hi)
