## Tattletale — PagedRadixTrie KV Cache
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.


## PagedRadixTrie — a compressed Radix/Patricia trie keyed by token sequences,
## where split/graft operations are at page granularity (256 tokens).
##
## DESIGN (see kvcache.lean for formalization):
##   The trie provides:
##     - O(prefix) LPM,
##     - O(prefix) interleaved and amortized forking, out of the decode hot-path
##     - O(log(depth)) amortized eviction with liveness guarantees.
##
##   It departs significantly from other implementations
##
##   1. On hashmaps
##
##   It doesn't use hashmaps as whether we use hashes or not, the whole sequence of tokens
##   needs to be processed, but hashmaps need collision resolution and cannot early exit.
##   This is an important optimization since after common system prompts
##   it is very likely that divergence occurs in the first 10~20 tokens.
##
##   Furthermore, after divergence, it's the conversation of an unique users and most are
##   unlikely to create more than 1~2 forks, maybe ~20 when scheduling teams of agents
##   so hashmaps are overkill after 2 forks.
##   The exact sequence of tokens need also to be compared anyway for forking.
##
##   So hashmaps need extra hash compute, save no memory bandwidth, cannot do early exits
##   and need collision resolution.
##
##   2. On amortized operations
##
##   The KVcache implements path compression. Due to uniqueness of LLM conversations
##   this significantly limit the number of nodes and the overhead of the KVCache metadata.
##   It also allows to amortize evictions as evicting a node will likely evict whole conversations.
##
##   Pages added during prefill/decode are accumulated and added to the tree at once.
##   This allows amortizing the trie modifications and remove them from the hot path
##   otherwise the KVCache would limit the DPS / TPS (decode per second / token per second)
##   of Tattletale.
##
##   3. On eviction
##
##   A single two-level data structure is used instead of using a global heapqueue or LRU cache.
##   The heap queue is extra complex iterations, more refcounting overhead.
##   Furthermore while on paper it does O(1) min finding, due to the locking mechanism
##   multiple nodes might have to be popped and reenqueued
##   and you would need to pop everything to realize all the leaves are locked.
##
##   Heapqueues are also problematic with eviction, since you can't delete arbitrarily from a heapqueue
##   so you would need tombstone and it breaks the assumption that GPU pages are returned
##   when the PagedRadixNode destructor is called (it wouldn't since the heap queue would still hold a reference).
##
##   At a high level, statistics about the oldest subtree are maintained and propagated after each graftPages operation
##   Due to path compression and relative uniqueness of prefix after 2 hops
##   Few comparisons O(log(path)) are needed to find an eviction target
##   and those are arguably cheaper than the bookkeeping to maintain O(1) in a binary heap.
##
##   Then once the at the leaves, each node maintains an intrusive WAVL tree that complements the children seq
##   and allow selecting the oldest unlocked child in O(lg n)
##
##   Furthermore eviction does not use an external clock as this requires a kernel syscall
##   thrashing the cache and easily introducing ~100ns of latency. You could do 500+ instructions
##   in that time on a 5GHz CPU.
##
##   4. On LPM dispatch — intrusive WAVL child index
##
##   A node with many children (e.g. root with 100K conversations) previously
##   scanned every child via `getCommonFirstPageLen` — O(N) per LPM, costing
##   690 µs for a miss in a 100K-child tree.
##
##   In IP-routing lc-tries are used (level-compressed Radix Tries).
##   An experiment was done with stride-16 lc-trie, which allowed extremely fast LPM (~7ns)
##   however deletion would require tombstones in the children seq to maintain index stability
##   which means unbounded memory growth for the root node.
##
##   WAVL trees are used here in a novel way — as a *longest prefix match*
##   index, not just a BST for exact-key lookups.  The comparator returns the
##   signed position of the first token mismatch (not plain -1/0/+1).  This
##   lets `wavlFindBestMatch` — itself a novel template — return the neighbor
##   with the longer shared prefix in O(log n), with zero redundant comparison
##   (the match length was already computed during BST navigation).
##
##   LPM descent uses O(log n) WAVL traversal with a full-page comparator.
##   `wavlFindBestMatch` returns the exact match or the best neighbor (pred or
##   succ with larger |cmp|, positive on tie) — the comparator's signed divergence
##   point gives the match length directly, zero redundant work.
##   Both hit and miss are pure O(log n).
##   The result (100K-child flat tree):
##   - LPM hit: 53 ns (was 9 ns stride‑16, still <0.1 µs)
##   - LPM miss: 32 ns (was 8 ns stride‑16, 690 µs original)
##   - graftPages last child: 203 ns (was 309 ns stride‑16)
##   - Tree build for 100K leaves: 47 ms (was 33 ms stride‑16)
##
##   The trade‑off: stride‑16's O(1) lookup is slightly faster for very wide
##   flat trees, but WAVL avoids the 256 KB allocation per node and actually
##   beats stride‑16 on deep trees (chain LPM −60%) where the per‑descent‑step
##   overhead of stride‑16 accumulates.
##
##
## Usage:
##   var cache: KVCache[TokenID, PageIdx]
##   let prefix = cache.lpm(tokens)    # pure query, locks path, returns matched pages
##   # ── caller uses tokens + pages however they like (prefill, decode, ...) ──
##   # ── when done, pass EVERYTHING back to the trie ──
##   cache.graftPages(allTokens, allPages)  # trie sorts out matching/fork/append
##   # eviction:
##   while not cache.evictable():
##     cache.evict()
##
## Invariants:
## - It is a PagedRadixTrie, the unit is a page of 256 tokens
##     Fork only happen at 256 token boundaries. Eviction at 256 token boundaries.
##     Data is stored as a sequence of tokens instead of seq[array[256, token]] for convenience
## - LPM (LongestPrefixMatch) doesn't modify the tree structure.
##   During traversal, the path taken is locked.
##   Assumption:
##     the path taken will always be followed up either by a prefill request
##     or a decode request and so it is active and is not candidate for eviction
## - graftPages
##     is done when a prefill or a decode request fills a page
##
## - It is a Patricia tree and implements path compression.
##   A parent usually cannot have a single child, it's 2 or more.
##   - graftPages appends pages directly to the parent if no fork are required
##   - evict would merge a child and its parent if the last sibling is evicted
##
## - Once grafted on the tree, pages are immutable.
##   A partially filled pages will be copied-on-write.
##   It will naturally be evicted when it's last access is the oldest.

import workspace/data_structures

# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════

const TokensPerPage* = 256
  ## Page granularity: all split/graft operations align to this.


type
  PagedRadixNode*[T, P] = ref object
    ## A node in the PagedRadixTrie.
    ## Tokens are compared per page of 256 tokens (TokensPerPage)
    ## Splitting can only occur are 256 token boundaries.
    ##
    ## The paged radix tree is augmented to provide low worst-case latency for:
    ## - Longest Prefix Match
    ## - Update: Forking and appending
    ## - Eviction
    ##
    ## It supports:
    ## - interleaved prefill and decode,
    ## - batch enqueueing pages
    ## - Deterministic bounded memory usage: deletion leaves no tombstone
    ## - No worst-case linear-time
    ## - No rebuilding, compacting, reindexing

    # ──── Functional fields ────────────────────────────────────────────────
    tokens: seq[T]
    pages: seq[P]            # Sequence of page indices attached to this leaf
    depth_in_pages: int32    # Shared number of pages before entering this node

    # ──── Trie structure ───────────────────────────────────────────────────
    children: seq[PagedRadixNode[T, P]]
    parent {.cursor.}: PagedRadixNode[T, P]
    childId: int32 = -1
      # index in parent's children seq (-1 = root)
      # This is a small optimization to avoid linear scan
      # when a node removes itself from its parent's children seq

    # ──── Acceleration structures — Intrusive WAVL tree ────────────────────
    # To avoid linear scanning in children for longest-prefix match or for eviction
    # we maintain 2 intrusive WAVL trees that maps an ordered property -> child index
    # This allows finding the best candidate in O(lg n).
    # Inserting and removing items is also O(lg n)
    # Furthermore in eviction if the best candidate is locked we can follow the next best.
    # Due to supporting online deletion (i.e. without compaction/rebuilding)
    # we have worst-case O(lg n) latency guarantees and no need for tombstones
    lpmLinks: seq[WavlLink]      # WAVL links for LPM child tree
    lpmRoot: int32 = WavlNil     # root of LPM WAVL tree (-1 = empty)
    evictLinks: seq[WavlLink]    # WAVL links for eviction child tree
    evictRoot: int32 = WavlNil   # root of eviction WAVL tree (-1 = empty)

    # ──── Cumulative metadata from subtree, current node included ──────────
    subtree_sum_pages: int32       # number of page in this subtree
    subtree_oldest_decode: uint64
    subtree_sum_locked: int32      # >0 means on an active decode path
    subtree_sum_leaves: int32      # 1 if leaf, else sum(child.subtree_sum_leaves)

  KVCache*[T, P] = ref object
    root: PagedRadixNode[T, P]
    kvClock: uint64

  LongestPrefixMatch*[P] = object
    ## Result of LPM. The returned `node` is locked and ready for graftPages.
    ## Walk up via `node.parent` to reach root.
    pages*: seq[P]
    totalTokenMatched*: int     # total tokens matched from root
    lastLevelMatched*: int      # tokens matched from the last page (to handle partial matches)

  KVCacheDefect = object of Defect

func new*[T, P](_: typedesc[KVCache[T, P]]): KVCache[T, P] =
  KVCache[T, P](
    root: PagedRadixNode[T, P](),
    kvClock: 0
  )

# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

func ceilDiv*(num, denom: int): int {.inline.} =
  (num + denom - 1) div denom

func isPowerOf2(n: SomeInteger): bool {.inline.} =
  ## Returns true if n is a power of 2
  (n and (n - 1)) == 0 and n > 0

func round_step_down(x: int, step: static int): int {.inline.} =
  ## Round the input to the previous multiple of "step"
  when step.isPowerOf2():
    # Step is a power of 2. (If compiler cannot prove that x>0 it does not make the optim)
    result = x and not(step - 1)
  else:
    result = x - x mod step

func round_step_up(x: int, step: static int): int {.inline.} =
  ## Round the input to the next multiple of "step"
  when step.isPowerOf2():
    # Step is a power of 2. (If C compiler cannot prove that x>0 it does not make the optim)
    result = (x + step - 1) and not(step - 1)
  else:
    result = ceilDiv(x, step) * step

# ──── Trie properties ────────────────────────────────────────────────────────────────────────

template isLeaf[T, P](n: PagedRadixNode[T, P]): bool = n.children.len == 0
template isLocked[T, P](n: PagedRadixNode[T, P]): bool = n.subtree_sum_locked > 0
template evictable[T, P](n: PagedRadixNode[T, P]): bool = n.isLeaf and not n.isLocked

func depthInTokens*[T, P](n: PagedRadixNode[T, P]): int {.inline.} =
  ## Number of tokens from root to the start of this node's data.
  n.depth_in_pages.int * TokensPerPage

func depthInTokens*[T, P](n: PagedRadixNode[T, P]; offsetTokens: int): int {.inline.} =
  ## Number of tokens from root to `offsetTokens` tokens into this node's data.
  n.depthInTokens + offsetTokens

func totalTokenCount*[T, P](n: PagedRadixNode[T, P]): int {.inline.} =
  ## Total number of tokens from root to the end of this node's data.
  n.depthInTokens + n.tokens.len

func getCommonFirstPageLen[T](a, b: openArray[T]): int {.inline.}

# ──── Child index (intrusive WAVL tree) ──────────────────────────────────

func lpmCmp[T](a, b: openArray[T], aPos = 0): int32 {.inline.} =
  ## WAVL comparator: strict total order on the full first page.
  ## Returns signed divergence point:
  ##   0       → identical first pages (exact match)
  ##   -(i+1)  → a[i] < b[i]; matched i tokens
  ##   +(i+1)  → a[i] > b[i]; matched i tokens
  ##   -(N+1)  → a is prefix of b; matched N tokens (N = min lengths)
  ##   +(N+1)  → b is prefix of a; matched N tokens
  ## INV A1 guarantees children diverge within the first page, so 0 only
  ## occurs when comparing input against a matching child (cache hit).
  let aLen = a.len - aPos
  let cmpLen = min(min(aLen, b.len), int(TokensPerPage))
  for i in 0 ..< cmpLen:
    if a[aPos + i] < b[i]: return -int32(i + 1)
    elif a[aPos + i] > b[i]: return int32(i + 1)
  if aLen < b.len: return -int32(cmpLen + 1)  # a is prefix of b
  elif aLen > b.len: return int32(cmpLen + 1)  # b is prefix of a
  return 0  # identical first pages (only between input and child, never two children)

proc addChild*[T, P](n: PagedRadixNode[T, P]; child: PagedRadixNode[T, P]) =
  ## Add a child node, maintaining the WAVL child index incrementally.
  n.children.add(child)
  let idx = int32(n.children.len - 1)
  child.childId = idx
  # Nim setLen uses exponential growth (×2 up to 32K, ×1.5 beyond),
  # matching children.add(). The link seqs stay in lockstep.
  n.lpmLinks.setLen(n.children.len)
  n.evictLinks.setLen(n.children.len)
  # INV A1 guarantees unique first-page keys — no tiebreaker needed
  wavlInsertTpl(n.lpmLinks, n.lpmRoot, idx):
    lpmCmp(n.children[a].tokens, n.children[b].tokens)
  wavlInsertTpl(n.evictLinks, n.evictRoot, idx):
    let ca = n.children[a]; let cb = n.children[b]
    if ca.subtree_oldest_decode < cb.subtree_oldest_decode: -1
    elif ca.subtree_oldest_decode > cb.subtree_oldest_decode: 1
    else: cmp(a, b)

proc findChild*[T, P](n: PagedRadixNode[T, P];
                       input: openArray[T]; pos: int): PagedRadixNode[T, P] =
  ## Find the best-matching child for input starting at pos.
  ##
  ## Uses WAVL with full-page key + predecessor/successor check.
  ## O(log n) for both cache hit and cache miss.
  if n.lpmRoot < 0: return nil
  let (found, best, bestCmp) = wavlFindBestMatch(n.lpmLinks, n.lpmRoot):
    lpmCmp(input, n.children[ti].tokens, pos)
  if found >= 0:
    return n.children[found]
  if best >= 0 and abs(bestCmp) - 1 > 0:  # bestCmp = signed divergence point; 0 → no match
    return n.children[best]
  return nil

{.push checks: off.}

func getCommonFirstPageLen[T](a, b: openArray[T]): int {.inline.} =
  ## Children diverge after first 256-token page (INV A1).
  let n = min(min(a.len, b.len), TokensPerPage)
  for i in 0 ..< n:
    if a[i] != b[i]:
      return i
  return n
func getCommonPrefixLen[T](data, prefix: openArray[T]): int {.inline.} =
  let n = min(data.len, prefix.len)
  for i in 0..<n:
    if data[i] != prefix[i]:
      return i
  return n

{.pop.}

# ──── Trie walk ────────────────────────────────────────────────────────────────────────

iterator walkUp[T, P](n: PagedRadixNode[T, P]): PagedRadixNode[T, P] =
  ## Walk up from node to root via parent pointers.
  var p {.cursor.} = n
  while p != nil:
    yield p
    p = p.parent

template walkDown[T, P](cache: KVCache[T, P]; tokens: openArray[T], prologueBody, descentIntoNodeBody, processMatchAndExitBody: untyped): untyped =
  ## walkDown the trie, following the `tokens` path
  ## and applying:
  ## - `descentIntoNodeBody` when descending into a new node level
  ## - `processMatchAndExitBody` when the best match is found
  ##
  ## Available injected variable for the caller are:
  ## - node
  ## - pos,       longest prefix match
  ## - numShared, shared prefix at last level
  ## - best,      the best children before recursion
  var node {.cursor, inject.} = cache.root
  var pos {.inject.} = 0

  prologueBody

  while true:
    # Invariants:
    #  1. We compare and split prefix at page level
    #  2. if we descended into this node
    #     we already checked a page worth of tokens at the previous level
    #  3. if we descended into this child
    #     it is at least TokensPerPage large.
    #  4. Children smaller than TokensPerPage are leaves and never written into.
    #     Due to page immutability and Copy-on-Write their parent forks and a sibling is created.
    #  TODO: We can still reuse their KV projections to avoid recomputing them.

    # Handle partial match (leaf or empty root)
    # -------------------------------------
    var numShared {.inject.}: int
    if node.tokens.len < TokensPerPage:
      let shared = getCommonPrefixLen(tokens.toOpenArray(pos, tokens.high), node.tokens)
      numShared = shared
      pos += shared
      if node.children.len == 0:
        # Leaf node — this is the best match
        processMatchAndExitBody
        break
      elif shared == node.tokens.len:
        # Root with fully matched content and children — check children
        discard
      else:
        # Root with partially matched content — best match found
        processMatchAndExitBody
        break
    else:
      # Compare from the start of this node's data — root may not have had
      # its first page pre-confirmed (children always have, via the descent's
      # getCommonFirstPageLen, so the first-page compare is a quick no-op).
      let firstPageShared = getCommonPrefixLen(
        tokens.toOpenArray(pos, tokens.high),
        node.tokens.toOpenArray(0, min(TokensPerPage, node.tokens.len) - 1)
      )
      if firstPageShared < TokensPerPage:
        numShared = firstPageShared
      elif node.tokens.len > TokensPerPage:
        numShared = TokensPerPage + getCommonPrefixLen(
          tokens.toOpenArray(pos + TokensPerPage, tokens.high),
          node.tokens.toOpenArray(TokensPerPage, node.tokens.high)
        )
      else:
        numShared = TokensPerPage

    pos += numShared

    # Guard: all input tokens consumed — stop before the child loop.
    # Without this, root with <256 tokens that matches all input would enter
    # the child loop with pos >= tokens.len, causing toOpenArray(pos, high) OOB
    # (e.g. pos=10, tokens.len=10 → tokens.toOpenArray(10, 9) crashes).
    if pos >= tokens.len:
      processMatchAndExitBody
      break

    # Match children for deeper descent
    # -------------------------------------
    var matched = false
    if node.children.len != 0:
      var best {.inject.}: PagedRadixNode[T, P]
      let found = node.findChild(tokens, pos)
      if found != nil:
        best = found
        descentIntoNodeBody
        matched = true

    if not matched:
      # Fallback, no child matches (or no children), this is the best node
      processMatchAndExitBody
      break

# ═══════════════════════════════════════════════════════════════════════════
# LPM — Longest Prefix Match (Read-only)
# ═══════════════════════════════════════════════════════════════════════════
#
# Lock lifecycle:
#   Prologue:      inc root.subtree_sum_locked
#   Each descent:  inc node.subtree_sum_locked
#   ── locks are released by the matching graftPages call ──
#
# Pages returned: sliced to the matched portion only (ceilDiv(numShared, 256)),
# NOT the full node's page array.  This avoids leaking page indices beyond
# the prefix that was actually matched.
#
# NOTE: The caller MUST call graftPages with the FULL token sequence and ALL
# pages to release the lock.  The trie handles matching, forking, and appending
# internally — you just pass back everything and let classifyGraft decide.
#
proc lpm*[T, P](cache: KVCache[T, P]; tokens: openArray[T]): LongestPrefixMatch[P] =
  cache.walkDown(tokens):
    # PrologueBody
    var pages: seq[P]
    inc cache.root.subtree_sum_locked
  do: # DescentIntoNodeBody -- full match
    # Add all pages
    pages.add node.pages

    # Descend into best child node
    node = best
    inc node.subtree_sum_locked
  do: # processMatchAndExitBody
    let matchPages = ceilDiv(numShared, TokensPerPage)
    if matchPages > 0:
      pages.add node.pages.toOpenArray(0, matchPages - 1)
    return LongestPrefixMatch[P](
      pages: pages,
      totalTokenMatched: pos,
      lastLevelMatched: numShared
    )

# ═══════════════════════════════════════════════════════════════════════════
# Update (graftPages)
# ═══════════════════════════════════════════════════════════════════════════
#
# The 5-branch dispatch is governed by `classifyGraft` (see below).
# Each case is handled by a dedicated proc:
#
#   gcFullMatch     → fullMatchOp        (update timestamps, release locks)
#   gcPartialMatch  → partialMatchOp     (COW sibling at sub-page fork)
#   gcRootNewChild  → rootNewChildOp     (add direct child under root)
#   gcFork          → forkPageOp         (newParent + 2 children)
#   gcAppend        → appendOp           (extend node in-place)
#
# All procs follow the same lock-release pattern:
#   target.subtree_sum_locked -= 1  +  walkUp from parent/ancestor
#
type GraftCase* = enum
  gcFullMatch, gcPartialMatch, gcRootNewChild, gcFork, gcAppend

func classifyGraft*(targetMatchLen, tokensLen, lastLevel, targetTokLen: int;
                    hasParent, rootHasChildren: bool): GraftCase =
  ## Pure-function decision: which graftPages branch applies?
  ## Each branch covers exactly one scenario — gcAppend is the fallthrough.
  if targetMatchLen == tokensLen:                        gcFullMatch
  elif lastLevel < TokensPerPage and hasParent and lastLevel == targetTokLen:
    gcPartialMatch
  elif lastLevel < TokensPerPage and not hasParent and rootHasChildren:
    gcRootNewChild
  elif lastLevel < targetTokLen:                         gcFork
  else:                                                  gcAppend

func walkUpUpdate[T, P](
    cache: KVCache[T, P];
    startNode: PagedRadixNode[T, P];
    pagesDelta, leavesDelta: int) =
  ## Walk up from startNode to root, re-keying eviction trees
  ## (oldest_decode changed), releasing locks, and updating counters.
  for up in startNode.walkUp():
    let upParent = up.parent
    if upParent != nil and upParent.evictRoot >= 0:
      wavlDelete(upParent.evictLinks, upParent.evictRoot, up.childId)
    up.subtree_oldest_decode = cache.kvClock
    if upParent != nil:
      wavlInsertTpl(upParent.evictLinks, upParent.evictRoot, up.childId):
        let ea = upParent.children[a]; let eb = upParent.children[b]
        if ea.subtree_oldest_decode < eb.subtree_oldest_decode: -1
        elif ea.subtree_oldest_decode > eb.subtree_oldest_decode: 1
        else: cmp(a, b)
    up.subtree_sum_locked -= 1
    up.subtree_sum_pages += pagesDelta.int32
    up.subtree_sum_leaves += leavesDelta.int32

# ──── Branch procs ──────────────────────────────────────────────────────
proc fullMatchOp[T, P](cache: var KVCache[T, P]; target: PagedRadixNode[T, P]) =
  ## All input already in cache — update timestamps, release locks.
  # Re-key target in parent's eviction tree
  let p = target.parent
  if p != nil and p.evictRoot >= 0:
    wavlDelete(p.evictLinks, p.evictRoot, target.childId)
  target.subtree_oldest_decode = cache.kvClock
  if p != nil:
    wavlInsertTpl(p.evictLinks, p.evictRoot, target.childId):
      let ca = p.children[a]; let cb = p.children[b]
      if ca.subtree_oldest_decode < cb.subtree_oldest_decode: -1
      elif ca.subtree_oldest_decode > cb.subtree_oldest_decode: 1
      else: cmp(a, b)
  target.subtree_sum_locked -= 1
  cache.walkUpUpdate(target.parent, pagesDelta = 0, leavesDelta = 0)

proc partialMatchOp[T, P](cache: var KVCache[T, P];
    target: PagedRadixNode[T, P]; tokens: openArray[T]; pages: openArray[P];
    lastLevelMatched: int) =
  ## Create a COW sibling at the divergence point (sub-page fork).
  ## All of target's content was matched; new input diverges within the
  ## first page of the sibling.  Sibling carries the new tail from
  ## target.depthInTokens + lastLevelMatched onward.
  let siblingTokenStart = target.depthInTokens + lastLevelMatched
  let siblingPageStart  = target.depth_in_pages.int
  let siblingPages      = pages.len - siblingPageStart
  let sibling = PagedRadixNode[T, P](
    tokens: tokens[siblingTokenStart .. ^1],
    pages: pages[siblingPageStart .. ^1],
    parent: target.parent,
    depth_in_pages: target.depth_in_pages,
    lpmRoot: WavlNil,
    evictRoot: WavlNil,
    subtree_sum_pages: siblingPages.int32,
    subtree_oldest_decode: cache.kvClock,
    subtree_sum_leaves: 1)
  target.parent.addChild(sibling)
  target.subtree_sum_locked -= 1
  cache.walkUpUpdate(sibling.parent, pagesDelta = siblingPages, leavesDelta = 1)

proc rootNewChildOp[T, P](cache: var KVCache[T, P];
    tokens: openArray[T]; pages: openArray[P]) =
  ## Root is a branching node (empty content, children populated) and the
  ## input shares zero prefix with any existing child. Add as a direct child.
  let sibling = PagedRadixNode[T, P](
    tokens: @tokens, pages: @pages, parent: cache.root,
    depth_in_pages: 0,
    subtree_sum_pages: pages.len.int32,
    subtree_oldest_decode: cache.kvClock, subtree_sum_leaves: 1)
  cache.root.addChild(sibling)
  cache.walkUpUpdate(cache.root, pagesDelta = pages.len, leavesDelta = 1)

proc forkPageOp[T, P](cache: var KVCache[T, P];
    target: PagedRadixNode[T, P]; tokens: openArray[T]; pages: openArray[P];
    lastLevelMatched: int) =
  ## Split the target node at a page boundary.  Creates a newParent that
  ## holds the shared prefix, with target (old tail) and sibling (new input)
  ## as its two children.
  let newParent = PagedRadixNode[T, P](
    tokens: move(target.tokens), pages: move(target.pages),
    parent: target.parent, depth_in_pages: target.depth_in_pages,
    subtree_sum_pages: target.subtree_sum_pages,
    subtree_sum_leaves: target.subtree_sum_leaves,
    subtree_sum_locked: target.subtree_sum_locked)

  let llBranchingPoint = lastLevelMatched.round_step_down(TokensPerPage)
  let llForkedPageOffset = (llBranchingPoint div TokensPerPage).int32
  let numCutPages = (target.pages.len - llForkedPageOffset.int).int32
  target.tokens = newParent.tokens[llBranchingPoint..^1]
  target.pages = newParent.pages[llForkedPageOffset..^1]
  target.parent = newParent
  target.depth_in_pages += llForkedPageOffset
  target.subtree_sum_pages -= numCutPages
  target.subtree_sum_locked -= 1

  newParent.tokens.setLen(llBranchingPoint)
  newParent.pages.setLen(llForkedPageOffset)

  let siblingTokenStart = target.depthInTokens
  let siblingPageStart  = target.depth_in_pages.int
  let extraPages        = (pages.len - siblingPageStart).int32
  let sibling = PagedRadixNode[T, P](
    tokens: tokens[siblingTokenStart .. ^1],
    pages: pages[siblingPageStart .. ^1],
    depth_in_pages: target.depth_in_pages,
    parent: newParent,
    subtree_sum_pages: extraPages,
    subtree_oldest_decode: cache.kvClock, subtree_sum_leaves: 1)

  newParent.addChild(target)
  newParent.addChild(sibling)

  if newParent.parent == nil:
    cache.root = newParent
  else:
    let grandparent {.cursor.} = newParent.parent
    grandparent.children[target.childId] = newParent
    newParent.childId = target.childId

  cache.walkUpUpdate(newParent, pagesDelta = extraPages, leavesDelta = 1)

proc appendOp[T, P](cache: var KVCache[T, P];
    target: PagedRadixNode[T, P]; tokens: openArray[T]; pages: openArray[P]) =
  ## Extend the target node with new tokens/pages.
  ## lastLevelMatched == target.tokens.len is guaranteed by classifyGraft.
  let existingPagesTotal = target.depth_in_pages.int + target.pages.len
  let existingTokensTotal = target.depthInTokens + target.tokens.len
  let extraPages = (pages.len - existingPagesTotal).int32
  target.tokens.add tokens.toOpenArray(existingTokensTotal, tokens.high)
  target.pages.add pages.toOpenArray(existingPagesTotal, pages.high)
  target.subtree_sum_pages += extraPages
  # Re-key target in parent's eviction tree
  let p = target.parent
  if p != nil and p.evictRoot >= 0:
    wavlDelete(p.evictLinks, p.evictRoot, target.childId)
  target.subtree_oldest_decode = cache.kvClock
  if p != nil:
    wavlInsertTpl(p.evictLinks, p.evictRoot, target.childId):
      let ca = p.children[a]; let cb = p.children[b]
      if ca.subtree_oldest_decode < cb.subtree_oldest_decode: -1
      elif ca.subtree_oldest_decode > cb.subtree_oldest_decode: 1
      else: cmp(a, b)
  target.subtree_sum_locked -= 1
  if target.subtree_sum_leaves == 0:
    target.subtree_sum_leaves = 1
  cache.walkUpUpdate(target.parent, pagesDelta = extraPages, leavesDelta = 0)

# ═══════════════════════════════════════════════════════════════════════════
# graftPages — public API
# ═══════════════════════════════════════════════════════════════════════════
#
# CONTRACT:
#
#   The caller manages exactly two things:
#     tokens: the COMPLETE token sequence for this request
#     pages:  ALL GPU page indices for the FULL token range
#
#   Call graftPages ONCE at sequence end (or whenever pages fill) by passing
#   back the FULL tokens + pages.  The trie internally:
#
#     1. walkDown(tokens)  — finds the deepest matching node
#     2. classifyGraft      — decides: fullMatch? partialMatch? fork? append?
#     3. branch proc        — attaches pages to the tree, releases locks
#
#   This means:
#     - Callers do NOT need to track "which pages are cached vs new"
#     - Callers do NOT need to reason about fork points or page boundaries
#     - The trie handles all tree-structure invariants internally
#     - Sequences can be started/finished in ANY order (no ordering assumption)
#     - The trie's locking (subtree_sum_locked) prevents eviction of active paths
#       regardless of interleaving
#
# Lock lifecycle:
#   lpm()  → inc subtree_sum_locked on each visited node
#   graftPages() → branch proc decrements subtree_sum_locked via walkUpUpdate
#   Locks are always released by graftPages. There is no separate unlock.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                     graftPages (after walkDown)                        │
# ├─────────────────────────────────────────────────────────────────────────┤
# │  classifyGraft: decision table                                        │
# │                                                                         │
# │  Condition                                 │ Branch    │ Action        │
# │  ──────────────────────────────────────────┼───────────┼───────────────│
# │  targetMatchLen == tokens.len              │ fullMatch │ update clock  │
# │  lastLevel < 256 AND parent!=nil           │ partial   │ COW sibling   │
# │    AND lastLevel == target.tokens.len      │ Match     │ at divergence  │
# │  ──────────────────────────────────────────┼───────────┼───────────────│
# │  lastLevel < 256 AND parent==nil           │ rootNew   │ add child     │
# │    AND children.len > 0                    │ Child     │ to root       │
# │  ──────────────────────────────────────────┼───────────┼───────────────│
# │  lastLevel < target.tokens.len             │ forkPage  │ newParent +   │
# │                                            │           │ 2 children    │
# │  ──────────────────────────────────────────┼───────────┼───────────────│
# │  (fallthrough)                             │ append    │ extend node   │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Lock release per branch:
#   fullMatch      target.subtree_sum_locked -= 1   + walkUp target.parent
#   partialMatch   target.subtree_sum_locked -= 1   + walkUp sibling.parent
#   rootNewChild   walkUp cache.root (handles root lock)
#   forkPage       target.subtree_sum_locked -= 1   + walkUp newParent
#   append         target.subtree_sum_locked -= 1   + walkUp target.parent
#
proc graftPages*[T, P](
      cache: var KVCache[T, P],
      tokens: openArray[T],
      pages: sink openArray[P]) =
  ## Commit a token sequence and its pages into the trie.
  ## Takes ownership of `pages`.
  ##
  ## Call this with the COMPLETE `tokens` and matching `pages` for this
  ## sequence.  The trie already holds some of these pages from earlier
  ## sequences (matched via LPM) — it sorts out which are new vs cached
  ## via classifyGraft.  You do NOT need to separate them.
  ##
  ## The lock acquired by lpm() is released as a side effect.

  var target {.cursor.}: PagedRadixNode[T, P]
  var targetMatchLen: int
  var lastLevelMatched: int

  cache.walkDown(tokens):
    discard
  do:
    node = best
  do:
    target = node
    targetMatchLen = pos
    lastLevelMatched = numShared

  cache.kvClock += 1

  case classifyGraft(targetMatchLen, tokens.len, lastLevelMatched,
                      target.tokens.len, target.parent != nil,
                      target.children.len > 0):
  of gcFullMatch:     fullMatchOp(cache, target)
  of gcPartialMatch:  partialMatchOp(cache, target, tokens, pages, lastLevelMatched)
  of gcRootNewChild:  rootNewChildOp(cache, tokens, pages)
  of gcFork:          forkPageOp(cache, target, tokens, pages, lastLevelMatched)
  of gcAppend:        appendOp(cache, target, tokens, pages)

# ═══════════════════════════════════════════════════════════════════════════
# Eviction
# ═══════════════════════════════════════════════════════════════════════════

proc compressPath(parent: PagedRadixNode) =
  if parent.children.len != 1:
    raise newException(
      KVCacheDefect,
      "[ttt] KVCache invariant violated: tried to compress a parentNode that didn't have an unique child. This is an internal bug.")

  let only = parent.children[0]
  # Absorb the child's tokens and payload into parent
  parent.tokens.add(only.tokens)
  parent.pages.add(only.pages)
  parent.children = only.children
  parent.subtree_sum_leaves = only.subtree_sum_leaves
  parent.subtree_oldest_decode = only.subtree_oldest_decode
  parent.subtree_sum_locked = only.subtree_sum_locked
  # Copy WAVL trees from the absorbed child (same indices as new children seq).
  # No rebuild needed — only's children become parent's children with same indices.
  parent.lpmLinks = only.lpmLinks
  parent.lpmRoot = only.lpmRoot
  parent.evictLinks = only.evictLinks
  parent.evictRoot = only.evictRoot
  # Fix parent pointers of grandchildren (childId stays correct — same indices)
  for gc in parent.children:
    gc.parent = parent

  # Recursive compress — grandparent may now be a single-child node
  # Fixes P3: Patricia tree invariant (no node has exactly 1 child).
  #
  # Note: subtree_oldest_decode is NOT re-keyed in grandparent's eviction
  # tree because it hasn't changed — parent and only were both updated to
  # the same kvClock during the last graftPages walk-up on this path.
  let gp = parent.parent
  if gp != nil and gp.children.len == 1:
    gp.compressPath()

proc findEvictionCandidate[T, P](cache: KVCache[T, P]): PagedRadixNode[T, P] =
  ## Find the coldest unlocked leaf for eviction.
  ## Uses the eviction WAVL tree for O(lg n) coldest-child selection.
  ## If the coldest child has all leaves locked, tries the next
  ## via wavlNext. Returns nil if no evictable child exists.
  ## Decrements subtree_sum_leaves on the way down.
  var n = cache.root
  if n.subtree_sum_locked >= n.subtree_sum_leaves:
    return nil
  while true:
    if n.isLeaf and not n.isLocked:
      return n
    # Scan children via eviction tree until finding an evictable one
    var candidateIdx = wavlMin(n.evictLinks, n.evictRoot)
    while candidateIdx >= 0:
      let candidate = n.children[candidateIdx]
      if candidate.subtree_sum_leaves > candidate.subtree_sum_locked:
        n = candidate
        n.subtree_sum_leaves -= 1
        break
      candidateIdx = wavlNext(n.evictLinks, candidateIdx)
    if candidateIdx < 0:
      return nil

proc evict*[T, P](cache: var KVCache[T, P]): int =
  ## Evict a leaf from the tree. Must be evictable (leaf + unlocked).
  ## Returns the number of pages freed (leaf.pages.len before removal).
  ##
  ## This implies:
  ## - leaf is removed from parent's children
  ## - subtree_sum_leaves is decremented all the way to root
  ## - if parent has 1 child left, parent is merged back (path compression)
  ##
  ## Exception:
  ## - Upon eviction, the root node is replaced with a new one
  ##   with no pages, no children.
  ##
  ## INVARIANTS: A5 maintained (counters accurate), C2 unchanged (unlocked).

  let leaf = cache.findEvictionCandidate()
  if leaf.isNil():
    return 0

  result = leaf.pages.len  # capture pages count before removal

  let parent {.cursor.} = leaf.parent
  if parent == nil:
    # Invariant, there is always a root node.
    # Replace the current one to drop all pages.
    # This can happen if there is a single conversation that consumes the whole KV cache
    # and a new one is restarted.
    cache.root = PagedRadixNode[T, P]()
    return

  # Remove leaf from parent's children — WAVL removal dance
  let idx = leaf.childId
  if idx < 0 or idx >= parent.children.len or parent.children[idx] != leaf:
    raise newException(
      KVCacheDefect,
      "[ttt] KVCache invariant violated: leaf detached from its parent. This is an internal bug.")

  let lastIdx = int32(parent.children.len - 1)

  # 1. Remove from LPM and eviction WAVL trees
  wavlDelete(parent.lpmLinks, parent.lpmRoot, idx)
  wavlDelete(parent.evictLinks, parent.evictRoot, idx)

  # 2. seq.del from children seq, fixing childId of swapped child
  parent.children.del(idx)
  if idx != lastIdx:
    parent.children[idx].childId = idx

  # 3. seq.del + fixLinks for both WAVL trees
  parent.lpmLinks.del(idx)
  fixLinksAfterIndexRemap(parent.lpmLinks, parent.lpmRoot, lastIdx, idx)
  parent.evictLinks.del(idx)
  fixLinksAfterIndexRemap(parent.evictLinks, parent.evictRoot, lastIdx, idx)

  # If parent now has exactly 1 child, merge it upward (path compression)
  # to maintain the invariant, 0 child, 2 children or more, never one.
  # A child can be merged into root (single user with only enough KV-cache for a single conversation)
  if parent.children.len == 1:
    parent.compressPath()

# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════
#
# Formal specification for PagedRadixTrie invariants (Nim)
# Mirrors the Lean formalization (kvcache.lean) and Patricia trie invariants.
#
# Invariants checked:
#  A1. Prefix entropy (children diverge within first page)
#  A3. Parent consistency (parent ↔ child bidirectional links)
#  A5. subtreeLeafCount correctness
#  A6. subtreeLeafCount monotonicity
#  C2. subtreeSumLocked partition
#  C3. subtreeSumLocked monotonicity
#  C4. Locked → locked descendant
#  E1. No single-child nodes (Patricia trie path compression invariant)
#  P1. Non-zero tokens/pages for valid nodes
#  P2. childId consistency with parent's children seq
#  W1. WAVL tree consistency (lpmLinks/evictLinks mirror children seq)

# Helper functions to compute subtree sums
func sumLeafCount[T, P](cs: openArray[PagedRadixNode[T, P]]): int32 =
  for c in cs: result += c.subtree_sum_leaves

func sumStagingLock[T, P](cs: openArray[PagedRadixNode[T, P]]): int32 =
  for c in cs: result += c.subtree_sum_locked


proc radixVerifyInvariants*[T, P](n: PagedRadixNode[T, P];
                                    ctx: string = "unknown") =
  ## Verify all structural and cumulative invariants on the subtree rooted at `n`.
  ## Recursively checks children. Raises `KVCacheDefect` on first violation.

  # ── A5: subtreeLeafCount correctness ──
  if n.children.len == 0:
    # Leaf nodes should have subtree_sum_leaves == 1
    # Exception: empty root (0 tokens, 0 pages) starts with 0 leaves
    if n.subtree_sum_leaves == 0 and n.tokens.len == 0 and n.pages.len == 0:
      discard # Empty root is allowed to have 0 leaves
    elif n.subtree_sum_leaves != 1:
      raise newException(KVCacheDefect,
        "[" & ctx & "] A5: leaf subtree_sum_leaves == " & $n.subtree_sum_leaves.int & ", expected 1")
  else:
    let sumLeaves = sumLeafCount(n.children)
    if n.subtree_sum_leaves != sumLeaves:
      raise newException(KVCacheDefect,
        "[" & ctx & "] A5: subtree_sum_leaves == " & $n.subtree_sum_leaves.int &
        " != sum children " & $sumLeaves.int)

  # ── C2: subtreeSumLocked partition ──
  if n.children.len > 0:
    let sumLocked = sumStagingLock(n.children)
    if n.subtree_sum_locked != sumLocked:
      raise newException(KVCacheDefect,
        "[" & ctx & "] C2: subtree_sum_locked == " & $n.subtree_sum_locked.int &
        " != sum children " & $sumLocked.int)

  # ── A6: subtreeLeafCount monotonicity ──
  for i, c in n.children:
    if c.subtree_sum_leaves > n.subtree_sum_leaves:
      raise newException(KVCacheDefect,
        "[" & ctx & "] A6: child[" & $i & "].subtree_sum_leaves (" & $c.subtree_sum_leaves.int &
        ") > parent (" & $n.subtree_sum_leaves.int & ")")

  # ── C3: subtreeSumLocked monotonicity ──
  for i, c in n.children:
    if c.subtree_sum_locked > n.subtree_sum_locked:
      raise newException(KVCacheDefect,
        "[" & ctx & "] C3: child[" & $i & "].subtree_sum_locked (" & $c.subtree_sum_locked.int &
        ") > parent (" & $n.subtree_sum_locked.int & ")")

  # ── C4: Locked implies locked descendant ──
  if n.subtree_sum_locked > 0 and not n.isLeaf:
    var hasLockedChild = false
    for c in n.children:
      if c.subtree_sum_locked > 0:
        hasLockedChild = true
        break
    if not hasLockedChild:
      raise newException(KVCacheDefect,
        "[" & ctx & "] C4: subtree_sum_locked > 0 but no locked child")

  # ── E1: No single-child nodes (Patricia trie path compression) ──
  if n.children.len == 1:
    raise newException(KVCacheDefect,
      "[" & ctx & "] E1: node has exactly 1 child (path compression violation)")

  # ── P2: childId consistency ──
  for i, c in n.children:
    if c.childId != i.int32:
      raise newException(KVCacheDefect,
        "[" & ctx & "] P2: child[" & $i & "].childId == " & $c.childId.int & ", expected " & $i)

  # ── A3: Parent consistency ──
  for c in n.children:
    if c.parent != n:
      raise newException(KVCacheDefect,
        "[" & ctx & "] A3: child.parent != this node")

  # ── P1: Non-zero tokens/pages for valid nodes ──
  if n.children.len > 0:
    # Internal nodes can have 0 tokens (e.g., root after fork)
    # But leaf nodes should have tokens
    for c in n.children:
      if c.isLeaf and c.tokens.len == 0:
        raise newException(KVCacheDefect,
          "[" & ctx & "] P1: leaf child has 0 tokens")
      if c.isLeaf and c.pages.len == 0:
        raise newException(KVCacheDefect,
          "[" & ctx & "] P1: leaf child has 0 pages")

  # ── W1: WAVL tree consistency ──
  if n.lpmRoot >= 0 or n.evictRoot >= 0:
    # WAVL links should match children seq length
    if n.lpmLinks.len != n.children.len:
      raise newException(KVCacheDefect,
        "[" & ctx & "] W1: lpmLinks.len (" & $n.lpmLinks.len &
        ") != children.len (" & $n.children.len & ")")
    if n.evictLinks.len != n.children.len:
      raise newException(KVCacheDefect,
        "[" & ctx & "] W1: evictLinks.len (" & $n.evictLinks.len &
        ") != children.len (" & $n.children.len & ")")

  # ── A1: Prefix entropy (children diverge within first page) ──
  # Check all pairs of children have different first pages
  for i in 0 ..< n.children.len:
    for j in i+1 ..< n.children.len:
      let ci = n.children[i]; let cj = n.children[j]
      if getCommonFirstPageLen(ci.tokens, cj.tokens) > 0:
        raise newException(KVCacheDefect,
          "[" & ctx & "] A1: children[" & $i & "] and children[" & $j &
          "] share first-page prefix")

  # ── Recurse into children ──
  for c in n.children:
    radixVerifyInvariants(c, ctx)
