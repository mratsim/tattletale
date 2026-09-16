# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Double-array Aho-Corasick automaton over a byte dictionary, encoded per Kanda, Akabe, Oda (arXiv 2207.13870).
##
## Goto-plus-fail transition, one input byte at a time:
##
##   step ──▶ t = base(s) xor b
##            ├─ check(t) == b ──▶ t is the child state, return it
##            ├─ s == root ──▶ stay at the root
##            └─ otherwise ──▶ s := fail(s), retry
##   each visited state ──▶ climb its output chain, report (value, end - length, end)
##
## Matching semantics:
## - overlapping matches, one report per pattern occurrence at its end position.
## - decision semantics (earliest start, first-declared ties) and leftmost selection live in the scanner (workspace/toktoktok/src/scan.nim).
##
## Base placement and safety:
##   | rule        | contract                                                                                                                       |
##   | ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
##   | free slots  | fixed 256-slot blocks, searched through a doubly linked vacancy list over the trailing 16-block window (Chain and SkipForward) |
##   | used-base   | a base value is claimed at most once, the XOR index keeps every child slot inside the 256-aligned block of its base            |
##   | block close | vacant slots receive check values keyed to the one never-assigned base, so no transition validates a vacant slot               |
##
## Match-policy boundary:
##   | case           | behavior                                                                                                                         |
##   | -------------- | -------------------------------------------------------------------------------------------------------------------------------- |
##   | same-start tie | kept as the last-visited (deepest) output, diverges from a first-declared rule when candidates relate through a suffix link      |
##   | reproducer     | "xabcd", patterns ["bcd", "abcd"] both match at start 1, first-declared wins in the consuming scanner, last-visited picks "abcd" |

import std/algorithm

const
  AhoCorasickBlockLen = 256
  ## Double-array block length, one aligned 256-slot block per push,
  ## the paper's block size for the byte alphabet (2^ceil(log2 256)).
  AhoCorasickNumFreeBlocks = 16
  ## Trailing blocks held for base search, the paper's SkipForward
  ## window (16 blocks).
  AhoCorasickRootIdx* = 0
  ## Slot index of the root state.
  AhoCorasickDeadIdx = 1
  ## Slot index marked used at init and never a transition target.
  ## The block-close check embedding skips it, so its check stays 0,
  ## a value no label can validate.

type
  AhoCorasickState {.final.} = object
    ## One double-array slot:
    ## - base == 0 marks a slot without transitions
    ## - fail and outputPos are meaningful only for real trie states
    ## - check validates a parent's transition that selects this slot
    base: int32
    fail: int32
    outputPos: int32
    check: uint8

  AhoCorasickOutput {.final.} = object
    ## One pattern occurrence class ending at a state, chained through
    ## parent (index+1 of the previous output in the chain, 0 ends it).
    value: int32
    length: int32
    parent: int32

  AhoCorasick* {.final.} = object
    ## Built automaton. Read-only after construction, shared freely.
    states: seq[AhoCorasickState]
    outputs: seq[AhoCorasickOutput]
    numStates: int32
    numPatterns: int32

proc outputHead*(d: AhoCorasick, state: int32): int32 {.inline.} =
  ## Head of the output chain of `state` (0 = none, else index+1).
  d.states[state].outputPos

proc outputValue*(d: AhoCorasick, outIdx: int32): int32 {.inline.} =
  ## Value of the chain entry at index+1 `outIdx`.
  d.outputs[outIdx - 1].value

proc outputLength*(d: AhoCorasick, outIdx: int32): int32 {.inline.} =
  ## Pattern byte length of the chain entry at index+1 `outIdx`.
  d.outputs[outIdx - 1].length

proc outputParent*(d: AhoCorasick, outIdx: int32): int32 {.inline.} =
  ## Next chain entry (index+1) of the entry at index+1 `outIdx`.
  d.outputs[outIdx - 1].parent

proc nextState*(d: AhoCorasick, state: int32, b: uint8): int32 {.inline.} =
  ## Goto-plus-fail transition (the automaton's extended transition):
  ## - follow the base/check children
  ## - on a miss, climb the fail links
  ## - settle at the root when nothing matches
  var s = state
  while true:
    let base = d.states[s].base
    if base != 0:
      let childIdx = base xor int32(b)
      if d.states[childIdx].check == b:
        return childIdx
    if s == AhoCorasickRootIdx:
      return AhoCorasickRootIdx
    s = d.states[s].fail

# ---------------------------------------------------------------------
# Base placement, the paper's vacant-search acceleration (Chain list, SkipForward window)
# ---------------------------------------------------------------------

type
  ListItem = object
    ## Per-slot build bookkeeping, held only for the active window
    ## (circular buffer over the last numFreeBlocks blocks).
    nxt: int32
    prev: int32
    usedBase: bool
    usedIndex: bool

  BuildHelper = object
    items: seq[ListItem]
    blockLen: int32
    numFreeBlocks: int32
    numBlocks: int32
    headIdx: int32

proc capacity(h: BuildHelper): int32 {.inline.} =
  int32(h.items.len)

proc numElements(h: BuildHelper): int32 {.inline.} =
  h.numBlocks * h.blockLen

proc activeBlockLo(h: BuildHelper): int32 {.inline.} =
  let sub = h.numBlocks - h.numFreeBlocks
  if sub < 0: 0 else: sub

proc slot(h: BuildHelper, idx: int32): int32 {.inline.} =
  ## Circular offset of an active-window index.
  idx mod h.capacity()

proc checkRange(h: BuildHelper, idx: int32) {.inline.} =
  let lo = h.activeBlockLo() * h.blockLen
  let hi = h.numElements()
  if idx < lo or idx >= hi:
    raise newException(RangeDefect, "BuildHelper index outside active window")

proc isUsedBase(h: BuildHelper, base: int32): bool {.inline.} =
  h.checkRange(base)
  h.items[h.slot(base)].usedBase

proc isUsedIndex(h: BuildHelper, idx: int32): bool {.inline.} =
  h.checkRange(idx)
  h.items[h.slot(idx)].usedIndex

proc useBase(h: var BuildHelper, base: int32) {.inline.} =
  h.checkRange(base)
  h.items[h.slot(base)].usedBase = true

proc useIndex(h: var BuildHelper, idx: int32) {.inline.} =
  ## Claims an index and unlinks it from the vacancy list.
  h.checkRange(idx)
  doAssert not h.items[h.slot(idx)].usedIndex
  h.items[h.slot(idx)].usedIndex = true
  let it = h.items[h.slot(idx)]
  let nxt = it.nxt
  let prev = it.prev
  h.items[h.slot(prev)].nxt = nxt
  h.items[h.slot(nxt)].prev = prev
  if h.headIdx == idx:
    if nxt != idx:
      h.headIdx = nxt
    else:
      h.headIdx = -1

proc droppedBlock(h: BuildHelper): int32 {.inline.} =
  ## Oldest active block index when the window is full, else -1.
  if h.capacity() <= h.numElements():
    h.activeBlockLo()
  else:
    -1

proc pushBlock(h: var BuildHelper) =
  ## Appends one fresh block, closing (and forgetting) the oldest active
  ## block when the window is full.
  if h.numElements() > high(int32) - h.blockLen:
    raise newException(ValueError, "double-array scale limit reached")
  let closed = h.droppedBlock()
  if closed >= 0:
    let endIdx = (closed + 1) * h.blockLen
    while h.headIdx >= 0 and h.headIdx < endIdx:
      h.useIndex(h.headIdx)
  let oldLen = h.numElements()
  h.numBlocks += 1
  let newLen = h.numElements()
  for idx in oldLen ..< newLen:
    h.items[h.slot(idx)] = ListItem()
    h.items[h.slot(idx)].nxt = int32(idx + 1)
    h.items[h.slot(idx)].prev = int32(idx - 1)
  if h.headIdx >= 0:
    let tail = h.items[h.slot(h.headIdx)].prev
    h.items[h.slot(oldLen)].prev = tail
    h.items[h.slot(tail)].nxt = int32(oldLen)
    h.items[h.slot(newLen - 1)].nxt = h.headIdx
    h.items[h.slot(h.headIdx)].prev = int32(newLen - 1)
  else:
    h.items[h.slot(oldLen)].prev = int32(newLen - 1)
    h.items[h.slot(newLen - 1)].nxt = int32(oldLen)
    h.headIdx = int32(oldLen)

proc unusedBaseInBlock(h: BuildHelper, blockIdx: int32): int32 =
  ## Smallest never-assigned BASE value in one block, else -1.
  let start = blockIdx * h.blockLen
  for base in start ..< (start + h.blockLen):
    if not h.isUsedBase(int32(base)):
      return int32(base)
  -1

proc removeInvalidChecks(h: BuildHelper, states: var seq[AhoCorasickState],
    blockIdx: int32) =
  ## Embeds CHECK values into the vacant slots of a closed block, keyed
  ## to the one BASE value of the block that no state ever claimed.
  ## Contract:
  ## - any transition from an assigned base reaching one of these slots fails the check
  ## - stale or default zero checks can never validate an invalid transition, including the NUL byte
  let unused = h.unusedBaseInBlock(blockIdx)
  if unused < 0:
    return
  for c in 0 ..< 256:
    let idx = unused xor int32(c)
    if idx == AhoCorasickRootIdx or idx == AhoCorasickDeadIdx or not h.isUsedIndex(int32(idx)):
      states[idx].check = uint8(c)

# ---------------------------------------------------------------------
# Sparse trie construction (the AC automaton's trie part)
# ---------------------------------------------------------------------

type
  NfaEdge = object
    label: uint8
    node: int32

  NfaNode = object
    edges: seq[NfaEdge]
    fail: int32
    output: seq[tuple[value, patLen: int32]]

proc nfaChild(nodes: seq[NfaNode], state: int32, c: uint8): int32 =
  ## Child state for byte c, -1 when absent.
  for e in nodes[state].edges:
    if e.label == c:
      return e.node
  -1

proc nfaAdd(nodes: var seq[NfaNode], pattern: openArray[char], value: int32) =
  ## Inserts one pattern, appending its (value, length) output at the terminal state.
  ## Duplicate patterns append a second output entry.
  var s = int32(0)
  for ch in pattern:
    let c = uint8(ch)
    let next = nodes.nfaChild(s, c)
    if next >= 0:
      s = next
    else:
      nodes.add NfaNode()
      let fresh = int32(nodes.len - 1)
      nodes[s].edges.add NfaEdge(label: c, node: fresh)
      s = fresh
  nodes[s].output.add (value, int32(pattern.len))

proc nfaBuildFails(nodes: var seq[NfaNode]): seq[int32] =
  ## BFS fail-link construction (the failure function f):
  ## - children of the root fail to the root
  ## - each deeper child fails to the longest proper suffix of its path
  ##   that is a trie state, found by climbing the parent's fail chain
  ## Returns the BFS order, which also drives the output merge.
  var q: seq[int32] = @[]
  for e in nodes[0].edges:
    q.add e.node
  var qi = 0
  while qi < q.len:
    let s = q[qi]
    inc qi
    for e in nodes[s].edges:
      var failId = nodes[s].fail
      var newFail = int32(0)
      while true:
        let via = nodes.nfaChild(failId, e.label)
        if via >= 0:
          newFail = via
          break
        let nextFail = nodes[failId].fail
        if failId == 0 and nextFail == 0:
          newFail = 0
          break
        failId = nextFail
      nodes[e.node].fail = newFail
      q.add e.node
  result = q

proc nfaBuildOutputs(outputs: var seq[AhoCorasickOutput],
    outputPos: var seq[int32], nodes: seq[NfaNode], q: seq[int32]) =
  ## Builds the output forest, the paper's Forest output representation.
  ## Each state's chain starts with its own outputs and continues into the fail target's chain, the shared suffix of the state's set.
  ## Chain contract:
  ## - entries keep the pattern order of insertion
  ## - outputPos holds the chain head (index+1, 0 = none)
  ## - entries push in reverse pattern order, so walking the chain
  ##   from the head yields the state's own outputs in insertion order
  outputPos.setLen(nodes.len)
  var last = int32(0)
  for i in countdown(nodes[0].output.len - 1, 0):
    let outItem = nodes[0].output[i]
    outputs.add AhoCorasickOutput(value: outItem.value, length: outItem.patLen,
        parent: last)
    last = int32(outputs.len)
  outputPos[0] = last
  for s in q:
    last = outputPos[nodes[s].fail]
    for i in countdown(nodes[s].output.len - 1, 0):
      let outItem = nodes[s].output[i]
      outputs.add AhoCorasickOutput(value: outItem.value, length: outItem.patLen,
          parent: last)
      last = int32(outputs.len)
    outputPos[s] = last

# ---------------------------------------------------------------------
# Double-array encoding
# ---------------------------------------------------------------------

proc findBase(h: BuildHelper, labels: openArray[uint8],
    statesLen: int): int32 =
  ## Smallest workable BASE for the given child labels.
  ## Scans the vacancy list of the active window, falls back to a fresh
  ## block index (the caller extends the array).
  if h.headIdx >= 0:
    var idx = h.headIdx
    while true:
      let base = idx xor int32(labels[0])
      if base != 0 and not h.isUsedBase(base):
        var ok = true
        for lab in labels:
          if h.isUsedIndex(base xor int32(lab)):
            ok = false
            break
        if ok:
          return base
      idx = h.items[h.slot(idx)].nxt
      if idx == h.headIdx:
        break
  int32(statesLen)

proc initArray(states: var seq[AhoCorasickState]): BuildHelper =
  states = newSeq[AhoCorasickState](AhoCorasickBlockLen)
  result = BuildHelper(
    items: newSeq[ListItem](AhoCorasickBlockLen * AhoCorasickNumFreeBlocks),
    blockLen: AhoCorasickBlockLen,
    numFreeBlocks: AhoCorasickNumFreeBlocks,
    numBlocks: 0,
    headIdx: -1)
  result.pushBlock()
  result.useIndex(AhoCorasickRootIdx)
  result.useIndex(AhoCorasickDeadIdx)

proc extendArray(states: var seq[AhoCorasickState], h: var BuildHelper) =
  let closed = h.droppedBlock()
  if closed >= 0:
    h.removeInvalidChecks(states, closed)
  h.pushBlock()
  states.setLen(states.len + AhoCorasickBlockLen)

proc sortEdges(nodes: var seq[NfaNode]) =
  ## Byte-sorts each state's edges, so fail-link BFS, output merge,
  ## and base placement (findBase anchors on the first label) all
  ## iterate deterministically in byte order.
  for i in 0 ..< nodes.len:
    let edges = nodes[i].edges
    var sorted = edges
    sorted.sort(proc (a, b: NfaEdge): int = cmp(a.label, b.label))
    nodes[i].edges = sorted

proc buildAhoCorasick*(patterns: openArray[string], values: openArray[int]): AhoCorasick =
  ## Builds the automaton for the given patterns.
  ## Args:
  ## - patterns
  ##   the dictionary patterns
  ## - values
  ##   values[i] is reported for occurrences of patterns[i]
  ##
  ## Deterministic for a fixed input.
  ## Identical pattern lists yield identical arrays.
  doAssert patterns.len == values.len
  var nodes: seq[NfaNode] = @[NfaNode()]
  for i in 0 ..< patterns.len:
    doAssert values[i] >= low(int32) and values[i] <= high(int32)
    nodes.nfaAdd(patterns[i], int32(values[i]))
  nodes.sortEdges()
  let q = nodes.nfaBuildFails()
  var outputs: seq[AhoCorasickOutput] = @[]
  var nfaOutputPos: seq[int32]
  nfaBuildOutputs(outputs, nfaOutputPos, nodes, q)

  var helper = initArray(result.states)
  var stateIdMap = newSeq[int32](nodes.len)
  for i in 0 ..< stateIdMap.len:
    stateIdMap[i] = AhoCorasickDeadIdx
  stateIdMap[0] = AhoCorasickRootIdx

  var stack: seq[int32] = @[int32(0)]
  var labels: seq[uint8] = @[]
  while stack.len > 0:
    let s = stack.pop()
    if nodes[s].edges.len == 0:
      continue
    labels.setLen(0)
    for e in nodes[s].edges:
      labels.add e.label
    let base = helper.findBase(labels, result.states.len)
    if base >= int32(result.states.len):
      result.states.extendArray(helper)
    for e in nodes[s].edges:
      let childIdx = base xor int32(e.label)
      helper.useIndex(childIdx)
      result.states[childIdx].check = e.label
      stateIdMap[e.node] = childIdx
      stack.add e.node
    result.states[stateIdMap[s]].base = base
    helper.useBase(base)

  for i in 0 ..< nodes.len:
    let idx = stateIdMap[i]
    result.states[idx].outputPos = nfaOutputPos[i]
    result.states[idx].fail = stateIdMap[nodes[i].fail]

  for b in helper.activeBlockLo() ..< helper.numBlocks:
    helper.removeInvalidChecks(result.states, b)

  result.outputs = outputs
  result.numStates = int32(nodes.len)
  result.numPatterns = int32(patterns.len)
