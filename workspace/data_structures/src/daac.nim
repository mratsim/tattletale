# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Double-array Aho-Corasick over a byte dictionary: build from a
## pattern list, query with overlapping match semantics. Ported from
## the daachorse BASE+CHECK design (compact double-array AC, reference
## crate daachorse) with the block-placement engineering of the
## double-array survey paper (arXiv 2207.13870).
##
## Layout: a trie over the dictionary patterns is encoded into two flat
## arrays. A state owns transitions through its BASE value: the candidate
## child slot for byte c is `base xor c`, and the transition exists iff
## that slot's CHECK equals c. Because the index transform is a byte XOR,
## all child slots of one state stay inside the 256-aligned block of its
## base. Failure links (proper-suffix trie states) and merged output
## chains (every pattern ending at a state, own outputs plus those
## inherited through fail links) sit alongside, mirroring the daachorse
## State record. Free-slot management uses the daachorse BuildHelper:
## fixed 256-wide blocks, a circular active window of blocks, a doubly
## linked vacancy list per window, and used-base marks so a BASE value is
## claimed at most once. On block close, CHECK values are embedded into
## the block's vacant slots (keyed to the one never-assigned BASE of the
## block) so no transition from any assigned base can false-positive into
## a vacant slot.
##
## Match kind: standard overlapping semantics. Every pattern occurrence
## is reported at its end position by walking the state's output chain.
## Decision semantics on top (earliest start, ties by dictionary
## priority) belong to the scanner layer that consumes this core
## (workspace/toktoktok/src/scan.nim). daachorse's leftmost
## kinds are deliberately not ported: their tie rule keeps the
## last-visited (deepest) output, which diverges from a first-declared
## tie rule when two same-start candidates are related through a suffix
## link from another trie branch (reproducer: patterns ["bcd", "abcd"]
## declared in this order over haystack "xabcd" - both match at start 1,
## first-declared wins per the scanner oracle, last-visited picks the
## deeper "abcd"). The pruned leftmost NFA construction is likewise not
## ported: pruning drops prefix-extension patterns the scanner oracle
## can still select.

import std/algorithm

const
  DaacBlockLen* = 256
  ## Double-array block length (daachorse BLOCK_LEN): one aligned
  ## 256-slot block per push.
  DaacNumFreeBlocks = 16
  ## Trailing blocks tracked for base search (daachorse num_free_blocks).
  DaacRootIdx* = 0
  ## Slot index of the root state.
  DaacDeadIdx* = 1
  ## Slot index reserved like the daachorse dead state (never a real
  ## transition target in the standard build, kept for layout parity and
  ## for the block-close embedding).

type
  DaacState* {.final.} = object
    ## One double-array slot. base == 0 marks a slot without transitions
    ## (daachorse stores Option<NonZeroU32>). fail and outputPos are
    ## meaningful only for real trie states. check validates a parent's
    ## transition that selects this slot.
    base*: int32
    fail*: int32
    outputPos*: int32
    check*: uint8

  DaacOutput* {.final.} = object
    ## One pattern occurrence class ending at a state, chained through
    ## parent (index+1 of the previous output in the chain, 0 ends it).
    value*: int32
    length*: int32
    parent*: int32

  Daac* {.final.} = object
    ## Built automaton: read-only after construction, shared freely.
    states*: seq[DaacState]
    outputs*: seq[DaacOutput]
    numStates*: int32
    numPatterns*: int32

proc outputHead*(d: Daac, state: int32): int32 {.inline.} =
  ## Head of the output chain of `state` (0 = none, else index+1).
  d.states[state].outputPos

proc outputValue*(d: Daac, outIdx: int32): int32 {.inline.} =
  ## Value of the chain entry at index+1 `outIdx`.
  d.outputs[outIdx - 1].value

proc outputLength*(d: Daac, outIdx: int32): int32 {.inline.} =
  ## Pattern byte length of the chain entry at index+1 `outIdx`.
  d.outputs[outIdx - 1].length

proc outputParent*(d: Daac, outIdx: int32): int32 {.inline.} =
  ## Next chain entry (index+1) of the entry at index+1 `outIdx`.
  d.outputs[outIdx - 1].parent

proc nextState*(d: Daac, state: int32, b: uint8): int32 {.inline.} =
  ## Standard goto-plus-fail transition: follow base/check children,
  ## climb fail links, and settle at the root when nothing matches.
  ## Mirrors daachorse next_state_id_unchecked.
  var s = state
  while true:
    let base = d.states[s].base
    if base != 0:
      let childIdx = base xor int32(b)
      if d.states[childIdx].check == b:
        return childIdx
    if s == DaacRootIdx:
      return DaacRootIdx
    s = d.states[s].fail

# ---------------------------------------------------------------------
# Build helper (daachorse build_helper.rs port)
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

proc removeInvalidChecks(h: BuildHelper, states: var seq[DaacState],
    blockIdx: int32) =
  ## Embeds CHECK values into the vacant slots of a closed block, keyed
  ## to the one BASE value of the block that no state ever claimed.
  ## After embedding, a transition from any assigned base reaching
  ## one of these slots fails the check, so stale or default zero checks
  ## can never validate an invalid transition (including the NUL byte).
  let unused = h.unusedBaseInBlock(blockIdx)
  if unused < 0:
    return
  for c in 0 ..< 256:
    let idx = unused xor int32(c)
    if idx == DaacRootIdx or idx == DaacDeadIdx or not h.isUsedIndex(int32(idx)):
      states[idx].check = uint8(c)

# ---------------------------------------------------------------------
# Sparse NFA construction (daachorse nfa_builder.rs port, standard kind)
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
  ## Inserts one pattern, appending its (value, length) output at the
  ## terminal state. Duplicate patterns append a second output entry.
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
  ## BFS fail-link construction, standard semantics (daachorse
  ## build_fails). Children of the root fail to the root. Returns the
  ## BFS order, which also drives the output merge.
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

proc nfaBuildOutputs(nodes: seq[NfaNode], q: seq[int32],
    outputs: var seq[DaacOutput], outputPos: var seq[int32]) =
  ## Merges each state's own outputs with its fail link's chain
  ## (daachorse build_outputs). Chain entries keep the pattern order of
  ## insertion, outputPos holds the head (index+1, 0 = none).
  ## Entries are pushed in reverse pattern order so the chain, walked
  ## from the head, yields own outputs in insertion (priority) order,
  ## exactly like the daachorse build_outputs push loop.
  outputPos.setLen(nodes.len)
  var last = int32(0)
  for i in countdown(nodes[0].output.len - 1, 0):
    let outItem = nodes[0].output[i]
    outputs.add DaacOutput(value: outItem.value, length: outItem.patLen,
        parent: last)
    last = int32(outputs.len)
  outputPos[0] = last
  for s in q:
    last = outputPos[nodes[s].fail]
    for i in countdown(nodes[s].output.len - 1, 0):
      let outItem = nodes[s].output[i]
      outputs.add DaacOutput(value: outItem.value, length: outItem.patLen,
          parent: last)
      last = int32(outputs.len)
    outputPos[s] = last

# ---------------------------------------------------------------------
# Double-array encoding (daachorse bytewise/builder.rs port)
# ---------------------------------------------------------------------

proc findBase(h: BuildHelper, labels: openArray[uint8],
    statesLen: int): int32 =
  ## Smallest workable BASE for the given child labels, scanning the
  ## vacancy list of the active window, falling back to a fresh block
  ## index (the caller extends the array).
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

proc initArray(states: var seq[DaacState]): BuildHelper =
  states = newSeq[DaacState](DaacBlockLen)
  result = BuildHelper(
    items: newSeq[ListItem](DaacBlockLen * DaacNumFreeBlocks),
    blockLen: DaacBlockLen,
    numFreeBlocks: DaacNumFreeBlocks,
    numBlocks: 0,
    headIdx: -1)
  result.pushBlock()
  result.useIndex(DaacRootIdx)
  result.useIndex(DaacDeadIdx)

proc extendArray(states: var seq[DaacState], h: var BuildHelper) =
  let closed = h.droppedBlock()
  if closed >= 0:
    h.removeInvalidChecks(states, closed)
  h.pushBlock()
  states.setLen(states.len + DaacBlockLen)

proc sortEdges(nodes: var seq[NfaNode]) =
  ## Byte-sorts each state's edges so label iteration (BFS order and
  ## find_base's first label) matches the daachorse BTreeMap order.
  for i in 0 ..< nodes.len:
    let edges = nodes[i].edges
    var sorted = edges
    sorted.sort(proc (a, b: NfaEdge): int = cmp(a.label, b.label))
    nodes[i].edges = sorted

proc buildDaac*(patterns: openArray[string], values: openArray[int]): Daac =
  ## Builds the automaton for the given patterns (values[i] is reported
  ## for occurrences of patterns[i]). Deterministic for a fixed input:
  ## identical pattern lists yield identical arrays.
  doAssert patterns.len == values.len
  var nodes: seq[NfaNode] = @[NfaNode()]
  for i in 0 ..< patterns.len:
    doAssert values[i] >= low(int32) and values[i] <= high(int32)
    nodes.nfaAdd(patterns[i], int32(values[i]))
  nodes.sortEdges()
  let q = nodes.nfaBuildFails()
  var outputs: seq[DaacOutput] = @[]
  var nfaOutputPos: seq[int32]
  nodes.nfaBuildOutputs(q, outputs, nfaOutputPos)

  var helper = initArray(result.states)
  var stateIdMap = newSeq[int32](nodes.len)
  for i in 0 ..< stateIdMap.len:
    stateIdMap[i] = DaacDeadIdx
  stateIdMap[0] = DaacRootIdx

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
