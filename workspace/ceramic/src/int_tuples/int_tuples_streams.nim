# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros
import ./int_tuples_compiletime

# ═══════════════════════════════════════════════════════════════════════
#  Tuple streams / iterators
# ═══════════════════════════════════════════════════════════════════════
#
# This module enables producer(s) -> transform(s) -> consumer(s) patterns
# for tuple.
# The goal is to generalize all higher order maps, folds, zips, filters, transforms
# and have a single canonical tuple walker.

type
  TupleStreamEventKind* = enum
    kOpen, kLeaf, kClose

  TupleStreamEvent* = object
    ## Stream a tuple
    ##
    ## Tracks modification so an identity stream for a tuple `t` = (11, 22, 33)
    ## can return `t` directly.
    ## Otherwise return tuple would be (t[0], t[1], t[2)), leading to unnecessary temporaries
    verbatim*: bool #
    path*: seq[int] # index chain root -> here, a top-level leaf = @[i]
    case kind*: TupleStreamEventKind
    of kLeaf:
      leaf*, leafTy*: NimNode
    else:
      discard

func depth*(ev: TupleStreamEvent): int {.inline.} =
  result = ev.path.len

func unwrapStmtListExpr(node: NimNode): NimNode =
  if node.kind == nnkStmtListExpr:
    node[^1]
  else: node

func getTupleIndex*(node: NimNode, idx: int): NimNode =
  if node.kind == nnkPar and node.len == 1:
    nnkBracketExpr.newTree(node, newLit(idx))
  elif node.kind in {nnkTupleConstr, nnkPar}:
    node[idx]
  else:
    nnkBracketExpr.newTree(node, newLit(idx))

# ═══════════════════════════════════════════════════════════════════════
#  Producer, a defunctionalized tuple tree walker
# ═══════════════════════════════════════════════════════════════════════

type
  TupleStream* = object
    stack: seq[tuple[elem, ty: NimNode, idx, depth: int, path: seq[int]]]
    hasPending: bool
    pending: TupleStreamEvent
    shallow: bool

func tupleStream*(s: NimNode): TupleStream =
  let ev = unwrapStmtListExpr(s)
  let ty = s.getTypeInst()
  if ty.isTupleTy():
    result.stack.add (ev, ty, 0, 0, @[])
    result.pending = TupleStreamEvent(path: @[], kind: kOpen, verbatim: true)
  else: # Scalars enter wrapped in a size-1 tuple, the stream is always tuple-shaped
    result.stack.add (nnkTupleConstr.newTree(ev), nnkTupleTy.newTree(ty), 0, 0, @[])
    result.pending = TupleStreamEvent(path: @[], kind: kOpen, verbatim: true)
  result.hasPending = true

func tupleDimsStream*(s: NimNode): TupleStream =
  ## Returns: the top-level elements as kLeaf events, one per
  ##   dimension in order, done() turns true after the last.
  ##
  ## - a sub-tuple element arrives whole in `leaf`
  ## - scalars wrap in a size-1 tuple
  ## - no root wrapper, the walk opens and closes on the leaves
  let ev = unwrapStmtListExpr(s)
  let ty = s.getTypeInst()
  result.shallow = true
  if ty.isTupleTy():
    if ty.len != 0:
      # idx starts at 1, the first leaf is already the pending event
      result.stack.add (ev, ty, 1, 0, @[])
      result.pending = TupleStreamEvent(path: @[0], kind: kLeaf, leaf: getTupleIndex(ev, 0), leafTy: ty[0], verbatim: true)
      result.hasPending = true
    # an empty tuple yields no events
  else:
    # scalars wrap in a size-1 tuple, the single leaf is the whole
    result.pending = TupleStreamEvent(path: @[0], kind: kLeaf, leaf: ev, leafTy: ty, verbatim: true)
    result.hasPending = true

func done*(s: var TupleStream): bool =
  not s.hasPending and s.stack.len == 0

func next*(s: var TupleStream): TupleStreamEvent =
  doAssert not s.done, "tupleStream exhausted"
  result = s.pending
  s.hasPending = false
  # Prepare the next pending event.
  if s.stack.len != 0:
    let f = s.stack[^1]
    if f.idx < f.ty.len:
      let childE = getTupleIndex(f.elem, f.idx)
      let childT = f.ty[f.idx]
      let childPath = f.path & f.idx
      inc s.stack[^1].idx
      if s.shallow:
        # a shallow stream yields the whole sub-tuple element as one leaf
        s.pending = TupleStreamEvent(path: childPath, kind: kLeaf, leaf: childE, leafTy: childT, verbatim: true)
      elif childT.isTupleTy():
        s.pending = TupleStreamEvent(path: childPath, kind: kOpen, verbatim: true)
        s.stack.add (childE, childT, 0, f.depth + 1, childPath)
      else:
        s.pending = TupleStreamEvent(path: childPath, kind: kLeaf, leaf: childE, leafTy: childT, verbatim: true)
    else:
      let depth = f.depth
      let path = f.path
      discard s.stack.pop()
      if s.shallow:
        # the shallow walk opens and closes on the leaves, no root wrapper
        s.hasPending = false
      else:
        s.pending = TupleStreamEvent(path: path, kind: kClose, verbatim: true)
        s.hasPending = true

iterator items*(s: TupleStream): TupleStreamEvent =
  var w = s
  while not w.done():
    yield w.next()

# ═══════════════════════════════════════════════════════════════════════
#  Builders, append/emit consumers
# ═══════════════════════════════════════════════════════════════════════

# ═════ Flat tuple builder ══════════════════════════════════════════

type
  TupleBuilderFlat* = object
    accums: seq[seq[NimNode]]
    verbatim: bool = true

func new*(T: type TupleBuilderFlat, numTuples = 1): T =
  result.accums.newSeq(numTuples)
  result.verbatim = true

func append*(tb: var TupleBuilderFlat, streamEvents: varargs[TupleStreamEvent]) =
  doAssert tb.accums.len == streamEvents.len
  for i, ev in streamEvents:
    if not ev.verbatim:
      tb.verbatim = false
    if ev.kind == kLeaf:
      tb.accums[i].add ev.leaf

template appendImpl(tb: var TupleBuilderFlat, leafValues: varargs[NimNode], isVerbatim: bool) =
  # vargards[NimNode] + default arguments doesn't seem to work
  doAssert tb.accums.len == leafValues.len
  if not isVerbatim:
    tb.verbatim = false
  for i, leaf in leafValues:
    tb.accums[i].add leaf

func append*(tb: var TupleBuilderFlat, leafValues: varargs[NimNode]) =
  tb.appendImpl(leafValues, isVerbatim = false)

func append*(tb: var TupleBuilderFlat, leafValues: varargs[NimNode], verbatim: bool) =
  tb.appendImpl(leafValues, isVerbatim = verbatim)

func prependBatch*(tb: var TupleBuilderFlat, batches: varargs[seq[NimNode]]) =
  doAssert tb.accums.len == batches.len
  tb.verbatim = false
  for i, batch in batches:
    tb.accums[i] = batch & tb.accums[i]

func markNonVerbatim*(tb: var TupleBuilderFlat) =
  tb.verbatim = false

func emit*(tb: TupleBuilderFlat, id: int, emitScalarForSize1 = false): tuple[resultTuple: NimNode, verbatim: bool] =
  var node = if emitScalarForSize1: nnkPar.newTree()
             else: nnkTupleConstr.newTree()
  for n in tb.accums[id]:
    node.add n
  result = (node, tb.verbatim)

# ═════ Nested tuple builder ══════════════════════════════════════════

type
  TupleBuilderNested* = object
    accums: seq[seq[NimNode]]
    completed: seq[NimNode]
    verbatim: bool = true

func new*(T: type TupleBuilderNested, numTuples = 1): T =
  result.accums.newSeq(numTuples)
  result.completed.newSeq(numTuples)
  result.verbatim = true

func append*(tb: var TupleBuilderNested, streamEvents: varargs[TupleStreamEvent]) =
  doAssert tb.accums.len == streamEvents.len
  for i, ev in streamEvents:
    if not ev.verbatim:
      tb.verbatim = false
    case ev.kind
    of kOpen:
      tb.accums[i].add newNimNode(nnkTupleConstr)
    of kLeaf:
      if tb.accums[i].len == 0:
        tb.completed[i] = ev.leaf
      else:
        tb.accums[i][^1].add ev.leaf
    of kClose:
      var node = tb.accums[i].pop()
      if node.len == 0:
        # every child of the subtree was dropped, keep nothing in the parent
        if tb.accums[i].len == 0:
          tb.completed[i] = nnkTupleConstr.newTree()
      else:
        if tb.accums[i].len == 0:
          tb.completed[i] = node
        else:
          tb.accums[i][^1].add node

template appendImpl(tb: var TupleBuilderNested, leafValues: varargs[NimNode], isVerbatim: bool) =
  # vargards[NimNode] + default arguments doesn't seem to work
  doAssert tb.accums.len == leafValues.len
  if not isVerbatim:
    tb.verbatim = false
  for i, leaf in leafValues:
    if tb.accums[i].len == 0:
      tb.completed[i] = leaf
    else:
      tb.accums[i][^1].add leaf

func append*(tb: var TupleBuilderNested, leafValues: varargs[NimNode]) =
  tb.appendImpl(leafValues, isVerbatim = false)

func append*(tb: var TupleBuilderNested, leafValues: varargs[NimNode], verbatim: bool) =
  tb.appendImpl(leafValues, isVerbatim = verbatim)

func markNonVerbatim*(tb: var TupleBuilderNested) =
  tb.verbatim = false

func emit*(tb: TupleBuilderNested, id: int, emitScalarForSize1 = false): tuple[resultTuple: NimNode, verbatim: bool] =
  doAssert id < tb.completed.len and tb.completed[id] != nil, "emit: slot not completed"
  var node = tb.completed[id]
  if emitScalarForSize1 and node.kind == nnkTupleConstr and node.len == 1:
    node = node[0]
  result = (node, tb.verbatim)

# ═══════════════════════════════════════════════════════════════════════
#  onLeaves, consumer-side event ingest
# ═══════════════════════════════════════════════════════════════════════

template onLeaves*(event: TupleStreamEvent, body: untyped): untyped =
  case event.kind
  of kOpen, kClose:
    builder.append(event)
  of kLeaf:
    body

# ═══════════════════════════════════════════════════════════════════════
#  Sinks
# ═══════════════════════════════════════════════════════════════════════

func leaves*(s: TupleStream): seq[tuple[leaf, leafTy: NimNode]] =
  for ev in s.items():
    if ev.kind == kLeaf:
      result.add (ev.leaf, ev.leafTy)

func tupleFlatten*(s: NimNode): seq[tuple[leaf, leafTy: NimNode]] =
  s.tupleStream().leaves()

# ═══════════════════════════════════════════════════════════════════════
#  Zip
# ═══════════════════════════════════════════════════════════════════════

type
  TupleZip* = object
    a, b: TupleStream

func zip*(a, b: TupleStream): TupleZip =
  TupleZip(a: a, b: b)

func done*(z: var TupleZip): bool =
  let a_done = z.a.done()
  let b_done = z.b.done()
  doAssert a_done == b_done, "zip: the trees are not congruent"
  a_done

func next*(z: var TupleZip): tuple[a, b: TupleStreamEvent] =
  let a_next = z.a.next()
  let b_next = z.b.next()
  if a_next.kind != b_next.kind or a_next.depth != b_next.depth:
    error "zip: the trees are not congruent"
  (a_next, b_next)

iterator items*(z: TupleZip): tuple[a, b: TupleStreamEvent] =
  var w = z
  while not w.done():
    yield w.next()

# ═══════════════════════════════════════════════════════════════════════
#  Filters
# ═══════════════════════════════════════════════════════════════════════

func hasType*(x: NimNode, t: static string): bool =
  ## Type equality in macro. Resolves through aliases.
  ## This does not handle generic matches (i.e. Int[4].hasType"Int")
  ##
  ## Example
  ##   x.hasType"int"
  sameType(x, bindSym(t))
