# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros

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
    depth*: int
    verbatim*: bool #
    case kind*: TupleStreamEventKind
    of kLeaf:
      leaf*, leafTy*: NimNode
    else:
      discard

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
    stack: seq[tuple[elem, ty: NimNode, idx, depth: int]]
    hasPending: bool
    pending: TupleStreamEvent

func tupleStream*(e: NimNode): TupleStream =
  var s: TupleStream
  let ev = unwrapStmtListExpr(e)
  let ty = e.getTypeInst()
  if ty.kind in {nnkTupleConstr, nnkTupleTy}:
    s.stack.add (ev, ty, 0, 0)
    s.pending = TupleStreamEvent(depth: 0, kind: kOpen, verbatim: true)
  else: # Scalars enter wrapped in a size-1 tuple, the stream is always tuple-shaped
    s.stack.add (nnkTupleConstr.newTree(ev), nnkTupleTy.newTree(ty), 0, 0)
    s.pending = TupleStreamEvent(depth: 0, kind: kOpen, verbatim: true)
  s.hasPending = true
  s

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
      inc s.stack[^1].idx
      if childT.kind in {nnkTupleConstr, nnkTupleTy}:
        s.pending = TupleStreamEvent(depth: f.depth + 1, kind: kOpen, verbatim: true)
        s.stack.add (childE, childT, 0, f.depth + 1)
      else:
        s.pending = TupleStreamEvent(depth: f.depth + 1, kind: kLeaf, leaf: childE, leafTy: childT, verbatim: true)
    else:
      let depth = f.depth
      discard s.stack.pop()
      s.pending = TupleStreamEvent(depth: depth, kind: kClose, verbatim: true)
    s.hasPending = true

iterator items*(s: var TupleStream): TupleStreamEvent =
  while not s.done:
    yield s.next()


# ═══════════════════════════════════════════════════════════════════════
#  Builders, append/emit consumers
# ═══════════════════════════════════════════════════════════════════════

# ═════ Flat tuple builder ══════════════════════════════════════════

type
  TupleBuilderFlat* = object
    accums*: seq[seq[NimNode]]
    verbatim*: bool = true

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

func emit*(tb: TupleBuilderFlat, id: int, emitScalarForSize1 = false): tuple[resultTuple: NimNode, verbatim: bool] =
  var node = if emitScalarForSize1: nnkPar.newTree()
             else: nnkTupleConstr.newTree()
  for n in tb.accums[id]:
    node.add n
  result = (node, tb.verbatim)

# ═════ Nested tuple builder ══════════════════════════════════════════

type
  TupleBuilderNested* = object
    accums*: seq[seq[NimNode]]
    completed*: seq[NimNode]
    verbatim*: bool = true

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
      let node = tb.accums[i].pop()
      if node.len == 0:
        # every child of the subtree was dropped, keep nothing in the parent
        if tb.accums[i].len == 0:
          tb.completed[i] = nnkTupleConstr.newTree()
      elif tb.accums[i].len == 0:
        tb.completed[i] = node
      else:
        tb.accums[i][^1].add node

func emit*(tb: TupleBuilderNested, id: int, emitScalarForSize1 = false): tuple[resultTuple: NimNode, verbatim: bool] =
  doAssert id < tb.completed.len and tb.completed[id] != nil, "emit: slot not completed"
  (tb.completed[id], tb.verbatim)

# ═══════════════════════════════════════════════════════════════════════
#  Sinks
# ═══════════════════════════════════════════════════════════════════════

func leaves*(s: TupleStream): seq[tuple[leaf, leafTy: NimNode]] =
  var src = s
  for ev in src.items():
    if ev.kind == kLeaf:
      result.add (ev.leaf, ev.leafTy)

func tupleFlatten*(e: NimNode): seq[tuple[leaf, leafTy: NimNode]] =
  tupleStream(e).leaves()

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

iterator items*(z: var TupleZip): tuple[a, b: TupleStreamEvent] =
  while not z.done():
    yield z.next()

# ═══════════════════════════════════════════════════════════════════════
#  Filters
# ═══════════════════════════════════════════════════════════════════════

func hasType*(x: NimNode; t: static string): bool =
  ## sameType wrapper: x has the type named `t`.
  sameType(x, bindSym(t))
