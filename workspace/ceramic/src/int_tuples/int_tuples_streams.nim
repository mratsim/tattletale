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
    depth*: int
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
    s.pending = TupleStreamEvent(depth: 0, kind: kOpen)
  else:
    s.pending = TupleStreamEvent(depth: 0, kind: kLeaf, leaf: ev, leafTy: ty)
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
        s.pending = TupleStreamEvent(depth: f.depth + 1, kind: kOpen)
        s.stack.add (childE, childT, 0, f.depth + 1)
      else:
        s.pending = TupleStreamEvent(depth: f.depth + 1, kind: kLeaf, leaf: childE, leafTy: childT)
    else:
      let depth = f.depth
      discard s.stack.pop()
      s.pending = TupleStreamEvent(depth: depth, kind: kClose)
    s.hasPending = true

iterator items*(s: var TupleStream): TupleStreamEvent =
  while not s.done:
    yield s.next()

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
#  Builders, append/emit consumers
# ═══════════════════════════════════════════════════════════════════════

# ═════ Flat tuple builder ══════════════════════════════════════════

type
  TupleBuilderFlat* = object
    accums*: seq[seq[NimNode]]

func new*(T: type TupleBuilderFlat, numTuples = 1): T =
  result.accums.newSeq(numTuples)

func append*(tb: var TupleBuilderFlat, streamEvents: varargs[TupleStreamEvent]) =
  doAssert tb.accums.len == streamEvents.len
  for i, ev in streamEvents:
    if ev.kind == kLeaf:
      tb.accums[i].add ev.leaf

func emit*(tb: TupleBuilderFlat, id: int, emitScalarForSize1 = false): NimNode =
  result = if emitScalarForSize1: nnkPar.newTree()
           else: nnkTupleConstr.newTree()
  for n in tb.accums[id]:
    result.add n

# ═════ Nested tuple builder ══════════════════════════════════════════

type
  TupleBuilderNested* = object
    accums*: seq[seq[NimNode]]
    completed*: seq[NimNode]

func new*(T: type TupleBuilderNested, numTuples = 1): T =
  result.accums.newSeq(numTuples)
  result.completed.newSeq(numTuples)

func append*(tb: var TupleBuilderNested, streamEvents: varargs[TupleStreamEvent]) =
  doAssert tb.accums.len == streamEvents.len
  for i, ev in streamEvents:
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
      if tb.accums[i].len == 0:
        tb.completed[i] = node
      else:
        tb.accums[i][^1].add node

func emit*(tb: TupleBuilderNested, id: int, emitScalarForSize1 = false): NimNode =
  doAssert id < tb.completed.len and tb.completed[id] != nil, "emit: slot not completed"
  tb.completed[id]