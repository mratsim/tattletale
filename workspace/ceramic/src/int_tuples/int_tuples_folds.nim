# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/macros, std/typetraits
import std/algorithm
import ./int_tuples_compiletime
import ./int_tuples_datatypes
import ./int_tuples_streams
import ./int_tuples_transforms
import workspace/ceramic/src/macros/replace_nodes

# ═══════════════════════════════════════════════════════════════
#  makeIntTuple, promotion to Int[V]() with value and type
# ═══════════════════════════════════════════════════════════════

proc appendLeaf(b: var TupleBuilderNested, ev: TupleStreamEvent) =
  case ev.kind
  of kOpen, kClose:
    b.append(ev)
  of kLeaf:
    let v = if ev.leaf.kind == nnkSym and ev.leafTy == nil: DynamicSentinel
            else: ev.leaf.getStaticInt()
    if v != DynamicSentinel:
      b.append((IntCT(v), intTy(v)))
    elif ev.leafTy == nil or ev.leafTy.kind == nnkIdentDefs:
      # no type known here, the type resolves in the caller's scope,
      # a declaration node is not a type, the leaf types as typeof(leaf)
      b.appendTyped(ev.leaf)
    else:
      b.append((ev.leaf, ev.leafTy))

proc makeIntTuple*(n: NimNode): tuple[val, ty: NimNode] {.compileTime.} =
  ## makeIntTuple on nodes: static leaves become Int[V]() constructors, dynamic leaves pass through and type as int
  ##
  ##  Examples:
  ##    makeIntTuple(newLit((2, 3)))
  ##        -> ((Int[2](), Int[3]()), (Int[2], Int[3]))
  ##    makeIntTuple(newLit(5))
  ##        -> (Int[5](), Int[5])
  # empty modes pass through, an explicit () mode carries zipped
  # gather semantics and must survive the promotion
  var acc = TupleBuilderNested.new(1, dropEmpty = false)
  for ev in n.tupleStream():
    acc.appendLeaf(ev)
  result = (acc.emit(0, not n.isTupleTy()), acc.emitTy(0, not n.isTupleTy()))

# ═══════════════════════════════════════════════════════════════
#  prefix and suffix product scans, value and type in one pass
# ═══════════════════════════════════════════════════════════════

type ProductAccum = object
  staticAcc: int
  flushing: bool
  astAcc: NimNode
  builder: TupleBuilderNested

proc init(a: var ProductAccum, seed: NimNode, seedTy: NimNode) =
  a.builder = TupleBuilderNested.new(1)
  var v = seed.getStaticInt()
  if v == DynamicSentinel and seedTy != nil:
    # a static-typed seed carries its value in the slot type
    v = seedTy.getStaticInt()
  if v != DynamicSentinel:
    a.staticAcc = v
  else:
    a.flushing = true
    a.astAcc = seed

proc update(a: var ProductAccum, ev: TupleStreamEvent) =
  case ev.kind
  of kOpen, kClose:
    a.builder.append(ev)
  of kLeaf:
    var v = if ev.leaf.kind == nnkSym and ev.leafTy == nil: DynamicSentinel
            else: ev.leaf.getStaticInt()
    if v == DynamicSentinel and ev.leafTy != nil:
      # a static-typed leaf carries its value in the slot type
      v = ev.leafTy.getStaticInt()
    if not a.flushing:
      a.builder.append((IntCT(a.staticAcc), intTy(a.staticAcc)))
      if v != DynamicSentinel:
        a.staticAcc = a.staticAcc * v
      else:
        a.flushing = true
        a.astAcc = IntCT(a.staticAcc) * ev.leaf
    else:
      # the chain's products carry their own types, the pair never disagrees,
      # Nim folds the value expression when the emitted tree instantiates
      a.builder.appendTyped(a.astAcc)
      if v != DynamicSentinel:
        a.astAcc = a.astAcc * newLit(v)
      else:
        a.astAcc = a.astAcc * ev.leaf

proc finish(a: var ProductAccum, emitScalarForSize1 = false): tuple[val, ty: NimNode] =
  result.val = a.builder.emit(0, emitScalarForSize1)
  result.ty = a.builder.emitTy(0, emitScalarForSize1)

proc productImpl*(n: NimNode): NimNode {.compileTime.} =
  var a: ProductAccum
  a.init(IntCT(1), IntCT(1)[0])
  for ev in n.tupleStream():
    a.update(ev)
  if a.flushing:
    result = a.astAcc
  else:
    result = IntCT(a.staticAcc)

proc prefixProductScan*(container, seed: NimNode, seedTy: NimNode = nil, reverse = false): tuple[val, ty: NimNode] {.compileTime.}
proc suffixProductScan*(container, seed: NimNode, seedTy: NimNode = nil, reverse = false): tuple[val, ty: NimNode] {.compileTime.}

proc scanPerMode(container, seed, seedTy: NimNode, reverse: bool): tuple[val, ty: NimNode] {.compileTime.} =
  doAssert seed.len == seedTy.len
  var r = TupleBuilderNested.new(1, dropEmpty = false)
  r.append(TupleStreamEvent(path: @[], kind: kOpen))
  var modeIdx = 0
  var collecting = false
  var modeB: TupleBuilderNested
  for ev in container.tupleStream(reverse):
    if ev.path.len == 0:
      continue
    if collecting:
      if ev.path.len == 1 and ev.kind == kClose:
        collecting = false
        modeB.append(ev)
        r.append(prefixProductScan(modeB.emit(0), seed[modeIdx],
                                   seedTy[modeIdx], reverse))
        modeIdx.inc
      else:
        modeB.append(ev)
    elif ev.kind == kOpen:
      modeB = TupleBuilderNested.new(1)
      modeB.append(ev)
      collecting = true
    else:
      doAssert seed[modeIdx].kind != nnkTupleConstr,
        "prefixProductScan: tuple seed leaf under a scalar shape mode"
      r.append((seed[modeIdx], seedTy[modeIdx]))
      modeIdx.inc
  r.append(TupleStreamEvent(path: @[], kind: kClose))
  result = (r.emit(0), r.emitTy(0))

proc prefixProductScan(container, seed: NimNode, seedTy: NimNode = nil, reverse = false): tuple[val, ty: NimNode] {.compileTime.} =
  ## Prefix product scan on nodes: each leaf's value is the seed times the leaves before it
  ##
  ## A tuple seed recurses per shape mode, a scalar seed under
  ## a tuple shape mode expands as that mode's base, congruent
  ## scalar pairs pass through verbatim
  ##
  ##  Examples:
  ##    prefixProductScan(newLit((2, 3, 4)), newLit(1))
  ##        -> ((Int[1](), Int[2](), Int[6]()), (Int[1], Int[2], Int[6]))
  ##    prefixProductScan(newLit(5), newLit(1))
  ##        -> (Int[1](), Int[1])
  ##    prefixProductScan(newLit(((2, 3), 4)), newLit((8, 1)))
  ##        -> (((Int[8](), Int[16]()), Int[1]()), ((Int[8], Int[16]), Int[1]))
  let containerIsTuple =
    container.kind == nnkTupleConstr and container.len > 0 or
    container.kind != nnkTupleConstr and container.isTupleTy()
  if seedTy != nil and seedTy.kind == nnkTupleConstr:
    if containerIsTuple:
      return scanPerMode(container, seed, seedTy, reverse)
    return (seed, seedTy)
  # a literal tuple seed and a symbol-bound tuple seed both go through
  # makeIntTuple, the leaves promote to Int[V] when static
  if seedTy.isNil and (seed.kind == nnkTupleConstr or
      seed.getTypeInst().isTupleTy()):
    let (sv, st) = makeIntTuple(seed)
    if containerIsTuple:
      return scanPerMode(container, sv, st, reverse)
    return (sv, st)
  var accu: ProductAccum
  accu.init(seed, seedTy)
  for ev in container.tupleStream():
    accu.update(ev)
  result = accu.finish(emitScalarForSize1 = not container.isTupleTy())

proc suffixProductScan*(container, seed: NimNode, seedTy: NimNode = nil, reverse = false): tuple[val, ty: NimNode] {.compileTime.} =
  ## Suffix product scan on nodes: each leaf's value is the seed times the leaves after it
  ##
  ##  Examples:
  ##    suffixProductScan(newLit((2, 3, 4)), newLit(1))
  ##        -> ((Int[12](), Int[4](), Int[1]()), (Int[12], Int[4], Int[1]))
  ##    suffixProductScan(newLit(5), newLit(1))
  ##        -> (Int[1](), Int[1])
  let containerIsTuple =
    container.kind == nnkTupleConstr and container.len > 0 or
    container.kind != nnkTupleConstr and container.isTupleTy()
  if seedTy != nil and seedTy.kind == nnkTupleConstr:
    if containerIsTuple:
      return scanPerMode(container, seed, seedTy, reverse)
    return (seed, seedTy)
  # a literal tuple seed and a symbol-bound tuple seed both go through
  # makeIntTuple, the leaves promote to Int[V] when static
  if seedTy.isNil and (seed.kind == nnkTupleConstr or
      seed.getTypeInst().isTupleTy()):
    let (sv, st) = makeIntTuple(seed)
    if containerIsTuple:
      return scanPerMode(container, sv, st, reverse)
    return (sv, st)
  var accu: ProductAccum
  accu.init(seed, seedTy)
  for ev in container.tupleStream(true):
    accu.update(ev)
  proc reverseTree(n: NimNode): NimNode {.compileTime.} =
    if n.kind == nnkTupleConstr:
      var items: seq[NimNode]
      for i in countdown(n.len - 1, 0):
        items.add reverseTree(n[i])
      result = nnkTupleConstr.newTree(items)
    else:
      result = n
  let (val, ty) = accu.finish(emitScalarForSize1 = not container.isTupleTy())
  result.val = reverseTree(val)
  result.ty = reverseTree(ty)

macro prefix_product_scan*(t: typed, seed: typed): untyped =
  ## Prefix product scan over all leaves of t: each leaf's value is `seed` times the leaves before it, the seed itself first.
  ##
  ##  Examples:
  ##    prefix_product_scan(5, 1)              -> 1
  ##    prefix_product_scan((2, 3, 4), 1)      -> (1, 2, 6)
  ##    prefix_product_scan(((4, 1), (8, 8)), Int[1]())
  ##        -> ((1, 4), (4, 32))
  let (val, _) = prefixProductScan(t, seed)
  result = val

macro suffix_product_scan*(t: typed, seed: typed): untyped =
  ## Suffix product scan over all leaves of t: each leaf's value is `seed` times the leaves after it
  ##
  ##  Examples:
  ##    suffix_product_scan(5, 1)              -> 1
  ##    suffix_product_scan((2, 3, 4), 1)      -> (12, 4, 1)
  ##    suffix_product_scan(((4, 1), (8, 8)), Int[1]())
  ##        -> ((64, 64), (8, 1))
  let (val, _) = suffixProductScan(t, seed)
  return val
