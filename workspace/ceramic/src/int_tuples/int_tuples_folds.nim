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
#  fold, left-fold reduction with Int[N] support
# ═══════════════════════════════════════════════════════════════

macro fold*(t: typed, startingAcc: typed, body: untyped): untyped =
  ## Fold over all leaves of t with an accumulator.
  ##
  ## The accumulator is called `acc` and the iteration variable `it`
  ##
  ##  Examples:
  ##    fold(5, 1, acc * it)            → 5
  ##    fold(Int[5](), 1, acc * it)     → 5  (Int[V] extracted via * overload)
  ##    fold((2,3,4), 1, acc * it)      → 24
  ##    fold((2,(3,4)), 1, acc * it)    → 24
  let accTy = startingAcc.getTypeInst()
  let intStart = accTy.sameType(bindSym"int")
  var leaves: seq[NimNode]

  if t.isTupleTy():
    var stream = t.tupleStream()
    while not stream.done():
      let event = stream.next()
      if event.kind == kLeaf:
        leaves.add event.leaf
  else:
    leaves.add t
  if leaves.len == 0:
    return newStmtList(startingAcc)

  var chain: NimNode = nil
  var acc: NimNode = if intStart: ident"acc" else: startingAcc

  for leaf in leaves:
    chain = body.replaceNodes(("acc", acc), ("it", leaf))
    acc = chain

  if intStart:
    result = quote do:
      block:
        let acc {.inject.} = `startingAcc`
        `chain`
  else:
    result = newStmtList(chain)

# ═══════════════════════════════════════════════════════════════
#  prefix_scanIt and suffix_scanIt, scans preserving constness
# ═══════════════════════════════════════════════════════════════

proc scanBuilder(node: NimNode, acc0: NimNode, body: NimNode, reversed: bool): NimNode =
  var stream = node.tupleStream(reversed)
  var builder = TupleBuilderNested.new(1)
  var acc = acc0
  while not stream.done():
    let ev = stream.next()
    case ev.kind
    of kLeaf:
      builder.append(acc)
      acc = body.replaceNodes(("acc", acc), ("it", ev.leaf))
    else:
      builder.append(ev)
  result = builder.emit(0)
  proc reverseTree(n: NimNode): NimNode =
    if n.kind == nnkTupleConstr:
      var items: seq[NimNode]
      for i in countdown(n.len - 1, 0):
        items.add reverseTree(n[i])
      result = nnkTupleConstr.newTree(items)
    else:
      result = n
  if reversed:
    result = reverseTree(result)

macro prefix_scanIt*(t: typed, startingAcc: typed, body: untyped): untyped =
  ## Left-to-right prefix scan over all leaves of t with an accumulator.
  ##
  ## The accumulator is called `acc` and the iteration variable `it`,
  ## `acc` binds the value before the element in scan direction
  ##
  ##  Examples:
  ##    prefix_scanIt(5, 1, acc * it)              → 1
  ##    prefix_scanIt((2, 3, 4), 1, acc * it)      → (1, 2, 6)
  ##    prefix_scanIt(((4, 1), (8, 8)), Int[1](), acc * it)
  ##        → ((1, 4), (4, 32))
  if t.isTupleTy():
    let scanned = t.scanBuilder(startingAcc, body, false)
    if startingAcc.getTypeInst().sameType(bindSym"int"):
      result = quote do:
        block:
          let acc {.inject.} = `startingAcc`
          `scanned`
    else:
      result = newStmtList(scanned)
  else:
    result = newStmtList(startingAcc)

macro suffix_scanIt*(t: typed, startingAcc: typed, body: untyped): untyped =
  ## Right-to-left suffix scan over all leaves of t with an accumulator.
  ##
  ## The accumulator is called `acc` and the iteration variable `it`,
  ## `acc` binds the value before the element in scan direction
  ##
  ##  Examples:
  ##    suffix_scanIt(5, 1, acc * it)              → 1
  ##    suffix_scanIt((2, 3, 4), 1, acc * it)      → (12, 4, 1)
  ##    suffix_scanIt(((4, 1), (8, 8)), Int[1](), acc * it)
  ##        → ((64, 64), (8, 1))
  if t.isTupleTy():
    let scanned = t.scanBuilder(startingAcc, body, true)
    if startingAcc.getTypeInst().sameType(bindSym"int"):
      result = quote do:
        block:
          let acc {.inject.} = `startingAcc`
          `scanned`
    else:
      result = newStmtList(scanned)
  else:
    result = newStmtList(startingAcc)
