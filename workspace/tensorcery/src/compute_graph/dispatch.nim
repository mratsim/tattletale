# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/tables
import std/macros
import workspace/crucible
import ../tensors_internals

# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

var OpRegistry {.compileTime.}: Table[string, NimNode]

macro registerOp*(op: static string, handler: typed) =
  OpRegistry[op] = handler

# ═══════════════════════════════════════════════════════════════════════
#  The walk
# ═══════════════════════════════════════════════════════════════════════

template withElemType(engine: HwEngine, T: untyped, dtype: Dtype, body: untyped) =
  case dtype
  of kFp32:
    type T = float32
    body
  of kBf16:
    type T = bfloat16
    body
  else:
    raise ValueError.newException("eval: dtype unsupported")

macro dispatchOp(E, T, engine, batch, n, rank: typed): untyped =
  result = nnkStmtList.newTree()
  var ifStmt = newNimNode(nnkIfStmt)
  for name, h in pairs(OpRegistry):
    let call = nnkCall.newTree(
      h, engine, batch, n, T, rank)
    ifStmt.add nnkElifBranch.newTree(call, nnkReturnStmt.newTree(newEmptyNode()))
  var elseBranch = newNimNode(nnkElse)
  elseBranch.add quote do:
    raiseAssert("Tensorcery internal error: no handlers for node with opName [" & n.opName() & "]")
  ifStmt.add elseBranch
  result.add ifStmt

proc evalWalk[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int)

proc evalNode*[E: HwEngine](engine: E, batch: BatchCommands[E], n: var TensorNode) =
  engine.withElemType(T, n.getImpl().dtype()):
    case n.getImpl().shape().p.len
    of 1: engine.evalWalk(batch, n, T, 1)
    of 2: engine.evalWalk(batch, n, T, 2)
    of 3: engine.evalWalk(batch, n, T, 3)
    of 4: engine.evalWalk(batch, n, T, 4)
    of 5: engine.evalWalk(batch, n, T, 5)
    else:
      raiseAssert("Tensorcery internal error: rank " & $n.getImpl().shape().p.len & " unsupported")

proc evalWalk[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int) =
  if n.isMaterialized():
    return
  dispatchOp(E, T, engine, batch, n, rank)
