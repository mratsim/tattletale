# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/tables
import std/macros
import workspace/crucible
import workspace/crucible/src/runtime/engines/engines_api
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

template withMem(engine: HwEngine, dtype: Dtype, body: untyped) =
  case dtype
  of kFp32:
    type Mem {.inject.} = typeof(engine.allocDevice(float32, 0))
    body
  of kBf16:
    type Mem {.inject.} = typeof(engine.allocDevice(bfloat16, 0))
    body
  else:
    raise ValueError.newException("eval: dtype unsupported")

macro dispatchOp(E, Sh, Mem, engine, batch, n: typed): untyped =
  result = nnkStmtList.newTree()
  var ifStmt = newNimNode(nnkIfStmt)
  for name, h in pairs(OpRegistry):
    let call = nnkCall.newTree(
      nnkBracketExpr.newTree(h, E, Sh, Mem), engine, batch, n)
    ifStmt.add nnkElifBranch.newTree(call, nnkReturnStmt.newTree(newEmptyNode()))
  var elseBranch = newNimNode(nnkElse)
  elseBranch.add quote do:
    raiseAssert("Tensorcery internal error: somehow there is a graph node with no handlers")
  ifStmt.add elseBranch
  result.add ifStmt

proc evalWalk[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode)

proc evalNode*[E: HwEngine](engine: E, batch: BatchCommands[E], n: var TensorNode) =
  engine.withMem(n.getImpl().dtype()):
    case n.getImpl().shape().len
    of 1: evalWalk[E, (int,), Mem](engine, batch, n)
    of 2: evalWalk[E, (int, int), Mem](engine, batch, n)
    of 3: evalWalk[E, (int, int, int), Mem](engine, batch, n)
    of 4: evalWalk[E, (int, int, int, int), Mem](engine, batch, n)
    of 5: evalWalk[E, (int, int, int, int, int), Mem](engine, batch, n)
    else:
      raiseAssert("Tensorcery internal error: rank " & $n.getImpl().shape().len & " unsupported")

proc evalWalk[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode) =
  if n.isMaterialized():
    return
  dispatchOp(E, Sh, Mem, engine, batch, n)
