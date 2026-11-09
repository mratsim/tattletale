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
  ## OpName to handler symbol, populated by op files at sem

macro registerOp*(op: static string, handler: typed) =
  ## Registration entry of an op file, the value is the generic symbol
  OpRegistry[op] = handler

# ═══════════════════════════════════════════════════════════════════════
#  The walk
# ═══════════════════════════════════════════════════════════════════════

template withMem(engine: HwEngine, dtype: Dtype, body: untyped) =
  ## Binds Mem over the body from the node's own dtype
  ## The one place the dtype case exists
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
  ## Expands one typed branch per registered op, each handler symbol
  ## instantiates with the walk's own bindings
  result = nnkStmtList.newTree()
  var ifStmt = newNimNode(nnkIfStmt)
  for name, h in pairs(OpRegistry):
    let call = nnkCall.newTree(
      nnkBracketExpr.newTree(h, E, Sh, Mem), engine, batch, n)
    ifStmt.add nnkElifBranch.newTree(call, nnkReturnStmt.newTree(newEmptyNode()))
  var elseBranch = newNimNode(nnkElse)
  elseBranch.add quote do:
    raise ValueError.newException(
      "eval: op '" & `n`.opName() & "' unsupported")
  ifStmt.add elseBranch
  result.add ifStmt

proc evalWalk[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: TensorNode)

proc evalNode*[E: HwEngine](engine: E, batch: BatchCommands[E], n: TensorNode) =
  ## Op enqueue handlers recurse into a node's children through this proc,
  ## the child walks at its own dtype and rank.
  engine.withMem(n.getImpl().dtype()):
    case n.getImpl().shape().len
    of 1: evalWalk[E, (int,), Mem](engine, batch, n)
    of 2: evalWalk[E, (int, int), Mem](engine, batch, n)
    of 3: evalWalk[E, (int, int, int), Mem](engine, batch, n)
    of 4: evalWalk[E, (int, int, int, int), Mem](engine, batch, n)
    of 5: evalWalk[E, (int, int, int, int, int), Mem](engine, batch, n)
    else:
      raise ValueError.newException(
        "eval: rank " & $n.getImpl().shape().len & " unsupported")

proc evalWalk[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: TensorNode) =
  if n.opName() == "":
    return
  dispatchOp(E, Sh, Mem, engine, batch, n)
