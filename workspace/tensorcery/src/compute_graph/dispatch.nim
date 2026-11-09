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

type VisitedNodes* = Table[pointer, TensorNode]
  ## Per-eval dedupe state. The key is the erased record's object pointer:
  ## `cast[pointer](n.private_obj)`, stable per record.
  ## The value is the node's materialized replacement, stored after enqueue.
  ##
  ## A re-visited node returns the stored replacement, so every producer computes once per eval and aliased consumers read one enqueued result:
  ##
  ## - a node in two input fields (w + w)
  ## - sibling roots sharing a producer (z1 = p+x, z2 = p+y)

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

macro dispatchOp(E, Sh, Mem, engine, batch, n, visited: typed): untyped =
  ## Calls the one registered handler that owns the node and yields its bool.
  ## An unknown op name raises.
  result = nnkStmtList.newTree()
  var ifStmt = newNimNode(nnkIfStmt)
  for name, h in pairs(OpRegistry):
    let call = nnkCall.newTree(
      nnkBracketExpr.newTree(h, E, Sh, Mem), engine, batch, n, visited)
    ifStmt.add nnkElifBranch.newTree(call, newLit(true))
  var elseBranch = newNimNode(nnkElse)
  elseBranch.add quote do:
    raise ValueError.newException(
      "eval: op '" & `n`.opName() & "' unsupported")
  ifStmt.add elseBranch
  result.add ifStmt

proc evalWalk[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes)

proc evalNode*[E: HwEngine](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes) =
  engine.withMem(n.getImpl().dtype()):
    case n.getImpl().shape().len
    of 1: evalWalk[E, (int,), Mem](engine, batch, n, visited)
    of 2: evalWalk[E, (int, int), Mem](engine, batch, n, visited)
    of 3: evalWalk[E, (int, int, int), Mem](engine, batch, n, visited)
    of 4: evalWalk[E, (int, int, int, int), Mem](engine, batch, n, visited)
    of 5: evalWalk[E, (int, int, int, int, int), Mem](engine, batch, n, visited)
    else:
      raise ValueError.newException(
        "eval: rank " & $n.getImpl().shape().len & " unsupported")

proc evalWalk[E: HwEngine, Sh: tuple, Mem](engine: E, batch: BatchCommands[E], n: var TensorNode, visited: var VisitedNodes) =
  if n.opName() == "":
    return
  let key = cast[pointer](n.private_obj)
  if key in visited:
    n = visited[key]
    return
  if dispatchOp(E, Sh, Mem, engine, batch, n, visited):
    visited[key] = n
