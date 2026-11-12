# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/[macros, sets, tables]
import workspace/crucible
import ../tensors_internal_types

# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

var OpRegistry {.compileTime.}: Table[string, NimNode]
var opsMarked {.compileTime.}: HashSet[string]

macro registerOp*(op: static string, handler: typed) =
  OpRegistry[op] = handler

proc markOpUsage*(op: static string) {.compileTime.} =
  ## Records `op` as used by this compilation. Only marked ops get a dispatch
  ## branch (see dispatchOp). Call it only as a direct statement of a macro
  ## body (the Tensor sugar macros execute per call site):
  ## - a `static:` block runs once at definition and marks unused ops
  ## - a compileTime call from a runtime proc body compiles but does nothing
  opsMarked.incl(op)

# ═══════════════════════════════════════════════════════════════════════
#  Graph batching
# ═══════════════════════════════════════════════════════════════════════

template withElemType(engine: HwEngine, ElemType: untyped, dtype: Dtype, body: untyped) =
  case dtype
  of kFp16:
    type ElemType = float16
    body
  of kFp32:
    type ElemType = float32
    body
  of kBf16:
    type ElemType = bfloat16
    body
  else:
    raise ValueError.newException("eval: dtype unsupported")

macro dispatchOp(E, T, engine, batch, n, rank: typed): untyped =
  # Expands one branch per marked registry entry. Unmarked handlers never instantiate,
  # and their kernels never accumulate.
  # This macro reads the marks at its expansion, which happens at the first
  # enqueueWalk instantiation. Ops marked afterwards are invisible to it.
  result = nnkStmtList.newTree()
  var ifStmt = newNimNode(nnkIfStmt)
  for name, h in pairs(OpRegistry):
    if name notin opsMarked:
      continue
    let call = nnkCall.newTree(
      h, engine, batch, n, T, rank)
    ifStmt.add nnkElifBranch.newTree(call, nnkReturnStmt.newTree(newEmptyNode()))
  var noHandler = quote do:
    raiseAssert("Tensorcery internal error: no handlers for node with opName [" & n.opName() & "]")
  if ifStmt.len == 0:
    # no op marked in this compilation: nothing can dispatch
    result.add noHandler
  else:
    var elseBranch = newNimNode(nnkElse)
    elseBranch.add noHandler
    ifStmt.add elseBranch
    result.add ifStmt

proc enqueueWalk[E: HwEngine, T](engine: E, batch: BatchCommands[E], n: var TensorNode, _: typedesc[T], rank: static int) =
  if n.isPreAllocated():
    return
  dispatchOp(E, T, engine, batch, n, rank)

proc enqueueNode*[E: HwEngine](engine: E, batch: BatchCommands[E], n: var TensorNode) =
  engine.withElemType(T, n.getImpl().dtype()):
    case n.getImpl().shape().p.len
    of 1: engine.enqueueWalk(batch, n, T, 1)
    of 2: engine.enqueueWalk(batch, n, T, 2)
    of 3: engine.enqueueWalk(batch, n, T, 3)
    of 4: engine.enqueueWalk(batch, n, T, 4)
    of 5: engine.enqueueWalk(batch, n, T, 5)
    else:
      raiseAssert("Tensorcery internal error: rank " & $n.getImpl().shape().p.len & " unsupported")
