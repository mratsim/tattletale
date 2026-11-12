# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}

import std/[macros, macrocache, tables]
import workspace/crucible
import ../tensors_internal_types

# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

var OpRegistry {.compileTime.}: Table[string, NimNode]

macro registerOp*(op: static string, handler: typed) =
  OpRegistry[op] = handler

proc renameDef(name: NimNode, suffix: string): NimNode =
  ## Renames a procdef name node.
  ## Quoted operator names are preserved. The export marker is dropped and the Impl is module-private.
  let inner = if name.kind == nnkPostfix: name[1] else: name
  case inner.kind
  of nnkIdent:
    ident($inner & suffix)
  of nnkAccQuoted:
    nnkAccQuoted.newTree(ident($inner[0] & suffix))
  else:
    error("hasKernel: unsupported proc name", name)

macro hasKernel*(ops: untyped, p: untyped): untyped =
  ## Defines `p` as `<name>Impl` plus an untyped forwarding template `<name>`
  ## with the same signature.
  ##
  ## Each template expansion increments the compile-time counter of every op
  ## in `ops` before forwarding. Calling the op is the mark that the dispatch
  ## reads (see dispatchOp).
  ##
  ## - Pragmas are preserved on the Impl.
  ## - The doc comment moves to the template.
  ## - `<name>Impl` is module-private.
  ## - Calling `<name>Impl` directly skips the mark: forward through the template.
  ##
  ## Usage:
  ##   proc `+`*(a, b: Tensor): Tensor {.hasKernel: "Add".} =
  ##     Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))
  var keys: seq[string]
  if ops.kind == nnkStrLit:
    keys.add($ops)
  else:
    for k in ops:
      keys.add($k)

  let name = p[0]
  let implName = renameDef(name, "Impl")

  # doc comment moves to the template
  var doc: NimNode = nil
  var body = p[6]
  if body.len > 0 and body[0].kind == nnkCommentStmt:
    doc = body[0]
    body = newTree(nnkStmtList, body[1 ..^ 1])

  var impl = p.copyNimTree()
  impl[0] = implName
  # strip the hasKernel pragma itself, keep the others
  var kept = newNimNode(nnkPragma)
  for pr in impl[4]:
    let head = if pr.kind in {nnkExprColonExpr, nnkCall}: pr[0] else: pr
    if $head != "hasKernel":
      kept.add(pr)
  impl[4] = kept
  impl[6] = body

  var params = p[3].copyNimTree()
  var args: seq[NimNode] = @[implName]
  for i in 1 ..< params.len:
    let defs = params[i]
    for j in 0 ..< defs.len - 2:
      args.add(defs[j])
  let forward = newTree(nnkCall, args)

  var tBody = newTree(nnkStmtList)
  for k in keys:
    let kNode = newLit(k)
    tBody.add quote do:
      static:
        CacheCounter(`kNode`).inc()
  tBody.add forward

  var tpl = newTree(nnkTemplateDef, name, newEmptyNode(),
      p[2].copyNimTree(), params.copyNimTree(), newEmptyNode(), newEmptyNode(), tBody)
  if not doc.isNil():
    tpl[6] = newTree(nnkStmtList, doc, tBody)
  result = newTree(nnkStmtList, impl, tpl)

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
    if CacheCounter(name).value == 0:
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
