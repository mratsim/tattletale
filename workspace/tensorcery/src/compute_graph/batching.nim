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

type
  OpHandlers = tuple[collect, kernel: NimNode]

  WalkFrame* = object
    ## One pending visit of the eval walk. The first visit pre-allocates
    ## the output and pushes the operands. The second visit enqueues the kernel.
    n*: TensorNode
    kernelPending*: bool

var OpRegistry {.compileTime.}: Table[string, OpHandlers]

macro registerOp*(op: static string, collect, kernel: typed) =
  OpRegistry[op] = (collect, kernel)

proc renameDef(name: NimNode, suffix: string): NimNode =
  ## Quoted operator names are preserved. The export marker is dropped.
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
  ## reads (see dispatchCollect and dispatchKernel).
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

  var doc: NimNode = nil
  var body = p[6]
  if body.len > 0 and body[0].kind == nnkCommentStmt:
    doc = body[0]
    body = newTree(nnkStmtList, body[1 ..^ 1])

  var impl = p.copyNimTree()
  impl[0] = implName
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
#  Eval-site dispatch
# ═══════════════════════════════════════════════════════════════════════

proc markedDispatch(collect: bool, engine, batch, n, stack, T: NimNode): NimNode =
  ## One branch per marked op: unmarked handlers and their kernels
  ## never instantiate. Each expansion re-reads the marks recorded so far.
  result = newNimNode(nnkIfStmt)
  for name, h in pairs(OpRegistry):
    if CacheCounter(name).value == 0:
      continue
    let handler = if collect: h.collect else: h.kernel
    var call = if collect:
        nnkCall.newTree(handler, engine, n, stack)
      else:
        nnkCall.newTree(handler, engine, batch, n)
    call.add(T)
    result.add nnkElifBranch.newTree(call, nnkDiscardStmt.newTree(newEmptyNode()))
  let noHandler = nnkCall.newTree(
    bindSym"raiseAssert",
    nnkInfix.newTree(ident"&",
      nnkInfix.newTree(ident"&",
        newLit("Tensorcery internal error: no handlers for node with opName ["),
        nnkCall.newTree(bindSym"opName", n)),
      newLit("]")))
  if result.len == 0:
    return noHandler
  result.add nnkElse.newTree(noHandler)

macro dispatchMarkedCollect*(T, engine, n, stack: typed): untyped =
  result = markedDispatch(true, engine, nil, n, stack, T)

macro dispatchMarkedKernel*(T, engine, batch, n: typed): untyped =
  result = markedDispatch(false, engine, batch, n, nil, T)

macro dispatchCollect*(engine, n, stack: typed): untyped =
  ## Dispatches one node's collect phase to its marked op's handler.
  result = quote do:
    case `n`.getImpl().dtype()
    of kFp16:
      type T = float16
      dispatchMarkedCollect(T, `engine`, `n`, `stack`)
    of kFp32:
      type T = float32
      dispatchMarkedCollect(T, `engine`, `n`, `stack`)
    of kBf16:
      type T = bfloat16
      dispatchMarkedCollect(T, `engine`, `n`, `stack`)
    else:
      raise ValueError.newException("eval: dtype unsupported")

macro dispatchKernel*(engine, n, batch: typed): untyped =
  ## Dispatches one node's kernel phase to its marked op's handler.
  result = quote do:
    case `n`.getImpl().dtype()
    of kFp16:
      type T = float16
      dispatchMarkedKernel(T, `engine`, `batch`, `n`)
    of kFp32:
      type T = float32
      dispatchMarkedKernel(T, `engine`, `batch`, `n`)
    of kBf16:
      type T = bfloat16
      dispatchMarkedKernel(T, `engine`, `batch`, `n`)
    else:
      raise ValueError.newException("eval: dtype unsupported")
