# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}
{.experimental: "codeReordering".}

import std/[macros, macrocache, tables]
import workspace/crucible
import workspace/iface
import ../tensors_internals
import ../tensors_internal_types
import ./metrics
export metrics

type
  OpHandlers = tuple[nodeType, kernel: NimNode]

  WalkFrame* = object
    ## One pending visit of the eval walk.
    ## The first visit pushes the operands.
    ## The second visit allocates the output and enqueues the kernel.
    n*: TensorGraph
    kernelPending*: bool

var OpRegistry {.compileTime.}: Table[string, OpHandlers]

proc collectNode[N](_: typedesc[N], n: var TensorGraph, stack: var seq[WalkFrame]): bool =
  var asX = n.to(N)
  if asX.isNil():
    return false
  for _, field in fieldPairs(asX[]):
    when field is TensorGraph:
      if not field.isNil():
        stack.add WalkFrame(n: field, kernelPending: false)
  return true

template evalRoots*(engine: HwEngine, roots: var openArray[TensorGraph]) =
  if roots.len != 0:
    incMetric(evals)
    var batch = engine.recordBatch()
    var stack: seq[WalkFrame]
    for i in 0 ..< roots.len:
      stack.add WalkFrame(n: roots[i], kernelPending: false)
    while stack.len > 0:
      let frame = stack.pop()
      var n = frame.n
      if not frame.kernelPending:
        if n.isPreAllocated():
          incMetric(nodesSkippedAllocated)
          continue
        stack.add WalkFrame(n: n, kernelPending: true)
        let v = n.to(ViewNode)
        if v.isNil():
          incMetric(nodesDispatched)
          engine.dispatchCollect(n, stack)
        else:
          stack.add WalkFrame(n: v.src, kernelPending: false)
      else:
        # Views are metadata, not ops: the walk wires them outside the marked registry.
        if n.to(ViewNode).isNil():
          engine.dispatchAlloc(n)
        if not engine.dispatchViewKernel(n, batch):
          incMetric(kernelsEnqueued)
          engine.dispatchKernel(n, batch)
    if batch.entries.len != 0:
      batch.submit()
    # Graph collapse: roots become leaves over their computed impls,
    # later evals walk nothing upstream.
    for i in 0 ..< roots.len:
      if not roots[i].isNil():
        let impl = roots[i].getImpl()
        case impl.dtype()
        of kFp16:
          roots[i] = impl.to(engine.getTensorImplType(float16)).to(TensorGraph)
        of kFp32:
          roots[i] = impl.to(engine.getTensorImplType(float32)).to(TensorGraph)
        of kBf16:
          roots[i] = impl.to(engine.getTensorImplType(bfloat16)).to(TensorGraph)

# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

macro registerOp*(N, kernel: typed, name: static string) =
  OpRegistry[name] = (nodeType: N, kernel: kernel)

proc renameDef(name: NimNode, suffix: string): NimNode =
  let inner = if name.kind == nnkPostfix: name[1] else: name
  case inner.kind
  of nnkIdent:
    ident($inner & suffix)
  of nnkAccQuoted:
    nnkAccQuoted.newTree(ident($inner[0] & suffix))
  else:
    error("hasKernel: unsupported proc name", name)

macro hasKernel*(ops: untyped, p: untyped): untyped =
  ## Defines `p` as a forwarding template with the same signature.
  ## Calling the op marks it for kernel generation and dispatch.
  ##
  ## Usage:
  ##   proc `+`*(a, b: Tensor): Tensor {.hasKernel: "Add".} =
  ##     Tensor(engine: a.engine, node: tn_add(a.engine, a.node, b.node))
  #
  # Mechanics:
  # - calling `<name>Impl` directly skips the mark: forward through the template
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
#
# eval(tensors)/item() expand evalRoots. Per node the walk calls:
#
#   dispatchCollect (first visit) / dispatchKernel (second visit)
#     │ template: binds T from the runtime dtype of n
#     ▼
#   dispatchMarked ── markedDispatch
#     │ macro: one elif-branch per op marked at this point in compilation
#     │ (CacheCounter, incremented by calling the op's public function);
#     │ unmarked collect and their kernels never instantiate
#     ▼
#   collect: collectNode[N](n, stack) ── pushes the node's TensorGraph fields
#   kernel: handler(engine, batch, n, T): bool ── first true wins, no match raises

proc markedDispatch(collect: bool, engine, batch, n, stack, T: NimNode): NimNode =
  result = newNimNode(nnkIfStmt)
  for name, h in pairs(OpRegistry):
    if CacheCounter(name).value == 0:
      continue
    var call: NimNode
    if collect:
      call = nnkCall.newTree(bindSym"collectNode", h.nodeType, n, stack)
    else:
      call = nnkCall.newTree(h.kernel, engine, batch, n)
      call.add(T)
    result.add nnkElifBranch.newTree(call, nnkDiscardStmt.newTree(newEmptyNode()))
  let noHandler = nnkCall.newTree(
    bindSym"raiseAssert",
    newLit("Tensorcery internal error: no handler for the dispatched node"))
  if result.len == 0:
    return noHandler
  result.add nnkElse.newTree(noHandler)

macro dispatchMarked*(engine, n, batch, stack, T: typed, collect: static bool): untyped =
  result = markedDispatch(collect, engine, batch, n, stack, T)

template dispatchCollect*(engine, n, stack: untyped) =
  case n.getImpl().dtype()
  of kFp16:
    type T = float16
    engine.dispatchMarked(n, nil, stack, T, true)
  of kFp32:
    type T = float32
    engine.dispatchMarked(n, nil, stack, T, true)
  of kBf16:
    type T = bfloat16
    engine.dispatchMarked(n, nil, stack, T, true)

template dispatchAlloc*(engine, n: untyped) =
  case n.getImpl().dtype()
  of kFp16:
    type T = float16
    engine.allocDataIn(n.getImpl().to(engine.getTensorImplType(T)))
  of kFp32:
    type T = float32
    engine.allocDataIn(n.getImpl().to(engine.getTensorImplType(T)))
  of kBf16:
    type T = bfloat16
    engine.allocDataIn(n.getImpl().to(engine.getTensorImplType(T)))

template dispatchKernel*(engine, n, batch: untyped) =
  case n.getImpl().dtype()
  of kFp16:
    type T = float16
    engine.dispatchMarked(n, batch, nil, T, false)
  of kFp32:
    type T = float32
    engine.dispatchMarked(n, batch, nil, T, false)
  of kBf16:
    type T = bfloat16
    engine.dispatchMarked(n, batch, nil, T, false)
