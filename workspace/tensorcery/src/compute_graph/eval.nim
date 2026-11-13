# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "views".}
{.experimental: "codeReordering".}

import std/[macros, macrocache, tables, sets, algorithm, strutils]
import workspace/crucible
import workspace/iface
import ../macros/ast_rebuilder
import ../tensors_internals
import ../tensors_internal_types
import ./metrics

export metrics

type
  OpHandlers = tuple[nodeType, kernel: NimNode]

  WalkFrame* = object
    n*: TensorGraph
    kernelPending*: bool

var OpRegistry {.compileTime.}: Table[string, OpHandlers]

proc magicSnapshotUsedOps*(): string {.compileTime.} =
  var names: seq[string]
  for name, _ in pairs(OpRegistry):
    if CacheCounter(name).value > 0:
      names.add name
  names.sort()
  names.join("\n")

proc collectNode[N](_: typedesc[N], n: var TensorGraph, stack: var seq[WalkFrame]): bool =
  var asX = n.to(N)
  if asX.isNil():
    return false
  for _, field in fieldPairs(asX[]):
    when field is TensorGraph:
      if not field.isNil():
        stack.add WalkFrame(n: field, kernelPending: false)
  return true

proc evalRoots*[E: HwEngine](engine: E, roots: var openArray[TensorGraph], snapshottedOps: static string) =
  ## Walks the graphs rooted at `roots` in LIFO post-order:
  ##
  ## - first visit: collect the operands
  ## - second visit: allocate the output and enqueue the kernel
  ## - a node whose op is absent from `snapshottedOps` raises the no-handler error
  ##
  ## Finally each evaluated root is replaced by its computed impl at the end.
  ## This drops ancestor nodes from the graph, potentially free-ing buffers.
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
          engine.dispatchCollect(n, stack, snapshottedOps)
        else:
          stack.add WalkFrame(n: v.src, kernelPending: false)
      else:
        # Views are metadata, not ops: the walk wires them outside the marked registry.
        if n.to(ViewNode).isNil():
          engine.dispatchAlloc(n)
        if not engine.dispatchViewKernel(n, batch):
          incMetric(kernelsEnqueued)
          engine.dispatchKernel(n, batch, snapshottedOps)
    if batch.entries.len != 0:
      batch.submit()
    # Replace the graph nodes by their computed tensors.
    # This drops ancestors and frees their buffers.
    for i in 0 ..< roots.len:
      if not roots[i].isNil():
        let matOffset = getMaterializedOffset(roots[i])
        let impl = roots[i].getImpl()
        case impl.dtype()
        of kFp16:
          let ci = impl.to(engine.getTensorImplType(float16))
          ci.offset = matOffset
          roots[i] = ci.to(TensorGraph)
        of kFp32:
          let ci = impl.to(engine.getTensorImplType(float32))
          ci.offset = matOffset
          roots[i] = ci.to(TensorGraph)
        of kBf16:
          let ci = impl.to(engine.getTensorImplType(bfloat16))
          ci.offset = matOffset
          roots[i] = ci.to(TensorGraph)

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
  of nnkSym:
    # A sem-checked name is a symbol node, operator symbol names print quoted
    let base = $inner
    if base.len > 1 and base[0] == '`' and base[^1] == '`':
      nnkAccQuoted.newTree(ident(base[1 ..^ 2] & suffix))
    else:
      ident(base & suffix)
  of nnkAccQuoted:
    nnkAccQuoted.newTree(ident($inner[0] & suffix))
  else:
    error("renameDef: unsupported proc name", name)

macro hasKernel*(ops: untyped, p: untyped): untyped =
  ## Marks an operation with a GPU kernel that needs to be compiled.
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

proc spliceSnapshot(body: NimNode): tuple[scanned: NimNode, uses: int] =
  ## Replaces every zero-argument `magicSnapshotUsedOps()` call in `body`
  ## with the `snapshottedOps` parameter and returns the count.
  #
  # Callee symbols stay resolved: converting a callee to an identifier
  # re-binds it ambiguously against the re-emitted template of the same name.
  proc walk(n: NimNode, count: var int): NimNode =
    if n.kind == nnkCall and n[0].kind in {nnkIdent, nnkSym} and
        $n[0] == "magicSnapshotUsedOps":
      if n.len != 1:
        error("magicSnapshotUsedOps takes no arguments", n)
      inc count
      return ident"snapshottedOps"
    if n.kind == nnkSym and n.symKind() in {nskParam, nskType}:
      return ident($n)
    if n.len == 0:
      return n
    result = copyNimNode(n)
    for i in 0 ..< n.len:
      result.add walk(n[i], count)
  var count = 0
  (body.walk(count), count)


proc makeEvalTemplate(counterKey: string, p: NimNode): NimNode =
  ## Splits an evaluating proc into the `<name>Impl` proc and the call-site template.
  ## The self-mark precedes the forward call, so the snapshot includes the surface's own op.
  var doc: NimNode = nil
  var body = p[6]
  if body.len > 0 and body[0].kind == nnkCommentStmt:
    doc = body[0]
    body = newTree(nnkStmtList, body[1 ..^ 1])

  var impl: NimNode = nil
  var forwardCall: NimNode = nil
  let (spliced, snapUses) = spliceSnapshot(body)
  # Only the caller-authored generics survive. A gensym'd implicit has no
  # binding in the template and is dropped.
  var genericParams = nnkGenericParams.newTree()
  for def in p[2]:
    if def.kind == nnkSym:
      if "gensym" notin $def:
        genericParams.add nnkIdentDefs.newTree(ident($def), newEmptyNode(), newEmptyNode())
    elif def.kind == nnkIdentDefs and "gensym" notin $def[0]:
      genericParams.add def
  let scannedBody = spliced
  if snapUses > 0:
    impl = p.copyNimTree()
    impl[0] = renameDef(p[0], "Impl")
    var kept = newNimNode(nnkPragma)
    for pr in impl[4]:
      let head = if pr.kind in {nnkExprColonExpr, nnkCall}: pr[0] else: pr
      if $head != "hasEval":
        kept.add(pr)
    impl[4] = kept
    impl[2] = genericParams
    impl[3] = rebuildUntypedAst(p[3])
    impl[3].add nnkIdentDefs.newTree(
        ident"snapshottedOps", nnkStaticTy.newTree(ident"string"), newEmptyNode())
    impl[6] = scannedBody

    var args: seq[NimNode] = @[impl[0]]
    let freshFormals = rebuildUntypedAst(p[3])
    for i in 1 ..< freshFormals.len:
      let defs = freshFormals[i]
      for j in 0 ..< defs.len - 2:
        if defs[j].eqIdent("_"):
          # `_` cannot be read, forward the generic bound by its typedesc[G] type
          let ty = defs[defs.len - 2]
          if not (ty.kind == nnkBracketExpr and $ty[0] == "typedesc" and ty[1].kind == nnkIdent):
            error("hasEval: a `_` parameter forwards only from a typedesc[G] type", defs[j])
          args.add ty[1]
        else:
          args.add defs[j]
    args.add nnkCall.newTree(ident"magicSnapshotUsedOps")
    forwardCall = nnkCall.newTree(args)

  var tBody = newTree(nnkStmtList)
  let kNode = newLit(counterKey)
  tBody.add quote do:
    static:
      CacheCounter(`kNode`).inc()
  if forwardCall.isNil():
    tBody.add body
  else:
    tBody.add forwardCall

  # A sem-checked name is a symbol. The identifier re-binds in the template
  # and the export marker is re-emitted from the symbol
  let tplName =
    if p[0].kind == nnkSym and isExported(p[0]):
      nnkPostfix.newTree(ident"*", rebuildUntypedAst(p[0]))
    else:
      rebuildUntypedAst(p[0])
  var tpl = newTree(nnkTemplateDef, tplName, p[1].copyNimTree(),
      genericParams,
      rebuildUntypedAst(p[3]),
      newEmptyNode(), newEmptyNode(), tBody)
  if not doc.isNil():
    tpl[6] = newTree(nnkStmtList, doc, tBody)
  if impl.isNil():
    return tpl
  result = newTree(nnkStmtList, impl, tpl)

macro hasEval*(name: static string, p: untyped): untyped =
  ## Mark an op that triggers evaluation
  ##
  ## Usage:
  ##   proc allClose*(t, r: Tensor): bool {.hasEval: "AllClose".} =
  ##     Tensor(engine: t.engine, node: tn_allClose(t.engine, t.node, r.node)).item(float32, magicSnapshotUsedOps()) == 0.0'f32
  makeEvalTemplate(name, p)

macro hasEval*(p: untyped): untyped =
  let inner = if p[0].kind == nnkPostfix: p[0][1] else: p[0]
  let key = if inner.kind == nnkAccQuoted: $inner[0] else: $inner
  makeEvalTemplate(key, p)

# ═══════════════════════════════════════════════════════════════════════
#  Eval-site dispatch
# ═══════════════════════════════════════════════════════════════════════
#
# Marked-op snapshots thread from the call site to evalRoots as a static parameter.
# Per node the walk calls:
#
#   dispatchCollect (first visit) / dispatchKernel (second visit)
#     │ template: binds T from the runtime dtype of n
#     ▼
#   dispatchMarked ── markedDispatch
#     │ macro: one elif-branch per registered op present in the threaded
#     │ snapshot. Ops absent from the snapshot and their kernels never
#     │ instantiate
#     ▼
#   collect: collectNode[N](n, stack) ── pushes the node's TensorGraph fields
#   kernel: handler(engine, batch, n, T): bool ── first true wins, no match raises

proc markedDispatch(collect: bool, engine, batch, n, stack, T: NimNode, snapshottedOps: string): NimNode =
  result = newNimNode(nnkIfStmt)
  let marked = snapshottedOps.split('\n').toHashSet()
  for name, h in pairs(OpRegistry):
    if name notin marked:
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

macro dispatchMarked*(engine, n, batch, stack, T: typed, snapshottedOps: static string, collect: static bool): untyped =
  result = markedDispatch(collect, engine, batch, n, stack, T, $snapshottedOps)

template dispatchCollect*(engine, n, stack: untyped, snapshottedOps: untyped) =
  case n.getImpl().dtype()
  of kFp16:
    type T = float16
    engine.dispatchMarked(n, nil, stack, T, snapshottedOps, true)
  of kFp32:
    type T = float32
    engine.dispatchMarked(n, nil, stack, T, snapshottedOps, true)
  of kBf16:
    type T = bfloat16
    engine.dispatchMarked(n, nil, stack, T, snapshottedOps, true)

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

template dispatchKernel*(engine, n, batch, snapshottedOps: untyped) =
  case n.getImpl().dtype()
  of kFp16:
    type T = float16
    engine.dispatchMarked(n, batch, nil, T, snapshottedOps, false)
  of kFp32:
    type T = float32
    engine.dispatchMarked(n, batch, nil, T, snapshottedOps, false)
  of kBf16:
    type T = bfloat16
    engine.dispatchMarked(n, batch, nil, T, snapshottedOps, false)
