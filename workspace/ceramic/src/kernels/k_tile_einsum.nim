# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Einsum over the tile algebra: index notation inside a device proc body.
##
## The macro parses the statement statically and lowers every operand
## into a relabeled view with one (size, stride) mode per einsum label,
## stride-0 size-1 modes for labels that the operand lacks.
##
## The general contraction body addresses threads as threadgroup-flat-id * 32 + lane.
## Launch it with one lane-width of threads per threadgroup (`blk = (32, 1)`).

import std/[macros, sequtils, strutils, tables]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_tile_sum
import workspace/ceramic/src/kernels/k_tile_gemm

# Mma schedule tile size: a multiple of the Metal atom's (8, 8, 8) cell,
# paddedLoad/guardedStore edges of the gemm handle tiles that do not divide it
const EinsumMmaTile = 8

proc mkLet(name, val: NimNode): NimNode =
  nnkLetSection.newTree(nnkIdentDefs.newTree(name, newEmptyNode(), val))

# ═══════════════════════════════════════════════════════════════════════
#  Static parse: statement -> labels
# ═══════════════════════════════════════════════════════════════════════

type EinsumOperand = object
  ## One `*`-chained right-side operand:
  ## - arg: the typed argument the operand symbol resolves to, nil before resolution
  ## - labels: the operand's labels in axis order
  ## - staticSizes: compile-time axis sizes, 0 where a leaf is a runtime int
  arg: NimNode
  sym: string
  labels: seq[string]
  staticSizes: seq[int]

type EinsumPlan = object
  ## Parsed statement:
  ## - dst: the left side's root identifier, emitted verbatim, it resolves
  ##   in the enclosing scope
  ## - lhsLabels: the left side's labels in output-axis order
  ## - operands: the right side's `*`-chained operands
  ## - bound: the contraction labels in first-appearance order
  dst: NimNode
  lhsLabels: seq[string]
  operands: seq[EinsumOperand]
  bound: seq[string]

proc parseLhsLabels(n: NimNode): seq[string] =
  case n.kind
  of nnkIdent:
    discard # scalar output, no labels
  of nnkBracketExpr:
    for i in 1 ..< n.len:
      n[i].expectKind(nnkIdent)
      result.add n[i].strVal
  else:
    error("einsum expects the explicit form `dst[indices] = rhs`, got " & $n.kind & " on the left side", n)

proc callHeadName(n: NimNode): string =
  n[0].toStrLit.strVal

proc collectRhsOperands(n: NimNode, ops: var seq[EinsumOperand]) =
  case n.kind
  of nnkInfix:
    if n[0].toStrLit.strVal != "*":
      error("einsum's right side joins operands with operator " & n[0].repr, n)
    collectRhsOperands(n[1], ops)
    collectRhsOperands(n[2], ops)
  of nnkBracketExpr:
    var op: EinsumOperand
    op.sym = n[0].strVal
    for i in 1 ..< n.len:
      n[i].expectKind(nnkIdent)
      op.labels.add n[i].strVal
    ops.add op
  of nnkCall:
    # generic-context form of `A[i, j]`
    if callHeadName(n) != "[]":
      error("einsum's right side operand must be an indexed view, got a call of " & callHeadName(n), n)
    var op: EinsumOperand
    op.sym = n[1].strVal
    for i in 2 ..< n.len:
      op.labels.add n[i].strVal
    ops.add op
  else:
    error("einsum's right side is a `*` chain of bracket-indexed operands, got " & $n.kind, n)

proc shapeLeafTypes(sh: NimNode): seq[NimNode] =
  ## Leaf type nodes of a layout shape type, nested modes flattened
  case sh.kind
  of nnkTupleConstr, nnkPar:
    for c in sh:
      result.add shapeLeafTypes(c)
  of nnkTupleTy:
    for c in sh:
      result.add shapeLeafTypes(c[^2])
  else:
    result.add sh

proc staticLeafSize(leaf: NimNode): int =
  ## Compile-time value of a shape leaf type, 0 when the leaf is a runtime int
  if leaf.kind == nnkBracketExpr and leaf.len == 2 and leaf[0].strVal == "Int":
    leaf[1].intVal.int
  else:
    0

proc resolveOperands(plan: var EinsumPlan, tensors: NimNode, stmt: NimNode) =
  ## Binds every right-side operand to one of the macro's typed arguments,
  ## checks the label count against the rank, records static sizes
  for op in plan.operands.mitems:
    for arg in tensors:
      if arg.strVal == op.sym:
        op.arg = arg
        break
    if op.arg.isNil:
      error("einsum operand `" & op.sym & "` names none of the macro's arguments: " & tensors.mapIt(it.strVal).join(", "), stmt)
    let ty = op.arg.getTypeInst()
    if ty.kind != nnkBracketExpr or ty.len != 4 or ty[0].strVal != "TensorView":
      error("einsum operates on TensorView arguments, `" & op.sym & "` is " & ty.repr, stmt)
    let leaves = shapeLeafTypes(ty[2])
    if leaves.len != op.labels.len:
      error("einsum operand `" & op.sym & "` takes " & $op.labels.len & " indices but its tensor has rank " & $leaves.len, stmt)
    op.staticSizes = leaves.mapIt(staticLeafSize(it))

proc checkLabelSizes(plan: EinsumPlan, stmt: NimNode) =
  ## Compile-time contradiction check over the labels whose sizes are static.
  ## Runtime leaves hide their sizes from the macro, the caller's shapes
  ## are trusted then
  var known = initTable[string, int]()
  for op in plan.operands:
    for k, l in op.labels:
      let s = op.staticSizes[k]
      if s == 0:
        continue
      if l in known and known[l] != s:
        error("einsum label `" & l & "` has size " & $s & " on `" & op.sym & "` but " & $known[l] & " elsewhere", stmt)
      known[l] = s

proc classifyLabels(plan: var EinsumPlan, stmt: NimNode) =
  ## Contraction labels: every right-side label the left side does not name.
  ## A left-side label that no right-side operand names is a parse error
  for l in plan.lhsLabels:
    if not plan.operands.anyIt(l in it.labels):
      error("einsum's left side names label `" & l & "` which no right-side operand names", stmt)
  for op in plan.operands:
    for l in op.labels:
      if l notin plan.lhsLabels and l notin plan.bound:
        plan.bound.add l

proc parseEinsumStmt(tensors, stmt: NimNode): EinsumPlan =
  ## Parses the one supported form: `dst[indices] = op1[i] * op2[j] * ...`
  var s = if stmt.kind == nnkStmtList and stmt.len == 1: stmt[0] else: stmt
  if s.kind == nnkCall and callHeadName(s) == "[]=":
    # generic-context form of `dst[i] = rhs`: the left side's arguments
    # flatten into the call, the right side is its last argument
    var lhs = nnkBracketExpr.newTree(s[1])
    for i in 2 ..< s.len - 1:
      lhs.add s[i]
    s = nnkAsgn.newTree(lhs, s[^1])
  if s.kind != nnkAsgn:
    error("einsum supports the explicit form `dst[indices] = rhs` only, got " & $s.kind, s)
  if s[0].kind == nnkBracketExpr:
    result.dst = s[0][0]
  else:
    result.dst = s[0]
  result.lhsLabels = parseLhsLabels(s[0])
  collectRhsOperands(s[1], result.operands)
  if result.operands.len == 0:
    error("einsum's right side is empty", s)
  resolveOperands(result, tensors, s)
  checkLabelSizes(result, s)
  classifyLabels(result, s)

# ═══════════════════════════════════════════════════════════════════════
#  Lowering: relabeled views + kernel class
# ═══════════════════════════════════════════════════════════════════════

proc isMmClass(plan: EinsumPlan): bool =
  ## mm-class routing test: exactly 2 operands sharing exactly one bound label.
  ## Free labels are disjoint across the operands with at least one per operand.
  ## More than one bound label needs a nested K mode, which the gemm kernel's
  ## flat K-tile iteration cannot traverse, so those contractions stay general
  if plan.operands.len != 2 or plan.bound.len != 1:
    return false
  let a = plan.operands[0]
  let b = plan.operands[1]
  for l in plan.bound:
    if l notin a.labels or l notin b.labels:
      return false
  let freeA = plan.lhsLabels.filterIt(it in a.labels)
  let freeB = plan.lhsLabels.filterIt(it in b.labels)
  result = freeA.len > 0 and freeB.len > 0 and not freeA.anyIt(it in freeB)

proc relabelCall(arg: NimNode, argLabels: seq[string], labels: seq[string]): NimNode =
  ## `make_view(arg.data, make_layout((label-order sizes), (label-order strides)))`:
  ## one (size, stride) mode per requested label, drawn from the tensor's layout
  ## with (1, 0) for labels that the tensor lacks
  var
    sizes = nnkTupleConstr.newTree()
    strides = nnkTupleConstr.newTree()
  for l in labels:
    let k = argLabels.find(l)
    let shape = nnkDotExpr.newTree(nnkDotExpr.newTree(arg, ident"layout"), ident"shape")
    let stride = nnkDotExpr.newTree(nnkDotExpr.newTree(arg, ident"layout"), ident"stride")
    if k >= 0:
      sizes.add nnkBracketExpr.newTree(shape, newLit(k))
      strides.add nnkBracketExpr.newTree(stride, newLit(k))
    else:
      sizes.add newLit(1)
      strides.add newLit(0)
  result = nnkCall.newTree(bindSym"make_view",
    nnkDotExpr.newTree(arg, ident"data"),
    nnkCall.newTree(bindSym"make_layout", sizes, strides))

proc emitSumTotal(plan: EinsumPlan): NimNode =
  ## A fully contracted operand's label order is already the view's order
  nnkCall.newTree(bindSym"sum", plan.dst, plan.operands[0].arg)

proc emitMm(plan: EinsumPlan): NimNode =
  ## Relabeled (M, K)/(N, K)/(M, N) views and the mma gemm of k_tile_gemm
  let
    a = plan.operands[0]
    b = plan.operands[1]
    freeA = plan.lhsLabels.filterIt(it in a.labels)
    freeB = plan.lhsLabels.filterIt(it in b.labels)
    aView = genSym(nskLet, "aViewMk")
    bView = genSym(nskLet, "bViewNk")
    dView = genSym(nskLet, "dViewMn")
    mma = genSym(nskConst, "mma")
    aT = nnkDotExpr.newTree(a.arg, ident"T")
    bT = nnkDotExpr.newTree(b.arg, ident"T")
    tile = newLit(EinsumMmaTile)
  result = newStmtList()
  result.add mkLet(aView, relabelCall(a.arg, a.labels, freeA & plan.bound))
  result.add mkLet(bView, relabelCall(b.arg, b.labels, freeB & plan.bound))
  # Dst view: the left-side labels are the dst's own axes in order
  result.add mkLet(dView,
    relabelCall(plan.dst, plan.lhsLabels, freeA & freeB))
  # Mma schedule: a const inside the kernel body, where the backend
  # resolves the atom and accumulates in fp32
  result.add quote do:
    const `mma` = tiledMma(float32, `aT`, `bT`, `tile`, `tile`, `tile`)
  result.add nnkCall.newTree(bindSym"matmul", mma, dView, aView, bView)

proc readComponent(plan: EinsumPlan, l: string, outCrd: NimNode, boundVars: seq[NimNode]): NimNode =
  ## Coordinate component of label `l` for an operand read: the output coordinate
  ## slot of a free label, the bound label's loop variable
  let p = plan.lhsLabels.find(l)
  if p >= 0:
    if plan.lhsLabels.len == 1:
      # rank-1 output: idx2crd returns the scalar coordinate
      outCrd
    else:
      nnkBracketExpr.newTree(outCrd, newLit(p))
  else:
    boundVars[plan.bound.find(l)]

proc boundEndExpr(plan: EinsumPlan, q: int, viewNodes: seq[NimNode]): NimNode =
  ## Exclusive loop end for bound label q: the size at the label's canonical
  ## slot of the first operand's relabeled view that names the label
  for i, op in plan.operands:
    if plan.bound[q] in op.labels:
      let
        shapeCall = nnkCall.newTree(bindSym"shape", viewNodes[i])
        slot = newLit(plan.lhsLabels.len + q)
      return quote do: `shapeCall`[`slot`]
  doAssert false, "unreachable: bound labels come from the operands"

proc emitGeneral(plan: EinsumPlan): NimNode =
  ## One general per-thread contraction body:
  ## - grid-stride over the output's flat size
  ## - per element, a serial product over the bound labels accumulates
  ##   into `acc`
  ## - one overwrite writes the output slot
  let canonical = plan.lhsLabels & plan.bound
  var boundVars: seq[NimNode]
  for l in plan.bound:
    boundVars.add genSym(nskForVar, l)

  result = newStmtList()
  var viewNodes: seq[NimNode]
  for op in plan.operands:
    let v = genSym(nskLet, op.sym & "View")
    result.add mkLet(v, relabelCall(op.arg, op.labels, canonical))
    viewNodes.add v

  let
    dstShape = nnkDotExpr.newTree(nnkDotExpr.newTree(plan.dst, ident"layout"), ident"shape")
    dstT = nnkDotExpr.newTree(plan.dst, ident"T")
    thrID = genSym(nskLet, "threadID")
    totalThreads = genSym(nskLet, "totalThreads")
    outSize = genSym(nskLet, "outSize")
    out0 = genSym(nskLet, "out0")
    outFlat = genSym(nskForVar, "outFlat")
    outCrd = genSym(nskLet, "outCrd")
    acc = genSym(nskVar, "acc")
    gridSizeCall = nnkCall.newTree(bindSym"getGridSize")
    laneWidthCall = nnkCall.newTree(bindSym"getLaneWidth")
    tileIdCall = nnkCall.newTree(bindSym"getLocalTileID", dstShape)
  result.add mkLet(thrID, nnkCall.newTree(bindSym"getLocalThreadID"))
  result.add mkLet(totalThreads, quote do: `gridSizeCall` * `laneWidthCall`)
  result.add mkLet(outSize,
    nnkCall.newTree(bindSym"toInt",
      nnkCall.newTree(bindSym"size", nnkDotExpr.newTree(plan.dst, ident"layout"))))
  result.add mkLet(out0,
    quote do: `tileIdCall` * `laneWidthCall` + `thrID`)

  var loopBody = newStmtList()
  loopBody.add mkLet(outCrd, nnkCall.newTree(bindSym"idx2crd", dstShape, outFlat))
  loopBody.add nnkVarSection.newTree(nnkIdentDefs.newTree(acc, newEmptyNode(),
    nnkCall.newTree(bindSym"zero", dstT)))

  var product: NimNode = nil
  for i, op in plan.operands:
    var read = nnkBracketExpr.newTree(viewNodes[i])
    for l in canonical:
      read.add readComponent(plan, l, outCrd, boundVars)
    product = if product.isNil: read
              else: quote do: `product` * `read`
  var inner: NimNode = quote do:
    `acc` += `product`
  for q in countdown(plan.bound.len - 1, 0):
    let
      boundVar = boundVars[q]
      boundEnd = boundEndExpr(plan, q, viewNodes)
    inner = quote do:
      for `boundVar` in 0 ..< `boundEnd`:
        `inner`
  loopBody.add inner
  let dstIdent = plan.dst
  loopBody.add quote do:
    `dstIdent`[`outCrd`] = `acc`

  result.add quote do:
    for `outFlat` in countup(`out0`, `outSize` - 1, `totalThreads`):
      `loopBody`

# ═══════════════════════════════════════════════════════════════════════
#  The macro
# ═══════════════════════════════════════════════════════════════════════

macro einsum*(tensors: varargs[typed], stmt: untyped): untyped =
  ## Einstein summation inside a device proc body over `distinct TensorView` operands
  ## parsed statically and lowered to relabeled views over the tile algebra.
  ##
  ## Explicit form only, the left side is the caller-provided output view:
  ##
  ##   einsum(A, B):
  ##     D[i, k] = A[i, j] * B[j, k]
  ##
  ## Left-side indices are never summed, their order is the output axis order.
  ## Right-side labels absent from the left side are contraction labels, summed with `+=`.
  ##
  ## Constraints:
  ## - operands name the macro's arguments, one index per rank
  ## - with static layout leaves, a size contradiction across operands is
  ##   a compile-time error
  ## - operand layouts need runtime int leaves at launch sites
  ## - a static `Int[N]` leaf breaks the generated runtime loops
  ##
  ## Kernel classes:
  ## - mm shape -> the mma gemm
  ## - 1 fully contracted operand -> k_tile_sum's `sum(dst, src)`, whose
  ##   zero-initialized-dst contract applies
  ## - everything else -> the general per-thread contraction body, which
  ##   overwrites every output slot it owns. Launch it with `blk = (32, 1)`
  let plan = parseEinsumStmt(tensors, stmt)
  if plan.lhsLabels.len == 0 and plan.operands.len == 1:
    result = emitSumTotal(plan)
  elif plan.isMmClass:
    result = emitMm(plan)
  else:
    result = emitGeneral(plan)
