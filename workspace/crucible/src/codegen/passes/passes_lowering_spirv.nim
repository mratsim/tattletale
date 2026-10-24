# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            SPIR-V lowering passes
#
# ############################################################
#
# Lowers the crucible IR for direct SPIR-V binary emission.
# The bundle registers the SPIR-V pipeline's passes.

import std/macros # for the `error` builtin (compile-time diagnostics)
import std/strutils # leaf param naming joins the field path
import std/tables # OrderedTable keys iterator
import ../ir/ast_datatypes
import ../ir/ast_type_constructors # initGpuPtrType
import ../ir/ast_type_resolvers # getExprType
import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_low_pointers
import ./passes_low_views

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSpirvLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = false)
  reg.registerSharedLoweringPasses()
  # The ternary rewrite registers before compound assignments and bool patching:
  # patchBoolToI32 wraps bool assignment right-hand sides in conv(gtInt32),
  # running first means the rewrite never sees those conv wrappers.
  # The fresh bool temps it introduces stay plain locals that patch ignores.
  reg.registerLowerTernaryToIf()
  # Compound assignments rewrite to plain assignments of the binop before
  # pointer-deref insertion, both copies of the address skeleton retype alike
  reg.registerLowerCompoundAssign()
  reg.registerInsertPointerDeref()
  # Object kernel params flatten to one scalar/pointer param per leaf.
  # The push block renders flat members only, bool leaves stay gtBool
  # here and the next pass retypes them.
  reg.registerFlattenObjectParams()
  # Bool kernel parameters declare i32 in the push-constant interface,
  # loads convert at the use boundary
  reg.registerPatchBoolToI32()

# ═══════════════════════════════════════════════════════════════════════
# Ternaries and lazy bools to if-statements
# ═══════════════════════════════════════════════════════════════════════

proc lowerTernaryExpr(ctx: var GpuContext, n: var GpuAst, pre: var seq[GpuAst])
proc lowerTernaryStmts(ctx: var GpuContext, stmts: var seq[GpuAst])
proc lowerTernaryStmt(ctx: var GpuContext, n: var GpuAst)

func registerLowerTernaryToIf(reg: var PassRegistry) =
  ## Rewrites ternaries and lazy bools into if-statements over hoisted temps,
  ## SPIR-V pipeline only, untyped ternary results refuse loudly
  reg.register("lowerTernaryToIf", pkTransform, phaseMain,
    "Rewrites gpuTernary nodes and lazy &&/|| into if-statements over hoisted temps, only the taken branch evaluates",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.lowerTernaryToIf()
  )

proc ternaryResultType(ctx: GpuContext, n: GpuAst): GpuType =
  ## Result type of a gpuTernary: the then-branch type.
  ## `true`/`false` read gtBool: the frontend emits bool literals as idents.
  ## Those idents carry no symbol type.
  if n.tThen.kind == gpuIdent and
      (n.tThen.symbol.name == "true" or n.tThen.symbol.name == "false"):
    return GpuType(kind: gtBool)
  result = ctx.getExprType(n.tThen)
  if result.isNil:
    error "passes_lowering_spirv: ternary result type is untyped"

proc rewriteTernary(ctx: var GpuContext, n: var GpuAst, resTy: GpuType, pre: var seq[GpuAst]) =
  ## Replaces one gpuTernary with a fresh temp: the temp declares
  ## uninitialized ahead of the statement, the if-statement assigns
  ## the taken branch only, and the expression reads the temp.
  ##
  ## Before (IR):   x = c ? a : b
  ## After (IR):    var _tn_N: T
  ##                if c: _tn_N = a else: _tn_N = b
  ##                x = _tn_N
  ##
  ## Ternaries nested in the condition hoist ahead of the group,
  ## ternaries nested in a branch hoist inside that branch
  ctx.lowerTernaryExpr(n.tCond, pre)
  let tempName = "_tn_" & $ctx.genSymCount
  inc ctx.genSymCount
  let tempIdent = GpuAst(kind: gpuIdent,
                         symbol: newSymbol(tempName, iSym = tempName,
                                           typ = resTy, symKind = gsLocal))
  var thenStmts = @[GpuAst(kind: gpuAssign, aLeft: tempIdent.clone(), aOp: "=",
                           aRight: n.tThen)]
  var elseStmts = @[GpuAst(kind: gpuAssign, aLeft: tempIdent.clone(), aOp: "=",
                           aRight: n.tElse)]
  ctx.lowerTernaryStmts(thenStmts)
  ctx.lowerTernaryStmts(elseStmts)
  let thenBody = if thenStmts.len == 1: thenStmts[0]
                 else: GpuAst(kind: gpuBlock, statements: thenStmts)
  let elseBody = if elseStmts.len == 1: elseStmts[0]
                 else: GpuAst(kind: gpuBlock, statements: elseStmts)
  pre.add GpuAst(kind: gpuVar, vName: tempIdent.clone(), vType: resTy,
                 vInit: GpuAst(kind: gpuDiscard), vMutable: true)
  pre.add GpuAst(kind: gpuIf, ifCond: n.tCond, ifThen: thenBody,
                 ifElifs: @[], ifElse: elseBody, ifIsExpr: false)
  n = tempIdent

proc rewriteLazyBool(ctx: var GpuContext, n: var GpuAst, pre: var seq[GpuAst]) =
  ## Normalizes a lazy bool to a ternary and rewrites it:
  ## `a && b` reads ternary(a, b, false), `a || b` reads ternary(a, true, b),
  ## the untaken branch never evaluates
  let isAnd = n.bOp.symbol.name == "&&"
  let taken = if isAnd: n.bRight
              else: GpuAst(kind: gpuIdent, symbol: newSymbol("true"))
  let untaken = if isAnd: GpuAst(kind: gpuIdent, symbol: newSymbol("false"))
                else: n.bRight
  var ternary = GpuAst(kind: gpuTernary, tCond: n.bLeft,
                       tThen: taken, tElse: untaken)
  ctx.rewriteTernary(ternary, GpuType(kind: gtBool), pre)
  n = ternary

proc lowerTernaryExpr(ctx: var GpuContext, n: var GpuAst, pre: var seq[GpuAst]) =
  ## Walks one expression subtree, rewriting every ternary and lazy bool,
  ## the hoisted [var, if] groups append to `pre` in evaluation order
  case n.kind
  of gpuTernary:
    let resTy = ctx.ternaryResultType(n)
    ctx.rewriteTernary(n, resTy, pre)
  of gpuBinOp:
    if n.bOp.symbol.name == "&&" or n.bOp.symbol.name == "||":
      ctx.rewriteLazyBool(n, pre)
    else:
      ctx.lowerTernaryExpr(n.bLeft, pre)
      ctx.lowerTernaryExpr(n.bRight, pre)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      ctx.lowerTernaryExpr(n.cArgs[i], pre)
  of gpuIndex:
    ctx.lowerTernaryExpr(n.iArr, pre)
    ctx.lowerTernaryExpr(n.iIndex, pre)
  of gpuDot:
    ctx.lowerTernaryExpr(n.dParent, pre)
  of gpuDeref:
    ctx.lowerTernaryExpr(n.dOf, pre)
  of gpuAddr:
    ctx.lowerTernaryExpr(n.aOf, pre)
  of gpuPrefix:
    ctx.lowerTernaryExpr(n.pVal, pre)
  of gpuConv:
    ctx.lowerTernaryExpr(n.convExpr, pre)
  of gpuCast:
    ctx.lowerTernaryExpr(n.cExpr, pre)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      ctx.lowerTernaryExpr(f.value, pre)
  of gpuArrayLit:
    for i in 0 ..< n.aValues.len:
      ctx.lowerTernaryExpr(n.aValues[i], pre)
  of gpuMaterialize:
    ctx.lowerTernaryExpr(n.mExpr, pre)
  of gpuBlock:
    ctx.lowerTernaryStmts(n.statements)
  of gpuConstexpr:
    ctx.lowerTernaryExpr(n.cValue, pre)
  of gpuIdent, gpuLit, gpuDiscard:
    discard
  else:
    error "passes_lowering_spirv: unsupported expression node " & $n.kind

proc lowerTernaryBranch(ctx: var GpuContext, body: var GpuAst) =
  ## Lowers one if-branch body: a block lowers its statement list in place,
  ## a bare statement lowers alone and rewraps when the rewrite hoists
  if body.kind == gpuBlock:
    ctx.lowerTernaryStmts(body.statements)
  else:
    var stmts = @[body]
    ctx.lowerTernaryStmts(stmts)
    if stmts.len == 1:
      body = stmts[0]
    else:
      body = GpuAst(kind: gpuBlock, statements: stmts)

func condNeedsRewrite(n: GpuAst): bool =
  ## Whether an expression subtree contains a node lowerTernaryExpr rewrites:
  ## a gpuTernary or a lazy &&/||. Only expression children are scanned,
  ## statement bodies lower through their own statement lists.
  case n.kind
  of gpuTernary:
    true
  of gpuBinOp:
    if n.bOp.symbol.name == "&&" or n.bOp.symbol.name == "||":
      true
    else:
      condNeedsRewrite(n.bLeft) or condNeedsRewrite(n.bRight)
  of gpuCall:
    for a in n.cArgs:
      if condNeedsRewrite(a):
        return true
    false
  of gpuIndex:
    condNeedsRewrite(n.iArr) or condNeedsRewrite(n.iIndex)
  of gpuDot:
    condNeedsRewrite(n.dParent)
  of gpuDeref:
    condNeedsRewrite(n.dOf)
  of gpuAddr:
    condNeedsRewrite(n.aOf)
  of gpuPrefix:
    condNeedsRewrite(n.pVal)
  of gpuConv:
    condNeedsRewrite(n.convExpr)
  of gpuCast:
    condNeedsRewrite(n.cExpr)
  of gpuObjConstr:
    for f in n.ocFields:
      if condNeedsRewrite(f.value):
        return true
    false
  of gpuArrayLit:
    for v in n.aValues:
      if condNeedsRewrite(v):
        return true
    false
  of gpuMaterialize:
    condNeedsRewrite(n.mExpr)
  of gpuBlock:
    for st in n.statements:
      if condNeedsRewrite(st):
        return true
    false
  of gpuConstexpr:
    condNeedsRewrite(n.cValue)
  else:
    false

proc lowerTernaryStmt(ctx: var GpuContext, n: var GpuAst) =
  ## Rewrites one statement: expression-level ternaries hoist into
  ## a `_ternary` block ahead of the statement, the caller's list walker
  ## flattens it back into the list
  var pre: seq[GpuAst]
  case n.kind
  of gpuBlock:
    ctx.lowerTernaryStmts(n.statements)
  of gpuVar:
    ctx.lowerTernaryExpr(n.vInit, pre)
  of gpuAssign:
    ctx.lowerTernaryExpr(n.aLeft, pre)
    ctx.lowerTernaryExpr(n.aRight, pre)
  of gpuIf:
    ctx.lowerTernaryExpr(n.ifCond, pre)
    ctx.lowerTernaryBranch(n.ifThen)
    # elif conditions evaluate after the then-branch (the backend emits the
    # chain in order), their hoisted temps must splice into the else branch,
    # never ahead of the whole if. When an elif condition needs a rewrite,
    # the chain nests into plain two-way ifs, each elif condition lowering
    # at its own evaluation position:
    #   if c1: B1 elif c2: B2 else: B3
    #     -> if c1: B1 else: (if c2: B2 else: B3)
    # A chain free of ternaries and lazy bools keeps the flat shape.
    var nestChain = false
    for el in n.ifElifs:
      if condNeedsRewrite(el.cond):
        nestChain = true
        break
    if nestChain:
      n.ifElse = GpuAst(kind: gpuIf, ifIsExpr: n.ifIsExpr,
                        ifCond: n.ifElifs[0].cond,
                        ifThen: n.ifElifs[0].body,
                        ifElifs: n.ifElifs[1 .. ^1],
                        ifElse: n.ifElse)
      n.ifElifs = @[]
      ctx.lowerTernaryBranch(n.ifElse)
    else:
      for el in n.ifElifs.mitems:
        ctx.lowerTernaryExpr(el.cond, pre)
        ctx.lowerTernaryBranch(el.body)
      if n.ifElse.kind != gpuDiscard:
        ctx.lowerTernaryBranch(n.ifElse)
  of gpuFor:
    ctx.lowerTernaryExpr(n.fStart, pre)
    ctx.lowerTernaryExpr(n.fEnd, pre)
    ctx.lowerTernaryExpr(n.fStep, pre)
    ctx.lowerTernaryStmt(n.fBody)
  of gpuReturn:
    ctx.lowerTernaryExpr(n.rValue, pre)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      ctx.lowerTernaryExpr(n.cArgs[i], pre)
  of gpuConstexpr:
    ctx.lowerTernaryExpr(n.cValue, pre)
  of gpuDiscard:
    discard
  else:
    error "passes_lowering_spirv: unsupported statement node " & $n.kind
  if pre.len > 0:
    n = GpuAst(kind: gpuBlock, blockLabel: "_ternary", statements: pre & @[n])

proc lowerTernaryStmts(ctx: var GpuContext, stmts: var seq[GpuAst]) =
  ## Walks a statement list in order, flattening `_ternary` splice blocks
  ## back into the list
  var outStmts: seq[GpuAst]
  for i in 0 ..< stmts.len:
    var st = stmts[i]
    ctx.lowerTernaryStmt(st)
    if st.kind == gpuBlock and st.blockLabel == "_ternary":
      for s in st.statements:
        outStmts.add s
    else:
      outStmts.add st
  stmts = outStmts

proc lowerTernaryToIf(ctx: var GpuContext) =
  ## Rewrites ternaries and lazy bools over every fn body and global block.
  ## Kernels and device fns live in allFnTab, fnTab aliases those entries,
  ## globalBlocks is the program-scope statement list
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      ctx.lowerTernaryStmt(fn.pBody)
  for blk in ctx.globalBlocks.mitems:
    if blk.kind == gpuBlock:
      ctx.lowerTernaryStmts(blk.statements)

# ═══════════════════════════════════════════════════════════════════════
# Compound assignment to plain assignment
# ═══════════════════════════════════════════════════════════════════════

proc lowerCompoundAssignStmts(ctx: var GpuContext, stmts: var seq[GpuAst])
proc lowerCompoundAssignStmt(ctx: var GpuContext, n: var GpuAst)

func registerLowerCompoundAssign(reg: var PassRegistry) =
  ## Rewrites compound assignments into plain assignments of the binop, SPIR-V pipeline only, unregistered element types refuse loudly
  reg.register("lowerCompoundAssign", pkTransform, phaseMain,
    "Rewrites compound assignments (+= -= *= /=) into plain assignments of the binop, the target address evaluates its side effects once",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.lowerCompoundAssign()
  )

func compoundBinop(aOp: string): string =
  ## Binop token a compound assignment applies, "" when none is registered
  case aOp
  of "+=": "+"
  of "-=": "-"
  of "*=": "*"
  of "/=": "/"
  else: ""

const compoundElemKinds = {gtUint8, gtUint16, gtInt16, gtUint32, gtInt32,
                           gtUint64, gtInt64, gtFloat32, gtFloat64, gtSize_t}
  ## Element kinds with a registered SPIR-V arithmetic binop,
  ## the OpIAdd/OpFAdd families over binopInst. Everything else refuses.

proc hoistAddressCalls(ctx: var GpuContext, n: var GpuAst, pre: var seq[GpuAst]) =
  ## Hoists every call inside an lvalue address path into a scalar temp
  ## assigned before the statement, the address expression evaluates
  ## each side effect exactly once and the rewritten assignment reads the temp
  case n.kind
  of gpuCall:
    # inner calls hoist first, the temp initializer stays call-free
    for i in 0 ..< n.cArgs.len:
      ctx.hoistAddressCalls(n.cArgs[i], pre)
    let t = ctx.getExprType(n)
    if t.isNil or t.kind notin compoundElemKinds + {gtBool}:
      let kindStr = if t.isNil: "untyped" else: $t.kind
      error "passes_lowering_spirv: address-path call result has no scalar temp type: " & kindStr
    let name = "_ca_" & $ctx.genSymCount
    inc ctx.genSymCount
    let tempIdent = GpuAst(kind: gpuIdent,
                           symbol: newSymbol(name, iSym = name, typ = t, symKind = gsLocal))
    # storage hoists to the fn setup block, the initializer stores here
    pre.add GpuAst(kind: gpuVar, vName: tempIdent.clone(), vType: t,
                   vInit: n, vMutable: true)
    n = tempIdent
  of gpuIndex:
    ctx.hoistAddressCalls(n.iArr, pre)
    ctx.hoistAddressCalls(n.iIndex, pre)
  of gpuDot:
    ctx.hoistAddressCalls(n.dParent, pre)
  of gpuDeref:
    ctx.hoistAddressCalls(n.dOf, pre)
  of gpuIdent, gpuLit:
    discard
  else:
    error "passes_lowering_spirv: unsupported address expression node " & $n.kind

proc rewriteCompoundAssign(ctx: var GpuContext, n: var GpuAst) =
  ## Rewrites one compound assignment into a plain assignment of the binop.
  ##
  ## Before (IR):   x[i] += y
  ## After (IR):    x[i] = x[i] + y
  ##
  ## Address-path calls hoist into scalar temps ahead of the statement:
  ## each address side effect runs once, the binop read and the store
  ## reference one pure address skeleton
  let binop = compoundBinop(n.aOp)
  if binop.len == 0:
    error "passes_lowering_spirv: compound assignment operator " & n.aOp & " has no rewrite"
  let elemTy = ctx.getExprType(n.aLeft)
  if elemTy.isNil or elemTy.kind notin compoundElemKinds:
    let kindStr = if elemTy.isNil: "untyped" else: $elemTy.kind
    error "passes_lowering_spirv: compound assignment " & n.aOp &
      " over " & kindStr & " has no registered SPIR-V binop"
  var pre: seq[GpuAst]
  if n.aLeft.kind != gpuIdent:
    ctx.hoistAddressCalls(n.aLeft, pre)
  let binopNode = GpuAst(kind: gpuBinOp,
                         bOp: GpuAst(kind: gpuIdent, symbol: newSymbol(binop)),
                         bLeft: n.aLeft.clone(),
                         bRight: n.aRight,
                         bIsOverloaded: false,
                         bType: elemTy)
  n = GpuAst(kind: gpuAssign, aLeft: n.aLeft, aOp: "=", aRight: binopNode)
  if pre.len > 0:
    # the caller splices the labeled block back into the statement list
    n = GpuAst(kind: gpuBlock, blockLabel: "_compound", statements: pre & @[n])

proc lowerCompoundAssignStmts(ctx: var GpuContext, stmts: var seq[GpuAst]) =
  ## Walks a statement list in order, splicing hoisted-temp preambles
  ## before their statement.
  var outStmts: seq[GpuAst]
  for i in 0 ..< stmts.len:
    var st = stmts[i]
    ctx.lowerCompoundAssignStmt(st)
    if st.kind == gpuBlock and st.blockLabel == "_compound":
      for s in st.statements:
        outStmts.add s
    else:
      outStmts.add st
  stmts = outStmts

proc lowerCompoundAssignStmt(ctx: var GpuContext, n: var GpuAst) =
  case n.kind
  of gpuBlock:
    ctx.lowerCompoundAssignStmts(n.statements)
  of gpuAssign:
    if n.aOp != "=":
      ctx.rewriteCompoundAssign(n)
  of gpuIf:
    ctx.lowerCompoundAssignStmt(n.ifThen)
    for el in n.ifElifs.mitems:
      ctx.lowerCompoundAssignStmt(el.body)
    if n.ifElse.kind != gpuDiscard:
      ctx.lowerCompoundAssignStmt(n.ifElse)
  of gpuFor:
    ctx.lowerCompoundAssignStmt(n.fBody)
  of gpuWhile:
    ctx.lowerCompoundAssignStmt(n.wBody)
  else:
    discard

proc lowerCompoundAssign(ctx: var GpuContext) =
  ## Rewrites compound assignments over every fn body and global block.
  ## Kernels and device fns live in allFnTab, fnTab aliases those entries,
  ## globalBlocks is the program-scope statement list
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      ctx.lowerCompoundAssignStmt(fn.pBody)
  for blk in ctx.globalBlocks.mitems:
    if blk.kind == gpuBlock:
      ctx.lowerCompoundAssignStmts(blk.statements)

# ═══════════════════════════════════════════════════════════════════════
# Object kernel params flatten to one param per leaf
# ═══════════════════════════════════════════════════════════════════════

func registerFlattenObjectParams(reg: var PassRegistry) =
  ## Replaces object-typed kernel params with one scalar/pointer param per
  ## field leaf and rewrites the body's field chains to the leaf params,
  ## SPIR-V pipeline only, whole-object uses refuse loudly
  reg.register("flattenObjectParams", pkTransform, phaseMain,
    "Decomposes object kernel params into leaf params in field order, body field chains read the leaf params, whole-object uses refuse",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.flattenObjectParams()
  )

func isStaticIntTy(t: GpuType): bool =
  ## Static-integer values: `static int` and instantiated `Int[static]`.
  ## They carry their value in the type, no push member and no param.
  t.kind == gtStatic or
      (t.kind == gtGenericInst and t.gName == "Int" and t.gArgs.len == 1 and
       t.gArgs[0].kind == gtStatic)

func staticIntTyValue(t: GpuType): int =
  if t.kind == gtStatic:
    t.sValue
  else:
    t.gArgs[0].sValue

type FlattenLeaf = object
  path: seq[string]  ## field names from the param root, declaration order
  typ: GpuType
  param: GpuAst      ## the flattened param ident, nil for static-int leaves
  shadow: GpuAst     ## Function-storage shadow for a written scalar leaf, nil otherwise

func flatScalarKinds(t: GpuType): bool =
  ## Leaf scalar kinds the SPIR-V push block renders, gtVoid cannot be a leaf
  t.kind in {gtBool, gtUint8, gtUint16, gtInt16, gtUint32, gtInt32,
             gtUint64, gtInt64, gtFloat32, gtFloat64, gtFloat16, gtSize_t}

proc collectFlattenLeaves(t: GpuType, path: seq[string], res: var seq[FlattenLeaf]) =
  ## Depth-first leaf collection in field order: the flattened params replace
  ## the object param at its position, the push-blob member order the backend
  ## renders stays field order.
  case t.kind
  of gtObject, gtGenericInst:
    if isStaticIntTy(t):
      res.add FlattenLeaf(path: path, typ: t)
      return
    for f in t.fields:
      collectFlattenLeaves(f.typ, path & f.name, res)
  of gtPtr, gtUncheckedArray, gtStatic:
    res.add FlattenLeaf(path: path, typ: t)
  else:
    if flatScalarKinds(t):
      res.add FlattenLeaf(path: path, typ: t)
    else:
      error "flattenObjectParams: kernel parameter member outside the SPIR-V subset: " & $t.kind

type FlattenState = object
  kernelName: string
  paramISym: string
  paramName: string
  leaves: seq[FlattenLeaf]
  written: seq[int]  ## indexes into leaves of scalar leaves assigned in the body

proc findFlatLeaf(st: FlattenState, fields: seq[string]): int =
  ## Leaf index for a field path, -1 when the path names an interior object
  ## (a whole sub-object use) or nothing.
  for i, l in st.leaves:
    if l.path == fields:
      return i
  -1

proc flatChain(n: GpuAst): tuple[root: GpuAst, fields: seq[string]] =
  ## Splits a member-access chain into its root node and the field names.
  doAssert n.kind == gpuDot, "not a member access"
  var cur = n
  while cur.kind == gpuDot:
    result.fields.insert(cur.dField.symbol.name)
    cur = cur.dParent
  result.root = cur

proc flatRefuse(st: FlattenState, msg: string) =
  error "flattenObjectParams: " & msg & " (kernel `" & st.kernelName &
    "`, parameter `" & st.paramName & "`)"

proc markFlatWrite(n: GpuAst, st: var FlattenState, marks: bool) =
  ## Records which scalar leaves the body assigns. Only a bare member chain
  ## on the left of `=` writes a leaf. Pointer-leaf and deref-target stores
  ## write device memory, never the parameter
  case n.kind
  of gpuDot:
    let (root, fields) = flatChain(n)
    if root.kind == gpuIdent and root.symbol.iSym == st.paramISym:
      if not marks:
        return
      let idx = st.findFlatLeaf(fields)
      if idx < 0:
        st.flatRefuse("whole-object use of the parameter's sub-value `" &
          st.paramName & "." & fields.join(".") & "`")
      let lt = st.leaves[idx].typ
      if isStaticIntTy(lt):
        st.flatRefuse("assignment to the static field `" &
          st.paramName & "." & fields.join(".") & "`: it holds no runtime storage")
      if lt.kind in {gtPtr, gtUncheckedArray}:
        st.flatRefuse("assignment to the pointer leaf parameter `" &
          st.paramName & "_" & fields.join("_") &
          "`: device-address parameters are read-only")
      if idx notin st.written:
        st.written.add idx
    else:
      markFlatWrite(n.dParent, st, marks)
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym == st.paramISym and marks:
      st.flatRefuse("assignment of the whole object")
  of gpuDeref:
    markFlatWrite(n.dOf, st, false)
  of gpuIndex:
    markFlatWrite(n.iArr, st, false)
  of gpuAddr:
    markFlatWrite(n.aOf, st, false)
  else:
    discard

proc markFlatWrites(n: GpuAst, st: var FlattenState) =
  case n.kind
  of gpuBlock:
    for s in n.statements:
      markFlatWrites(s, st)
  of gpuAssign:
    markFlatWrite(n.aLeft, st, true)
  of gpuIf:
    markFlatWrites(n.ifThen, st)
    for el in n.ifElifs:
      markFlatWrites(el.body, st)
    if n.ifElse.kind != gpuDiscard:
      markFlatWrites(n.ifElse, st)
  of gpuFor:
    markFlatWrites(n.fBody, st)
  of gpuWhile:
    markFlatWrites(n.wBody, st)
  else:
    discard

proc rewriteFlatUses(n: var GpuAst, st: FlattenState, underAddr: bool) =
  ## Rewrites every use of the object parameter. Member chains rooted at it
  ## consume their whole chain into the leaf param, the shadow of a written
  ## scalar leaf or the literal of a static leaf. Any other occurrence is
  ## a whole-object use and refuses
  case n.kind
  of gpuDot:
    let (root, fields) = flatChain(n)
    if root.kind == gpuIdent and root.symbol.iSym == st.paramISym:
      let idx = st.findFlatLeaf(fields)
      if idx < 0:
        st.flatRefuse("whole-object use of the parameter's sub-value `" &
          st.paramName & "." & fields.join(".") & "`")
      let leaf = st.leaves[idx]
      if isStaticIntTy(leaf.typ):
        n = GpuAst(kind: gpuLit, lType: GpuType(kind: gtInt32),
                   lValue: $staticIntTyValue(leaf.typ))
      elif underAddr and leaf.typ.kind notin {gtPtr, gtUncheckedArray}:
        st.flatRefuse("address of the scalar leaf parameter `" &
          st.paramName & "_" & fields.join("_") &
          "`: a push-constant scalar has no address")
      elif not leaf.shadow.isNil:
        n = leaf.shadow
      else:
        n = leaf.param
    else:
      rewriteFlatUses(n.dParent, st, underAddr)
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym == st.paramISym:
      st.flatRefuse("whole-object use of the parameter")
  of gpuAddr:
    rewriteFlatUses(n.aOf, st, true)
  of gpuDeref:
    rewriteFlatUses(n.dOf, st, underAddr)
  of gpuIndex:
    rewriteFlatUses(n.iArr, st, underAddr)
    rewriteFlatUses(n.iIndex, st, underAddr)
  of gpuBinOp:
    rewriteFlatUses(n.bLeft, st, underAddr)
    rewriteFlatUses(n.bRight, st, underAddr)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      rewriteFlatUses(n.cArgs[i], st, underAddr)
  of gpuConv:
    rewriteFlatUses(n.convExpr, st, underAddr)
  of gpuCast:
    rewriteFlatUses(n.cExpr, st, underAddr)
  of gpuPrefix:
    rewriteFlatUses(n.pVal, st, underAddr)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      rewriteFlatUses(f.value, st, underAddr)
  of gpuArrayLit:
    for i in 0 ..< n.aValues.len:
      rewriteFlatUses(n.aValues[i], st, underAddr)
  of gpuMaterialize:
    rewriteFlatUses(n.mExpr, st, underAddr)
  of gpuConstexpr:
    rewriteFlatUses(n.cValue, st, underAddr)
  of gpuLit, gpuDiscard:
    discard
  # statements
  of gpuBlock:
    for s in n.statements.mitems:
      rewriteFlatUses(s, st, underAddr)
  of gpuAssign:
    rewriteFlatUses(n.aLeft, st, false)
    rewriteFlatUses(n.aRight, st, underAddr)
  of gpuVar:
    rewriteFlatUses(n.vInit, st, underAddr)
  of gpuIf:
    rewriteFlatUses(n.ifCond, st, underAddr)
    rewriteFlatUses(n.ifThen, st, underAddr)
    for el in n.ifElifs.mitems:
      rewriteFlatUses(el.cond, st, underAddr)
      rewriteFlatUses(el.body, st, underAddr)
    if n.ifElse.kind != gpuDiscard:
      rewriteFlatUses(n.ifElse, st, underAddr)
  of gpuFor:
    rewriteFlatUses(n.fStart, st, underAddr)
    rewriteFlatUses(n.fEnd, st, underAddr)
    rewriteFlatUses(n.fStep, st, underAddr)
    rewriteFlatUses(n.fBody, st, underAddr)
  of gpuWhile:
    rewriteFlatUses(n.wCond, st, underAddr)
    rewriteFlatUses(n.wBody, st, underAddr)
  of gpuReturn:
    rewriteFlatUses(n.rValue, st, underAddr)
  else:
    error "flattenObjectParams: unsupported node " & $n.kind

proc scanBodyIdents(n: GpuAst, taken: var seq[string]) =
  ## Collects every identifier name the body already uses, leaf param names
  ## must not collide with them
  if n.isNil:
    return
  if n.kind == gpuIdent and n.symbol != nil:
    taken.add n.symbol.name
  for ch in n.items:
    scanBodyIdents(ch, taken)

proc flattenKernelObjectParam(fn: GpuAst, pIdx: int): int =
  ## Flattens the object param at `pIdx` into leaf params, rewrites the body
  ## to read them. Returns the number of params inserted in its place.
  let kernelName = fn.pName.ident()
  let p = fn.pParams[pIdx]
  var st = FlattenState(kernelName: kernelName,
                        paramISym: p.ident.symbol.iSym,
                        paramName: p.ident.symbol.name)
  st.leaves = @[]
  collectFlattenLeaves(p.typ, @[], st.leaves)
  # names must be collision-free: against the remaining params and every
  # identifier the body already uses, the emitter keys storage by name
  var taken: seq[string]
  for q in fn.pParams:
    taken.add q.ident.symbol.name
  scanBodyIdents(fn.pBody, taken)
  for i, leaf in st.leaves.mpairs:
    if isStaticIntTy(leaf.typ):
      # no runtime storage: the leaf value substitutes as a literal,
      # no push member and no param, like the backend's member loop skip
      continue
    let name = p.ident.symbol.name & "_" & leaf.path.join("_")
    if name in taken:
      error "flattenObjectParams: flattened leaf name `" & name &
        "` collides with an existing name in kernel `" & kernelName & "`"
    taken.add name
    var paramTy = leaf.typ
    if paramTy.kind == gtUncheckedArray:
      # a flexible-array leaf crosses as a device buffer, the ptr type
      # the emitter's address route renders as one u64 push member
      paramTy = initGpuPtrType(paramTy.uaTo, implicitPtr = false)
    let sym = newSymbol(name, typ = paramTy, symKind = gsGlobalKernelParam)
    leaf.param = GpuAst(kind: gpuIdent, symbol: sym)
    st.leaves[i] = leaf
  markFlatWrites(fn.pBody, st)
  # a scalar leaf the body assigns needs Function storage: the emitter can
  # store only through locals, so the shadow var initializes from the push load
  for idx in st.written:
    var leaf = st.leaves[idx]
    let shadowName = leaf.param.symbol.name & "_shadow"
    if shadowName in taken:
      error "flattenObjectParams: shadow variable name `" & shadowName &
        "` collides with an existing name in kernel `" & kernelName & "`"
    taken.add shadowName
    let shadowSym = newSymbol(shadowName, typ = leaf.typ, symKind = gsLocal)
    leaf.shadow = GpuAst(kind: gpuIdent, symbol: shadowSym)
    st.leaves[idx] = leaf
  # splice the leaf params in at the object param's position so the push
  # block member order stays the backend's field-order expansion
  var params: seq[GpuParam]
  for i in 0 ..< pIdx:
    params.add fn.pParams[i]
  var inserted = 0
  for leaf in st.leaves:
    if leaf.param.isNil:
      continue
    params.add GpuParam(ident: leaf.param, typ: leaf.param.symbol.typ,
                        addressSpace: p.addressSpace, passByRef: false,
                        flavor: p.flavor)
    inc inserted
  for i in pIdx + 1 ..< fn.pParams.len:
    params.add fn.pParams[i]
  fn.pParams = params
  # shadow declarations hoist to the emitter's setup block, they are body
  # var statements and the initializer reads the leaf param's push load
  if st.written.len > 0:
    if fn.pBody.kind != gpuBlock:
      fn.pBody = GpuAst(kind: gpuBlock, statements: @[fn.pBody])
    for leaf in st.leaves:
      if not leaf.shadow.isNil:
        fn.pBody.statements.insert(
          GpuAst(kind: gpuVar, vName: leaf.shadow, vType: leaf.typ,
                 vInit: leaf.param, vMutable: true, addressSpace: asRMEM),
          0)
  rewriteFlatUses(fn.pBody, st, false)
  result = inserted

proc flattenObjectParams(ctx: var GpuContext) =
  ## Flattens object-typed kernel params over every `{.global.}` kernel.
  ## Kernels and device fns live in allFnTab, fnTab aliases those entries,
  ## device fns keep their object params: only kernel signatures flatten.
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind != gpuProc or attGlobal notin fn.pAttributes:
      continue
    var i = 0
    while i < fn.pParams.len:
      let t = fn.pParams[i].typ
      if t.kind in {gtObject, gtGenericInst} and not isStaticIntTy(t):
        i += flattenKernelObjectParam(fn, i)
      else:
        inc i
