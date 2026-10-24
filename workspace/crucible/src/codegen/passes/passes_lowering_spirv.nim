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
import std/sets # pad-name collision check over body identifiers
import std/strutils # leaf param naming joins the field path
import std/tables # the reconstruction-local table
import ../ir/ast_datatypes
import ../ir/ast_type_constructors # initGpuPtrType
import ../ir/ast_type_resolvers # getExprType
import ./pass_datatypes
import ./pass_helpers # collectIdentNames
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
  reg.registerLowerCompoundAssign(deps = @["lowerTernaryToIf"])
  reg.registerInsertPointerDeref()
  # Object kernel params explode into leaf params and a reconstruction local,
  # pure-value objects included: the push block renders flat members only.
  # Bool leaves stay gtBool, the next pass retypes them
  reg.registerExplodeKernelParams(deps = @["insertPointerDeref"])
  # u32 pad members hold the device push block at the offsets the host
  # launch serialization packs: the object head rounds up to the object
  # alignment and the tail to the host-rounded end
  reg.registerSpirvPushBlockPad(deps = @["explodeKernelParams"])
  # Bool kernel parameters declare i32 in the push-constant interface,
  # loads convert at the use boundary
  reg.registerPatchBoolToI32(deps = @["explodeKernelParams"])

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
    # the rewrite assigns the branch expression to the temp, single-statement block wrappers around the branches unwrap first
    n.tThen = unwrapSingleStmtBlocks(n.tThen)
    if n.tElse.kind != gpuDiscard:
      n.tElse = unwrapSingleStmtBlocks(n.tElse)
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
    if condNeedsRewrite(n.fStep):
      # the step re-evaluates per iteration, the emitter computes the step
      # inside the continue block:
      #
      # - the temp storage declares ahead of the loop
      # - the assignment splices at the body's end
      # - the continue block reads the fresh value
      var stepPre: seq[GpuAst]
      ctx.lowerTernaryExpr(n.fStep, stepPre)
      var bodyStmts: seq[GpuAst]
      for h in stepPre:
        if h.kind == gpuVar:
          pre.add h
        else:
          bodyStmts.add h
      if n.fBody.kind == gpuBlock:
        n.fBody.statements = n.fBody.statements & bodyStmts
      else:
        n.fBody = GpuAst(kind: gpuBlock, statements: @[n.fBody] & bodyStmts)
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

func registerLowerCompoundAssign(reg: var PassRegistry, deps: seq[string] = @[]) =
  ## Rewrites compound assignments into plain assignments of the binop, SPIR-V pipeline only, unregistered element types refuse loudly
  reg.register("lowerCompoundAssign", pkTransform, phaseMain,
    "Rewrites compound assignments (+= -= *= /=) into plain assignments of the binop, the target address evaluates its side effects once",
    dependsOn = deps,
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
  of gpuBinOp:
    # pure-expression nodes have no side effects of their own, the walk only
    # descends into children and the gpuCall variant hoists the temp
    ctx.hoistAddressCalls(n.bLeft, pre)
    ctx.hoistAddressCalls(n.bRight, pre)
  of gpuConv:
    ctx.hoistAddressCalls(n.convExpr, pre)
  of gpuCast:
    ctx.hoistAddressCalls(n.cExpr, pre)
  of gpuPrefix:
    ctx.hoistAddressCalls(n.pVal, pre)
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
# Push-block padding (SPIR-V launch policy)
# ═══════════════════════════════════════════════════════════════════════

func registerSpirvPushBlockPad(reg: var PassRegistry, deps: seq[string] = @[]) =
  ## Inserts u32 pad params between the exploded leaves so the device push
  ## block's member offsets match the host launch serialization.
  ## SPIR-V pipeline only: the padding is the SPIR-V launch ABI's policy
  ## and stays out of the shared explode pass
  reg.register("spirvPushBlockPad", pkTransform, phaseMain,
    "Inserts u32 pad params after the exploded leaves so the device push-block offsets match the host serialization's object head and tail rounding",
    dependsOn = deps,
    run = proc(ctx: var GpuContext): void =
      ctx.spirvPushBlockPad()
  )

func pushHostWidth(t: GpuType): int =
  ## Byte width of one flat push member under the host launch serialization:
  ##
  ## - scalars read the shared gpuScalarByteSize table
  ## - a pointer crosses as its 8-byte device address
  ## - an array packs at the element width
  case t.kind
  of gtPtr, gtVoidPtr, gtUncheckedArray: 8
  of gtArray: gpuScalarByteSize(t.aTyp) * t.aLen
  else: gpuScalarByteSize(t)

func pushHostAlign(t: GpuType): int =
  ## Byte alignment of one flat push member under the host serialization:
  ## arrays align at the element width
  case t.kind
  of gtArray: pushHostWidth(t.aTyp)
  else: pushHostWidth(t)

type PadGroup = object
  ## Leaf params one exploded object contributed, in field order
  leaves: seq[tuple[name: string, typ: GpuType]]

proc collectLeafIdents(fn: GpuAst, n: GpuAst, leaves: var seq[tuple[name: string, typ: GpuType]]) =
  ## Collects the leaf param references of a reconstruction construction
  ## in field order, nested sub-object constructions recurse.
  ## Types come from the kernel's param list: its idents carry none
  case n.kind
  of gpuObjConstr:
    for f in n.ocFields:
      fn.collectLeafIdents(f.value, leaves)
  of gpuIdent:
    for p in fn.pParams:
      if p.ident.symbol.name == n.symbol.name:
        leaves.add (n.symbol.name, p.typ)
        return
    error "spirvPushBlockPad: kernel `" & fn.pName.ident() &
      "`: reconstruction references non-parameter `" & n.symbol.name & "`"
  else:
    discard

proc collectPadGroups(fn: GpuAst, groups: var seq[PadGroup]) =
  ## Explode leaves one reconstruction local per exploded param:
  ## a global-kernel-param-named var whose initializer is an object
  ## construction over the leaf params. Its fields name the group's
  ## leaves in field order
  if fn.pBody.kind != gpuBlock:
    return
  for st in fn.pBody.statements:
    if st.kind == gpuVar and st.vInit.kind == gpuObjConstr and
        st.vName.symbol.symKind == gsGlobalKernelParam:
      var g = PadGroup()
      fn.collectLeafIdents(st.vInit, g.leaves)
      groups.add g

proc spirvPushBlockPad(ctx: var GpuContext) =
  ## Rebuilds every `{.global.}` kernel's params with u32 pads interleaved.
  ##
  ## - the host packs each object at roundUp(off, objAlign) head and tail
  ## - the device push block rounds flat members to their own alignment
  ## - the pads equalize the two offsets
  ##
  ## objAlign is the max leaf alignment, matching the host serializer's deviceLayout for the object
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind != gpuProc or attGlobal notin fn.pAttributes:
      continue
    var groups: seq[PadGroup]
    fn.collectPadGroups(groups)
    if groups.len == 0:
      continue
    var leafOf = initTable[string, tuple[g, l: int]]()
    var objAlign: seq[int]
    for gi, g in groups:
      for li, leaf in g.leaves:
        leafOf[leaf.name] = (gi, li)
      var a = 1
      for leaf in g.leaves:
        a = max(a, pushHostAlign(leaf.typ))
      objAlign.add a
    var used: HashSet[string]
    for p in fn.pParams:
      used.incl p.ident.symbol.name
    fn.pBody.collectIdentNames(used)
    var params: seq[GpuParam]
    var hostOff = 0
    var devOff = 0
    var padCount = 0
    for p in fn.pParams:
      let name = p.ident.symbol.name
      let w = pushHostWidth(p.typ)
      let a = pushHostAlign(p.typ)
      if name in leafOf:
        let (gi, li) = leafOf[name]
        if li == 0:
          # the host rounds the object's head to the object alignment
          hostOff = roundUpMultiple(hostOff, objAlign[gi])
      let hostPlace = roundUpMultiple(hostOff, a)
      # the device rounds flat members to their own alignment.
      # U32 pads advance the device offset to the host placement.
      # All members are at least 32 bits wide, both offsets keep their 4-byte granularity
      while devOff < hostPlace:
        let padName = name & "_pad" & $padCount
        if padName in used:
          error "spirvPushBlockPad: padding member name `" & padName &
            "` collides with an existing name in kernel `" & fn.pName.ident() & "`"
        used.incl padName
        let padSym = newSymbol(padName, typ = GpuType(kind: gtUint32),
                               symKind = gsGlobalKernelParam)
        params.add GpuParam(ident: GpuAst(kind: gpuIdent, symbol: padSym),
                            typ: padSym.typ, addressSpace: asRMEM,
                            passByRef: false)
        devOff = roundUpMultiple(devOff, 4) + 4
        inc padCount
      doAssert devOff == hostPlace,
        "spirvPushBlockPad: device offset " & $devOff & " missed host placement " & $hostPlace
      devOff = hostPlace + w
      hostOff = hostPlace + w
      if name in leafOf:
        let (gi, li) = leafOf[name]
        if li == groups[gi].leaves.len - 1:
          # the host rounds the object's packed end to the object alignment
          hostOff = roundUpMultiple(hostOff, objAlign[gi])
      params.add p
    fn.pParams = params
