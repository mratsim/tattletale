# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ═══════════════════════════════════════════════════════════════════════
#  Compound assignment lowering
# ═══════════════════════════════════════════════════════════════════════

import std/macros # for the `error` builtin (compile-time diagnostics)
import std/tables # OrderedTable keys iterator
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers # getExprType
import ./pass_datatypes
import ./pass_registry

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerLowerCompoundAssign*(reg: var PassRegistry) =
  ## Rewrites compound assignments into plain assignments of the binop, SPIR-V pipeline only, unregistered element types refuse loudly
  reg.register("lowerCompoundAssign", pkTransform, phaseMain,
    "Rewrites compound assignments (+= -= *= /=) into plain assignments of the binop, the target address evaluates its side effects once",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.lowerCompoundAssign()
  )

# ═══════════════════════════════════════════════════════════════════════
# Compound assignment to plain assignment
# ═══════════════════════════════════════════════════════════════════════

proc lowerCompoundAssignStmts(ctx: var GpuContext, stmts: var seq[GpuAst])
proc lowerCompoundAssignStmt(ctx: var GpuContext, n: var GpuAst)

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
      error "passes_low_compound_assign: address-path call result has no scalar temp type: " & kindStr
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
    error "passes_low_compound_assign: unsupported address expression node " & $n.kind

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
    error "passes_low_compound_assign: compound assignment operator " & n.aOp & " has no rewrite"
  let elemTy = ctx.getExprType(n.aLeft)
  if elemTy.isNil or elemTy.kind notin compoundElemKinds:
    let kindStr = if elemTy.isNil: "untyped" else: $elemTy.kind
    error "passes_low_compound_assign: compound assignment " & n.aOp &
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
  ## globalBlocks carries the program-scope statements
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      ctx.lowerCompoundAssignStmt(fn.pBody)
  for blk in ctx.globalBlocks.mitems:
    if blk.kind == gpuBlock:
      ctx.lowerCompoundAssignStmts(blk.statements)
