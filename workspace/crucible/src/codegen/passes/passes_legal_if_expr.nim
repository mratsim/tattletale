# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
# If-expression mapping
#
# ############################################################
#
# Expression-only gpuIf branches convert to gpuTernary, backends emit
# `cond ? then : else` from the lowered form.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

import std / [macros, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_helpers

func registerMapIfExprToTernary*(reg: var PassRegistry) =
  reg.register("mapIfExprToTernary", pkTransform, phaseEarly,
    "Converts gpuIf(isExpr:true) with expression-only branches to gpuTernary",
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          mapIfExprToTernaryImpl(fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          mapIfExprToTernaryImpl(fn.pBody)
  )

# ═══════════════════════════════════════════════════════════════════════
# If-expression mapping
# ═══════════════════════════════════════════════════════════════════════

func mapIfExprToTernaryImpl(n: var GpuAst) =
  ## Backends emit `cond ? then : else` from the lowered gpuTernary.
  ## Branches carrying statements stay gpuIf: they lower in blitBlockExprs
  ## as a statement if assigning a fresh temp, so the unselected branch's
  ## statements never run.
  case n.kind
  of gpuIf:
    if n.ifIsExpr:
      if isExprOnlyBranch(n.ifThen) and isExprOnlyBranch(n.ifElse):
        let tern = GpuAst(
          kind: gpuTernary,
          tCond: n.ifCond,
          tThen: n.ifThen,
          tElse: n.ifElse)
        n = tern
        mapIfExprToTernaryImpl(n.tCond)
        mapIfExprToTernaryImpl(n.tThen)
        if n.tElse.kind != gpuDiscard:
          mapIfExprToTernaryImpl(n.tElse)
      else:
        mapIfExprToTernaryImpl(n.ifCond)
        mapIfExprToTernaryImpl(n.ifThen)
        if n.ifElse.kind != gpuDiscard:
          mapIfExprToTernaryImpl(n.ifElse)
    else:
      mapIfExprToTernaryImpl(n.ifCond)
      mapIfExprToTernaryImpl(n.ifThen)
      for el in n.ifElifs.mitems:
        mapIfExprToTernaryImpl(el.cond)
        mapIfExprToTernaryImpl(el.body)
      if n.ifElse.kind != gpuDiscard:
        mapIfExprToTernaryImpl(n.ifElse)
  of gpuBlock:
    for i in 0 ..< n.statements.len:
      mapIfExprToTernaryImpl(n.statements[i])
  of gpuFor:
    mapIfExprToTernaryImpl(n.fStart)
    mapIfExprToTernaryImpl(n.fEnd)
    mapIfExprToTernaryImpl(n.fStep)
    mapIfExprToTernaryImpl(n.fBody)
  of gpuWhile:
    mapIfExprToTernaryImpl(n.wCond)
    mapIfExprToTernaryImpl(n.wBody)
  of gpuTernary:
    mapIfExprToTernaryImpl(n.tCond)
    mapIfExprToTernaryImpl(n.tThen)
    if n.tElse.kind != gpuDiscard:
      mapIfExprToTernaryImpl(n.tElse)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      mapIfExprToTernaryImpl(n.cArgs[i])
  of gpuBinOp:
    mapIfExprToTernaryImpl(n.bLeft)
    mapIfExprToTernaryImpl(n.bRight)
  of gpuVar:
    mapIfExprToTernaryImpl(n.vInit)
  of gpuAssign:
    mapIfExprToTernaryImpl(n.aLeft)
    mapIfExprToTernaryImpl(n.aRight)
  of gpuReturn:
    mapIfExprToTernaryImpl(n.rValue)
  of gpuDot:
    mapIfExprToTernaryImpl(n.dParent)
    mapIfExprToTernaryImpl(n.dField)
  of gpuIndex:
    mapIfExprToTernaryImpl(n.iArr)
    mapIfExprToTernaryImpl(n.iIndex)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      mapIfExprToTernaryImpl(f.value)
  of gpuPrefix:
    mapIfExprToTernaryImpl(n.pVal)
  of gpuAddr:
    mapIfExprToTernaryImpl(n.aOf)
  of gpuDeref:
    mapIfExprToTernaryImpl(n.dOf)
  of gpuConv:
    mapIfExprToTernaryImpl(n.convExpr)
  of gpuCast:
    mapIfExprToTernaryImpl(n.cExpr)
  of gpuConstexpr:
    mapIfExprToTernaryImpl(n.cValue)
  of gpuMaterialize:
    mapIfExprToTernaryImpl(n.mExpr)
  of gpuArrayLit:
    for v in n.aValues.mitems:
      mapIfExprToTernaryImpl(v)
  else:
    discard
