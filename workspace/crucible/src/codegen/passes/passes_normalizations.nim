# Tattletale
# Copyright (c) 2026 Mamy Andre-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Normalization passes over the IR (GpuAst),
## run after construction but before legalization.

import std / [sequtils, tables, sets, strutils]
import ../ir/ast_datatypes
import ../ir/ast_type_constructors
import ../../builtins/builtins_functions
import ./pass_datatypes

# ############################################################
#
# Normalizations
#
# ############################################################
#
# Canonical reshaping right after IR construction establishes the forms
# every later stage assumes. Every backend registers them identically.

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerNormalizationPasses*(reg: var PassRegistry) =
  reg.registerFilterPragmas()
  reg.registerResolveOverloadedOperators()

  # After resolveOverloadedOperators: it turns binops into calls, the fnTab
  # closure over call edges must run after it.
  reg.register("deadCodeElimination", pkTransform, phaseEarly,
    "Fills fnTab with the transitively called subset of allFnTab, the printers emit only that",
    dependsOn = @[],
    run = deadCodeElimination
  )

  reg.registerDeEmbedForRangeAdjustment()

  reg.registerMangleNames()

func registerFilterPragmas(reg: var PassRegistry) =
  reg.register("filterPragmas", pkTransform, phaseEarly,
    "Filters Nim-specific pragmas: gpuProc pRawPragmas into pAttributes, statement pragmas into their GPU meaning (unroll marks the enclosing loop)",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          filterPragmasImpl(fn)
          markExplicitUnrollsImpl(fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          filterPragmasImpl(fn)
          markExplicitUnrollsImpl(fn.pBody)
  )

func registerResolveOverloadedOperators(reg: var PassRegistry) =
  reg.register("resolveOverloadedOperators", pkTransform, phaseEarly,
    "Converts gpuBinOp with non-primitive types to gpuCall",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          resolveOverloadedOperatorsImpl(ctx, fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          resolveOverloadedOperatorsImpl(ctx, fn.pBody)
  )

func registerDeEmbedForRangeAdjustment(reg: var PassRegistry) =
  reg.register("deEmbedForRangeAdjustment", pkTransform, phaseEarly,
    "Normalizes gpuFor range expressions, removing embedded +1 adjustments",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          deEmbedForRangeAdjustmentImpl(fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          deEmbedForRangeAdjustmentImpl(fn.pBody)
  )

func registerMangleNames(reg: var PassRegistry) =
  reg.register("mangleNames", pkTransform, phaseMain,
    "Finalizes fnTable display names: plain names keep their name, collisions get the iSym hash",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      mangleNamesImpl(ctx)
  )

proc dceCalleesFrom(ctx: var GpuContext, n: GpuAst): void =
  ## Adds every fn called in `n` (transitively) to `fnTab`:
  ## the printing world is the called subset, the rest is dead and dropped.
  case n.kind
  of gpuCall:
    let fn = n.cName
    if fn in ctx.allFnTab and fn notin ctx.fnTab:
      let fnCalled = ctx.allFnTab[fn]
      ctx.fnTab[fn] = fnCalled
      for ch in fnCalled:
        ctx.dceCalleesFrom(ch)
    for arg in n.cArgs:
      ctx.dceCalleesFrom(arg)
  else:
    for ch in n:
      ctx.dceCalleesFrom(ch)

proc deadCodeElimination(ctx: var GpuContext) =
  ## fnTab closure over call edges: the printers emit the called subset,
  ## unreferenced fns are dead.
  for fnKey in toSeq(ctx.fnTab.keys):
    dceCalleesFrom(ctx, ctx.fnTab[fnKey])

# ═══════════════════════════════════════════════════════════════════════
# Name mangling
# ═══════════════════════════════════════════════════════════════════════

func mangleNamesImpl(ctx: var GpuContext) =
  ## Finalize fnTable display names:
  ## - fkBuiltin entries emit their plain name as-is
  ## - fkDefined plain names keep the name, on collision the first keeps
  ##   it and every later entry gets an iSym hash (7 base58 chars)
  ## - operator overloads were finalized at registration
  ##   (getOperatorMangledName in nim_to_crucible) and are skipped here
  var seenNames = initHashSet[string]()

  for key, entry in ctx.fnTable.mpairs:
    let ident = entry.ident
    if ident.isNil or ident.symbol.isNil:
      continue
    if fkBuiltin in entry.kind:
      continue
    let name = ident.symbol.name
    if not name.isPlainIdentifier:
      continue

    if name.len == 0 or name in seenNames:
      # Use the iSym as the hash source (stable across compilation units)
      ident.symbol.name = mangledHashName(name, ident.symbol.iSym)

    seenNames.incl ident.symbol.name

# ═══════════════════════════════════════════════════════════════════════
# filterPragmas
# ═══════════════════════════════════════════════════════════════════════

func filterPragmasImpl(n: var GpuAst) =
  ## Nim-specific pragmas drop, GPU-meaningful ones populate pAttributes
  if n.kind != gpuProc:
    for child in n.mitems:
      filterPragmasImpl(child)
    return

  var attrs: set[GpuAttribute]
  for p in n.pRawPragmas:
    case p
    of "device": attrs.incl attDevice
    of "global": attrs.incl attGlobal
    of "inline", "forceinline": attrs.incl attForceInline
    of "builtin", "importc", "magic":
      # treated as builtin, no body needed
      discard
    of "varargs":
      continue
    of "noinit", "noInit", "raises", "cudaName":
      discard
    of "noSideEffect", "nimcall", "closure", "shallow":
      discard
    else:
      discard
  n.pAttributes = attrs

func markExplicitUnrollsImpl(n: var GpuAst) =
  if n.kind == gpuFor:
    var unroll = false
    var kept: seq[GpuAst]
    for stmt in n.fBody.statements:
      if stmt.kind == gpuComment and stmt.comment == "{.unroll.}":
        unroll = true
      else:
        kept.add stmt
    if unroll:
      n.fUnroll = true
      n.fBody.statements = kept
  for child in n.mitems:
    markExplicitUnrollsImpl(child)

# ═══════════════════════════════════════════════════════════════════════
# resolveOverloadedOperators
# ═══════════════════════════════════════════════════════════════════════

proc resolveOverloadedOperatorsImpl(ctx: var GpuContext, n: var GpuAst) =
  ## An overloaded gpuBinOp becomes a gpuCall, the ident found by iSym lookup
  case n.kind
  of gpuBinOp:
    if n.bIsOverloaded:
      let opISym = n.bOp.symbol.iSym
      var fnIdent: GpuAst = nil
      for key in ctx.allFnTab.keys:
        if key.symbol != nil and key.symbol.iSym == opISym:
          fnIdent = key
          break
      if fnIdent.isNil:
        # fallback, may lack the signature hash
        fnIdent = n.bOp
      var call = GpuAst(kind: gpuCall)
      call.cName = fnIdent
      # dropped static VALUE params erase the matching operand, as nnkCall does at construction
      var staticSkip: seq[int]
      if fnIdent in ctx.processedProcs:
        staticSkip = ctx.processedProcs[fnIdent].staticParamPositions
      call.cArgs = @[]
      for operandIdx in 0 ..< 2:
        if operandIdx in staticSkip: continue
        let operand = if operandIdx == 0: n.bLeft else: n.bRight
        call.cArgs.add operand
      n = call
      for i in 0 ..< call.cArgs.len:
        resolveOverloadedOperatorsImpl(ctx, call.cArgs[i])
      return
    else:
      resolveOverloadedOperatorsImpl(ctx, n.bLeft)
      resolveOverloadedOperatorsImpl(ctx, n.bRight)

  of gpuBlock:
    for i in 0 ..< n.statements.len:
      resolveOverloadedOperatorsImpl(ctx, n.statements[i])
  of gpuFor:
    resolveOverloadedOperatorsImpl(ctx, n.fStart)
    resolveOverloadedOperatorsImpl(ctx, n.fEnd)
    resolveOverloadedOperatorsImpl(ctx, n.fBody)
  of gpuWhile:
    resolveOverloadedOperatorsImpl(ctx, n.wCond)
    resolveOverloadedOperatorsImpl(ctx, n.wBody)
  of gpuTernary:
    resolveOverloadedOperatorsImpl(ctx, n.tCond)
    resolveOverloadedOperatorsImpl(ctx, n.tThen)
    if n.tElse.kind != gpuDiscard:
      resolveOverloadedOperatorsImpl(ctx, n.tElse)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      resolveOverloadedOperatorsImpl(ctx, n.cArgs[i])
  of gpuVar:
    resolveOverloadedOperatorsImpl(ctx, n.vInit)
  of gpuAssign:
    resolveOverloadedOperatorsImpl(ctx, n.aLeft)
    resolveOverloadedOperatorsImpl(ctx, n.aRight)
  of gpuReturn:
    resolveOverloadedOperatorsImpl(ctx, n.rValue)
  of gpuDot:
    resolveOverloadedOperatorsImpl(ctx, n.dParent)
    resolveOverloadedOperatorsImpl(ctx, n.dField)
  of gpuIndex:
    resolveOverloadedOperatorsImpl(ctx, n.iArr)
    resolveOverloadedOperatorsImpl(ctx, n.iIndex)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      resolveOverloadedOperatorsImpl(ctx, f.value)
  of gpuPrefix:
    resolveOverloadedOperatorsImpl(ctx, n.pVal)
  of gpuAddr:
    resolveOverloadedOperatorsImpl(ctx, n.aOf)
  of gpuDeref:
    resolveOverloadedOperatorsImpl(ctx, n.dOf)
  of gpuConv:
    resolveOverloadedOperatorsImpl(ctx, n.convExpr)
  of gpuCast:
    resolveOverloadedOperatorsImpl(ctx, n.cExpr)
  of gpuConstexpr:
    resolveOverloadedOperatorsImpl(ctx, n.cValue)
  of gpuMaterialize:
    resolveOverloadedOperatorsImpl(ctx, n.mExpr)
  of gpuArrayLit:
    for v in n.aValues.mitems:
      resolveOverloadedOperatorsImpl(ctx, v)
  of gpuIf:
    resolveOverloadedOperatorsImpl(ctx, n.ifCond)
    resolveOverloadedOperatorsImpl(ctx, n.ifThen)
    for el in n.ifElifs.mitems:
      resolveOverloadedOperatorsImpl(ctx, el.cond)
      resolveOverloadedOperatorsImpl(ctx, el.body)
    if n.ifElse.kind != gpuDiscard:
      resolveOverloadedOperatorsImpl(ctx, n.ifElse)
  of gpuProc:
    resolveOverloadedOperatorsImpl(ctx, n.pBody)
  else:
    discard

# ═══════════════════════════════════════════════════════════════════════
# deEmbedForRangeAdjustment
# ═══════════════════════════════════════════════════════════════════════

func deEmbedForRangeAdjustmentImpl(n: var GpuAst) =
  ## Resolves embedded +1 adjustments left by Slice/HSlice object ranges
  case n.kind
  of gpuFor:
    let start = n.fStart
    let endVal = n.fEnd
    if endVal.kind == gpuBinOp and endVal.bOp.kind == gpuIdent and
       endVal.bOp.symbol.name == "+":
      if endVal.bRight.kind == gpuLit and endVal.bRight.lValue == "1":
        n.fEnd = endVal.bLeft
        n.fRangeKind = rkInclusive
      elif endVal.bLeft.kind == gpuLit and endVal.bLeft.lValue == "1":
        n.fEnd = endVal.bRight
        n.fRangeKind = rkInclusive
    deEmbedForRangeAdjustmentImpl(n.fBody)

  of gpuBlock:
    for i in 0 ..< n.statements.len:
      deEmbedForRangeAdjustmentImpl(n.statements[i])
  of gpuWhile:
    deEmbedForRangeAdjustmentImpl(n.wBody)
  of gpuIf:
    deEmbedForRangeAdjustmentImpl(n.ifCond)
    deEmbedForRangeAdjustmentImpl(n.ifThen)
    for el in n.ifElifs.mitems:
      deEmbedForRangeAdjustmentImpl(el.cond)
      deEmbedForRangeAdjustmentImpl(el.body)
    if n.ifElse.kind != gpuDiscard:
      deEmbedForRangeAdjustmentImpl(n.ifElse)
  of gpuTernary:
    deEmbedForRangeAdjustmentImpl(n.tCond)
    deEmbedForRangeAdjustmentImpl(n.tThen)
    if n.tElse.kind != gpuDiscard:
      deEmbedForRangeAdjustmentImpl(n.tElse)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      deEmbedForRangeAdjustmentImpl(n.cArgs[i])
  of gpuBinOp:
    deEmbedForRangeAdjustmentImpl(n.bLeft)
    deEmbedForRangeAdjustmentImpl(n.bRight)
  of gpuVar:
    deEmbedForRangeAdjustmentImpl(n.vInit)
  of gpuAssign:
    deEmbedForRangeAdjustmentImpl(n.aLeft)
    deEmbedForRangeAdjustmentImpl(n.aRight)
  of gpuReturn:
    deEmbedForRangeAdjustmentImpl(n.rValue)
  of gpuDot:
    deEmbedForRangeAdjustmentImpl(n.dParent)
    deEmbedForRangeAdjustmentImpl(n.dField)
  of gpuIndex:
    deEmbedForRangeAdjustmentImpl(n.iArr)
    deEmbedForRangeAdjustmentImpl(n.iIndex)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      deEmbedForRangeAdjustmentImpl(f.value)
  of gpuPrefix:
    deEmbedForRangeAdjustmentImpl(n.pVal)
  of gpuAddr:
    deEmbedForRangeAdjustmentImpl(n.aOf)
  of gpuDeref:
    deEmbedForRangeAdjustmentImpl(n.dOf)
  of gpuConv:
    deEmbedForRangeAdjustmentImpl(n.convExpr)
  of gpuCast:
    deEmbedForRangeAdjustmentImpl(n.cExpr)
  of gpuConstexpr:
    deEmbedForRangeAdjustmentImpl(n.cValue)
  of gpuMaterialize:
    deEmbedForRangeAdjustmentImpl(n.mExpr)
  of gpuArrayLit:
    for v in n.aValues.mitems:
      deEmbedForRangeAdjustmentImpl(v)
  of gpuProc:
    deEmbedForRangeAdjustmentImpl(n.pBody)
  else:
    discard
