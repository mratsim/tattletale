## Tattletale
## Copyright (c) 2026 Mamy Andre-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

## Phase 4: Frontend Extraction
##
## Normalization passes extracted from `nim_to_gpu.nim` (the IR construction monolith)
## into named, testable passes that run after IR construction but before legalization.
##
## Each pass replaces inline logic that was baked into `toGpuAst`:

import std / [sequtils, tables, sets, strutils]
import ../ir/gpu_types
import ../builtins/nim_builtins
import ./pass_datatypes
import ./passes_preprocessing


# ═══════════════════════════════════════════════════════════════════════
# Pass 1: lowerIfExpr
# ═══════════════════════════════════════════════════════════════════════

proc lowerIfExprImpl*(n: var GpuAst) =
  ## Convert `gpuIf(isExpr: true)` nodes into `gpuTernary` nodes.
  ##
  ## During IR construction (`toGpuAst`), `nnkIfExpr` is emitted as
  ## `gpuBlock(gpuIf(isExpr: true, ...))`. This pass lowers that to
  ## `gpuBlock(gpuTernary(cond, then, else))` so backends emit e.g.
  ## `cond ? then : else` in CUDA C.
  case n.kind
  of gpuIf:
    if n.ifIsExpr:
      # Convert to gpuTernary
      let tern = GpuAst(
        kind: gpuTernary,
        tCond: n.ifCond,
        tThen: n.ifThen,
        tElse: n.ifElse
      )
      n = tern
      # Recurse into ternary children (catches nested if-expr chains)
      lowerIfExprImpl(n.tCond)
      lowerIfExprImpl(n.tThen)
      if n.tElse.kind != gpuDiscard:
        lowerIfExprImpl(n.tElse)
    else:
      # Recurse into children
      lowerIfExprImpl(n.ifCond)
      lowerIfExprImpl(n.ifThen)
      if n.ifElse.kind != gpuDiscard:
        lowerIfExprImpl(n.ifElse)
  of gpuBlock:
    for i in 0 ..< n.statements.len:
      lowerIfExprImpl(n.statements[i])
  of gpuFor:
    lowerIfExprImpl(n.fStart)
    lowerIfExprImpl(n.fEnd)
    lowerIfExprImpl(n.fBody)
  of gpuWhile:
    lowerIfExprImpl(n.wCond)
    lowerIfExprImpl(n.wBody)
  of gpuTernary:
    lowerIfExprImpl(n.tCond)
    lowerIfExprImpl(n.tThen)
    if n.tElse.kind != gpuDiscard:
      lowerIfExprImpl(n.tElse)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      lowerIfExprImpl(n.cArgs[i])
  of gpuBinOp:
    lowerIfExprImpl(n.bLeft)
    lowerIfExprImpl(n.bRight)
  of gpuVar:
    lowerIfExprImpl(n.vInit)
  of gpuAssign:
    lowerIfExprImpl(n.aLeft)
    lowerIfExprImpl(n.aRight)
  of gpuReturn:
    lowerIfExprImpl(n.rValue)
  of gpuDot:
    lowerIfExprImpl(n.dParent)
    lowerIfExprImpl(n.dField)
  of gpuIndex:
    lowerIfExprImpl(n.iArr)
    lowerIfExprImpl(n.iIndex)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      lowerIfExprImpl(f.value)
  of gpuPrefix:
    lowerIfExprImpl(n.pVal)
  of gpuAddr:
    lowerIfExprImpl(n.aOf)
  of gpuDeref:
    lowerIfExprImpl(n.dOf)
  of gpuConv:
    lowerIfExprImpl(n.convExpr)
  of gpuCast:
    lowerIfExprImpl(n.cExpr)
  of gpuConstexpr:
    lowerIfExprImpl(n.cValue)
  of gpuMaterialize:
    lowerIfExprImpl(n.mExpr)
  of gpuArrayLit:
    for v in n.aValues.mitems:
      lowerIfExprImpl(v)
  else:
    discard


# ═══════════════════════════════════════════════════════════════════════
# Pass 2: mapOperators
# ═══════════════════════════════════════════════════════════════════════

proc maybePatchFnName(n: var GpuAst) =
  ## Renames operator function names whose symbols are not valid C++
  ## identifier characters (e.g. `+`->`add`, `-`->`sub`).
  doAssert n.kind == gpuIdent
  template patch(arg, by: untyped): untyped =
    arg.symbol.iSym = arg.symbol.iSym.replace(arg.symbol.name, by)
    arg.symbol.name = by
  let name = n.symbol.name
  case name
  of "+": patch(n, "add")
  of "-": patch(n, "sub")
  of "*": patch(n, "mul")
  of "/": patch(n, "div")
  of "..": patch(n, "range")
  else: discard

proc mapOperatorsImpl*(n: var GpuAst) =
  ## Walk all function-name `gpuIdent` nodes and patch operator names.
  ## Only patches idents with symKind == gsProc (function identifiers),
  ## NOT operator symbols inside gpuBinOp (which CUDA C understands natively).
  case n.kind
  of gpuIdent:
    # Only patch function names (gsProc), not operator symbols (gsNone)
    if n.symbol.symKind == gsProc:
      maybePatchFnName(n)
  of gpuBlock:
    for i in 0 ..< n.statements.len:
      mapOperatorsImpl(n.statements[i])
  of gpuFor:
    mapOperatorsImpl(n.fStart)
    mapOperatorsImpl(n.fEnd)
    mapOperatorsImpl(n.fBody)
    mapOperatorsImpl(n.fVar)
  of gpuWhile:
    mapOperatorsImpl(n.wCond)
    mapOperatorsImpl(n.wBody)
  of gpuTernary:
    mapOperatorsImpl(n.tCond)
    mapOperatorsImpl(n.tThen)
    if n.tElse.kind != gpuDiscard:
      mapOperatorsImpl(n.tElse)
  of gpuCall:
    mapOperatorsImpl(n.cName)
    for i in 0 ..< n.cArgs.len:
      mapOperatorsImpl(n.cArgs[i])
  of gpuBinOp:
    # Do NOT patch bOp — operator symbols (+, -, etc.) are valid C++
    mapOperatorsImpl(n.bLeft)
    mapOperatorsImpl(n.bRight)
  of gpuVar:
    mapOperatorsImpl(n.vName)
    mapOperatorsImpl(n.vInit)
  of gpuAssign:
    mapOperatorsImpl(n.aLeft)
    mapOperatorsImpl(n.aRight)
  of gpuReturn:
    mapOperatorsImpl(n.rValue)
  of gpuDot:
    mapOperatorsImpl(n.dParent)
    mapOperatorsImpl(n.dField)
  of gpuIndex:
    mapOperatorsImpl(n.iArr)
    mapOperatorsImpl(n.iIndex)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      mapOperatorsImpl(f.value)
  of gpuPrefix:
    mapOperatorsImpl(n.pVal)
  of gpuAddr:
    mapOperatorsImpl(n.aOf)
  of gpuDeref:
    mapOperatorsImpl(n.dOf)
  of gpuConv:
    mapOperatorsImpl(n.convExpr)
  of gpuCast:
    mapOperatorsImpl(n.cExpr)
  of gpuConstexpr:
    mapOperatorsImpl(n.cIdent)
    mapOperatorsImpl(n.cValue)
  of gpuMaterialize:
    mapOperatorsImpl(n.mExpr)
  of gpuArrayLit:
    for v in n.aValues.mitems:
      mapOperatorsImpl(v)
  of gpuIf:
    mapOperatorsImpl(n.ifCond)
    mapOperatorsImpl(n.ifThen)
    if n.ifElse.kind != gpuDiscard:
      mapOperatorsImpl(n.ifElse)
  of gpuProc:
    mapOperatorsImpl(n.pName)
    for p in n.pParams.mitems:
      mapOperatorsImpl(p.ident)
    mapOperatorsImpl(n.pBody)
  else:
    discard


# ═══════════════════════════════════════════════════════════════════════
# Pass 3: filterPragmas
# ═══════════════════════════════════════════════════════════════════════

proc filterPragmasImpl*(n: var GpuAst) =
  ## Walk all `gpuProc` nodes and filter their `pRawPragmas` to
  ## populate `pAttributes`, dropping Nim-specific pragmas that
  ## have no GPU backend meaning.
  if n.kind != gpuProc:
    # Recurse into children
    for child in n.mitems:
      filterPragmasImpl(child)
    return

  # Process raw pragmas
  var attrs: set[GpuAttribute]
  for p in n.pRawPragmas:
    case p
    of "device": attrs.incl attDevice
    of "global": attrs.incl attGlobal
    of "inline", "forceinline": attrs.incl attForceInline
    of "nimonly", "builtin", "importc", "magic":
      # These cause the proc to be treated as a builtin (no body needed)
      discard
    of "varargs":
      continue
    of "noinit", "noInit", "raises", "cudaName":
      discard
    # Common Nim pragmas that are not relevant for GPU codegen:
    of "noSideEffect", "nimcall", "closure", "shallow":
      discard
    else:
      # Unknown pragmas are silently dropped (they are Nim-specific)
      discard
  n.pAttributes = attrs


# ═══════════════════════════════════════════════════════════════════════
# Pass 4: resolveOverloadedOperators
# ═══════════════════════════════════════════════════════════════════════

proc resolveOverloadedOperatorsImpl*(ctx: var GpuContext; n: var GpuAst) =
  ## Walk all `gpuBinOp` nodes and check `bIsOverloaded` flag.
  ## If true, convert the `gpuBinOp` into a `gpuCall` to the operator function.
  ## Uses fnTable/iSym lookup to get the correct function identifier.
  case n.kind
  of gpuBinOp:
    if n.bIsOverloaded:
      # Find the matching function identifier in allFnTab
      let opISym = n.bOp.symbol.iSym
      var fnIdent: GpuAst = nil
      for key in ctx.allFnTab.keys:
        if key.symbol != nil and key.symbol.iSym == opISym:
          fnIdent = key
          break
      if fnIdent.isNil:
        # Fallback: use the binop's operator ident (may lack signature hash)
        fnIdent = n.bOp
      var call = GpuAst(kind: gpuCall)
      call.cName = fnIdent
      call.cArgs = @[n.bLeft, n.bRight]
      n = call
      # Recurse into children
      for i in 0 ..< call.cArgs.len:
        resolveOverloadedOperatorsImpl(ctx, call.cArgs[i])
      return
    else:
      # Recurse into children (overloaded flag already set at construction time)
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
    if n.ifElse.kind != gpuDiscard:
      resolveOverloadedOperatorsImpl(ctx, n.ifElse)
  of gpuProc:
    resolveOverloadedOperatorsImpl(ctx, n.pBody)
  else:
    discard


# ═══════════════════════════════════════════════════════════════════════
# Pass 5: deEmbedForRangeAdjustment
# ═══════════════════════════════════════════════════════════════════════

proc deEmbedForRangeAdjustmentImpl*(n: var GpuAst) =
  ## Normalize gpuFor range expressions: ensure fStart/fEnd/fRangeKind
  ## are consistent and no embedded adjustments remain.
  ##
  ## Phase 3 already replaced the `..` range +1 with `fRangeKind` on `gpuFor`.
  ## This pass handles any remaining edge cases (e.g. Slice/HSlice object
  ## construction ranges that may have embedded adjustments).
  case n.kind
  of gpuFor:
    # The range is already normalized from IR construction via fRangeKind.
    # Validate consistency: if fStart or fEnd is a gpuBinOp with +1 adjustment,
    # resolve it and correct fRangeKind if needed.
    let start = n.fStart
    let endVal = n.fEnd

    # Check if fEnd has a `+ 1` adjustment pattern (binOp with `+` and lit `1`)
    if endVal.kind == gpuBinOp and endVal.bOp.kind == gpuIdent and
       endVal.bOp.symbol.name == "+":
      # Check if second operand is literal 1
      if endVal.bRight.kind == gpuLit and endVal.bRight.lValue == "1":
        # This is a `+ 1` adjustment for inclusive range
        n.fEnd = endVal.bLeft
        n.fRangeKind = rkInclusive
      elif endVal.bLeft.kind == gpuLit and endVal.bLeft.lValue == "1":
        n.fEnd = endVal.bRight
        n.fRangeKind = rkInclusive

    # Recurse into body
    deEmbedForRangeAdjustmentImpl(n.fBody)

  of gpuBlock:
    for i in 0 ..< n.statements.len:
      deEmbedForRangeAdjustmentImpl(n.statements[i])
  of gpuWhile:
    deEmbedForRangeAdjustmentImpl(n.wBody)
  of gpuIf:
    deEmbedForRangeAdjustmentImpl(n.ifCond)
    deEmbedForRangeAdjustmentImpl(n.ifThen)
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


# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc registerNormalizationPasses*(reg: var PassRegistry) =
  ## Register normalization passes extracted from `nim_to_gpu.nim`.
  ## These run after IR construction but before legalization passes.

  reg.register("lowerIfExpr", pkTransform, phaseEarly,
    "Converts gpuIf(isExpr:true) to gpuTernary",
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          lowerIfExprImpl(fn.pBody)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          lowerIfExprImpl(fn.pBody)
  )

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

  reg.register("mapOperators", pkTransform, phaseEarly,
    "Patches operator function names (+,*,/,-) to valid C++ identifiers",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          mapOperatorsImpl(fn)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          mapOperatorsImpl(fn)
  )

  reg.register("filterPragmas", pkTransform, phaseEarly,
    "Filters Nim-specific pragmas from gpuProc pRawPragmas into pAttributes",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          filterPragmasImpl(fn)
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          filterPragmasImpl(fn)
  )

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

  # ── Pass: rewriteCompoundAssignment (common, ALL backends) ──
  # Registered here — NOT in registerPreprocessingPasses — because runPasses
  # executes passes in registration order and legalization (blitBlockExprs)
  # must see the desugared gpuAssign: blitting a compound-assign binop LHS
  # would by-value-blit the addressed block into a discarded temp
  # (`((&_blit_N) += (...))` — not a modifiable lvalue on CUDA, and the
  # accumulation would be silently lost). Runs after resolveOverloadedOperators
  # so non-primitive `+=` (already gpuCall) is never touched.
  reg.register("rewriteCompoundAssignment", pkTransform, phaseEarly,
    "Rewrites compound-assign binops (x += y) into plain assignments (x = x + y) " &
      "so the LHS is a real modifiable lvalue on all backends",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          fn.pBody.walk(proc(n: var GpuAst): void =
            if n.kind == gpuBinOp:
              n = rewriteCompoundAssignmentImpl(n)
          )
      for fnKey in ctx.genericInsts.keys:
        var fn = ctx.genericInsts[fnKey]
        if fn.kind == gpuProc:
          fn.pBody.walk(proc(n: var GpuAst): void =
            if n.kind == gpuBinOp:
              n = rewriteCompoundAssignmentImpl(n)
          )
  )
