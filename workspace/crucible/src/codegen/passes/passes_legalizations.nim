## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [macros, sequtils, tables]
import ../ir/gpu_types
import ./pass_datatypes

proc insertResult(ctx: var GpuContext; fn: GpuAst) =
  ## Insert `var result` and `return result` into a function body.
  if fn.pRetType.kind == gtVoid: return

  proc lastIsReturn(n: GpuAst): bool =
    doAssert n.kind == gpuBlock
    if n.statements[^1].kind == gpuReturn: return true

  if not lastIsReturn(fn.pBody):
    let resId = GpuAst(kind: gpuIdent, iName: "result",
                       iSym: "result",
                       iTyp: fn.pRetType,
                       symbolKind: gsLocal)
    let res = GpuAst(kind: gpuVar, vName: resId,
                     vType: fn.pRetType,
                     vInit: GpuAst(kind: gpuDiscard),
                     vRequiresMemcpy: false,
                     vMutable: true)
    fn.pBody.statements.insert(res, 0)

    if not lastIsReturn(fn.pBody):
      fn.pBody.statements.add GpuAst(kind: gpuReturn, rValue: resId)

    for i in countdown(fn.pBody.statements.high, 0):
      let stmt = fn.pBody.statements[i]
      if stmt.kind notin {gpuVar, gpuComment, gpuDiscard, gpuReturn, gpuIf, gpuFor, gpuWhile}:
        if stmt.kind == gpuBlock and stmt.isExpr:
          if stmt.statements.len == 1:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt.statements[0])
          else:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        elif stmt.kind != gpuAssign:
          fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        break

proc firstNonBlock(n: GpuAst): GpuAst =
  ## Unwrap single-statement gpuBlock wrappers.
  result = n
  while result.kind == gpuBlock and result.statements.len == 1:
    result = result.statements[0]

proc liftConstexprFrom(n: var GpuAst; lifts: var seq[GpuAst]) =
  ## Lift gpuConstexpr nodes from expression children, replacing them with their cIdent.
  case n.kind
  of gpuConstexpr:
    lifts.add n
    n = n.cIdent
  of gpuVar:
    liftConstexprFrom(n.vInit, lifts)
  of gpuAssign:
    liftConstexprFrom(n.aRight, lifts)
  of gpuDot:
    liftConstexprFrom(n.dParent, lifts)
    liftConstexprFrom(n.dField, lifts)
  of gpuIndex:
    liftConstexprFrom(n.iArr, lifts)
    liftConstexprFrom(n.iIndex, lifts)
  of gpuCall:
    for i in 0 ..< n.cArgs.len:
      liftConstexprFrom(n.cArgs[i], lifts)
  of gpuObjConstr:
    for i in 0 ..< n.ocFields.len:
      liftConstexprFrom(n.ocFields[i].value, lifts)
  of gpuReturn:
    liftConstexprFrom(n.rValue, lifts)
  of gpuAddr:
    liftConstexprFrom(n.aOf, lifts)
  of gpuDeref:
    liftConstexprFrom(n.dOf, lifts)
  of gpuConv:
    liftConstexprFrom(n.convExpr, lifts)
  of gpuCast:
    liftConstexprFrom(n.cExpr, lifts)
  of gpuBinOp:
    liftConstexprFrom(n.bLeft, lifts)
    liftConstexprFrom(n.bRight, lifts)
  of gpuPrefix:
    liftConstexprFrom(n.pVal, lifts)
  of gpuIf:
    liftConstexprFrom(n.ifCond, lifts)
  of gpuFor:
    liftConstexprFrom(n.fStart, lifts)
    liftConstexprFrom(n.fEnd, lifts)
  of gpuWhile:
    liftConstexprFrom(n.wCond, lifts)
  of gpuArrayLit:
    for i in 0 ..< n.aValues.len:
      liftConstexprFrom(n.aValues[i], lifts)
  of gpuBlock:
    for i in 0 ..< n.statements.len:
      if n.statements[i].kind == gpuConstexpr:
        lifts.add n.statements[i]
        n.statements[i] = GpuAst(kind: gpuDiscard)
      else:
        liftConstexprFrom(n.statements[i], lifts)
  else:
    discard

proc dedupVar(stmt: var GpuAst; usedNames: var CountTable[string]) =
  ## If stmt is a variable declaration whose name is already used,
  ## assign a unique name with a numeric suffix.
  if stmt.kind == gpuVar:
    let name = stmt.vName.ident()
    if name in usedNames:
      var uniqueName = name & "_0"
      var counter = 0
      while uniqueName in usedNames:
        counter += 1
        uniqueName = name & "_" & $counter
      stmt.vName.iName = uniqueName
      stmt.vName.iSym = uniqueName
    usedNames.inc stmt.vName.ident()

proc collectStmt(stmt: var GpuAst; usedNames: var CountTable[string]; newStmts: var seq[GpuAst]) =
  ## Add a statement to the flat block, lifting constexpr children first
  ## and deduplicating variable names.
  if stmt.kind != gpuConstexpr:
    var lifts: seq[GpuAst]
    liftConstexprFrom(stmt, lifts)
    for i in 0 ..< lifts.len:
      dedupVar(lifts[i], usedNames)
      newStmts.add lifts[i]
  dedupVar(stmt, usedNames)
  newStmts.add stmt

proc hoistBlockPreamble(slot: var GpuAst; usedNames: var CountTable[string]; newStmts: var seq[GpuAst]) =
  ## If `slot` is a gpuBlock with multiple statements, hoist all but the last
  ## into `newStmts` and replace `slot` with the last statement.
  ## Repeats via firstNonBlock until `slot` is no longer a multi-stmt block.
  while true:
    let inner = firstNonBlock(slot)
    if inner.kind == gpuBlock and inner.statements.len > 1:
      for j in 0 ..< inner.statements.len - 1:
        var innerStmt = inner.statements[j]
        collectStmt(innerStmt, usedNames, newStmts)
      slot = inner.statements[^1]
    else:
      break

proc hoistFromExprs(n: var GpuAst; usedNames: var CountTable[string]; newStmts: var seq[GpuAst]) =
  ## Recursively walk expression children of n. If any is a
  ## gpuBlock(isExpr: true) with multiple statements, hoist the
  ## preamble statements into newStmts and replace with the last stmt.
  ## Then recurse on the replacement (handles multi-level nesting).
  case n.kind
  of gpuBlock:
    if n.isExpr and n.statements.len > 1:
      hoistBlockPreamble(n, usedNames, newStmts)
      hoistFromExprs(n, usedNames, newStmts)
  of gpuCall:
    for arg in n.cArgs.mitems:
      hoistFromExprs(arg, usedNames, newStmts)
  of gpuReturn:
    hoistFromExprs(n.rValue, usedNames, newStmts)
  of gpuAssign:
    hoistFromExprs(n.aLeft, usedNames, newStmts)
    hoistFromExprs(n.aRight, usedNames, newStmts)
  of gpuVar:
    hoistFromExprs(n.vInit, usedNames, newStmts)
  of gpuAddr:
    hoistFromExprs(n.aOf, usedNames, newStmts)
  of gpuDeref:
    hoistFromExprs(n.dOf, usedNames, newStmts)
  of gpuConv:
    hoistFromExprs(n.convExpr, usedNames, newStmts)
  of gpuCast:
    hoistFromExprs(n.cExpr, usedNames, newStmts)
  of gpuBinOp:
    hoistFromExprs(n.bLeft, usedNames, newStmts)
    hoistFromExprs(n.bRight, usedNames, newStmts)
  of gpuPrefix:
    hoistFromExprs(n.pVal, usedNames, newStmts)
  of gpuIf:
    hoistFromExprs(n.ifCond, usedNames, newStmts)
  of gpuFor:
    hoistFromExprs(n.fStart, usedNames, newStmts)
    hoistFromExprs(n.fEnd, usedNames, newStmts)
  of gpuWhile:
    hoistFromExprs(n.wCond, usedNames, newStmts)
  of gpuIndex:
    hoistFromExprs(n.iArr, usedNames, newStmts)
    hoistFromExprs(n.iIndex, usedNames, newStmts)
  of gpuDot:
    hoistFromExprs(n.dParent, usedNames, newStmts)
    hoistFromExprs(n.dField, usedNames, newStmts)
  of gpuObjConstr:
    for f in n.ocFields.mitems:
      hoistFromExprs(f.value, usedNames, newStmts)
  of gpuArrayLit:
    for v in n.aValues.mitems:
      hoistFromExprs(v, usedNames, newStmts)
  else:
    discard

proc unnest(n: var GpuAst; usedNames: var CountTable[string]) =
  ## Flatten nested gpuBlock scopes into a single flat statement list.
  ## Six cases must be handled:
  ##
  ## 1. Single-stmt gpuBlock wrapper (from nnkConstSection, nnkLetSection):
  ##    Unwrapped by firstNonBlock — the inner statement replaces the block.
  ##
  ## 2. gpuVar with block-valued vInit (from liftConstexpr hoisting):
  ##    vInit is a gpuBlock containing temps + a final expression.
  ##    The temps become separate statements; the last expr becomes vInit.
  ##    Handled recursively by hoistFromExprs (unwraps multi-level nesting).
  ##
  ## 3. gpuAssign with block-valued aLeft/aRight (from call-operator template expansion):
  ##    The preamble stmts become separate statements; the last stmt replaces the
  ##    expression slot (aLeft or aRight).
  ##
  ## 4. gpuCall with block-valued arguments (from flatten() and similar):
  ##    Same hoisting: preamble stmts from each block arg are spliced into the
  ##    parent block.
  ##
  ## 5. gpuReturn with block-valued rValue (from constexpr/compile-time folding):
  ##    Preamble hoisted, last stmt becomes the return value.
  ##
  ## 6. Multi-stmt gpuBlock as a regular statement (from template expansion):
  ##    Handled by collectStmt like any other statement type.
  ##    collectStmt lifts gpuConstexpr from expressions and deduplicates
  ##    {.inject.} variable names across unrolled iterations.
  case n.kind
  of gpuBlock:
    var newStmts: seq[GpuAst]
    for stmtIdx in 0 ..< n.statements.len:
      var stmt = n.statements[stmtIdx]
      var inner = firstNonBlock(stmt)
      if inner.kind == gpuVar:
        # gpuVar vInit blocks (from nnkBlockStmt) may lack isExpr=true,
        # so hoistBlockPreamble's firstNonBlock handles them regardless.
        hoistBlockPreamble(inner.vInit, usedNames, newStmts)
        # Also hoist expression-position blocks deeper in the vInit tree
        # (e.g. gpuCall arguments that are gpuBlock(isExpr: true))
        hoistFromExprs(inner, usedNames, newStmts)
        collectStmt(stmt, usedNames, newStmts)
      else:
        hoistFromExprs(inner, usedNames, newStmts)
        collectStmt(stmt, usedNames, newStmts)
    n.statements = newStmts
    for i in 0 ..< n.statements.len:
      unnest(n.statements[i], usedNames)
  of gpuIf:
    unnest(n.ifThen, usedNames)
    if n.ifElse.kind != gpuDiscard:
      unnest(n.ifElse, usedNames)
  of gpuFor:
    unnest(n.fBody, usedNames)
  of gpuWhile:
    unnest(n.wBody, usedNames)
  else:
    discard


proc registerLegalizationPasses*(reg: var PassRegistry) =
  ## Register passes that make the IR well-formed.

  reg.register("ensureBlock", pkTransform, phaseEarly,
    "Wraps non-block bodies in gpuBlock",
    proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.pBody.kind != gpuBlock:
          fn.pBody = GpuAst(kind: gpuBlock, statements: @[fn.pBody])
        fn.pBody.walk(proc(n: var GpuAst): void =
          case n.kind
          of gpuIf:
            if n.ifThen.kind != gpuBlock:
              n.ifThen = GpuAst(kind: gpuBlock, statements: @[n.ifThen])
            if n.ifElse.kind != gpuDiscard and n.ifElse.kind != gpuBlock:
              n.ifElse = GpuAst(kind: gpuBlock, statements: @[n.ifElse])
          of gpuFor:
            if n.fBody.kind != gpuBlock:
              n.fBody = GpuAst(kind: gpuBlock, statements: @[n.fBody])
          of gpuWhile:
            if n.wBody.kind != gpuBlock:
              n.wBody = GpuAst(kind: gpuBlock, statements: @[n.wBody])
          else: discard
        )
    )

  reg.register("maybeInsertResult", pkTransform, phaseMain,
    "Inserts var result and return result",
    dependsOn = @["ensureNoCustomResult", "ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        insertResult(ctx, fn)
    )


  reg.register("unnestBlockExprs", pkTransform, phaseMain,
    "Hoists preamble stmts out of block-valued expressions (variable init, LHS/RHS, call args, return value)",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        var usedNames: CountTable[string]
        unnest(fn.pBody, usedNames)
    )
