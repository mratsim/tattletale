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

proc flattenAssignLhs(n: var GpuAst) =
  ## When a gpuAssign's LHS is a gpuBlock(isExpr: true), hoist the block's
  ## preamble statements into the parent block and keep only the last
  ## statement as the assignment target. This prevents the backend from
  ## emitting a spurious `;` after the value-producing statement of an
  ## expression block used as an assignment lvalue.
  case n.kind
  of gpuBlock:
    var newStmts: seq[GpuAst]
    for stmt in n.statements.mitems:
      if stmt.kind == gpuAssign and
         stmt.aLeft.kind == gpuBlock and
         stmt.aLeft.isExpr:
        let lhsBlock = stmt.aLeft
        if lhsBlock.statements.len == 1:
          # Single statement: unwrap directly
          stmt.aLeft = lhsBlock.statements[0]
          newStmts.add stmt
        else:
          # Multi-statement: hoist preamble, last stmt becomes LHS
          for j in 0 ..< lhsBlock.statements.len - 1:
            newStmts.add lhsBlock.statements[j]
          stmt.aLeft = lhsBlock.statements[^1]
          newStmts.add stmt
      else:
        newStmts.add stmt
    n.statements = newStmts
    # Recurse into children
    for i in 0 ..< n.statements.len:
      flattenAssignLhs(n.statements[i])
  of gpuIf:
    flattenAssignLhs(n.ifThen)
    if n.ifElse.kind != gpuDiscard:
      flattenAssignLhs(n.ifElse)
  of gpuFor:
    flattenAssignLhs(n.fBody)
  of gpuWhile:
    flattenAssignLhs(n.wBody)
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

  reg.register("flattenAssignLhs", pkTransform, phaseMain,
    "Hoists preamble stmts out of block-valued LHS in assignments",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        flattenAssignLhs(fn.pBody)
    )

  reg.register("unnestBlockInits", pkTransform, phaseMain,
    "Lifts preceding stmts out of var/let block-inits and constexpr from expression slots",
    dependsOn = @["ensureBlock"],
    run = proc(ctx: var GpuContext): void =
      proc firstNonBlock(n: GpuAst): GpuAst =
        result = n
        while result.kind == gpuBlock and result.statements.len == 1:
          result = result.statements[0]

      # Lifts any gpuConstexpr found in expression children of `n`,
      # adding them to `lifts` and replacing them with their cIdent.
      proc liftConstexprFrom(n: var GpuAst; lifts: var seq[GpuAst]) =
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
          # Walk into expression blocks to find constexprs in their statements.
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

      proc unnest(n: var GpuAst; usedNames: var CountTable[string]) =
        ## Flatten nested gpuBlock scopes into a single flat statement list.
        ## Three cases of gpuBlock must be handled:
        ##
        ## 1. Single-stmt gpuBlock wrapper (from nnkConstSection, nnkLetSection):
        ##    Unwrapped by firstNonBlock — the inner statement replaces the block.
        ##
        ## 2. gpuVar with block-valued vInit (from liftConstexpr hoisting):
        ##    vInit is a gpuBlock containing temps + a final expression.
        ##    The temps become separate statements; the last expr becomes vInit.
        ##    Repeated via while loop for multi-level nesting.
        ##
        ## 3. Multi-stmt gpuBlock as a regular statement (from template expansion):
        ##    Handled by collectStmt like any other statement type.
        ##    collectStmt lifts gpuConstexpr from expressions and deduplicates
        ##    {.inject.} variable names across unrolled iterations.
        case n.kind
        of gpuBlock:
          var newStmts: seq[GpuAst]
          for stmtIdx in 0 ..< n.statements.len:
            var stmt = n.statements[stmtIdx]
            let inner = firstNonBlock(stmt)
            # Unwrap block-valued vInit — extract inner declarations
            if inner.kind == gpuVar:
              while true:
                let blockVal = firstNonBlock(inner.vInit)
                if blockVal.kind == gpuBlock and blockVal.statements.len > 1:
                  for j in 0 ..< blockVal.statements.len - 1:
                    var innerStmt = blockVal.statements[j]
                    collectStmt(innerStmt, usedNames, newStmts)
                  inner.vInit = blockVal.statements[^1]
                else:
                  break
              collectStmt(stmt, usedNames, newStmts)
            else:
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
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        var usedNames: CountTable[string]
        unnest(fn.pBody, usedNames)
    )
