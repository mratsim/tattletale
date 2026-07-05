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
                     vInit: GpuAst(kind: gpuVoid),
                     vRequiresMemcpy: false,
                     vMutable: true)
    fn.pBody.statements.insert(res, 0)

    if not lastIsReturn(fn.pBody):
      fn.pBody.statements.add GpuAst(kind: gpuReturn, rValue: resId)

    for i in countdown(fn.pBody.statements.high, 0):
      let stmt = fn.pBody.statements[i]
      if stmt.kind notin {gpuVar, gpuComment, gpuVoid, gpuReturn, gpuIf, gpuFor, gpuWhile}:
        if stmt.kind == gpuBlock and stmt.isExpr:
          if stmt.statements.len == 1:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt.statements[0])
          else:
            fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        elif stmt.kind != gpuAssign:
          fn.pBody.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        break

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
            if n.ifElse.kind != gpuVoid and n.ifElse.kind != gpuBlock:
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
              n.statements[i] = GpuAst(kind: gpuVoid)
            else:
              liftConstexprFrom(n.statements[i], lifts)
        else:
          discard

      proc unnest(n: var GpuAst) =
        case n.kind
        of gpuBlock:
          var newStmts: seq[GpuAst]
          for stmtIdx in 0 ..< n.statements.len:
            var stmt = n.statements[stmtIdx]
            let inner = firstNonBlock(stmt)
            # First: lift block inits
            if inner.kind in {gpuVar}:
              # Repeatedly unwrap nested blocks in vInit.
              # Deduplicate let/var names — recursive template expansion
              # (foldZipWith_recurse) reuses {.inject.} names (acc, it_a,
              # it_b) across iterations. Nim's C backend assigns each a
              # unique mangled name; crucible must do the same.
              while true:
                let vInitInner = firstNonBlock(inner.vInit)
                if vInitInner.kind == gpuBlock and vInitInner.statements.len > 1:
                  for j in 0 ..< vInitInner.statements.len - 1:
                    let stmt = vInitInner.statements[j]
                    # Unwrap single-var gpuBlock (from nnkLetSection wrapper)
                    var innerStmt = stmt
                    while innerStmt.kind == gpuBlock and innerStmt.statements.len == 1:
                      innerStmt = innerStmt.statements[0]
                    if innerStmt.kind == gpuVar:
                      let name = innerStmt.vName.ident()
                      var dup = false
                      for ex in newStmts:
                        var exInner = ex
                        while exInner.kind == gpuBlock and exInner.statements.len == 1:
                          exInner = exInner.statements[0]
                        if exInner.kind == gpuVar and exInner.vName.ident() == name:
                          dup = true
                          break
                      if dup:
                        let uniqueName = name & "_" & $newStmts.len
                        innerStmt.vName.iName = uniqueName
                        innerStmt.vName.iSym = uniqueName & "_" & $newStmts.len
                    newStmts.add stmt
                  inner.vInit = vInitInner.statements[^1]
                else:
                  break
            # Second: lift any gpuConstexpr from expression children
            if stmt.kind != gpuConstexpr:
              var lifts: seq[GpuAst]
              liftConstexprFrom(stmt, lifts)
              for l in lifts:
                newStmts.add l
            newStmts.add stmt
          n.statements = newStmts
          for i in 0 ..< n.statements.len:
            unnest(n.statements[i])
        of gpuIf:
          unnest(n.ifThen)
          if n.ifElse.kind != gpuVoid:
            unnest(n.ifElse)
        of gpuFor:
          unnest(n.fBody)
        of gpuWhile:
          unnest(n.wBody)
        else:
          discard
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        unnest(fn.pBody)
    )
