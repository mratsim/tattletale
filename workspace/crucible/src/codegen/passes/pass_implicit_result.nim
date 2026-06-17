# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Implicit result variable pass.
##
## Validates and transforms GPU IR for the implicit `result` variable:
##
## 1. `ensureNoCustomResult` — rejects `var result` at any nesting depth
## 2. `ensureResultAssignedBeforeRead` — warns if `result` is read before write
## 3. `maybeInsertResult` — inserts `var result` declaration + `return result`

import std / [macros, sequtils]
import ../ir/gpu_types

proc ensureNoCustomResult*(n: GpuAst; fnName: string) =
  ## Raise a compile-time error if a `gpuVar` named `result` exists
  ## at any nesting depth (including inside gpuIf/gpuFor/gpuWhile).
  case n.kind
  of gpuBlock:
    for ch in n:
      ensureNoCustomResult(ch, fnName)
  of gpuVar:
    if n.vName.ident() == "result":
      error fnName & " has a custom `result` variable which shadows the implicit `result` (not allowed in GPU code)"
  of gpuIf:
    ensureNoCustomResult(n.ifThen, fnName)
    if n.ifElse.kind != gpuVoid:
      ensureNoCustomResult(n.ifElse, fnName)
  of gpuFor:
    ensureNoCustomResult(n.fBody, fnName)
  of gpuWhile:
    ensureNoCustomResult(n.wBody, fnName)
  else:
    discard

proc ensureResultAssignedBeforeRead*(n: GpuAst; fnName: string): bool =
  ## Warn if `result` is read before being written. Returns true if `result`
  ## is definitely assigned before any read in `n`. For conditional branches,
  ## returns true only if BOTH branches assign.
  case n.kind
  of gpuBlock:
    var assigned = false
    for ch in n:
      if ch.kind == gpuAssign and ch.aLeft.kind == gpuIdent and ch.aLeft.ident() == "result":
        assigned = true
      elif ch.kind == gpuIf:
        let thenAssigned = ch.ifThen.ensureResultAssignedBeforeRead(fnName)
        let elseAssigned = ch.ifElse.kind != gpuVoid and ch.ifElse.ensureResultAssignedBeforeRead(fnName)
        assigned = thenAssigned and elseAssigned
      elif ch.kind in {gpuFor, gpuWhile}:
        discard  # loop may not execute, so assignment inside is not guaranteed
      else:
        if not assigned:
          warning "result may be used before being initialized in " & fnName & ". " &
            "Assign to `result = ...` before reading it."
    result = assigned
  of gpuAssign:
    result = n.aLeft.kind == gpuIdent and n.aLeft.ident() == "result"
  else:
    result = false

proc maybeInsertResult*(ast: var GpuAst, retType: GpuType, fnName: string) =
  ## Insert `var result: T` at the top and `return result` at the bottom,
  ## unless the last statement is already a return.
  ##
  ## Pre-conditions (call before this):
  ## - `ensureNoCustomResult` — no `var result` in body
  ## - `ensureResultAssignedBeforeRead` — result is assigned before read
  if retType.kind == gtVoid: return

  proc lastIsReturn(n: GpuAst): bool =
    doAssert n.kind == gpuBlock
    if n.statements[^1].kind == gpuReturn: return true

  if not lastIsReturn(ast):
    let resId = GpuAst(kind: gpuIdent, iName: "result",
                       iSym: "result",
                       iTyp: retType,
                       symbolKind: gsLocal)
    let res = GpuAst(kind: gpuVar, vName: resId,
                     vType: retType,
                     vInit: GpuAst(kind: gpuVoid),
                     vRequiresMemcpy: false,
                     vMutable: true)
    ast.statements.insert(res, 0)

    if not lastIsReturn(ast):
      ast.statements.add GpuAst(kind: gpuReturn, rValue: resId)

    # For generic instantiations: the Nim compiler hasn't rewritten
    # single-expression bodies to `result = expr`. Fix up the last
    # standalone expression (if any) to assign to result.
    for i in countdown(ast.statements.high, 0):
      let stmt = ast.statements[i]
      if stmt.kind notin {gpuVar, gpuComment, gpuVoid, gpuReturn, gpuIf, gpuFor, gpuWhile}:
        if stmt.kind == gpuBlock and stmt.isExpr:
          if stmt.statements.len == 1:
            ast.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt.statements[0])
          else:
            ast.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        elif stmt.kind != gpuAssign:
          ast.statements[i] = GpuAst(kind: gpuAssign, aLeft: resId, aRight: stmt)
        break
