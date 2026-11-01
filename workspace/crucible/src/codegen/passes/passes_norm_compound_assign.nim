# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Normalization: call-temp binding for compound assignments.
##
##   - asgnS(target, arith(target, rhs)) with target bearing a call
##     becomes temp declarations + asgnS(target, arith(temp, rhs))
##
## The frontend emits a scalar compound assignment whose target and arithmetic
## read share one tree: Nim evaluates the desugared `op=` left-hand side once,
## a call in the shared tree would print and evaluate twice.
##
## This pass binds every call-bearing operand of the shared tree into a fresh temp.
## Only the shared-tree shape binds: a target call the user wrote twice keeps both evaluations.
##
## Runs first in the normalization stage: the shared-tree detection compares references, so reshaping passes run after it.

{.experimental: "codeReordering".}

import std/sets
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

func registerCompoundAssignTemps*(reg: var PassRegistry) =
  reg.register("bindCompoundAssignTemps", pkTransform,
    "Binds the call-bearing operands of a compound-assign target into temps" &
    " so the single Nim evaluation prints once",
    run = bindCompoundAssignTempsPass)

# ═══════════════════════════════════════════════════════════════════════
# Temp binding
# ═══════════════════════════════════════════════════════════════════════

type TempBinder = object
  used: HashSet[string]
  fresh: int

proc tempName(b: var TempBinder): string =
  while true:
    let candidate = if b.fresh == 0: "lhs_val" else: "lhs_val_" & $b.fresh
    b.fresh.inc
    if candidate notin b.used:
      b.used.incl candidate
      return candidate

proc tempBinderFor(fn: CirProc): TempBinder =
  result.used.init
  for p in fn.params:
    result.used.incl p.name
  var collected = result.used
  let onExpr = proc(e: CirExpr) =
    if e.kind == eSym:
      collected.incl e.name
  let onStmt = proc(s: CirStmt) =
    if s.kind in {sVar, sConst}:
      collected.incl s.name
  fn.body.walkStmts(onStmt, onExpr)
  result.used = collected

proc bindOperandTemp(b: var TempBinder, hoists: var seq[CirStmt], e: CirExpr): CirExpr =
  ## Binds one call-bearing operand into a fresh temp, returns the temp symbol.
  ## The operand evaluates once, both use sites read the temp.
  let typ = getExprType(e)
  let name = b.tempName()
  let tmp = sym(name, typ)
  hoists.add(varD(name, typ))
  hoists.add(asgnS(tmp, e))
  return tmp

proc bindCallBearingOperands(b: var TempBinder, hoists: var seq[CirStmt], e: CirExpr): CirExpr =
  ## Binds each call-bearing operand of an lvalue tree into a fresh temp.
  ## Only operand positions bind: the lvalue spine (`x`, `x[i]`, `x.f`) stays structural.
  ## An operand temp is legal on every target, an address temp is not:
  ## GLSL dissolves pointer locals and keeps no function storage.
  if e.kind notin {eAt, ePat, eDot, eDeref}:
    if e.hasCallDeep():
      return b.bindOperandTemp(hoists, e)
    return e
  case e.kind
  of eAt:
    e.atBase = b.bindCallBearingOperands(hoists, e.atBase)
    if e.atIndex.hasCallDeep():
      e.atIndex = b.bindOperandTemp(hoists, e.atIndex)
  of ePat:
    e.patBase = b.bindCallBearingOperands(hoists, e.patBase)
    if e.patIndex.hasCallDeep():
      e.patIndex = b.bindOperandTemp(hoists, e.patIndex)
  of eDot:
    e.dotBase = b.bindCallBearingOperands(hoists, e.dotBase)
  of eDeref:
    e.derefOperand = b.bindCallBearingOperands(hoists, e.derefOperand)
  else:
    # unreachable: the guard above admits only the four spine kinds, and this branch keeps the case total
    discard
  return e

# ═══════════════════════════════════════════════════════════════════════
# Statement walk
# ═══════════════════════════════════════════════════════════════════════

proc isSharedArithTarget(s: CirStmt): bool =
  ## True for the compound-assign emission: the assignment target and the arithmetic
  ## read are the same tree reference.
  ## A target call the user wrote twice lowers to two distinct trees and never matches.
  if s.rhs.kind notin {eAdd, eSub, eMul, eDiv}:
    return false
  result = not s.rhs.left.isNil and s.rhs.left == s.lhs

proc bindStmtTemps(b: var TempBinder, body: var seq[CirStmt]) =
  var lowered: seq[CirStmt]
  for s in body.mitems:
    case s.kind
    of sAsgn:
      if s.isSharedArithTarget():
        var hoists: seq[CirStmt]
        s.lhs = b.bindCallBearingOperands(hoists, s.lhs)
        lowered.add(hoists & @[s])
      else:
        lowered.add(s)
    of sIte:
      b.bindStmtTemps(s.thenBody)
      b.bindStmtTemps(s.elseBody)
      lowered.add(s)
    of sLoop:
      b.bindStmtTemps(s.loopBody)
      lowered.add(s)
    else:
      lowered.add(s)
  body = lowered

proc bindCompoundAssignTempsPass(m: CirModule): string =
  for fn in m.procs:
    var b = tempBinderFor(fn)
    b.bindStmtTemps(fn.body)
