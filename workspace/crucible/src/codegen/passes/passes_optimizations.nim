# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Optimization stage over CIRv2: min/max builtin folding and dead-local elimination.
##
## - min/max folding: a min/max-idiom ternary over scalar operands becomes the native min/max builtin call
## - dead-local elimination: a local that nothing reads afterwards is deleted, its type never reaches the printer
## - the min/max fold runs first so its eliminations cascade into the dead-local pass

import std/[sequtils, sets]
import ../ir/[cirv2_lang, cirv2_build]
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerOptimizationPasses*(reg: var PassRegistry) =
  ## Registers the optimization stage passes, folding before dead-local
  ## elimination so the folds cascade into the deletions.
  reg.register("foldMinMaxBuiltins", pkTransform, phaseMain,
    "Folds a min/max-idiom ternary over scalar operands to the native min/max builtin call",
    run = foldMinMaxBuiltins)
  reg.register("removeDeadLocals", pkTransform, phaseMain,
    "Deletes the locals bound to constants nothing reads afterwards",
    run = eliminateDeadLocals)

# ═══════════════════════════════════════════════════════════════════════
# Min/max ternary folding
# ═══════════════════════════════════════════════════════════════════════

func sameExpr(a, b: CirExpr): bool =
  ## Structural equality of two expression trees: same kind, same payload, same children in order.
  ## Operand match test for the min/max fold: a ternary's branch operands against its condition operands.
  if a.isNil or b.isNil:
    return a.isNil and b.isNil
  if a.kind != b.kind:
    return false
  case a.kind
  of eSym:
    if a.name != b.name:
      return false
  of eIntLit:
    if a.value != b.value:
      return false
  of eFloatLit:
    if a.fValue != b.fValue:
      return false
  of eDot:
    if a.fieldName != b.fieldName:
      return false
  else:
    discard
  let ca = getExprChildren(a)
  let cb = getExprChildren(b)
  if ca.len != cb.len:
    return false
  for i in 0 ..< ca.len:
    if not sameExpr(ca[i], cb[i]):
      return false
  return true

func isNanGuard(e: CirExpr): bool =
  ## True for the negated self-equality `!(x == x)`, Nim's float min/max NaN test.
  ## `x != x` lowers to the same node shape.
  e.kind == eNot and e.notOperand.kind == eEq and
    sameExpr(e.notOperand.a, e.notOperand.b)

func foldMinMaxTernary(e: CirExpr): CirExpr =
  ## Replaces a min/max-idiom ternary with the native builtin call, nil when no idiom applies.
  ##
  ## - max folds `(P <= Q) ? Q : P`, min folds `(P <= Q) ? P : Q`
  ## - float operands fold only in the NaN-guard form `(P <= Q) || (X != X)`, on either `||` side.
  ##   A NaN operand unguarded fails `P <= Q` and selects the other branch.
  ##   The native builtin returns the other operand the same way, so only the guarded form folds.
  ## - both branch operands must be basic numeric of the same kind
  if e.kind != eTernary:
    return nil
  var le: CirExpr = nil
  var guarded = false
  case e.cond.kind
  of eLe:
    le = e.cond
  of eOr:
    for (cmp, guard) in [(e.cond.andLeft, e.cond.andRight),
                         (e.cond.andRight, e.cond.andLeft)]:
      if cmp.kind == eLe and isNanGuard(guard):
        le = cmp
        guarded = true
        break
  else:
    return nil
  if le.isNil:
    return nil
  let p = le.a
  let q = le.b
  var builtin: string
  if sameExpr(e.thenE, p) and sameExpr(e.elseE, q):
    builtin = "min"
  elif sameExpr(e.thenE, q) and sameExpr(e.elseE, p):
    builtin = "max"
  else:
    return nil
  let tA = getExprType(e.thenE)
  let tB = getExprType(e.elseE)
  if tA.isNil or tB.isNil:
    return nil
  if tA.kind notin {ctInt, ctUint, ctFloat} or tA.kind != tB.kind:
    return nil
  if tA.kind == ctFloat and not guarded:
    return nil
  var folded = CirExpr(kind: eCall)
  folded.callee = sym(builtin, freshProcType(@[tA, tB], e.ternType), true)
  folded.args = @[e.thenE, e.elseE]
  return folded

proc foldMinMaxExprs(e: CirExpr) =
  ## Folds the min/max ternary idioms of one expression tree in place.
  ## A folded node becomes the builtin call. The walk then descends into the new arguments.
  e.walkExprs(proc(x: CirExpr) =
    if x.kind == eTernary:
      let folded = foldMinMaxTernary(x)
      if folded != nil:
        x[] = folded[])

proc foldMinMaxBuiltins(m: CirModule): string =
  for fn in m.procs:
    fn.body.walkStmts(nil, foldMinMaxExprs)
  m.globals.walkStmts(nil, foldMinMaxExprs)
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Dead-local elimination
# ═══════════════════════════════════════════════════════════════════════

proc eliminateDeadLocals(m: CirModule): string =
  ## Deletes every local bound to a constant that nothing reads, to a fixpoint.
  while m.procs.anyIt(removeDeadLocals(it)):
    discard
  return ""


proc collectLocalReads(e: CirExpr, reads: var HashSet[string]) =
  if e.isNil:
    return
  if e.kind == eSym:
    reads.incl e.name
    return
  for ch in getExprChildren(e):
    collectLocalReads(ch, reads)

proc collectLocalReads(stmts: seq[CirStmt], reads: var HashSet[string]) =
  for s in stmts:
    case s.kind
    of sVar, sConst:
      collectLocalReads(s.init, reads)
    of sAsgn:
      collectLocalReads(s.lhs, reads)
      collectLocalReads(s.rhs, reads)
    of sIte:
      collectLocalReads(s.cond, reads)
      collectLocalReads(s.thenBody, reads)
      collectLocalReads(s.elseBody, reads)
    of sLoop:
      collectLocalReads(s.loopCond, reads)
      collectLocalReads(s.loopBody, reads)
    of sScope:
      collectLocalReads(s.scopeBody, reads)
    of sAssume:
      collectLocalReads(s.assumeCond, reads)
    of sDiscard:
      collectLocalReads(s.discardVal, reads)
    of sRet:
      collectLocalReads(s.retVal, reads)
    of sCall:
      collectLocalReads(s.callCallee, reads)
      for a in s.callArgs:
        collectLocalReads(a, reads)
    of sAsm:
      # an asm operand reads its symbol, so the dead-local pass keeps it
      for op in s.asmOperands:
        collectLocalReads(op, reads)
    of sEmit:
      # an emit operand reads its symbol, so the dead-local pass keeps it
      for op in s.emitOperands:
        collectLocalReads(op, reads)
    else:
      discard

proc removeDeadLocals(fn: CirProc): bool =
  ## Deletes the locals that nothing reads, their writes included, one round.
  ## True when something was removed. The caller iterates to a fixpoint.
  var reads = initHashSet[string]()
  fn.body.collectLocalReads(reads)
  var removed = false
  var kept: seq[CirStmt]
  for s in fn.body:
    case s.kind
    of sVar, sConst:
      if s.name notin reads and s.name != "result":
        removed = true
        continue
      kept.add s
    of sAsgn:
      if s.lhs.kind == eSym and s.lhs.name notin reads and s.lhs.name != "result":
        removed = true
        continue
      kept.add s
    else:
      kept.add s
  fn.body = kept
  return removed
