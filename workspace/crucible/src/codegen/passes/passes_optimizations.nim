# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Optimization stage over CIRv2: arithmetic identity folding, dead-local elimination.
##
## - identity folding: a `mul`/`div` by the literal 1 or an integer `add`/`sub`
##   with the literal 0 collapses to the surviving operand, one bottom-up walk
## - dead-local elimination: a local nothing reads afterwards is deleted,
##   its type never reaches the printer
## - folding runs first, the eliminations it enables cascade into the dead-local pass

import std/[sequtils, sets]
import ../ir/cirv2_lang
import ./pass_datatypes

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerOptimizationPasses*(reg: var PassRegistry) =
  ## Registers the optimization stage passes, folding before dead-local
  ## elimination so the folds cascade into the deletions.
  reg.register("foldArithmeticIdentities", pkTransform, phaseMain,
    "Folds mul/div by the literal 1 and integer add/sub with 0 to the surviving operand",
    run = foldArithmeticIdentities)
  reg.register("removeDeadLocals", pkTransform, phaseMain,
    "Deletes the locals bound to constants nothing reads afterwards",
    run = eliminateDeadLocals)

# ═══════════════════════════════════════════════════════════════════════
# Arithmetic identity folding
# ═══════════════════════════════════════════════════════════════════════

func isLitValue(e: CirExpr, value: int64): bool =
  ## True for the integer or float literal carrying `value` exactly.
  case e.kind
  of eIntLit: e.value == value
  of eFloatLit: e.fValue == float64(value)
  else: false

proc foldArithIdentity(e: CirExpr): CirExpr =
  ## Identity fold of one already-rewritten arithmetic node,
  ## nil when no identity applies.
  ##
  ## Soundness:
  ## - `x * 1` and `x / 1` are exact for every int and IEEE float value,
  ##   the bit pattern is preserved (including -0.0, NaN, Inf), so they fold
  ## - integer `add`/`sub` with 0 fold, float `add` with 0 does not:
  ##   `-0.0 + 0.0` is `+0.0` and the bit pattern changes
  ## - `x * 0` is not the identity (NaN, Inf), it never folds
  case e.kind
  of eMul:
    if isLitValue(e.left, 1):
      return e.right
    if isLitValue(e.right, 1):
      return e.left
  of eDiv:
    if isLitValue(e.right, 1):
      return e.left
  of eAdd, eSub:
    if e.opType.kind in {ctInt, ctUint}:
      if e.kind == eAdd:
        if isLitValue(e.left, 0):
          return e.right
        if isLitValue(e.right, 0):
          return e.left
      else:
        if isLitValue(e.right, 0):
          return e.left
  else:
    discard

proc foldIdentitiesExpr(e: CirExpr) =
  ## Folds the arithmetic identities in one expression, bottom-up,
  ## in place: a folded node's surviving operand moves into the node.
  if e.isNil:
    return
  case e.kind
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    foldIdentitiesExpr(e.left)
    foldIdentitiesExpr(e.right)
    let folded = foldArithIdentity(e)
    if folded != nil:
      e[] = folded[]
  of eAt:
    foldIdentitiesExpr(e.atBase)
    foldIdentitiesExpr(e.atIndex)
  of ePat:
    foldIdentitiesExpr(e.patBase)
    foldIdentitiesExpr(e.patIndex)
  of eDot:
    foldIdentitiesExpr(e.dotBase)
  of eDeref:
    foldIdentitiesExpr(e.derefOperand)
  of eAddr:
    foldIdentitiesExpr(e.addrOperand)
  of eEq, eNeq, eLt, eLe:
    foldIdentitiesExpr(e.a)
    foldIdentitiesExpr(e.b)
  of eAnd, eOr:
    foldIdentitiesExpr(e.andLeft)
    foldIdentitiesExpr(e.andRight)
  of eNot:
    foldIdentitiesExpr(e.notOperand)
  of eNeg, eBitNot:
    foldIdentitiesExpr(e.unOperand)
  of eCast, eConv:
    foldIdentitiesExpr(e.castOperand)
  of eOconstr:
    for f in e.kvFields.mitems:
      foldIdentitiesExpr(f.value)
  of eAconstr:
    for el in e.elems.mitems:
      foldIdentitiesExpr(el)
  of eCall:
    foldIdentitiesExpr(e.callee)
    for a in e.args.mitems:
      foldIdentitiesExpr(a)
  of eNil, eTrue, eFalse, eIntLit, eFloatLit, eSym, eSizeof:
    discard

proc foldIdentitiesStmts(ss: var seq[CirStmt]) =
  ## Folds the arithmetic identities through one statement list,
  ## nested bodies included.
  for s in ss:
    case s.kind
    of sVar, sConst:
      foldIdentitiesExpr(s.init)
    of sAsgn:
      foldIdentitiesExpr(s.lhs)
      foldIdentitiesExpr(s.rhs)
    of sIte:
      foldIdentitiesExpr(s.cond)
      foldIdentitiesStmts(s.thenBody)
      foldIdentitiesStmts(s.elseBody)
    of sLoop:
      foldIdentitiesStmts(s.loopBody)
    of sCase:
      foldIdentitiesExpr(s.selector)
      for b in s.branches.mitems:
        foldIdentitiesStmts(b.body)
      foldIdentitiesStmts(s.caseElse)
    of sScope:
      foldIdentitiesStmts(s.scopeBody)
    of sAssume:
      foldIdentitiesExpr(s.assumeCond)
    of sRet:
      foldIdentitiesExpr(s.retVal)
    of sDiscard:
      foldIdentitiesExpr(s.discardVal)
    of sCall:
      foldIdentitiesExpr(s.callCallee)
      for a in s.callArgs.mitems:
        foldIdentitiesExpr(a)
    of sAsm:
      for op in s.asmOperands.mitems:
        foldIdentitiesExpr(op)
    of sEmit:
      for op in s.emitOperands.mitems:
        foldIdentitiesExpr(op)
    of sLab, sJmp:
      discard

proc foldArithmeticIdentities(m: CirModule): string =
  ## Folds the arithmetic identities in every proc body and module global.
  ## Returns "" always, the pass is unconditional.
  for fn in m.procs:
    foldIdentitiesStmts(fn.body)
  foldIdentitiesStmts(m.globals)
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Dead-local elimination
# ═══════════════════════════════════════════════════════════════════════

proc eliminateDeadLocals(m: CirModule): string =
  ## Deletes every local bound to a constant nothing reads, to a fixpoint.
  ## Returns "" always, the pass is unconditional.
  while m.procs.anyIt(removeDeadLocals(it)):
    discard
  return ""


proc collectLocalReads(e: CirExpr, reads: var HashSet[string]) =
  ## Every local name read in the expression.
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
      collectLocalReads(s.loopBody, reads)
    of sScope:
      collectLocalReads(s.scopeBody, reads)
    of sRet:
      collectLocalReads(s.retVal, reads)
    of sCall:
      collectLocalReads(s.callCallee, reads)
      for a in s.callArgs:
        collectLocalReads(a, reads)
    of sAsm:
      # an asm operand reads its symbol, the dead-local pass must keep it
      for op in s.asmOperands:
        collectLocalReads(op, reads)
    of sEmit:
      # an emit operand reads its symbol. The dead-local pass must keep it.
      for op in s.emitOperands:
        collectLocalReads(op, reads)
    else:
      discard

proc removeDeadLocals(fn: CirProc): bool =
  ## Deletes the locals nothing reads, their writes included, one round.
  ## True when something was removed, the caller iterates to a fixpoint.
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
