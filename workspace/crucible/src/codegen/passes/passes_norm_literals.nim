# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Normalization: literal types settle at the use sites.
##
##   - a literal operand re-types to the operand type of its operation,
##     of a comparison, of a call formal, or of an assignment destination
##   - a comparison whose first operand is a literal reads its operand type
##     from the other operand: the frontend swaps `>` and `>=` operands, so
##     the literal can sit first
##
## The frontend lowers a literal at the width its source node carries.
## This pass decides the final type from the typed IR, before the feature
## scans read the literal widths.

{.experimental: "codeReordering".}

import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerRetypeLiterals*(reg: var PassRegistry) =
  reg.register("retypeLiterals", pkTransform,
    "Re-types literals to their use-site type: the operation's operand type," &
    " the callee formals' types, the assignment destination",
    run = retypeLiteralsPass)

# ═══════════════════════════════════════════════════════════════════════
# Literal retyping
# ═══════════════════════════════════════════════════════════════════════

proc retypeOperand(e: CirExpr, target: CirType) =
  ## Re-types a literal operand to `target` in place, an exact conversion.
  ## Non-literals and kind-mismatched pairs return unchanged.
  if e.isNil or target.isNil:
    return
  case e.kind
  of eFloatLit:
    if target.kind == ctFloat and not sameType(e.fLitType, target):
      e.fLitType = target
  of eIntLit:
    if target.kind in {ctInt, ctUint} and not sameType(e.litType, target):
      e.litType = target
  else:
    discard

proc retypeCallArgs(callee: CirExpr, args: seq[CirExpr]) =
  ## Re-types literal call arguments to the callee formals' types.
  ## The runtime arguments and the proc type's parameters align positionally:
  ## both skip the compile-time formals.
  if callee.isNil:
    return
  let fnType = getExprType(callee)
  if fnType.isNil or fnType.kind != ctProc:
    return
  for i in 0 ..< min(args.len, fnType.params.len):
    args[i].retypeOperand(fnType.params[i])

proc retypeLiteralsPass(m: CirModule): string =
  for fn in m.procs:
    let onExpr = proc(e: CirExpr) =
      case e.kind
      of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
        e.left.retypeOperand(e.opType)
        e.right.retypeOperand(e.opType)
      of eEq, eNeq, eLt, eLe:
        e.a.retypeOperand(e.cmpType)
        e.b.retypeOperand(e.cmpType)
        if e.a.kind in {eIntLit, eFloatLit} and
            e.b.kind notin {eIntLit, eFloatLit} and
            getExprType(e.b) != nil:
          e.cmpType = getExprType(e.b)
      of eCall:
        e.callee.retypeCallArgs(e.args)
      else:
        discard
    let onStmt = proc(s: CirStmt) =
      case s.kind
      of sAsgn:
        s.rhs.retypeOperand(getExprType(s.lhs))
      of sCall:
        s.callCallee.retypeCallArgs(s.callArgs)
      else:
        discard
    fn.body.walkStmts(onStmt, onExpr)
