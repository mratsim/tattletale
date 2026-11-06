# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ############################################################
#
#            Legalization stage
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerRewriteStmtsExprs*(reg: var PassRegistry) =
  reg.register("rewriteStmtsExprs", pkTransform,
    "Rewrites statement-expressions so their statements run in statement position," &
    " in front of the statement reading the value or inside the branch computing it",
    run = rewriteStmtsExprsPass)

func registerResolveBoolDomain*(reg: var PassRegistry) =
  reg.register("resolveBoolDomain", pkTransform,
    "Rewrites `and`/`or` over integer operands to the bit form",
    run = resolveBoolDomainPass)

# ═══════════════════════════════════════════════════════════════════════
#  Statement-expression elimination
# ═══════════════════════════════════════════════════════════════════════

proc rewriteStmt(el: var TempBinder, s: CirStmt): seq[CirStmt]
proc rewriteBody(el: var TempBinder, body: seq[CirStmt]): seq[CirStmt]
proc rewriteExpr(el: var TempBinder, e: CirExpr): seq[CirStmt]

proc rewriteBranchIntoTemp(el: var TempBinder, e: CirExpr, tmp: CirExpr): seq[CirStmt] =
  if e.kind == eStmts:
    var arm: seq[CirStmt]
    for s in e.stmts:
      arm.add(el.rewriteStmt(s))
    arm.add(el.rewriteExpr(e.valueExpr))
    arm.add(asgnS(tmp, e.valueExpr))
    return arm
  let pre = el.rewriteExpr(e)
  return pre & @[asgnS(tmp, e)]

proc rewriteExpr(el: var TempBinder, e: CirExpr): seq[CirStmt] =
  ## A C expression has no statement position.
  ## Rewrites every statement-expression in place, its statements returned
  ## for the front.
  ##
  ## Before (IR):
  ##   x = (stmts t = a (t + b))
  ##
  ## After (IR):
  ##   t = a
  ##   x = (t + b)
  ##
  ## A statement-expression inside a ternary branch or a bool `and`/`or`
  ## operand assigns a temp instead.
  ## An operand skipped by short-circuit evaluation runs no statements.
  ##
  ## Before (IR):
  ##   x = (ite c (stmts t = a (t + b)) 0)
  ##
  ## After (IR):
  ##   var _branch_0                (the temp, both branches assign it)
  ##   if c:
  ##     t = a
  ##     _branch_0 = (t + b)
  ##   else:
  ##     _branch_0 = 0
  ##   x = _branch_0
  if e.isNil:
    return @[]
  case e.kind
  of eStmts:
    var pre: seq[CirStmt]
    for s in e.stmts:
      pre.add(el.rewriteStmt(s))
    pre.add(el.rewriteExpr(e.valueExpr))
    e[] = e.valueExpr[]
    return pre
  of eTernary:
    let condPre = el.rewriteExpr(e.cond)
    if not (e.thenE.containsStmtsDeep() or e.elseE.containsStmtsDeep()):
      return condPre & el.rewriteExpr(e.thenE) & el.rewriteExpr(e.elseE)
    let t = e.ternType
    let cond = e.cond
    let name = el.claimTempName("branch_val")
    let tmp = sym(name, t)
    let thenArm = el.rewriteBranchIntoTemp(e.thenE, tmp)
    let elseArm = el.rewriteBranchIntoTemp(e.elseE, tmp)
    e[] = tmp[]
    return condPre & @[varD(name, t), iteS(cond, thenArm, elseArm)]
  of eAnd, eOr:
    if getExprType(e.andLeft).kind != ctBool or not e.andRight.containsStmtsDeep():
      # an integer `and`/`or` evaluates both operands, its statements hoist eagerly
      var pre: seq[CirStmt]
      for ch in e.getExprChildren():
        pre.add(el.rewriteExpr(ch))
      return pre
    let leftPre = el.rewriteExpr(e.andLeft)
    let cond = e.andLeft
    let name = el.claimTempName("short_circuit")
    let tmp = sym(name, tBool())
    let arm = el.rewriteBranchIntoTemp(e.andRight, tmp)
    let skipped = @[asgnS(tmp, if e.kind == eAnd: falseE() else: trueE())]
    let (thenArm, elseArm) = if e.kind == eAnd: (arm, skipped) else: (skipped, arm)
    e[] = tmp[]
    return leftPre & @[varD(name, tBool()), iteS(cond, thenArm, elseArm)]
  else:
    # statement-expressions in child expressions contribute their statements
    # to the return value, in encounter order
    var pre: seq[CirStmt]
    for ch in e.getExprChildren():
      pre.add(el.rewriteExpr(ch))
    return pre

proc perTestClone(s: CirStmt): CirStmt =
  ## One condition statement cloned for the re-run at the end of a loop body.
  ##
  ## A condition declaration's scope ends at the condition:
  ## - a declaration with an initial value assigns the pre-loop binding,
  ##   the next test reads the recomputed value
  ## - a declaration without an initial value recomputes nothing and drops
  ## - any other statement clones as is
  if s.kind == sVar:
    if s.init.isNil:
      return nil
    return asgnS(sym(s.name, s.declType), cloneExpr(s.init))
  return cloneStmts(@[s])[0]

proc rewriteStmt(el: var TempBinder, s: CirStmt): seq[CirStmt] =
  ## One statement with its expressions eliminated.
  ## The statements run in front, the statement last.
  case s.kind
  of sVar:
    if s.init.isNil:
      return @[s]
    let pre = el.rewriteExpr(s.init)
    return pre & @[s]
  of sAsgn:
    var pre: seq[CirStmt]
    pre.add(el.rewriteExpr(s.lhs))
    pre.add(el.rewriteExpr(s.rhs))
    return pre & @[s]
  of sIte:
    # a condition's statements run where its test runs: the first branch's
    # statements run in front of the whole chain, a later branch's statements
    # run at the head of the else body, in front of that branch's own test
    let condPre = el.rewriteExpr(s.cond)
    s.thenBody = el.rewriteBody(s.thenBody)
    s.elseBody = el.rewriteBody(s.elseBody)
    return condPre & @[s]
  of sWhile:
    # Nim evaluates the condition before every test, the first included.
    #
    # Before (IR):
    #   while (stmts n = n + 1 (n < 3)):
    #     i = i + 1
    #
    # After (IR):
    #   n = n + 1
    #   while (n < 3):
    #     i = i + 1
    #     n = n + 1
    let condPre = el.rewriteExpr(s.whileCond)
    s.whileBody = el.rewriteBody(s.whileBody)
    if condPre.len == 0:
      return @[s]
    for pre in condPre:
      let replay = perTestClone(pre)
      if replay != nil:
        s.whileBody.add(replay)
    return condPre & @[s]
  of sRet:
    let pre = el.rewriteExpr(s.retVal)
    return pre & @[s]
  of sDiscard:
    let pre = el.rewriteExpr(s.discardVal)
    return pre & @[s]
  of sCall:
    var pre: seq[CirStmt]
    pre.add(el.rewriteExpr(s.callCallee))
    for a in s.callArgs:
      pre.add(el.rewriteExpr(a))
    return pre & @[s]
  of sAsm:
    var pre: seq[CirStmt]
    for op in s.asmOperands:
      pre.add(el.rewriteExpr(op))
    return pre & @[s]
  of sEmit:
    var pre: seq[CirStmt]
    for op in s.emitOperands:
      pre.add(el.rewriteExpr(op))
    return pre & @[s]

proc rewriteBody(el: var TempBinder, body: seq[CirStmt]): seq[CirStmt] =
  for s in body:
    result.add(el.rewriteStmt(s))

proc resolveBoolDomainPass(m: CirModule) =
  for fn in m.procs:
    fn.body.walkStmts(nil, proc(e: CirExpr) =
      case e.kind
      of eAnd, eOr:
        let t = getExprType(e.andLeft)
        if t.kind notin {ctInt, ctUint}:
          return
        let a = e.andLeft
        let b = e.andRight
        if e.kind == eAnd:
          e[] = bitAndE(t, a, b)[]
        else:
          e[] = bitOrE(t, a, b)[]
      of eBitXor:
        if e.opType.kind == ctBool:
          raiseAssert("bool xor reached the bool-domain pass as the bit form")
      else:
        discard)

proc rewriteStmtsExprsPass(m: CirModule) =
  for fn in m.procs:
    var el = tempBinderFor(fn)
    fn.body = el.rewriteBody(fn.body)

