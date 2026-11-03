# Tattletale
# Copyright (c) 2026 Mamy Andre-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/sets
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ############################################################
#
#            Normalization stage
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSettleExprForms*(reg: var PassRegistry) =
  reg.register("settleExprForms", pkTransform,
    "Settles expression forms over the typed IR: literal use-site types," &
    " the bool operator domain, literal conversions, constructor field order",
    run = settleExprFormsPass)

func registerCompoundAssignTemps*(reg: var PassRegistry) =
  reg.register("bindCompoundAssignTemps", pkTransform,
    "Binds the call-bearing operands of a compound-assign target into temps" &
    " so the single Nim evaluation prints once",
    run = bindCompoundAssignTempsPass)

func registerMarkExplicitUnrolls*(reg: var PassRegistry) =
  reg.register("markExplicitUnrolls", pkTransform,
    "Sets the unroll flag on the loops whose body states the `{.unroll.}`" &
    " marker statement and drops every marker",
    run = markExplicitUnrollsPass)

func registerNormalizationPasses*(reg: var PassRegistry) =
  # settleExprForms decides the literal widths, the bool operator domain,
  # constructor field order, before the later stages read them
  reg.registerSettleExprForms()
  # bindCompoundAssignTemps runs first among the reshaping passes:
  # shared-tree detection compares references
  reg.registerCompoundAssignTemps()
  reg.registerMarkExplicitUnrolls()

# ═══════════════════════════════════════════════════════════════════════
#  Expression settling
# ═══════════════════════════════════════════════════════════════════════


proc zeroValue(t: CirType): CirExpr =
  ## Zero value of a CIRv2 type, the fill for an omitted constructor field.
  case t.kind
  of ctInt, ctUint:
    result = intLit(0, t)
  of ctFloat:
    result = fltLit(0.0, t)
  of ctString:
    result = strLit("")
  of ctBf16:
    # bfloat16 has no literal token form, the zero is the f32 zero converted
    result = convE(t, fltLit(0.0, tFloat(32)))
  of ctBool:
    result = falseE()
  of ctPtr, ctAptr:
    result = nilE(t)
  of ctArray:
    var elems: seq[CirExpr] = @[]
    for i in 0 ..< t.aN:
      elems.add(zeroValue(t.aElem))
    result = aconstrE(t, elems)
  of ctStruct:
    var fields: seq[CirKv] = @[]
    for f in t.fields:
      fields.add(kv(f.name, zeroValue(f.typ)))
    result = oconstrE(t, fields)
  of ctVoid, ctVector, ctProc, ctSpan:
    result = nil

proc retypeOperand(e: CirExpr, target: CirType) =
  ## Re-types a literal operand to `target` in place, an exact conversion.
  ## The pass leaves non-literals and kind-mismatched pairs unchanged.
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

proc fillOmittedFields(e: CirExpr) =
  ## Reorders a constructor's fields to the type's declaration order,
  ## omitted fields fill with the type's zero value.
  if e.kind != eOconstr or e.ocType.isNil or e.ocType.kind != ctStruct:
    return
  var ordered: seq[CirKv] = @[]
  for f in e.ocType.fields:
    var value: CirExpr = nil
    for kvf in e.kvFields:
      if kvf.name == f.name:
        value = kvf.value
        break
    ordered.add(kv(f.name, if value != nil: value else: zeroValue(f.typ)))
  e.kvFields = ordered

proc settleExprFormsPass(m: CirModule): string =
  for fn in m.procs:
    let onExpr = proc(e: CirExpr) =
      case e.kind
      of eConv:
        # a conversion over a literal is exact, the node folds to a literal
        # at the conversion target
        if e.castType != nil and e.castType.kind != ctBf16 and
            e.castOperand.kind in {eIntLit, eFloatLit}:
          let typ = e.castType
          let op = e.castOperand
          if typ.kind == ctFloat and op.kind == eIntLit:
            e.kind = eFloatLit
            e.fValue = op.value.float
            e.fLitType = typ
          elif op.kind == eIntLit:
            e.kind = eIntLit
            e.value = op.value
            e.litType = typ
          else:
            # Float-literal operand, the cast retypes it in place
            e.kind = eFloatLit
            e.fValue = op.fValue
            e.fLitType = typ
      of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
        if e.kind in {eBitAnd, eBitOr, eBitXor} and e.opType != nil and
            e.opType.kind == ctBool:
          # bool domain: the logical form, `xor` prints as the != comparison
          let a = e.left
          let b = e.right
          let t = e.opType
          case e.kind
          of eBitAnd:
            e.kind = eAnd
            e.andLeft = a
            e.andRight = b
          of eBitOr:
            e.kind = eOr
            e.andLeft = a
            e.andRight = b
          else:
            e.kind = eNeq
            e.cmpType = t
            e.a = a
            e.b = b
        else:
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
      of eOconstr:
        e.fillOmittedFields()
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

# ═══════════════════════════════════════════════════════════════════════
#  Compound-assign temps
# ═══════════════════════════════════════════════════════════════════════


type TempBinder = object
  used: HashSet[string]
  fresh: int

proc claimTempName(b: var TempBinder): string =
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
    if s.kind == sVar:
      collected.incl s.name
  fn.body.walkStmts(onStmt, onExpr)
  result.used = collected

proc bindOperandTemp(b: var TempBinder, hoists: var seq[CirStmt], e: CirExpr): CirExpr =
  ## One evaluation per operand, both use sites read the temp.
  let typ = getExprType(e)
  let name = b.claimTempName()
  let tmp = sym(name, typ)
  hoists.add(varD(name, typ))
  hoists.add(asgnS(tmp, e))
  return tmp

proc bindCallBearingOperands(b: var TempBinder, hoists: var seq[CirStmt], e: CirExpr): CirExpr =
  ## Only operand positions bind: the lvalue spine (`x`, `x[i]`, `x.f`) is structural.
  ## An operand temp is legal on every target, an address temp is not:
  ## GLSL rewrites pointer locals to their pointees and keeps no function storage.
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
    # unreachable: the guard above admits only the four spine kinds
    discard
  return e

proc isSharedArithTarget(s: CirStmt): bool =
  ## A target call the user wrote twice lowers to two distinct trees, so the reference-identity match never fires on it.
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
    of sWhile:
      b.bindStmtTemps(s.whileBody)
      lowered.add(s)
    else:
      lowered.add(s)
  body = lowered

proc bindCompoundAssignTempsPass(m: CirModule): string =
  for fn in m.procs:
    var b = tempBinderFor(fn)
    b.bindStmtTemps(fn.body)

# ═══════════════════════════════════════════════════════════════════════
#  Unroll marking
# ═══════════════════════════════════════════════════════════════════════


const unrollMarker = "{.unroll.}"

func isUnrollMarker(s: CirStmt): bool =
  s.kind == sEmit and s.emitOperands.len == 0 and s.emitParts == @[unrollMarker]

proc markExplicitUnrolls(ss: var seq[CirStmt]) =
  var kept: seq[CirStmt]
  for s in mitems(ss):
    case s.kind
    of sWhile:
      var body: seq[CirStmt]
      var marked = false
      for b in s.whileBody:
        if b.isUnrollMarker:
          marked = true
        else:
          body.add b
      if marked:
        s.whileUnroll = true
      markExplicitUnrolls(body)
      s.whileBody = body
      kept.add s
    of sIte:
      markExplicitUnrolls(s.thenBody)
      markExplicitUnrolls(s.elseBody)
      kept.add s
    else:
      if not s.isUnrollMarker:
        kept.add s
  ss = kept

proc markExplicitUnrollsPass(m: CirModule): string =
  for fn in m.procs:
    markExplicitUnrolls(fn.body)
  return ""
