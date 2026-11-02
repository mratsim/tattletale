# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Normalization: expression forms settle over the typed IR.
##
##   - a literal re-types to the operand type at its use site: operation, comparison, call formal, or assignment destination
##   - a conversion over a literal folds to the literal at the conversion target, a bfloat16
##     the pass keeps the node for a bfloat16 target: bfloat16 has no literal token form, the conversion prints
##   - a bitwise `and`/`or`/`xor` over bool operands becomes the logical form, `xor` prints as the `!=` comparison
##   - an object constructor normalizes to its type's declaration order, omitted fields fill with the type's zero value
##
## The frontend lowers syntactic shape only: the source node's literal width,
## the bitwise operator form, the constructor's given fields.
## This pass decides the final forms from the typed IR.
## The feature scans read the literal widths after this pass runs.

{.experimental: "codeReordering".}

import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSettleExprForms*(reg: var PassRegistry) =
  reg.register("settleExprForms", pkTransform,
    "Settles expression forms over the typed IR: literal use-site types," &
    " the bool operator domain, literal conversions, constructor field order",
    run = settleExprFormsPass)

# ═══════════════════════════════════════════════════════════════════════
# Rules
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
