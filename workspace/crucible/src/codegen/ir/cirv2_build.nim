# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cirv2 builders: ergonomic constructors for the CIRv2 tree.
##
## Builders produce well-shaped nodes by construction:
##
## - typed ops get their type from the caller
## - comparisons derive their operand type from the first operand
## - calls exist only as statements (`callS`) or destination-bound expressions (`callE`)
##
## Anything the builders cannot express correctly is a verifier violation, and `cirv2_verify` catches it.

import cirv2_lang
import std/macros

# Type constructors

proc tVoid*(): CirType =
  CirType(kind: ctVoid)

proc tBool*(): CirType =
  CirType(kind: ctBool)

proc tInt*(bits: int): CirType =
  CirType(kind: ctInt, bits: bits)

proc tUint*(bits: int): CirType =
  CirType(kind: ctUint, bits: bits)

proc tFloat*(bits: int): CirType =
  CirType(kind: ctFloat, bits: bits)

proc tVec*(n: int, elem: CirType): CirType =
  CirType(kind: ctVector, vN: n, vElem: elem)

proc tPtr*(space: CirAddrSpace, elem: CirType): CirType =
  CirType(kind: ctPtr, space: space, pElem: elem)

proc tAptr*(space: CirAddrSpace, elem: CirType): CirType =
  CirType(kind: ctAptr, space: space, pElem: elem)

proc fld*(name: string, typ: CirType): CirField =
  CirField(name: name, typ: typ)

proc tStruct*(name: string, fields: varargs[CirField]): CirType =
  CirType(kind: ctStruct, name: name, fields: @fields)

proc tArray*(n: int, elem: CirType): CirType =
  CirType(kind: ctArray, aN: n, aElem: elem)

proc tProcT*(params: varargs[CirType], ret: CirType): CirType =
  CirType(kind: ctProc, params: @params, ret: ret)

proc withOrigin*(t: CirType, n: NimNode): CirType =
  ## Macro-time only: the origin survives `withSpace` copies but never participates in `sameType`.
  t.origin = n
  t

# Expression constructors

proc sym*(name: string, typ: CirType, builtin = false,
          coordBuiltin = cbkNone, synchroBuiltin = sbkNone,
          reductionBuiltin = rbkNone, atomicsBuiltin = abkNone,
          shuffleBuiltin = shkNone): CirExpr =
  ## One symbol reference. `builtin` marks a `{.builtin.}` read.
  ##
  ## Contract:
  ## - the canonicalization pass fills the typed builtin kinds from the name tables
  ## - the printers dispatch on the kinds
  CirExpr(kind: eSym, name: name, typ: typ, builtin: builtin,
    coordBuiltin: coordBuiltin, synchroBuiltin: synchroBuiltin,
    reductionBuiltin: reductionBuiltin, atomicsBuiltin: atomicsBuiltin,
    shuffleBuiltin: shuffleBuiltin)

proc intLit*(value: int64, typ: CirType): CirExpr =
  CirExpr(kind: eIntLit, value: value, litType: typ)

proc fltLit*(value: float64, typ: CirType): CirExpr =
  CirExpr(kind: eFloatLit, fValue: value, fLitType: typ)

proc nilE*(typ: CirType): CirExpr =
  CirExpr(kind: eNil, nilType: typ)

proc trueE*(): CirExpr =
  CirExpr(kind: eTrue)

proc falseE*(): CirExpr =
  CirExpr(kind: eFalse)

proc derefE*(operand: CirExpr): CirExpr =
  CirExpr(kind: eDeref, derefOperand: operand)

proc atE*(base, index: CirExpr): CirExpr =
  CirExpr(kind: eAt, atBase: base, atIndex: index)

proc patE*(base, index: CirExpr): CirExpr =
  CirExpr(kind: ePat, patBase: base, patIndex: index)

proc dotE*(base: CirExpr, fieldName: string, inherit = 0): CirExpr =
  CirExpr(kind: eDot, dotBase: base, fieldName: fieldName, inherit: inherit)

proc arithE*(opKind: CirExprKind, opType: CirType, left, right: CirExpr): CirExpr =
  ## One constructor for the typed binary arithmetic/bitwise family (eAdd/eSub/eMul/eDiv/eMod/eShl/eShr/eBitAnd/eBitOr/eBitXor).
  doAssert opKind in {eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor}
  result = CirExpr(kind: opKind)
  result.opType = opType
  result.left = left
  result.right = right

proc addE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eAdd, opType, left, right)
proc subE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eSub, opType, left, right)
proc mulE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eMul, opType, left, right)
proc divE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eDiv, opType, left, right)
proc modE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eMod, opType, left, right)
proc shlE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eShl, opType, left, right)
proc shrE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eShr, opType, left, right)
proc bitAndE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eBitAnd, opType, left, right)
proc bitOrE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eBitOr, opType, left, right)
proc bitXorE*(opType: CirType, left, right: CirExpr): CirExpr = arithE(eBitXor, opType, left, right)

proc bitNotE*(opType: CirType, operand: CirExpr): CirExpr =
  CirExpr(kind: eBitNot, unType: opType, unOperand: operand)

proc negE*(opType: CirType, operand: CirExpr): CirExpr =
  CirExpr(kind: eNeg, unType: opType, unOperand: operand)

proc cmpE*(opKind: CirExprKind, a, b: CirExpr): CirExpr =
  ## One constructor for the comparison family (eEq/eNeq/eLt/eLe).
  ## The operand type comes from `a`.
  doAssert opKind in {eEq, eNeq, eLt, eLe}
  result = CirExpr(kind: opKind)
  result.cmpType = getExprType(a)
  result.a = a
  result.b = b

proc eqE*(a, b: CirExpr): CirExpr = cmpE(eEq, a, b)
proc neqE*(a, b: CirExpr): CirExpr = cmpE(eNeq, a, b)
proc ltE*(a, b: CirExpr): CirExpr = cmpE(eLt, a, b)
proc leE*(a, b: CirExpr): CirExpr = cmpE(eLe, a, b)

proc notE*(operand: CirExpr): CirExpr =
  CirExpr(kind: eNot, notOperand: operand)

proc andE*(left, right: CirExpr): CirExpr =
  CirExpr(kind: eAnd, andLeft: left, andRight: right)

proc orE*(left, right: CirExpr): CirExpr =
  CirExpr(kind: eOr, andLeft: left, andRight: right)

proc castE*(typ: CirType, operand: CirExpr): CirExpr =
  CirExpr(kind: eCast, castType: typ, castOperand: operand)

proc convE*(typ: CirType, operand: CirExpr): CirExpr =
  CirExpr(kind: eConv, castType: typ, castOperand: operand)

proc addrE*(typ: CirType, operand: CirExpr): CirExpr =
  CirExpr(kind: eAddr, addrType: typ, addrOperand: operand)

proc sizeofE*(ofType: CirType): CirExpr =
  CirExpr(kind: eSizeof, ofType: ofType)

proc kv*(name: string, value: CirExpr): CirKv =
  CirKv(name: name, value: value)

proc oconstrE*(typ: CirType, fields: varargs[CirKv]): CirExpr =
  CirExpr(kind: eOconstr, ocType: typ, kvFields: @fields)

proc aconstrE*(typ: CirType, elems: varargs[CirExpr]): CirExpr =
  CirExpr(kind: eAconstr, acType: typ, elems: @elems)

proc callE*(callee: CirExpr, args: varargs[CirExpr]): CirExpr =
  ## A call bound to its destination: legal only as the RHS of `asgn`.
  CirExpr(kind: eCall, callee: callee, args: @args)

# Statement constructors

proc varD*(name: string, typ: CirType, init: CirExpr = nil, space: CirAddrSpace = caThread): CirStmt =
  ## Declares a local:
  ## - `space` names the memory class
  ## - `caThread` is the per-thread register default, `caThreadgroup` a `{.smem.}` local, `caConstant` a `{.const_mem.}` local
  CirStmt(kind: sVar, name: name, declType: typ, init: init, space: space)

proc constD*(name: string, typ: CirType, value: CirExpr): CirStmt =
  CirStmt(kind: sConst, name: name, declType: typ, init: value)

proc asgnS*(lhs, rhs: CirExpr): CirStmt =
  CirStmt(kind: sAsgn, lhs: lhs, rhs: rhs)

proc iteS*(cond: CirExpr, thenBody: seq[CirStmt], elseBody: seq[CirStmt] = @[]): CirStmt =
  CirStmt(kind: sIte, cond: cond, thenBody: thenBody, elseBody: elseBody)

proc loopS*(loopBody: seq[CirStmt], unroll = false): CirStmt =
  ## `unroll` marks a `{.unroll.}`-hinted loop: every printer emits a full-unroll directive in front of it.
  CirStmt(kind: sLoop, loopBody: loopBody, unroll: unroll)

proc rng*(lo, hi: int64): CirRange =
  CirRange(lo: lo, hi: hi)

proc branch*(ranges: seq[CirRange], body: seq[CirStmt]): CirCaseBranch =
  CirCaseBranch(ranges: ranges, body: body)

proc caseS*(selector: CirExpr, branches: seq[CirCaseBranch], caseElse: seq[CirStmt] = @[]): CirStmt =
  CirStmt(kind: sCase, selector: selector, branches: branches, caseElse: caseElse)

proc labS*(labName: string): CirStmt =
  CirStmt(kind: sLab, labName: labName)

proc jmpS*(jmpTarget: string): CirStmt =
  CirStmt(kind: sJmp, jmpTarget: jmpTarget)

proc scopeS*(scopeBody: seq[CirStmt]): CirStmt =
  CirStmt(kind: sScope, scopeBody: scopeBody)

proc assumeS*(assumeCond: CirExpr): CirStmt =
  CirStmt(kind: sAssume, assumeCond: assumeCond)

proc retS*(retVal: CirExpr = nil): CirStmt =
  CirStmt(kind: sRet, retVal: retVal)

proc discardS*(discardVal: CirExpr = nil): CirStmt =
  CirStmt(kind: sDiscard, discardVal: discardVal)

proc callS*(callCallee: CirExpr, callArgs: varargs[CirExpr]): CirStmt =
  ## A void call in statement position.
  CirStmt(kind: sCall, callCallee: callCallee, callArgs: @callArgs)

proc asmS*(asmParts: seq[string], asmOperands: seq[CirExpr]): CirStmt =
  ## One inline-assembly statement: verbatim template text alternating with symbol operands.
  ## The target printer splices the operands' names.
  CirStmt(kind: sAsm, asmParts: asmParts, asmOperands: asmOperands)

proc emitS*(emitParts: seq[string], emitOperands: seq[CirExpr]): CirStmt =
  ## One `{.emit.}` statement: verbatim target-source text alternating with symbol operands.
  ##
  ## Contract:
  ## - the target printer splices the operands' names and appends no terminator
  ## - the emitted text supplies its terminators
  CirStmt(kind: sEmit, emitParts: emitParts, emitOperands: emitOperands)

# Proc / module constructors

proc param*(name: string, typ: CirType): CirParam =
  CirParam(name: name, typ: typ)

proc procD*(name: string, params: seq[CirParam], ret: CirType, body: seq[CirStmt]): CirProc =
  CirProc(name: name, params: params, ret: ret, body: body)

proc moduleD*(name: string, types: seq[CirType], procs: seq[CirProc]): CirModule =
  CirModule(name: name, types: types, procs: procs)
