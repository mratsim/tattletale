# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import cirv2_lang
import std/macros

# ############################################################
#
#            CIRv2 builders
#
# ############################################################

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

proc tBf16*(): CirType =
  CirType(kind: ctBf16, bits: 16)

proc tVec*(n: int, elem: CirType): CirType =
  CirType(kind: ctVector, vN: n, vElem: elem)

proc tPtr*(addressSpace: AddressSpace, elem: CirType): CirType =
  CirType(kind: ctPtr, addressSpace: addressSpace, pElem: elem)

proc tAptr*(addressSpace: AddressSpace, elem: CirType): CirType =
  CirType(kind: ctAptr, addressSpace: addressSpace, pElem: elem)

proc fld*(name: string, typ: CirType): CirField =
  CirField(name: name, typ: typ)

proc tStruct*(name: string, fields: varargs[CirField]): CirType =
  CirType(kind: ctStruct, name: name, fields: @fields)

proc tArray*(n: int, elem: CirType): CirType =
  CirType(kind: ctArray, aN: n, aElem: elem)

proc tSpan*(addressSpace: AddressSpace, elem: CirType): CirType =
  ## Runtime-length view over `elem` storage in `addressSpace`:
  ##
  ## - an `openArray[T]` parameter or the result of a `toOpenArray` call
  ## - the span split pass decomposes every span into an element pointer
  ##   plus an int32 length, no span survives to the printers
  CirType(kind: ctSpan, sSpace: addressSpace, sElem: elem)

proc tProcT*(params: varargs[CirType], ret: CirType): CirType =
  CirType(kind: ctProc, params: @params, ret: ret)

# Expression constructors

proc sym*(name: string, typ: CirType, builtin = false): CirExpr =
  CirExpr(kind: eSym, name: name, typ: typ, builtin: builtin)

proc intLit*(value: int64, typ: CirType): CirExpr =
  CirExpr(kind: eIntLit, value: value, litType: typ)

proc fltLit*(value: float64, typ: CirType): CirExpr =
  CirExpr(kind: eFloatLit, fValue: value, fLitType: typ)

proc tString*(): CirType =
  CirType(kind: ctString)

proc strLit*(value: string): CirExpr =
  CirExpr(kind: eStrLit, sValue: value)

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

proc dotE*(base: CirExpr, fieldName: string): CirExpr =
  CirExpr(kind: eDot, dotBase: base, fieldName: fieldName)

proc arithE*(opKind: CirExprKind, opType: CirType, left, right: CirExpr): CirExpr =
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

proc negE*(opType: CirType, operand: CirExpr): CirExpr =
  CirExpr(kind: eNeg, unType: opType, unOperand: operand)

proc cmpE*(opKind: CirExprKind, a, b: CirExpr): CirExpr =
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

proc ternaryE*(ternType: CirType, cond, thenE, elseE: CirExpr): CirExpr =
  ## A conditional value, `cond` selects between `thenE` and `elseE`.
  ## Branch values must already carry `ternType`.
  doAssert cond != nil and thenE != nil and elseE != nil,
    "ternaryE: a conditional value takes a condition and two branches"
  CirExpr(kind: eTernary, ternType: ternType, cond: cond, thenE: thenE, elseE: elseE)

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
  CirExpr(kind: eCall, callee: callee, args: @args)

# Statement constructors

proc varD*(name: string, typ: CirType, init: CirExpr = nil, addressSpace: AddressSpace = caThread): CirStmt =
  ## `caThreadgroup` is a `{.smem.}` local, `caConstant` a `{.const_mem.}` local.
  CirStmt(kind: sVar, name: name, declType: typ, init: init, addressSpace: addressSpace)

proc asgnS*(lhs, rhs: CirExpr): CirStmt =
  CirStmt(kind: sAsgn, lhs: lhs, rhs: rhs)

proc iteS*(cond: CirExpr, thenBody: seq[CirStmt], elseBody: seq[CirStmt] = @[]): CirStmt =
  CirStmt(kind: sIte, cond: cond, thenBody: thenBody, elseBody: elseBody)

proc whileS*(cond: CirExpr, whileBody: seq[CirStmt], unroll = false): CirStmt =
  ## A loop re-testing `cond` before every iteration.
  doAssert cond != nil, "whileS: a conditioned loop takes a condition"
  CirStmt(kind: sWhile, whileCond: cond, whileBody: whileBody, whileUnroll: unroll)

proc retS*(retVal: CirExpr = nil): CirStmt =
  CirStmt(kind: sRet, retVal: retVal)

proc discardS*(discardVal: CirExpr = nil): CirStmt =
  CirStmt(kind: sDiscard, discardVal: discardVal)

proc callS*(callCallee: CirExpr, callArgs: varargs[CirExpr]): CirStmt =
  CirStmt(kind: sCall, callCallee: callCallee, callArgs: @callArgs)

proc asmS*(asmParts: seq[string], asmOperands: seq[CirExpr]): CirStmt =
  CirStmt(kind: sAsm, asmParts: asmParts, asmOperands: asmOperands)

proc emitS*(emitParts: seq[string], emitOperands: seq[CirExpr]): CirStmt =
  CirStmt(kind: sEmit, emitParts: emitParts, emitOperands: emitOperands)

# Proc / module constructors

proc param*(name: string, typ: CirType): CirParam =
  CirParam(name: name, typ: typ)

proc procD*(name: string, params: seq[CirParam], ret: CirType, body: seq[CirStmt]): CirProc =
  CirProc(name: name, params: params, ret: ret, body: body)

proc moduleD*(name: string, types: seq[CirType], procs: seq[CirProc]): CirModule =
  CirModule(name: name, types: types, procs: procs)

# ═══════════════════════════════════════════════════════════════════════
#  Proc pragmas
# ═══════════════════════════════════════════════════════════════════════

proc applyProcPragmas*(fn: NimNode, p: CirProc): string =
  ## Args:
  ## - `p` the CirProc annotated in place
  ##
  ## Other pragmas are legal and dropped.
  ## Returns "" when clean, else the first refusal.
  for prag in fn.pragma:
    case prag.kind
    of nnkIdent, nnkSym:
      case prag.strVal
      of "global":
        p.isKernel = true
      of "inline":
        p.inline = true
      else:
        discard
    of nnkExprColonExpr, nnkCall, nnkCommand:
      if prag.len >= 2:
        case prag[0].strVal
        of "exportc":
          # a proc keeps its own name under an argumentless `exportc`
          p.exportName = prag[1].strVal
        of "workgroup":
          var dims: NimNode =
            if prag.kind == nnkExprColonExpr:
              prag[1]
            else:
              prag
          if dims.kind in {nnkPar, nnkTupleConstr, nnkBracket}:
            for d in 0 ..< 3:
              # a missing trailing dimension defaults to 1: workgroup: (4, 2) is the (4, 2, 1) launch shape
              let v = if d < dims.len: dims[d].intVal.int else: 1
              if v <= 0:
                return "workgroup dimension " & $(d + 1) & " is " & $v &
                  ", only positive dimensions are supported"
              p.workgroup[d] = v
          elif dims.kind in nnkIntLit .. nnkUInt64Lit:
            if dims.intVal.int <= 0:
              return "workgroup dimension 1 is " & $dims.intVal.int &
                ", only positive dimensions are supported"
            p.workgroup[0] = dims.intVal.int
            p.workgroup[1] = 1
            p.workgroup[2] = 1
        else:
          discard
    else:
      discard
  return ""

