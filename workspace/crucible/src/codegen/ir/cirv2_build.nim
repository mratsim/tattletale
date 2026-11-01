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
## - typed ops and ternaries get their type from the caller
## - comparisons derive their operand type from the first operand
## - statement calls go through `callS`
##
## A construct that the builders cannot express correctly has no node: build no such shape.

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

proc tPtr*(space: AddressSpace, elem: CirType): CirType =
  CirType(kind: ctPtr, space: space, pElem: elem)

proc tAptr*(space: AddressSpace, elem: CirType): CirType =
  CirType(kind: ctAptr, space: space, pElem: elem)

proc fld*(name: string, typ: CirType): CirField =
  CirField(name: name, typ: typ)

proc tStruct*(name: string, fields: varargs[CirField]): CirType =
  CirType(kind: ctStruct, name: name, fields: @fields)

proc tArray*(n: int, elem: CirType): CirType =
  CirType(kind: ctArray, aN: n, aElem: elem)

proc tSpan*(space: AddressSpace, elem: CirType): CirType =
  ## Runtime-length view over `elem` storage in `space`:
  ##
  ## - an `openArray[T]` parameter or the result of a `toOpenArray` call
  ## - the span split pass decomposes every span into an element pointer
  ##   plus an int32 length, no span survives to the printers
  CirType(kind: ctSpan, sSpace: space, sElem: elem)

proc tProcT*(params: varargs[CirType], ret: CirType): CirType =
  CirType(kind: ctProc, params: @params, ret: ret)

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

proc dotE*(base: CirExpr, fieldName: string): CirExpr =
  CirExpr(kind: eDot, dotBase: base, fieldName: fieldName)

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

proc ternaryE*(ternType: CirType, cond, thenE, elseE: CirExpr): CirExpr =
  ## A conditional value, `cond` selects between `thenE` and `elseE`.
  ##
  ## Args:
  ## - `ternType` types the node and both branches
  ## - printers emit the target's conditional-value form: a `?:` operator
  ##   where one exists, `select(elseE, thenE, cond)` on WGSL
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
  ## A call in value position: its result may sit in any operand slot
  ## (see the `eCall` contract in cirv2_lang). Statement-position calls use `callS`.
  CirExpr(kind: eCall, callee: callee, args: @args)

# Statement constructors

proc varD*(name: string, typ: CirType, init: CirExpr = nil, space: AddressSpace = caThread): CirStmt =
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

proc whileS*(cond: CirExpr, loopBody: seq[CirStmt], unroll = false): CirStmt =
  ## A loop re-testing `cond` before every iteration.
  ##
  ## Args:
  ## - `cond` is an ordinary expression, calls may appear in it
  doAssert cond != nil, "whileS: a conditioned loop takes a condition"
  CirStmt(kind: sLoop, loopCond: cond, loopBody: loopBody, unroll: unroll)

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

# ═══════════════════════════════════════════════════════════════════════
# Proc pragmas
# ═══════════════════════════════════════════════════════════════════════

proc applyProcPragmas*(fn: NimNode, p: CirProc): string =
  ## Args:
  ## - `p` the CirProc annotated in place
  ##
  ## `global` marks the compute entry point, `exportc` renames the emitted entry point,
  ## `workgroup(x, y, z)` sets the baked launch shape.
  ## `{.inline.}` marks an inline device function.
  ##
  ## Other pragmas are legal and dropped.
  ##
  ## Returns "" when clean, else the first refusal.
  ## A workgroup dimension of 0 or less is outside the supported set:
  ## every target emits it verbatim as an invalid launch shape.
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
          # `exportc: "entry"` renames the emitted entry point.
          # Bare `exportc` keeps the proc's own name.
          p.exportName = prag[1].strVal
        of "workgroup":
          # both call and colon forms: `workgroup(x, y, z)`, `workgroup: (x, y, z)`
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

