# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Binary round-trip for a `CirModule`: `serializeCir` writes a binary payload and `deserializeCir`
## rebuilds the module.
##
## Invariants:
## - shared `CirType`/`CirExpr`/`CirStmt` subgraphs stay shared across the round trip
## - cyclic structures terminate: the first visit writes the node fields,
##   every later visit writes a backreference
## - `features`, `callGraph` and `refBlocks` are pass outputs. The serializer skips them:
##   a frontend module never sets them and the rebuilt module leaves them empty
## - the walk is linear in the module size and runs in the Nim VM and at runtime

import std/tables
import cirv2_lang

# ═══════════════════════════════════════════════════════════════════════
#  Writer
# ═══════════════════════════════════════════════════════════════════════

type
  CirWriter = object
    buf: string
    seenTypes: Table[pointer, int]
    seenExprs: Table[pointer, int]
    seenStmts: Table[pointer, int]

proc putU8(w: var CirWriter, x: uint8) {.inline.} =
  w.buf.add char(x)

proc putU32(w: var CirWriter, x: uint32) {.inline.} =
  for i in 0 ..< 4:
    w.buf.add char((x shr (i * 8)) and 0xFF)

proc putI64(w: var CirWriter, x: int64) {.inline.} =
  let u = cast[uint64](x)
  for i in 0 ..< 8:
    w.buf.add char((u shr (i * 8)) and 0xFF)

proc putF64(w: var CirWriter, x: float64) {.inline.} =
  w.putI64(cast[int64](x))

proc putBool(w: var CirWriter, x: bool) {.inline.} =
  w.putU8(uint8(x))

proc putStr(w: var CirWriter, s: string) {.inline.} =
  w.putU32(uint32(s.len))
  w.buf.add s

# putCirExpr/putCirStmt recurse into each other: eStmts contains statements and statements contain expressions
proc putCirExpr(w: var CirWriter, e: CirExpr)
proc putCirStmt(w: var CirWriter, s: CirStmt)

template putRef(w: var CirWriter, seen: var Table[pointer, int], node: untyped, body: untyped) =
  ## Reference cell: 0 = nil, 1 = first occurrence with its fields, n >= 2 = backreference to the n - 2 th node of this kind.
  if node.isNil():
    w.putU32(0)
  else:
    let key = cast[pointer](node)
    let idx = seen.getOrDefault(key, -1)
    if idx < 0:
      seen[key] = seen.len
      w.putU32(1)
      body
    else:
      w.putU32(uint32(idx) + 2)

proc putPartsOperands(w: var CirWriter, parts: seq[string], operands: seq[CirExpr]) =
  ## Shared body of `sAsm` and `sEmit`: verbatim text pieces and spliced operands.
  w.putU32(uint32(parts.len))
  for p in parts:
    w.putStr(p)
  w.putU32(uint32(operands.len))
  for o in operands:
    w.putCirExpr(o)

proc putCirType(w: var CirWriter, t: CirType) =
  w.putRef(w.seenTypes, t):
    w.putU8(uint8(t.kind))
    case t.kind
    of ctVoid, ctBool, ctString:
      discard
    of ctInt, ctUint, ctFloat, ctBf16:
      w.putI64(t.bits)
    of ctVector:
      w.putI64(t.vN)
      w.putCirType(t.vElem)
    of ctPtr, ctAptr:
      w.putU8(uint8(t.addressSpace))
      w.putCirType(t.pElem)
    of ctStruct:
      w.putStr(t.name)
      w.putU32(uint32(t.fields.len))
      for f in t.fields:
        w.putStr(f.name)
        w.putCirType(f.typ)
    of ctArray:
      w.putI64(t.aN)
      w.putCirType(t.aElem)
    of ctSpan:
      w.putU8(uint8(t.sSpace))
      w.putCirType(t.sElem)
    of ctProc:
      w.putU32(uint32(t.params.len))
      for p in t.params:
        w.putCirType(p)
      w.putCirType(t.ret)

proc putCirExpr(w: var CirWriter, e: CirExpr) =
  w.putRef(w.seenExprs, e):
    w.putU8(uint8(e.kind))
    case e.kind
    of eNil:
      w.putCirType(e.nilType)
    of eTrue, eFalse:
      discard
    of eSym:
      w.putStr(e.name)
      w.putCirType(e.typ)
      w.putBool(e.builtin)
      w.putU8(uint8(e.coordBuiltin))
      w.putU8(uint8(e.synchroBuiltin))
      w.putU8(uint8(e.reductionBuiltin))
      w.putU8(uint8(e.shuffleBuiltin))
      w.putU8(uint8(e.atomicsBuiltin))
    of eIntLit:
      w.putI64(e.value)
      w.putCirType(e.litType)
    of eFloatLit:
      w.putF64(e.fValue)
      w.putCirType(e.fLitType)
    of eStrLit:
      w.putStr(e.sValue)
    of eDeref:
      w.putCirExpr(e.derefOperand)
    of eAt:
      w.putCirExpr(e.atBase)
      w.putCirExpr(e.atIndex)
    of ePat:
      w.putCirExpr(e.patBase)
      w.putCirExpr(e.patIndex)
    of eDot:
      w.putCirExpr(e.dotBase)
      w.putStr(e.fieldName)
    of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
      w.putCirType(e.opType)
      w.putCirExpr(e.left)
      w.putCirExpr(e.right)
    of eNeg:
      w.putCirType(e.unType)
      w.putCirExpr(e.unOperand)
    of eEq, eNeq, eLt, eLe:
      w.putCirType(e.cmpType)
      w.putCirExpr(e.a)
      w.putCirExpr(e.b)
    of eNot:
      w.putCirExpr(e.notOperand)
    of eAnd, eOr:
      w.putCirExpr(e.andLeft)
      w.putCirExpr(e.andRight)
    of eTernary:
      w.putCirType(e.ternType)
      w.putCirExpr(e.cond)
      w.putCirExpr(e.thenE)
      w.putCirExpr(e.elseE)
    of eCast, eConv:
      w.putCirType(e.castType)
      w.putCirExpr(e.castOperand)
    of eAddr:
      w.putCirType(e.addrType)
      w.putCirExpr(e.addrOperand)
    of eSizeof:
      w.putCirType(e.ofType)
    of eOconstr:
      w.putCirType(e.ocType)
      w.putU32(uint32(e.kvFields.len))
      for kv in e.kvFields:
        w.putStr(kv.name)
        w.putCirExpr(kv.value)
    of eAconstr:
      w.putCirType(e.acType)
      w.putU32(uint32(e.elems.len))
      for x in e.elems:
        w.putCirExpr(x)
    of eCall:
      w.putCirExpr(e.callee)
      w.putU32(uint32(e.args.len))
      for a in e.args:
        w.putCirExpr(a)
    of eStmts:
      w.putU32(uint32(e.stmts.len))
      for s in e.stmts:
        w.putCirStmt(s)
      w.putCirExpr(e.valueExpr)

proc putCirStmt(w: var CirWriter, s: CirStmt) =
  w.putRef(w.seenStmts, s):
    w.putU8(uint8(s.kind))
    case s.kind
    of sVar:
      w.putStr(s.name)
      w.putCirType(s.declType)
      w.putCirExpr(s.init)
      w.putU8(uint8(s.addressSpace))
    of sAsgn:
      w.putCirExpr(s.lhs)
      w.putCirExpr(s.rhs)
    of sIte:
      w.putCirExpr(s.cond)
      w.putU32(uint32(s.thenBody.len))
      for b in s.thenBody:
        w.putCirStmt(b)
      w.putU32(uint32(s.elseBody.len))
      for b in s.elseBody:
        w.putCirStmt(b)
    of sWhile:
      w.putCirExpr(s.whileCond)
      w.putU32(uint32(s.whileBody.len))
      for b in s.whileBody:
        w.putCirStmt(b)
      w.putBool(s.whileUnroll)
    of sRet:
      w.putCirExpr(s.retVal)
    of sDiscard:
      w.putCirExpr(s.discardVal)
    of sCall:
      w.putCirExpr(s.callCallee)
      w.putU32(uint32(s.callArgs.len))
      for a in s.callArgs:
        w.putCirExpr(a)
    of sAsm:
      w.putPartsOperands(s.asmParts, s.asmOperands)
    of sEmit:
      w.putPartsOperands(s.emitParts, s.emitOperands)

proc putCirParam(w: var CirWriter, p: CirParam) =
  w.putStr(p.name)
  w.putCirType(p.typ)
  w.putBool(p.varParam)
  w.putBool(p.passByRef)

proc putCirProc(w: var CirWriter, p: CirProc) =
  ## Procs are unique per module, no backreference cell.
  w.putStr(p.name)
  w.putU32(uint32(p.params.len))
  for param in p.params:
    w.putCirParam(param)
  w.putCirType(p.ret)
  w.putU32(uint32(p.body.len))
  for s in p.body:
    w.putCirStmt(s)
  w.putBool(p.isGlobalFn)
  w.putBool(p.inline)
  w.putStr(p.exportName)
  for d in p.workgroup:
    w.putI64(d)

# ═══════════════════════════════════════════════════════════════════════
#  Reader
# ═══════════════════════════════════════════════════════════════════════

type
  CirReader = object
    buf: string
    pos: int
    types: Table[int, CirType]
    exprs: Table[int, CirExpr]
    stmts: Table[int, CirStmt]

# getCirExpr/getCirStmt recurse into each other, forward declared
proc getCirExpr(r: var CirReader): CirExpr
proc getCirStmt(r: var CirReader): CirStmt

proc getU8(r: var CirReader): uint8 {.inline.} =
  result = uint8(r.buf[r.pos])
  inc r.pos

proc getU32(r: var CirReader): uint32 {.inline.} =
  for i in 0 ..< 4:
    result = result or (uint32(r.buf[r.pos]) shl (i * 8))
    inc r.pos

proc getI64(r: var CirReader): int64 {.inline.} =
  var u: uint64
  for i in 0 ..< 8:
    u = u or (uint64(r.buf[r.pos]) shl (i * 8))
    inc r.pos
  result = cast[int64](u)

proc getF64(r: var CirReader): float64 {.inline.} =
  cast[float64](r.getI64())

proc getBool(r: var CirReader): bool {.inline.} =
  r.getU8() != 0

proc getStr(r: var CirReader): string {.inline.} =
  let len = int(r.getU32())
  result = newString(len)
  for i in 0 ..< len:
    result[i] = r.buf[r.pos + i]
  inc r.pos, len

template getRef[T](r: var CirReader, seen: var Table[int, T], node: var T, readKind: untyped, body: untyped) =
  ## Reference cell: 0 = nil, 1 = first occurrence with its fields, n >= 2 = backreference to the n - 2 th node of this kind.
  ##
  ## `readKind` reads the node discriminator from the payload byte that comes after the tag.
  ## The reader constructs the object with that branch at once.
  ## Assigning the discriminator of a live case object raises FieldDefect.
  let tag = r.getU32()
  if tag == 0:
    node = nil
  elif tag == 1:
    node = T(kind: readKind)
    seen[seen.len] = node
    body
  else:
    node = seen[int(tag) - 2]

proc getPartsOperands(r: var CirReader): tuple[parts: seq[string], operands: seq[CirExpr]] =
  ## Shared body of `sAsm` and `sEmit`: verbatim text pieces and spliced operands.
  let n = int(r.getU32())
  result.parts = newSeq[string](n)
  for i in 0 ..< n:
    result.parts[i] = r.getStr()
  let m = int(r.getU32())
  result.operands = newSeq[CirExpr](m)
  for i in 0 ..< m:
    result.operands[i] = r.getCirExpr()

proc getCirType(r: var CirReader): CirType =
  r.getRef(r.types, result, CirTypeKind(r.getU8())):
    case result.kind
    of ctVoid, ctBool, ctString:
      discard
    of ctInt, ctUint, ctFloat, ctBf16:
      result.bits = r.getI64()
    of ctVector:
      result.vN = r.getI64()
      result.vElem = r.getCirType()
    of ctPtr, ctAptr:
      result.addressSpace = AddressSpace(r.getU8())
      result.pElem = r.getCirType()
    of ctStruct:
      result.name = r.getStr()
      let n = int(r.getU32())
      result.fields = newSeq[CirField](n)
      for i in 0 ..< n:
        result.fields[i].name = r.getStr()
        result.fields[i].typ = r.getCirType()
    of ctArray:
      result.aN = r.getI64()
      result.aElem = r.getCirType()
    of ctSpan:
      result.sSpace = AddressSpace(r.getU8())
      result.sElem = r.getCirType()
    of ctProc:
      let n = int(r.getU32())
      result.params = newSeq[CirType](n)
      for i in 0 ..< n:
        result.params[i] = r.getCirType()
      result.ret = r.getCirType()

proc getCirExpr(r: var CirReader): CirExpr =
  r.getRef(r.exprs, result, CirExprKind(r.getU8())):
    case result.kind
    of eNil:
      result.nilType = r.getCirType()
    of eTrue, eFalse:
      discard
    of eSym:
      result.name = r.getStr()
      result.typ = r.getCirType()
      result.builtin = r.getBool()
      result.coordBuiltin = CirCoordBuiltinKind(r.getU8())
      result.synchroBuiltin = CirSynchroBuiltinKind(r.getU8())
      result.reductionBuiltin = CirReductionBuiltinKind(r.getU8())
      result.shuffleBuiltin = CirShuffleBuiltinKind(r.getU8())
      result.atomicsBuiltin = CirAtomicsBuiltinKind(r.getU8())
    of eIntLit:
      result.value = r.getI64()
      result.litType = r.getCirType()
    of eFloatLit:
      result.fValue = r.getF64()
      result.fLitType = r.getCirType()
    of eStrLit:
      result.sValue = r.getStr()
    of eDeref:
      result.derefOperand = r.getCirExpr()
    of eAt:
      result.atBase = r.getCirExpr()
      result.atIndex = r.getCirExpr()
    of ePat:
      result.patBase = r.getCirExpr()
      result.patIndex = r.getCirExpr()
    of eDot:
      result.dotBase = r.getCirExpr()
      result.fieldName = r.getStr()
    of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
      result.opType = r.getCirType()
      result.left = r.getCirExpr()
      result.right = r.getCirExpr()
    of eNeg:
      result.unType = r.getCirType()
      result.unOperand = r.getCirExpr()
    of eEq, eNeq, eLt, eLe:
      result.cmpType = r.getCirType()
      result.a = r.getCirExpr()
      result.b = r.getCirExpr()
    of eNot:
      result.notOperand = r.getCirExpr()
    of eAnd, eOr:
      result.andLeft = r.getCirExpr()
      result.andRight = r.getCirExpr()
    of eTernary:
      result.ternType = r.getCirType()
      result.cond = r.getCirExpr()
      result.thenE = r.getCirExpr()
      result.elseE = r.getCirExpr()
    of eCast, eConv:
      result.castType = r.getCirType()
      result.castOperand = r.getCirExpr()
    of eAddr:
      result.addrType = r.getCirType()
      result.addrOperand = r.getCirExpr()
    of eSizeof:
      result.ofType = r.getCirType()
    of eOconstr:
      result.ocType = r.getCirType()
      let n = int(r.getU32())
      result.kvFields = newSeq[CirKv](n)
      for i in 0 ..< n:
        result.kvFields[i].name = r.getStr()
        result.kvFields[i].value = r.getCirExpr()
    of eAconstr:
      result.acType = r.getCirType()
      let n = int(r.getU32())
      result.elems = newSeq[CirExpr](n)
      for i in 0 ..< n:
        result.elems[i] = r.getCirExpr()
    of eCall:
      result.callee = r.getCirExpr()
      let n = int(r.getU32())
      result.args = newSeq[CirExpr](n)
      for i in 0 ..< n:
        result.args[i] = r.getCirExpr()
    of eStmts:
      let n = int(r.getU32())
      result.stmts = newSeq[CirStmt](n)
      for i in 0 ..< n:
        result.stmts[i] = r.getCirStmt()
      result.valueExpr = r.getCirExpr()

proc getCirStmt(r: var CirReader): CirStmt =
  r.getRef(r.stmts, result, CirStmtKind(r.getU8())):
    case result.kind
    of sVar:
      result.name = r.getStr()
      result.declType = r.getCirType()
      result.init = r.getCirExpr()
      result.addressSpace = AddressSpace(r.getU8())
    of sAsgn:
      result.lhs = r.getCirExpr()
      result.rhs = r.getCirExpr()
    of sIte:
      result.cond = r.getCirExpr()
      let n = int(r.getU32())
      result.thenBody = newSeq[CirStmt](n)
      for i in 0 ..< n:
        result.thenBody[i] = r.getCirStmt()
      let m = int(r.getU32())
      result.elseBody = newSeq[CirStmt](m)
      for i in 0 ..< m:
        result.elseBody[i] = r.getCirStmt()
    of sWhile:
      result.whileCond = r.getCirExpr()
      let n = int(r.getU32())
      result.whileBody = newSeq[CirStmt](n)
      for i in 0 ..< n:
        result.whileBody[i] = r.getCirStmt()
      result.whileUnroll = r.getBool()
    of sRet:
      result.retVal = r.getCirExpr()
    of sDiscard:
      result.discardVal = r.getCirExpr()
    of sCall:
      result.callCallee = r.getCirExpr()
      let n = int(r.getU32())
      result.callArgs = newSeq[CirExpr](n)
      for i in 0 ..< n:
        result.callArgs[i] = r.getCirExpr()
    of sAsm:
      let (parts, operands) = r.getPartsOperands()
      result.asmParts = parts
      result.asmOperands = operands
    of sEmit:
      let (parts, operands) = r.getPartsOperands()
      result.emitParts = parts
      result.emitOperands = operands

proc getCirParam(r: var CirReader): CirParam =
  result.name = r.getStr()
  result.typ = r.getCirType()
  result.varParam = r.getBool()
  result.passByRef = r.getBool()

proc getCirProc(r: var CirReader): CirProc =
  result = CirProc()
  result.name = r.getStr()
  let n = int(r.getU32())
  result.params = newSeq[CirParam](n)
  for i in 0 ..< n:
    result.params[i] = r.getCirParam()
  result.ret = r.getCirType()
  let b = int(r.getU32())
  result.body = newSeq[CirStmt](b)
  for i in 0 ..< b:
    result.body[i] = r.getCirStmt()
  result.isGlobalFn = r.getBool()
  result.inline = r.getBool()
  result.exportName = r.getStr()
  for i in 0 ..< 3:
    result.workgroup[i] = r.getI64()

# ═══════════════════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════════════════

proc serializeCir*(m: CirModule): string =
  ## Serializes a module to the binary payload that `deserializeCir` reads back.
  var w = CirWriter()
  w.putStr(m.name)
  w.putU32(uint32(m.types.len))
  for t in m.types:
    w.putCirType(t)
  w.putU32(uint32(m.procs.len))
  for p in m.procs:
    w.putCirProc(p)
  w.putU32(uint32(m.globals.len))
  for g in m.globals:
    w.putCirStmt(g)
  w.putBool(m.verifyIR)
  w.buf

proc deserializeCir*(payload: string): CirModule =
  ## Rebuilds the module `serializeCir` wrote.
  var r = CirReader(buf: payload)
  result = CirModule()
  result.name = r.getStr()
  let nTypes = int(r.getU32())
  result.types = newSeq[CirType](nTypes)
  for i in 0 ..< nTypes:
    result.types[i] = r.getCirType()
  let nProcs = int(r.getU32())
  result.procs = newSeq[CirProc](nProcs)
  for i in 0 ..< nProcs:
    result.procs[i] = r.getCirProc()
  let nGlobals = int(r.getU32())
  result.globals = newSeq[CirStmt](nGlobals)
  for i in 0 ..< nGlobals:
    result.globals[i] = r.getCirStmt()
  result.verifyIR = r.getBool()
