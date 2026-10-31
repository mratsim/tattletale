# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Cirv2 debug renderer for tests and pass development:
## one node per line, s-expression shaped, in the Leng vocabulary
## (`(add Type a b)`, `(i +32)`, `aptr`, `ite`, `lab`/`jmp`).

import cirv2_lang
import std/strutils

type
  RenderBuf* = object
    lines*: seq[string]
    indent*: int

proc add(b: var RenderBuf, text: string) =
  b.lines.add(spaces(b.indent) & text)

proc open(b: var RenderBuf, text: string) =
  b.add(text)
  b.indent.inc

proc close(b: var RenderBuf) =
  b.add(")")
  b.indent.dec

proc renderStmt(b: var RenderBuf, s: CirStmt)

proc getSpaceName(s: CirAddrSpace): string =
  case s
  of caThread: "thread"
  of caThreadgroup: "threadgroup"
  of caConstant: "constant"
  of caDevice: "device"

proc renderType(t: CirType): string =
  ## Leng rendering of one type: `(i +32)`, `(aptr device (f +32))`, `:Name`.
  if t.isNil:
    return "<no-type>"
  case t.kind
  of ctVoid: result = "(void)"
  of ctBool: result = "(bool)"
  of ctInt: result = "(i " & (if t.bits < 0: "-" else: "+") & $abs(t.bits) & ")"
  of ctUint: result = "(u " & (if t.bits < 0: "-" else: "+") & $abs(t.bits) & ")"
  of ctFloat: result = "(f " & (if t.bits < 0: "-" else: "+") & $abs(t.bits) & ")"
  of ctVector: result = "(vector " & $t.vN & " " & renderType(t.vElem) & ")"
  of ctPtr: result = "(ptr " & getSpaceName(t.space) & " " & renderType(t.pElem) & ")"
  of ctAptr: result = "(aptr " & getSpaceName(t.space) & " " & renderType(t.pElem) & ")"
  of ctStruct: result = ":" & t.name
  of ctArray: result = "(array " & renderType(t.aElem) & " " & $t.aN & ")"
  of ctSpan: result = "(span " & getSpaceName(t.sSpace) & " " & renderType(t.sElem) & ")"
  of ctProc:
    result = "(proctype (params"
    for p in t.params:
      result.add(" " & renderType(p))
    result.add(") " & renderType(t.ret) & ")")

proc renderExpr(e: CirExpr): string =
  ## Leng rendering of one expression.
  if e.isNil:
    return "<no-expr>"
  case e.kind
  of eNil: result = "(nil)"
  of eTrue: result = "(true)"
  of eFalse: result = "(false)"
  of eSym: result = e.name
  of eIntLit: result = $e.value
  of eFloatLit: result = $e.fValue
  of eDeref: result = "(deref " & renderExpr(e.derefOperand) & ")"
  of eAt: result = "(at " & renderExpr(e.atBase) & " " & renderExpr(e.atIndex) & ")"
  of ePat: result = "(pat " & renderExpr(e.patBase) & " " & renderExpr(e.patIndex) & ")"
  of eDot:
    result = "(dot " & renderExpr(e.dotBase) & " " & e.fieldName & " " & $e.inherit & ")"
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result = "(" & toLowerAscii($e.kind)[1 ..^ 1] & " " & renderType(e.opType) &
      " " & renderExpr(e.left) & " " & renderExpr(e.right) & ")"
  of eBitNot, eNeg:
    result = "(" & toLowerAscii($e.kind)[1 ..^ 1] & " " & renderType(e.unType) &
      " " & renderExpr(e.unOperand) & ")"
  of eEq, eNeq, eLt, eLe:
    result = "(" & toLowerAscii($e.kind)[1 ..^ 1] & " " & renderType(e.cmpType) &
      " " & renderExpr(e.a) & " " & renderExpr(e.b) & ")"
  of eNot: result = "(not " & renderExpr(e.notOperand) & ")"
  of eAnd: result = "(and " & renderExpr(e.andLeft) & " " & renderExpr(e.andRight) & ")"
  of eOr: result = "(or " & renderExpr(e.andLeft) & " " & renderExpr(e.andRight) & ")"
  of eCast: result = "(cast " & renderType(e.castType) & " " & renderExpr(e.castOperand) & ")"
  of eConv: result = "(conv " & renderType(e.castType) & " " & renderExpr(e.castOperand) & ")"
  of eAddr: result = "(addr " & renderExpr(e.addrOperand) & ")"
  of eSizeof: result = "(sizeof " & renderType(e.ofType) & ")"
  of eOconstr:
    result = "(oconstr " & renderType(e.ocType)
    for kvField in e.kvFields:
      result.add(" (kv " & kvField.name & " " & renderExpr(kvField.value) & ")")
    result.add(")")
  of eAconstr:
    result = "(aconstr " & renderType(e.acType)
    for el in e.elems:
      result.add(" " & renderExpr(el))
    result.add(")")
  of eCall:
    result = "(call " & renderExpr(e.callee)
    for arg in e.args:
      result.add(" " & renderExpr(arg))
    result.add(")")

proc renderStmtList*(b: var RenderBuf, body: seq[CirStmt], header: string) =
  ## A branch body list under `header`, `.` when empty.
  if body.len == 0:
    b.add(header & " .")
  else:
    b.open(header)
    for s in body:
      b.renderStmt(s)
    b.close

proc renderRange(r: CirRange): string =
  if r.lo == r.hi:
    result = $r.lo
  else:
    result = "(range " & $r.lo & " " & $r.hi & ")"

proc renderStmt(b: var RenderBuf, s: CirStmt) =
  ## Compound nodes indent their children, branch bodies print under their header.
  if s.isNil:
    b.add("<no-stmt>")
    return
  case s.kind
  of sVar:
    # register default stays unannotated, a placed local shows its class
    let spc = if s.space == caThread: ""
              else: " " & toLowerAscii($s.space)[2 ..^ 1]
    if s.init != nil:
      b.add("(var" & spc & " " & s.name & " " & renderType(s.declType) & " " &
        renderExpr(s.init) & ")")
    else:
      b.add("(var" & spc & " " & s.name & " " & renderType(s.declType) & " .)")
  of sConst:
    b.add("(const " & s.name & " " & renderType(s.declType) & " " &
      renderExpr(s.init) & ")")
  of sAsgn:
    b.add("(asgn " & renderExpr(s.lhs) & " " & renderExpr(s.rhs) & ")")
  of sIte:
    b.open("(ite " & renderExpr(s.cond))
    b.renderStmtList(s.thenBody, "(then")
    if s.elseBody.len > 0:
      b.renderStmtList(s.elseBody, "(else")
    b.close
  of sLoop:
    b.open("(loop")
    for inner in s.loopBody:
      b.renderStmt(inner)
    b.close
  of sCase:
    b.open("(case " & renderExpr(s.selector))
    for br in s.branches:
      var rangesText = "(ranges"
      for r in br.ranges:
        rangesText.add(" " & renderRange(r))
      rangesText.add(")")
      b.open("(of " & rangesText)
      for inner in br.body:
        b.renderStmt(inner)
      b.close
    if s.caseElse.len > 0:
      b.open("(else")
      for inner in s.caseElse:
        b.renderStmt(inner)
      b.close
    b.close
  of sLab:
    b.add("(lab " & s.labName & ")")
  of sJmp:
    b.add("(jmp " & s.jmpTarget & ")")
  of sScope:
    b.open("(scope")
    for inner in s.scopeBody:
      b.renderStmt(inner)
    b.close
  of sAssume:
    b.add("(assume " & renderExpr(s.assumeCond) & ")")
  of sRet:
    if s.retVal != nil:
      b.add("(ret " & renderExpr(s.retVal) & ")")
    else:
      b.add("(ret .)")
  of sDiscard:
    if s.discardVal != nil:
      b.add("(discard " & renderExpr(s.discardVal) & ")")
    else:
      b.add("(discard .)")
  of sCall:
    var text = "(call " & renderExpr(s.callCallee)
    for arg in s.callArgs:
      text.add(" " & renderExpr(arg))
    b.add(text & ")")
  of sAsm:
    var text = "(asm"
    for i, part in s.asmParts:
      text.add(" \"" & part & "\"")
      if i < s.asmOperands.len:
        text.add(" " & renderExpr(s.asmOperands[i]))
    b.add(text & ")")
  of sEmit:
    var text = "(emit"
    for i, part in s.emitParts:
      text.add(" \"" & part & "\"")
      if i < s.emitOperands.len:
        text.add(" " & renderExpr(s.emitOperands[i]))
    b.add(text & ")")

proc renderProc*(fn: CirProc): string =
  ## Leng rendering of one proc.
  var b = RenderBuf()
  var header = "(proc " & fn.name
  for p in fn.params:
    header.add(" (param " & p.name & " " & renderType(p.typ) & ")")
  header.add(" " & renderType(fn.ret))
  b.open(header)
  for s in fn.body:
    b.renderStmt(s)
  b.close
  result = b.lines.join("\n") & "\n"

proc renderModule*(m: CirModule): string =
  ## Renders a whole module: nominal types, then the program-scope globals
  ## when any were hoisted, then procs.
  result = "(module " & m.name & ")\n"
  for t in m.types:
    result.add("(type " & renderType(t) & ")\n")
  if m.globals.len > 0:
    var b = RenderBuf()
    b.renderStmtList(m.globals, "(globals")
    result.add(b.lines.join("\n") & "\n")
  for fn in m.procs:
    result.add(renderProc(fn))