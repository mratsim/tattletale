# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/tables
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./passes_low_pointers
import ./pass_helpers

# ############################################################
#
#            OpenCL lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerOpenclHoistProgramScopeConstants*(reg: var PassRegistry) =
  reg.register("openclHoistProgramScopeConstants", pkTransform,
    "Relocates const_mem locals into the module's program scope",
    run = hoistConstMemLocals
  )

func registerOpenclLowerByrefParams*(reg: var PassRegistry) =
  reg.register("lowerByrefParams", pkTransform,
    "Rewrites passByRef params into const Type* ptr params",
    run = openclLowerByrefParams
  )

# ═══════════════════════════════════════════════════════════════════════
#  By-ref parameter lowering
# ═══════════════════════════════════════════════════════════════════════

proc addrWrapByrefArgs(pre: var seq[CirStmt], tmpCount: var int, e: CirExpr, byref: Table[string, seq[int]]) =
  ## Wraps a call argument in an address when the callee formal expects the struct
  ## as a hidden reference and the argument is still the struct value.
  ##
  ## A non-lvalue argument (a call result, a constructor) copies into a fresh local first, `&` of a temporary is invalid C.
  ## Hoisted declarations insert in front of the statement, nested calls hoist first, so a call node sees its arguments already rewritten.
  if e.isNil:
    return
  for ch in getExprChildren(e):
    addrWrapByrefArgs(pre, tmpCount, ch, byref)
  if e.kind != eCall:
    return
  for i in byref.getOrDefault(e.callee.name):
    if i >= e.args.len:
      break
    let at = getExprType(e.args[i])
    if at != nil and at.kind == ctStruct:
      let ptrTy = tPtr(caThread, at)
      if isLvalue(e.args[i]):
        e.args[i] = addrE(ptrTy, e.args[i])
      else:
        inc tmpCount
        let tmpName = "byref_tmp" & $tmpCount
        let tmp = sym(tmpName, at)
        pre.add(varD(tmpName, at))
        pre.add(asgnS(tmp, e.args[i]))
        e.args[i] = addrE(ptrTy, tmp)

proc addrWrapByrefStmts(stmts: var seq[CirStmt], tmpCount: var int, byref: Table[string, seq[int]])
  ## Forward declaration: nested statement lists recurse into each other.

proc addrWrapByrefStmt(pre: var seq[CirStmt], tmpCount: var int, s: CirStmt, byref: Table[string, seq[int]]) =
  case s.kind
  of sAsgn:
    addrWrapByrefArgs(pre, tmpCount, s.rhs, byref)
  of sIte:
    addrWrapByrefArgs(pre, tmpCount, s.cond, byref)
    addrWrapByrefStmts(s.thenBody, tmpCount, byref)
    addrWrapByrefStmts(s.elseBody, tmpCount, byref)
  of sWhile:
    addrWrapByrefArgs(pre, tmpCount, s.whileCond, byref)
    addrWrapByrefStmts(s.whileBody, tmpCount, byref)
  of sRet:
    addrWrapByrefArgs(pre, tmpCount, s.retVal, byref)
  of sDiscard:
    addrWrapByrefArgs(pre, tmpCount, s.discardVal, byref)
  of sCall:
    let callE = CirExpr(kind: eCall)
    callE.callee = s.callCallee
    callE.args = s.callArgs
    addrWrapByrefArgs(pre, tmpCount, callE, byref)
    s.callCallee = callE.callee
    s.callArgs = callE.args
  of sVar:
    addrWrapByrefArgs(pre, tmpCount, s.init, byref)
  of sAsm, sEmit:
    discard

proc addrWrapByrefStmts(stmts: var seq[CirStmt], tmpCount: var int, byref: Table[string, seq[int]]) =
  var i = 0
  while i < stmts.len:
    var pre: seq[CirStmt] = @[]
    addrWrapByrefStmt(pre, tmpCount, stmts[i], byref)
    for h in pre:
      stmts.insert(h, i)
      i.inc
    i.inc

proc openclLowerByrefParams(m: CirModule) =
  ## Large by-value struct formals pass as hidden `const T*` pointers.
  ##
  ## Contract:
  ## - the callee renames the formal to `_p_<name>`, its type stays the struct,
  ##   the printer emits `const T*` from `passByRef`
  ## - body reads deref the hidden pointer, `s.data[0]` becomes `(*_p_s).data[0]`
  ## - call sites pass addresses, a non-lvalue argument copies into a fresh local first
  ## - kernels keep by-value struct parameters, only device functions rewrite
  var byref = initTable[string, seq[int]]()
  for fn in m.procs:
    if fn.isGlobalFn:
      continue
    for i, p in fn.params.mpairs:
      if not p.passByRef:
        continue
      let oldName = p.name
      p.name = "_p_" & oldName
      let ptrSym = sym(p.name, tPtr(caThread, p.typ))
      fn.body.walkStmts(nil, proc(e: CirExpr) =
        if e.kind == eSym and e.name == oldName:
          # Nim 2 case objects forbid in-place branch switching
          e[] = CirExpr(kind: eDeref, derefOperand: ptrSym)[])
      byref.mgetOrPut(fn.name, @[]).add(i)
  for fn in m.procs:
    var tmpCount = 0
    addrWrapByrefStmts(fn.body, tmpCount, byref)
