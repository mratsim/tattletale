# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared pass helpers over CIRv2.
##
## - call-target collection
## - the root symbol of a dotted or indexed chain
## - declaration and clone utilities
## - the kernel-parameter leaf walk
##
## Every pass file may import these, no pass owns them.

import std/sets
import ../ir/cirv2_lang
import ../ir/cirv2_build

proc exprRootSym*(e: CirExpr): CirExpr =
  ## Symbol at the root of a dotted or indexed chain, nil otherwise.
  result = e
  while result != nil and result.kind in {eDot, ePat, eAt}:
    result = case result.kind
      of eDot: result.dotBase
      of ePat: result.patBase
      else: result.atBase
  if result != nil and result.kind != eSym:
    result = nil

proc appendCallTargetsExpr*(e: CirExpr, targets: var HashSet[string]) =
  ## Collects the callee names of one expression into `targets`.
  if e.isNil:
    return
  if e.kind == eCall and e.callee.kind == eSym:
    targets.incl e.callee.name
  for ch in getExprChildren(e):
    appendCallTargetsExpr(ch, targets)

proc appendCallTargetsStmts*(ss: seq[CirStmt], targets: var HashSet[string]) =
  ## Collects the callee names of one statement list, nested bodies included.
  for s in ss:
    case s.kind
    of sVar, sConst:
      appendCallTargetsExpr(s.init, targets)
    of sAsgn:
      appendCallTargetsExpr(s.lhs, targets)
      appendCallTargetsExpr(s.rhs, targets)
    of sIte:
      appendCallTargetsExpr(s.cond, targets)
      appendCallTargetsStmts(s.thenBody, targets)
      appendCallTargetsStmts(s.elseBody, targets)
    of sLoop:
      appendCallTargetsStmts(s.loopBody, targets)
    of sScope:
      appendCallTargetsStmts(s.scopeBody, targets)
    of sRet:
      appendCallTargetsExpr(s.retVal, targets)
    of sCall:
      if s.callCallee.kind == eSym:
        targets.incl s.callCallee.name
      for a in s.callArgs:
        appendCallTargetsExpr(a, targets)
    else:
      discard


proc collectCallTargets*(stmts: seq[CirStmt], targets: var HashSet[string]) =
  ## Collects every callee name appearing in the statement list.
  ## Nested bodies are included.
  appendCallTargetsStmts(stmts, targets)

proc collectDeclaredLocals*(stmts: seq[CirStmt], locals: var seq[tuple[name: string, typ: CirType]]) =
  ## Every local declared in the statement list with its type, nested bodies included.
  for s in stmts:
    case s.kind
    of sVar, sConst:
      locals.add((name: s.name, typ: s.declType))
    of sIte:
      collectDeclaredLocals(s.thenBody, locals)
      collectDeclaredLocals(s.elseBody, locals)
    of sLoop:
      collectDeclaredLocals(s.loopBody, locals)
    of sScope:
      collectDeclaredLocals(s.scopeBody, locals)
    else:
      discard

proc cloneExpr*(e: CirExpr): CirExpr =
  ## Deep copy of one expression: statements and expressions are ref objects,
  ## a splice renames its copy in place, the callee's own body must not move.
  if e.isNil:
    return nil
  result = CirExpr(kind: e.kind)
  case e.kind
  of eSym:
    result.name = e.name
    result.typ = e.typ
    result.builtin = e.builtin
    result.coordBuiltin = e.coordBuiltin
    result.synchroBuiltin = e.synchroBuiltin
    result.reductionBuiltin = e.reductionBuiltin
    result.atomicsBuiltin = e.atomicsBuiltin
  of eIntLit:
    result.value = e.value
    result.litType = e.litType
  of eFloatLit:
    result.fValue = e.fValue
    result.fLitType = e.fLitType
  of eNil:
    result.nilType = e.nilType
  of eTrue, eFalse:
    discard
  of eAt:
    result.atBase = cloneExpr(e.atBase)
    result.atIndex = cloneExpr(e.atIndex)
  of ePat:
    result.patBase = cloneExpr(e.patBase)
    result.patIndex = cloneExpr(e.patIndex)
  of eDot:
    result.dotBase = cloneExpr(e.dotBase)
    result.fieldName = e.fieldName
    result.inherit = e.inherit
  of eDeref:
    result.derefOperand = cloneExpr(e.derefOperand)
  of eAddr:
    result.addrType = e.addrType
    result.addrOperand = cloneExpr(e.addrOperand)
  of eAdd, eSub, eMul, eDiv, eMod, eShl, eShr, eBitAnd, eBitOr, eBitXor:
    result.opType = e.opType
    result.left = cloneExpr(e.left)
    result.right = cloneExpr(e.right)
  of eEq, eNeq, eLt, eLe:
    result.cmpType = e.cmpType
    result.a = cloneExpr(e.a)
    result.b = cloneExpr(e.b)
  of eAnd, eOr:
    result.andLeft = cloneExpr(e.andLeft)
    result.andRight = cloneExpr(e.andRight)
  of eNot:
    result.notOperand = cloneExpr(e.notOperand)
  of eNeg, eBitNot:
    result.unOperand = cloneExpr(e.unOperand)
  of eCast, eConv:
    result.castType = e.castType
    result.castOperand = cloneExpr(e.castOperand)
  of eSizeof:
    result.ofType = e.ofType
  of eOconstr:
    result.ocType = e.ocType
    for f in e.kvFields:
      result.kvFields.add kv(f.name, cloneExpr(f.value))
  of eAconstr:
    result.acType = e.acType
    for el in e.elems:
      result.elems.add cloneExpr(el)
  of eCall:
    result.callee = cloneExpr(e.callee)
    for a in e.args:
      result.args.add cloneExpr(a)

proc cloneStmts*(stmts: seq[CirStmt]): seq[CirStmt] =
  ## Deep copy of a statement list.
  for s in stmts:
    var c = CirStmt(kind: s.kind)
    case s.kind
    of sVar:
      c.name = s.name
      c.declType = s.declType
      c.init = cloneExpr(s.init)
      c.space = s.space
    of sConst:
      c.name = s.name
      c.declType = s.declType
      c.init = cloneExpr(s.init)
    of sAsgn:
      c.lhs = cloneExpr(s.lhs)
      c.rhs = cloneExpr(s.rhs)
    of sIte:
      c.cond = cloneExpr(s.cond)
      c.thenBody = cloneStmts(s.thenBody)
      c.elseBody = cloneStmts(s.elseBody)
    of sLoop:
      c.loopBody = cloneStmts(s.loopBody)
      c.unroll = s.unroll
    of sLab:
      c.labName = s.labName
    of sJmp:
      c.jmpTarget = s.jmpTarget
    of sScope:
      c.scopeBody = cloneStmts(s.scopeBody)
    of sAssume:
      c.assumeCond = cloneExpr(s.assumeCond)
    of sRet:
      c.retVal = cloneExpr(s.retVal)
    of sCall:
      c.callCallee = cloneExpr(s.callCallee)
      for a in s.callArgs:
        c.callArgs.add cloneExpr(a)
    of sAsm:
      c.asmParts = s.asmParts
      for op in s.asmOperands:
        c.asmOperands.add cloneExpr(op)
    of sEmit:
      c.emitParts = s.emitParts
      for op in s.emitOperands:
        c.emitOperands.add cloneExpr(op)
    of sDiscard, sCase:
      discard
    result.add c

func containsPtrDeep*(t: CirType): bool =
  ## Returns true when `t` names a pointer directly, in a field, or nested.
  case t.kind
  of ctPtr, ctAptr:
    true
  of ctStruct:
    for f in t.fields:
      if containsPtrDeep(f.typ):
        return true
    false
  of ctArray:
    containsPtrDeep(t.aElem)
  else:
    false

type
  CirParamLeaf* = object
    ## One exploded kernel parameter: a pointer leaf or a whole value leaf.
    name*: string
    typ*: CirType
    isBuffer*: bool

proc leavesWalk(t: CirType, name: string, leaves: var seq[CirParamLeaf]) =
  ## One step of the leaf walk.
  ## - pointer leaves stay whole, `isBuffer` set
  ## - structs holding pointers recurse under `name_field` names
  ## - value structs and arrays cross whole
  ## - an array leaf holding pointers refuses
  case t.kind
  of ctPtr, ctAptr:
    leaves.add CirParamLeaf(name: name, typ: t, isBuffer: true)
  of ctStruct:
    if t.containsPtrDeep:
      for f in t.fields:
        if f.typ.kind != ctVoid:
          leavesWalk(f.typ, name & "_" & f.name, leaves)
    else:
      leaves.add CirParamLeaf(name: name, typ: t, isBuffer: false)
  of ctArray:
    doAssert not t.containsPtrDeep,
      "kernelParamLeaves: array leaf '" & name & "' holds pointers, not supported as a kernel parameter"
    leaves.add CirParamLeaf(name: name, typ: t, isBuffer: false)
  else:
    leaves.add CirParamLeaf(name: name, typ: t, isBuffer: false)

proc kernelParamLeaves*(t: CirType, name: string): seq[CirParamLeaf] =
  ## Returns the leaves of `t` in field order, the walk of `leavesWalk`.
  leavesWalk(t, name, result)

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program scope
# ═══════════════════════════════════════════════════════════════════════

proc freshProcType*(params: seq[CirType], ret: CirType): CirType =
  ## A fresh proc type over `params`.
  ##
  ## Returns a `CirType` ref sharing nothing with the interned signature types:
  ##
  ## - structurally identical call-site callee types share one `CirType` ref
  ## - a call site re-typing its callee swaps the symbol type for this fresh copy, the shared ref stays untouched
  result = CirType(kind: ctProc)
  result.params = params
  result.ret = ret

proc hoistConstMemStmts*(m: CirModule, fn: CirProc, ss: var seq[CirStmt]): string =
  ## Relocates the `{.const_mem.}` locals of one statement list into `m.globals` in encounter order,
  ## nested scopes and case branches included, the declarations deleted from the list.
  ## Returns "" when clean, else the first refusal.
  ##
  ## Kernel and device-function locals relocate alike.
  ## Program-scope constant storage is visible to all functions in every target running this walk.
  ##
  ## The initializer is required: a program-scope constant declaration has no other initialization site.
  ## An initializer only the target's compiler can fold (a read of another local, say) prints verbatim and refuses there.
  var i = 0
  while i < ss.len:
    let s = ss[i]
    if s.kind == sVar and s.space == caConstant:
      if s.init == nil:
        return "at " & fn.name & ": const_mem local " & s.name &
          " without an initializer"
      m.globals.add s
      ss.delete i
      continue
    case s.kind
    of sIte:
      let thenErr = hoistConstMemStmts(m, fn, s.thenBody)
      if thenErr.len > 0:
        return thenErr
      let elseErr = hoistConstMemStmts(m, fn, s.elseBody)
      if elseErr.len > 0:
        return elseErr
    of sLoop:
      let loopErr = hoistConstMemStmts(m, fn, s.loopBody)
      if loopErr.len > 0:
        return loopErr
    of sCase:
      for bi in 0 ..< s.branches.len:
        let branchErr = hoistConstMemStmts(m, fn, s.branches[bi].body)
        if branchErr.len > 0:
          return branchErr
      let elseErr = hoistConstMemStmts(m, fn, s.caseElse)
      if elseErr.len > 0:
        return elseErr
    of sScope:
      let scopeErr = hoistConstMemStmts(m, fn, s.scopeBody)
      if scopeErr.len > 0:
        return scopeErr
    else:
      discard
    i.inc
  return ""

proc hoistConstMemLocals*(m: CirModule): string =
  ## Relocates every `{.const_mem.}` local of every proc into `m.globals` in encounter order,
  ## scopes and case branches included, the declarations dropped from their bodies.
  ## Returns "" when clean, else the first refusal.
  ##
  ## Contract:
  ## - the initializer is required, a const_mem local without one refuses
  ## - runs after the reserved-word rename, relocated declaration names match
  ##   the renamed body references
  for fn in m.procs:
    let err = hoistConstMemStmts(m, fn, fn.body)
    if err.len > 0:
      return err
  return ""
