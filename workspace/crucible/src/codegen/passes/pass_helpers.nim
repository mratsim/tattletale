# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Shared pass helpers over CIRv2.
##
## Every pass file may import these, no pass owns them.
## Rewrite helpers return the empty string when clean, else a refusal text naming the site.

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build

proc exprRootSym*(e: CirExpr): CirExpr =
  result = e
  while result != nil and result.kind in {eDot, ePat, eAt}:
    result = case result.kind
      of eDot: result.dotBase
      of ePat: result.patBase
      else: result.atBase
  if result != nil and result.kind != eSym:
    result = nil

proc walkExprs*(e: CirExpr, visit: proc(e: CirExpr)) =
  ## Contract:
  ## - a visitor may rewrite the visited node in place, expressions are shared refs
  ## - the children are read after the visit, so a rewrite steers the descent
  ##   into the rewritten shape
  if e.isNil:
    return
  visit(e)
  for ch in getExprChildren(e):
    walkExprs(ch, visit)

proc walkStmts*(ss: seq[CirStmt], onStmt: proc(s: CirStmt), onExpr: proc(e: CirExpr)) =
  ## Walks one statement list through two optional visitors.
  ##
  ## Contract:
  ## - `onStmt` runs last for a statement, after the expression visits and after the nested bodies recurse
  ## - the walk iterates a shallow copy of the list, insertions made by `onStmt` do not disturb the iteration
  ## - visitors return nothing: a refusing visitor records its refusal itself
  ##   and stops work once recorded, so the first refusal in walk order wins

  proc slot(e: CirExpr) =
    if onExpr != nil:
      e.walkExprs(onExpr)
  var walked: seq[CirStmt]
  for s in ss:
    walked.add s
  for s in walked:
    case s.kind
    of sVar, sConst:
      slot(s.init)
    of sAsgn:
      slot(s.lhs)
      slot(s.rhs)
    of sIte:
      slot(s.cond)
      walkStmts(s.thenBody, onStmt, onExpr)
      walkStmts(s.elseBody, onStmt, onExpr)
    of sWhile:
      slot(s.whileCond)
      walkStmts(s.whileBody, onStmt, onExpr)
    of sRet:
      slot(s.retVal)
    of sDiscard:
      slot(s.discardVal)
    of sCall:
      slot(s.callCallee)
      for a in s.callArgs:
        slot(a)
    of sAsm:
      for op in s.asmOperands:
        slot(op)
    of sEmit:
      for op in s.emitOperands:
        slot(op)
    if onStmt != nil:
      onStmt(s)

proc collectCallTargets*(stmts: seq[CirStmt], targets: var HashSet[string]) =
  var collected = targets
  let onExpr = proc(e: CirExpr) =
    if e.kind == eCall and e.callee.kind == eSym:
      collected.incl e.callee.name
  let onStmt = proc(s: CirStmt) =
    if s.kind == sCall and s.callCallee.kind == eSym:
      collected.incl s.callCallee.name
  stmts.walkStmts(onStmt, onExpr)
  targets = collected

proc collectDeclaredLocals*(stmts: seq[CirStmt], locals: var seq[tuple[name: string, typ: CirType]]) =
  var collected = locals
  let collect = proc(s: CirStmt) =
    if s.kind in {sVar, sConst}:
      collected.add((name: s.name, typ: s.declType))
  stmts.walkStmts(collect, nil)
  locals = collected

proc walkTypeTree*(t: CirType, visit: proc(t: CirType), descendPtrs: bool = true) =
  ## Contract:
  ## - a printer that names no type behind a pointer kind passes `descendPtrs = false`
  ##   and leaves the pointer targets unvisited
  ## - a collecting visitor cannot loop, value type trees are acyclic
  if t.isNil:
    return
  visit(t)
  case t.kind
  of ctStruct:
    for f in t.fields:
      walkTypeTree(f.typ, visit, descendPtrs)
  of ctArray:
    walkTypeTree(t.aElem, visit, descendPtrs)
  of ctVector:
    walkTypeTree(t.vElem, visit, descendPtrs)
  of ctPtr, ctAptr:
    if descendPtrs:
      walkTypeTree(t.pElem, visit, descendPtrs)
  else:
    discard

proc singleKernel*(m: CirModule): CirProc =
  ## A module without exactly one kernel contributes no kernel-side registrations,
  ## such a module refuses at the printer.
  for fn in m.procs:
    if fn.isKernel:
      if result != nil:
        return nil
      result = fn

proc scanStmtRefs*(ss: seq[CirStmt], onDecl: proc(s: CirStmt): bool, onExpr: proc(e: CirExpr)) =
  ## `onDecl` takes each `var`/`const` declaration and returns `false` to skip
  ## that declaration's initializer walk, for a declaration that never prints
  ## on the refusing target.
  for s in ss:
    case s.kind
    of sVar, sConst:
      if onDecl(s):
        s.init.walkExprs(onExpr)
    of sAsgn:
      s.lhs.walkExprs(onExpr)
      s.rhs.walkExprs(onExpr)
    of sIte:
      s.cond.walkExprs(onExpr)
      scanStmtRefs(s.thenBody, onDecl, onExpr)
      scanStmtRefs(s.elseBody, onDecl, onExpr)
    of sWhile:
      scanStmtRefs(s.whileBody, onDecl, onExpr)
    of sRet:
      s.retVal.walkExprs(onExpr)
    of sDiscard:
      s.discardVal.walkExprs(onExpr)
    of sCall:
      s.callCallee.walkExprs(onExpr)
      for a in s.callArgs:
        a.walkExprs(onExpr)
    of sAsm, sEmit:
      discard

proc cloneExpr*(e: CirExpr): CirExpr =
  ## Deep copy: expressions are shared refs, a rewrite of the clone must not
  ## reach the source tree.
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
  of eTernary:
    result.ternType = e.ternType
    result.cond = cloneExpr(e.cond)
    result.thenE = cloneExpr(e.thenE)
    result.elseE = cloneExpr(e.elseE)
  of eNot:
    result.notOperand = cloneExpr(e.notOperand)
  of eNeg:
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
    of sWhile:
      c.whileCond = cloneExpr(s.whileCond)
      c.whileBody = cloneStmts(s.whileBody)
      c.whileUnroll = s.whileUnroll
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
    of sDiscard:
      c.discardVal = cloneExpr(s.discardVal)
    result.add c

func containsPtrDeep*(t: CirType): bool =
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
    ## `isBuffer` marks a pointer leaf that binds as a device buffer, a value leaf binds by value.
    name*: string
    typ*: CirType
    isBuffer*: bool

proc leavesWalk(t: CirType, name: string, leaves: var seq[CirParamLeaf]) =
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
  leavesWalk(t, name, result)

# ═══════════════════════════════════════════════════════════════════════
# Pointer-local explosion
# ═══════════════════════════════════════════════════════════════════════

type
  ExplodeLeaf* = tuple[sym: string, typ: CirType]
    ## One exploded field read: the leaf symbol standing in for it plus the leaf type.

  ExplodeCtx* = object
    pendingPtrs*: HashSet[string]
      ## Pointer locals awaiting their address-of binding, a read refuses.
    aliasOf*: Table[string, string]
      ## Pointer local -> pointee local: one binding fixes the pair, a rebinding refuses.
    structLocals*: HashSet[string]
      ## Pointer-holding struct locals exploded into field rewrites: a whole-value use refuses.
    leafOf*: Table[string, ExplodeLeaf]
      ## Dotted field path of an exploded struct local -> its bound leaf symbol:
      ## `v.data` -> the symbol bound to `data`, `v.inner.f` -> the symbol bound to the nested `f`.
    byRefParams*: HashSet[string]
      ## `var T` parameters of the proc in flight: targets that bind them `inout`
      ## unwrap their derefs to the bare parameter.
    procs*: Table[string, CirProc]
      ## Module procs by name: targets that check call-site argument shapes against
      ## callee parameter types read them here.

proc refuseAt*(fn: CirProc, detail: string): string =
  ## Text the pass runner prefixes with the pass name.
  "at " & fn.name & ": " & detail

proc dottedPath*(e: CirExpr): string =
  if e.kind == eSym:
    return e.name
  if e.kind == eDot:
    let base = dottedPath(e.dotBase)
    if base.len > 0:
      return base & "." & e.fieldName
  return ""

proc structRoot*(e: CirExpr): CirExpr =
  ## Unlike `exprRootSym`, an indexed or pattern base stops this walk, nil.
  result = e
  while result != nil and result.kind == eDot:
    result = result.dotBase
  if result != nil and result.kind != eSym:
    result = nil

proc spliceSym*(e: CirExpr, name: string, typ: CirType) =
  ## Expressions are shared refs, so a rewrite applies to the node every sharer sees.
  e[] = sym(name, typ)[]

proc registerExplodedStruct*(fn: CirProc, ctx: var ExplodeCtx, path: string, t: CirType, c: CirExpr, target: string): string =
  ## Records the field rewrites of one pointer-holding struct local whose
  ## object construction binds leaf symbols, nested constructions recurse.
  ##
  ## Args:
  ## - `path`: the dotted path the local is exploded under, `v` at the top
  ## - `t`: the declared type of the local, `c` must construct it
  ## - `target`: the target's name in the refusal texts, `WGSL` or `HLSL`
  if c.kind != eOconstr:
    return refuseAt(fn, "struct local " & path &
      " initializer is not a construction, " & target &
      " has no pointer struct members")
  if c.ocType.name != t.name:
    return refuseAt(fn, "struct local " & path & " initializer has type " &
      c.ocType.name & ", expected " & t.name)
  var bound: seq[string]
  for kvv in c.kvFields:
    bound.add kvv.name
    var ft: CirType = nil
    for f in t.fields:
      if f.name == kvv.name:
        ft = f.typ
    if ft.isNil:
      return refuseAt(fn, "field " & kvv.name & " is not on " & t.name)
    if kvv.value.kind == eOconstr:
      let err = registerExplodedStruct(fn, ctx, path & "." & kvv.name, ft, kvv.value, target)
      if err.len > 0:
        return err
    elif kvv.value.kind == eSym:
      ctx.leafOf[path & "." & kvv.name] = (kvv.value.name, ft)
    else:
      return refuseAt(fn, "field " & path & "." & kvv.name &
        " initializer is not a leaf symbol, " & target &
        " has no pointer struct members")
  for f in t.fields:
    if f.name notin bound:
      return refuseAt(fn, "field " & f.name & " of " & path &
        " is not bound by the construction")
  return ""

proc explodeStructVarDecl*(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt, target: string): string =
  if s.init == nil or s.init.kind != eOconstr:
    return refuseAt(fn, "struct local " & s.name &
      " without a construction, " & target & " has no pointer struct members")
  let err = registerExplodedStruct(fn, ctx, s.name, s.declType, s.init, target)
  if err.len > 0:
    return err
  ctx.structLocals.incl(s.name)
  return ""

proc rewritePointerExprShared*(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr,
    target: string,
    ownArm: proc(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string): string =
  ## Shared skeleton of the WGSL and HLSL pointer-local expression rewrites.
  ##
  ## A deref operand stands for its pointee:
  ## - a bound alias reads as the pointee local that the address-of binding recorded
  ## - a by-reference parameter reads as the bare parameter itself
  ##
  ## `ownArm` handles every shape outside the skeleton, refusal set and recursion included.
  ## `target` is interpolated into the refusal texts.
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in ctx.structLocals:
      return refuseAt(fn, "struct local " & e.name &
        " used whole, " & target & " has no pointer struct members, read the fields")
    if e.name in ctx.pendingPtrs:
      return refuseAt(fn, "pointer local " & e.name &
        " read before its address-of binding")
    if e.name in ctx.aliasOf:
      return refuseAt(fn, "pointer local " & e.name &
        " used as a value, " & target & " has no pointer values")
    return ""
  of eDot:
    let root = structRoot(e)
    if root != nil and root.name in ctx.structLocals:
      let path = dottedPath(e)
      if ctx.leafOf.hasKey(path):
        let leaf = ctx.leafOf[path]
        e.spliceSym(leaf.sym, leaf.typ)
        return ""
    return ownArm(fn, ctx, e)
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym:
      if op.name in ctx.aliasOf:
        e.spliceSym(ctx.aliasOf[op.name], op.typ.pElem)
        return ""
      if op.name in ctx.byRefParams:
        e.spliceSym(op.name, op.typ.pElem)
        return ""
    return ownArm(fn, ctx, e)
  else:
    return ownArm(fn, ctx, e)

# ═══════════════════════════════════════════════════════════════════════
# Const_mem locals to program scope
# ═══════════════════════════════════════════════════════════════════════

proc freshProcType*(params: seq[CirType], ret: CirType): CirType =
  ## A fresh proc type over `params`, sharing nothing with the interned signature types.
  ##
  ## - structurally identical call-site callee types share one `CirType` ref
  ## - a call site re-typing its callee swaps the symbol type for the fresh copy, so the shared ref of the other call sites stays untouched
  result = CirType(kind: ctProc)
  result.params = params
  result.ret = ret

proc hoistConstMemStmts*(m: CirModule, fn: CirProc, ss: var seq[CirStmt]): string =
  ## Relocates the `{.const_mem.}` locals of one statement list into `m.globals`, deleting the declarations from the list.
  ##
  ## Contract:
  ## - relocation is in encounter order, loop and branch bodies included
  ## - kernel and device-function locals relocate alike: program-scope constant
  ##   storage is visible to all functions in every target running this walk
  ## - the initializer is required, a program-scope constant declaration has no
  ##   other initialization site
  ## - an initializer only the target's compiler can fold prints verbatim and refuses there (a read of another local, say)
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
    of sWhile:
      let loopErr = hoistConstMemStmts(m, fn, s.whileBody)
      if loopErr.len > 0:
        return loopErr
    else:
      discard
    i.inc
  return ""

proc hoistConstMemLocals*(m: CirModule): string =
  ## Relocates every `{.const_mem.}` local of every proc into the module's
  ## constant section:
  ## - runs after the reserved-word rename, so the relocated declaration
  ##   names match the renamed body references
  for fn in m.procs:
    let err = hoistConstMemStmts(m, fn, fn.body)
    if err.len > 0:
      return err
  return ""
