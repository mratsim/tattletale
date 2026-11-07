# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./passes_lowering_cfamily
import ./passes_low_pointers
import ./pass_helpers

# ############################################################
#
#            WGSL lowering
#
# ############################################################

# ═══════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════

func registerWgslExplodePointerLocals*(reg: var PassRegistry) =
  registerExplodePointerLocals(reg, "wgslExplodePointerLocals",
    "Rewrites pointer locals to their pointees and pointer-holding struct" &
    " locals to their leaf symbols",
    "WGSL")

func registerWgslHoistProgramScopeLocals*(reg: var PassRegistry) =
  registerCfamilyHoistProgramScopeLocals(reg, "wgslHoistProgramScopeLocals",
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope")

func registerWgslInlinePointerParamCalls*(reg: var PassRegistry) =
  reg.register("wgslInlinePointerParamCalls", pkTransform,
    "Inlines calls whose pointer arguments cannot cross a WGSL function" &
    " signature",
    run = inlinePointerParamCalls)

# ═══════════════════════════════════════════════════════════════════════
#  Pointer-param call inlining
# ═══════════════════════════════════════════════════════════════════════

proc crossesWgslSignature(t: CirType): bool =
  ## WGSL function signatures accept values and ptr<function> pointers only.
  if t.isNil:
    return false
  case t.kind
  of ctAptr:
    result = true
  of ctPtr:
    result = t.addressSpace != caThread
  else:
    result = false

proc hasCrossingPointerParam(fn: CirProc): bool =
  for p in fn.params:
    if p.typ.crossesWgslSignature:
      return true

proc containsReturn(ss: seq[CirStmt]): bool =
  ## A frontend trailing void-body `return` does not count here: the caller drops it before splicing.
  for s in ss:
    case s.kind
    of sRet:
      return true
    of sIte:
      if s.thenBody.containsReturn or s.elseBody.containsReturn:
        return true
    of sWhile:
      if s.whileBody.containsReturn:
        return true
    else:
      discard
  return false

proc callsCrossingCalleeTransitively(seen: var HashSet[string], fn: CirProc, procs: Table[string, CirProc]): bool =
  ## Returns true when `fn` calls itself or reaches a proc whose parameters
  ## cannot cross a WGSL function signature. Splicing such a cycle would never terminate.
  if fn.name in seen:
    return false
  seen.incl(fn.name)
  var targets: HashSet[string]
  collectCallTargets(targets, fn.body)
  for t in targets:
    let callee = procs.getOrDefault(t)
    if callee.isNil:
      continue
    if callee.hasCrossingPointerParam:
      return true
    if callsCrossingCalleeTransitively(seen, callee, procs):
      return true
  return false

proc substituteArgsExpr(e: CirExpr, subst: Table[string, CirExpr]) =
  if e.isNil:
    return
  if e.kind == eSym:
    if subst.hasKey(e.name):
      let arg = subst[e.name]
      e[] = arg[]
    return
  for ch in getExprChildren(e):
    ch.substituteArgsExpr(subst)

proc substituteArgsStmts(ss: var seq[CirStmt], subst: Table[string, CirExpr]) =
  ## Statement-walk overload of `substituteArgsExpr`.
  ## Hand-rolled walk: a substitution splices the call argument into the symbol node
  ## and never re-walks the spliced content, its symbols belong to the caller.
  ## A shared walk would descend into the rewritten node and substitute a second parameter inside the argument.
  for s in mitems(ss):
    case s.kind
    of sVar:
      s.init.substituteArgsExpr(subst)
    of sAsgn:
      s.lhs.substituteArgsExpr(subst)
      s.rhs.substituteArgsExpr(subst)
    of sIte:
      s.cond.substituteArgsExpr(subst)
      s.thenBody.substituteArgsStmts(subst)
      s.elseBody.substituteArgsStmts(subst)
    of sWhile:
      s.whileCond.substituteArgsExpr(subst)
      s.whileBody.substituteArgsStmts(subst)
    of sRet:
      s.retVal.substituteArgsExpr(subst)
    of sDiscard:
      s.discardVal.substituteArgsExpr(subst)
    of sCall:
      s.callCallee.substituteArgsExpr(subst)
      for a in mitems(s.callArgs):
        a.substituteArgsExpr(subst)
    of sAsm, sEmit:
      discard

proc inlineInStmts(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, procs: Table[string, CirProc])
  ## Forward declaration, nested statement lists recurse into each other.

proc refuseValuePositionCalls(fn: CirProc, e: CirExpr, procs: Table[string, CirProc]) =
  ## A call in value position cannot inline.
  ## Only a void callee splices into statement position, a call whose result
  ## feeds an expression has no destination for the inlined body's writes,
  ## so the pass refuses the call.
  if e.isNil:
    return
  if e.kind == eCall and e.callee.kind == eSym:
    let callee = procs.getOrDefault(e.callee.name)
    if callee != nil and not callee.isGlobalFn and callee.hasCrossingPointerParam:
      raiseAssert("value-position call of " & callee.name & ", its pointer parameter cannot cross a WGSL signature")
  for ch in getExprChildren(e):
    refuseValuePositionCalls(fn, ch, procs)

proc inlineCallSite(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, i: int, procs: Table[string, CirProc]): bool =
  ## Returns whether the statement spliced, refusing calls the splice cannot host.
  let s = ss[i]
  let callee = procs.getOrDefault(s.callCallee.name)
  if callee != nil and not callee.isGlobalFn and callee.hasCrossingPointerParam:
    if callee.ret.kind != ctVoid:
      raiseAssert("inlining " & callee.name & " needs a void return, its pointer parameter cannot cross a WGSL signature")
    var body = cloneStmts(callee.body)
    if body.len > 0 and body[^1].kind == sRet and body[^1].retVal == nil:
      # an implicit fall-through return of the frontend, dropping it is
      # semantics-preserving at the splice position
      body.setLen(body.len - 1)
    if body.containsReturn:
      raiseAssert("inlining " & callee.name & " needs a return-free body, its pointer parameter cannot cross a WGSL signature")
    var seen: HashSet[string]
    if callsCrossingCalleeTransitively(seen, callee, procs):
      raiseAssert("inlining " & callee.name & " reaches a crossing call cycle, its pointer parameter cannot cross a WGSL signature")
    # substitution onto fresh aliases evaluates a call argument once
    var bindings: seq[CirStmt]
    var subst = initTable[string, CirExpr]()
    for j, p in callee.params:
      if p.typ.kind in {ctPtr, ctAptr}:
        # a pointer argument has no WGSL local form, it substitutes directly
        subst[p.name] = s.callArgs[j]
        continue
      var alias = "inl" & $fresh & "_" & p.name
      fresh.inc
      while alias in used:
        alias.add "_"
      used.incl(alias)
      bindings.add CirStmt(kind: sVar, name: alias, declType: p.typ, init: s.callArgs[j])
      subst[p.name] = sym(alias, p.typ)
    body.substituteArgsStmts(subst)
    ss = ss[0 ..< i] & bindings & body & ss[i + 1 .. ^1]
    return true
  for a in s.callArgs:
    refuseValuePositionCalls(fn, a, procs)
  return false

proc walkValuePositions(used: var HashSet[string], fresh: var int, fn: CirProc, s: CirStmt, procs: Table[string, CirProc]) =
  ## One non-call statement: its subexpressions refuse value-position calls,
  ## nested statement lists recurse through `inlineInStmts`.
  case s.kind
  of sVar:
    refuseValuePositionCalls(fn, s.init, procs)
  of sAsgn:
    refuseValuePositionCalls(fn, s.lhs, procs)
    refuseValuePositionCalls(fn, s.rhs, procs)
  of sIte:
    refuseValuePositionCalls(fn, s.cond, procs)
    inlineInStmts(s.thenBody, used, fresh, fn, procs)
    inlineInStmts(s.elseBody, used, fresh, fn, procs)
  of sWhile:
    refuseValuePositionCalls(fn, s.whileCond, procs)
    inlineInStmts(s.whileBody, used, fresh, fn, procs)
  of sRet:
    refuseValuePositionCalls(fn, s.retVal, procs)
  of sDiscard:
    refuseValuePositionCalls(fn, s.discardVal, procs)
  of sCall, sAsm, sEmit:
    discard

proc inlineInStmts(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, procs: Table[string, CirProc]) =
  var i = 0
  while i < ss.len:
    let s = ss[i]
    if s.kind == sCall and s.callCallee.kind == eSym:
      let spliced = inlineCallSite(ss, used, fresh, fn, i, procs)
      if not spliced:
        i.inc
      continue
    walkValuePositions(used, fresh, fn, s, procs)
    i.inc

proc inlinePointerParamCalls(m: CirModule) =
  ## Placement contract:
  ## - the pass runs after the pointer-local explosion pass, so the spliced
  ##   bodies carry no pointer locals
  ## - the per-address-space callee instantiation runs first, a re-spaced formal
  ##   refuses exactly when the new signature cannot print
  var procs = initTable[string, CirProc]()
  for fn in m.procs:
    procs[fn.name] = fn
  for fn in mitems(m.procs):
    var used = initHashSet[string]()
    var locals: seq[tuple[name: string, typ: CirType]]
    collectDeclaredLocals(locals, fn.body)
    for l in locals:
      used.incl(l.name)
    for p in fn.params:
      used.incl(p.name)
    var fresh = 0
    while true:
      inlineInStmts(fn.body, used, fresh, fn, procs)
      var changed = false
      var targets: HashSet[string]
      collectCallTargets(targets, fn.body)
      for t in targets:
        let callee = procs.getOrDefault(t)
        if callee != nil and not callee.isGlobalFn and callee.hasCrossingPointerParam:
          changed = true
      if not changed:
        break
  var kept: seq[CirProc]
  for fn in m.procs:
    if not fn.isGlobalFn and fn.hasCrossingPointerParam:
      # every call site inlined, the signature has no WGSL form
      continue
    kept.add fn
  m.procs = kept

