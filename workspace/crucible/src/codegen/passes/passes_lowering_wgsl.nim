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
    "Inlines device functions that have no WGSL form and value-position" &
    " calls to them. A crossing pointer or pointer-holding struct" &
    " argument, or a coordinate-builtin read, removes the form",
    run = inlinePointerParamCalls)

# ═══════════════════════════════════════════════════════════════════════
#  Pointer-param call inlining
# ═══════════════════════════════════════════════════════════════════════

proc crossesWgslSignature(t: CirType): bool =
  ## WGSL function signatures accept values and ptr<function> pointers only.
  ## A pointer-holding struct crosses too: the WGSL struct declaration drops
  ## its pointer fields, so the callee's field reads have no target type.
  if t.isNil:
    return false
  case t.kind
  of ctAptr:
    result = true
  of ctPtr:
    result = t.addressSpace != caThread
  of ctStruct:
    result = t.containsPtrDeep
  else:
    result = false

proc hasCrossingPointerParam(fn: CirProc): bool =
  for p in fn.params:
    if p.typ.crossesWgslSignature:
      return true

proc collectCoordReads(kinds: var HashSet[CirCoordBuiltinKind], e: CirExpr) =
  if e.kind == eSym and e.builtin and e.coordBuiltin != cbkNone:
    kinds.incl(e.coordBuiltin)

proc collectCoordReadsStmts(ss: seq[CirStmt]): HashSet[CirCoordBuiltinKind] =
  ## Collects the coordinate builtins a body reads.
  ## WGSL binds coordinate builtins as entrypoint parameters only.
  ## A device function reading one has no callable form.
  var kinds: HashSet[CirCoordBuiltinKind]
  let collect = proc(e: CirExpr) =
    collectCoordReads(kinds, e)
  ss.walkStmts(nil, collect)
  return kinds

proc needsInlineForWgsl(fn: CirProc): bool =
  ## A device function with no WGSL form inlines into its caller.
  ## Its signature declares a pointer or a pointer-holding struct parameter.
  ## Its body may also read a coordinate builtin, an entrypoint-only value.
  fn.hasCrossingPointerParam or fn.body.collectCoordReadsStmts().len > 0

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

proc reachesInlineCycle(path: var HashSet[string], fn: CirProc, procs: Table[string, CirProc]): bool =
  ## Returns true when `fn` reaches itself through callees the pass inlines.
  ## Splicing such a cycle would never terminate.
  ## An acyclic chain inlines one link per fixpoint round.
  if fn.name in path:
    return true
  path.incl(fn.name)
  var targets: HashSet[string]
  collectCallTargets(targets, fn.body)
  for t in targets:
    let callee = procs.getOrDefault(t)
    if callee.isNil or callee.isGlobalFn or not callee.needsInlineForWgsl:
      continue
    if reachesInlineCycle(path, callee, procs):
      return true
  path.excl(fn.name)
  return false

proc substituteArgsExpr(e: CirExpr, subst: Table[string, CirExpr]) =
  ## Substitutes parameter reads with their spliced arguments.
  ## The callee symbol of a call is a function reference and never substitutes.
  if e.isNil:
    return
  if e.kind == eSym:
    if subst.hasKey(e.name):
      let arg = subst[e.name]
      e[] = arg[]
    return
  if e.kind == eCall:
    for a in mitems(e.args):
      a.substituteArgsExpr(subst)
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

proc renameLocalsExpr(e: CirExpr, rename: Table[string, string]) =
  ## Renames the local reads of a cloned callee body to their spliced names.
  ## The callee symbol of a call is a function reference and never renames.
  ## Builtin-born symbol reads keep their name.
  if e.isNil:
    return
  case e.kind
  of eSym:
    if not e.builtin and rename.hasKey(e.name):
      e.name = rename[e.name]
  of eCall:
    for a in mitems(e.args):
      a.renameLocalsExpr(rename)
  else:
    for ch in getExprChildren(e):
      ch.renameLocalsExpr(rename)

proc renameLocalsStmts(ss: var seq[CirStmt], rename: Table[string, string]) =
  ## Statement-walk overload of `renameLocalsExpr`.
  for s in mitems(ss):
    case s.kind
    of sVar:
      # a declaration renames too: the reads already carry the spliced name
      if rename.hasKey(s.name):
        s.name = rename[s.name]
      s.init.renameLocalsExpr(rename)
    of sAsgn:
      s.lhs.renameLocalsExpr(rename)
      s.rhs.renameLocalsExpr(rename)
    of sIte:
      s.cond.renameLocalsExpr(rename)
      s.thenBody.renameLocalsStmts(rename)
      s.elseBody.renameLocalsStmts(rename)
    of sWhile:
      s.whileCond.renameLocalsExpr(rename)
      s.whileBody.renameLocalsStmts(rename)
    of sRet:
      s.retVal.renameLocalsExpr(rename)
    of sDiscard:
      s.discardVal.renameLocalsExpr(rename)
    of sCall:
      for a in mitems(s.callArgs):
        a.renameLocalsExpr(rename)
    of sAsm, sEmit:
      discard

proc claimSplicedNames(used: var HashSet[string], fresh: var int, callee: CirProc): Table[string, string] =
  ## Renames every declared local of a spliced callee under one `inl<n>_` prefix.
  ## The caller declares locals of the same names and WGSL forbids shadowing.
  ## Renaming happens before the argument substitution so a spliced argument
  ## expression never catches the callee-local rename.
  fresh.inc
  var locals: seq[tuple[name: string, typ: CirType]]
  collectDeclaredLocals(locals, callee.body)
  for l in locals:
    var name = "inl" & $fresh & "_" & l.name
    while name in used:
      name.add "_"
    used.incl(name)
    result[l.name] = name

proc inlineInStmts(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, procs: Table[string, CirProc])
  ## Forward declaration, nested statement lists recurse into each other.

proc retsToAsgns(ss: var seq[CirStmt], host: CirExpr) =
  ## Writes each return of a spliced value body as an assignment to `host`.
  ## `host` is the caller's destination of the call result.
  ## A return mid-list or inside a loop refuses: the statements after it would run.
  ## A conditional return with one falling-through branch refuses: `host` then reads a value the original never wrote.
  for i in 0 ..< ss.len:
    let last = i == ss.len - 1
    case ss[i].kind
    of sRet:
      if not last:
        raiseAssert("inlining a value call needs its return at the end of the statement list")
      if ss[i].retVal == nil:
        raiseAssert("inlining a value call needs a returned value, the body returns nothing")
      ss[i] = CirStmt(kind: sAsgn, lhs: host, rhs: ss[i].retVal)
    of sIte:
      let thenRets = ss[i].thenBody.containsReturn
      let elseRets = ss[i].elseBody.containsReturn
      if thenRets != elseRets:
        raiseAssert("inlining a value call needs a return in both branches of a conditional return")
      if (thenRets or elseRets) and not last:
        raiseAssert("inlining a value call needs a conditional return at the end of the statement list")
      ss[i].thenBody.retsToAsgns(host)
      ss[i].elseBody.retsToAsgns(host)
    of sWhile:
      if ss[i].whileBody.containsReturn:
        raiseAssert("inlining a value call needs a return-free loop body")
    else:
      discard

proc spliceCalleeBody(callee: CirProc, callArgs: seq[CirExpr], used: var HashSet[string], fresh: var int, procs: Table[string, CirProc], host: CirExpr): seq[CirStmt] =
  ## One inlined call site: the callee's body with its locals renamed, parameters substituted, and returns writing `host`.
  ## A nil `host` splices a void callee in statement position.
  ## A non-nil `host` splices a value call whose result it receives.
  var body = cloneStmts(callee.body)
  if body.len > 0 and body[^1].kind == sRet and body[^1].retVal == nil:
    # an implicit fall-through return of the frontend, dropping it is
    # semantics-preserving at the splice position
    body.setLen(body.len - 1)
  var path: HashSet[string]
  if reachesInlineCycle(path, callee, procs):
    raiseAssert("inlining " & callee.name & " reaches an inline cycle, the splice would never terminate")
  if host == nil:
    if body.containsReturn:
      raiseAssert("inlining " & callee.name & " needs a return-free body, the callee has no WGSL function form")
  else:
    if callee.ret.kind == ctVoid:
      raiseAssert("inlining " & callee.name & " in value position needs a non-void return")
    if callee.ret.containsPtrDeep:
      raiseAssert("inlining " & callee.name & " returns a pointer-holding struct, WGSL has no local form for the value")
  let rename = claimSplicedNames(used, fresh, callee)
  body.renameLocalsStmts(rename)
  # substitution onto fresh aliases evaluates a call argument once
  var bindings: seq[CirStmt]
  var subst = initTable[string, CirExpr]()
  for j, p in callee.params:
    if p.typ.kind in {ctPtr, ctAptr}:
      # a pointer argument has no WGSL local form, it substitutes directly
      subst[p.name] = callArgs[j]
      continue
    if p.typ.kind == ctStruct and p.typ.containsPtrDeep:
      # a struct argument substitutes directly: its pointer fields have no
      # WGSL local form. The pointer-local explosion pass dissolves spliced
      # field reads into the caller's leaf symbols.
      if callArgs[j].kind != eSym:
        raiseAssert("struct argument " & p.name & " of " & callee.name &
          " is not a symbol read, WGSL has no local form for a pointer-holding struct value")
      subst[p.name] = callArgs[j]
      continue
    var alias = "inl" & $fresh & "_" & p.name
    while alias in used:
      alias.add "_"
    used.incl(alias)
    bindings.add CirStmt(kind: sVar, name: alias, declType: p.typ, init: callArgs[j])
    subst[p.name] = sym(alias, p.typ)
  body.substituteArgsStmts(subst)
  if host != nil:
    body.retsToAsgns(host)
    doAssert not body.containsReturn
  result = bindings & body

proc refuseValuePositionCalls(fn: CirProc, e: CirExpr, procs: Table[string, CirProc]) =
  ## A call in a value position the splice cannot host: a direct initializer
  ## or assignment splices, every other position refuses.
  if e.isNil:
    return
  if e.kind == eCall and e.callee.kind == eSym:
    let callee = procs.getOrDefault(e.callee.name)
    if callee != nil and not callee.isGlobalFn and callee.needsInlineForWgsl:
      raiseAssert("value-position call of " & callee.name & ", the callee has no WGSL function form")
  for ch in getExprChildren(e):
    refuseValuePositionCalls(fn, ch, procs)

proc inlineCallSite(ss: var seq[CirStmt], used: var HashSet[string], fresh: var int, fn: CirProc, i: int, procs: Table[string, CirProc]): bool =
  ## Returns whether the statement spliced, refusing calls the splice cannot host.
  let s = ss[i]
  let callee = procs.getOrDefault(s.callCallee.name)
  if callee != nil and not callee.isGlobalFn and callee.needsInlineForWgsl:
    if callee.ret.kind != ctVoid:
      raiseAssert("inlining " & callee.name & " in statement position needs a void return")
    ss = ss[0 ..< i] & spliceCalleeBody(callee, s.callArgs, used, fresh, procs, nil) & ss[i + 1 .. ^1]
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
    if s.kind == sVar and s.init != nil and s.init.kind == eCall and s.init.callee.kind == eSym:
      let callee = procs.getOrDefault(s.init.callee.name)
      if callee != nil and not callee.isGlobalFn and callee.needsInlineForWgsl:
        let host = sym(s.name, s.declType)
        let splicedBody = spliceCalleeBody(callee, s.init.args, used, fresh, procs, host)
        s.init = nil
        ss = ss[0 ..< i] & @[s] & splicedBody & ss[i + 1 .. ^1]
        continue
    if s.kind == sAsgn and s.rhs != nil and s.rhs.kind == eCall and s.rhs.callee.kind == eSym:
      let callee = procs.getOrDefault(s.rhs.callee.name)
      if callee != nil and not callee.isGlobalFn and callee.needsInlineForWgsl:
        let splicedBody = spliceCalleeBody(callee, s.rhs.args, used, fresh, procs, s.lhs)
        ss = ss[0 ..< i] & splicedBody & ss[i + 1 .. ^1]
        continue
    walkValuePositions(used, fresh, fn, s, procs)
    i.inc

proc inlinePointerParamCalls(m: CirModule) =
  ## Placement contract:
  ## - the pass runs before the pointer-local explosion pass, which dissolves
  ##   the spliced struct field reads into the caller's leaves
  ## - the spliced bodies carry no pointer locals: pointer and pointer-holding
  ##   struct arguments substitute directly
  ## - the per-address-space callee instantiation runs after this pass
  ##   and reads the splice-substituted concrete argument types
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
        if callee != nil and not callee.isGlobalFn and callee.needsInlineForWgsl:
          changed = true
      if not changed:
        break
  var kept: seq[CirProc]
  for fn in m.procs:
    if not fn.isGlobalFn and fn.needsInlineForWgsl:
      # every call site inlined, the signature has no WGSL form
      continue
    kept.add fn
  m.procs = kept
