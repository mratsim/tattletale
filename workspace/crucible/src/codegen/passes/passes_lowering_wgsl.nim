# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## WGSL lowering: per-target transforms for the WGSL storage model.
##
## - kernel parameters of struct type with pointer fields expand into flat ptr/scalar leaf parameters, WebGPU is a flat resource target
## - pointer locals rewrite to their pointee, pointer-holding struct locals into per-field leaf rewrites
## - `{.smem.}`/`{.const_mem.}` kernel locals relocate into `CirModule.globals`, top-level kernel-body locals only
## - a statement call to a device function taking a buffer-reference or out-of-space pointer parameter inlines the callee at the call site
##
## The pass leaves kernel and device-function pointer parameters untouched, the printer emits `ptr<function, T>` parameters as-is.
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
  ## Rewritten aliases and relocated declarations use the renamed names that their body already references.
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
  ## WGSL function signatures accept values and `ptr<function>` pointers only,
  ## so a buffer reference and a pointer in any other address space refuse.
  if t.isNil:
    return false
  case t.kind
  of ctAptr:
    result = true
  of ctPtr:
    result = t.space != caThread
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

proc callsCrossingCalleeTransitively(fn: CirProc, procs: Table[string, CirProc], seen: var HashSet[string]): bool =
  ## Returns true when `fn` calls itself or reaches a proc whose parameters
  ## cannot cross a WGSL function signature. Splicing such a cycle would never terminate.
  if fn.name in seen:
    return false
  seen.incl(fn.name)
  var targets: HashSet[string]
  fn.body.collectCallTargets(targets)
  for t in targets:
    let callee = procs.getOrDefault(t)
    if callee.isNil:
      continue
    if callee.hasCrossingPointerParam:
      return true
    if callee.callsCrossingCalleeTransitively(procs, seen):
      return true
  return false

proc substituteArgsExpr(e: CirExpr, subst: Table[string, CirExpr]) =
  ## Replaces every symbol named in `subst` with its call-site argument, in place.
  ## One substitution pass covers every parameter: the spliced argument is not
  ## re-walked, its symbols belong to the caller.
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
    of sVar, sConst:
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

proc inlineInStmts(fn: CirProc, ss: var seq[CirStmt], procs: Table[string, CirProc], used: var HashSet[string], fresh: var int): string
  ## Forward declaration, nested statement lists recurse into each other.

proc refuseValuePositionCalls(fn: CirProc, e: CirExpr, procs: Table[string, CirProc]): string =
  ## A call in value position cannot inline.
  ## Only a void callee splices into statement position, a call whose result
  ## feeds an expression has no destination for the inlined body's writes,
  ## so the pass refuses the call.
  if e.isNil:
    return ""
  if e.kind == eCall and e.callee.kind == eSym:
    let callee = procs.getOrDefault(e.callee.name)
    if callee != nil and not callee.isKernel and callee.hasCrossingPointerParam:
      return refuseAt(fn, "value-position call of " & callee.name & ", its pointer parameter cannot cross a WGSL signature")
  for ch in getExprChildren(e):
    let err = refuseValuePositionCalls(fn, ch, procs)
    if err.len > 0:
      return err
  return ""

proc inlineCallSite(fn: CirProc, ss: var seq[CirStmt], i: int, procs: Table[string, CirProc], used: var HashSet[string], fresh: var int): tuple[err: string, spliced: bool] =
  let s = ss[i]
  let callee = procs.getOrDefault(s.callCallee.name)
  if callee != nil and not callee.isKernel and callee.hasCrossingPointerParam:
    if callee.ret.kind != ctVoid:
      return (refuseAt(fn, "inlining " & callee.name & " needs a void return, its pointer parameter cannot cross a WGSL signature"), false)
    var body = cloneStmts(callee.body)
    if body.len > 0 and body[^1].kind == sRet and body[^1].retVal == nil:
      # an implicit fall-through return of the frontend, dropping it is
      # semantics-preserving at the splice position
      body.setLen(body.len - 1)
    if body.containsReturn:
      return (refuseAt(fn, "inlining " & callee.name & " needs a return-free body, its pointer parameter cannot cross a WGSL signature"), false)
    var seen: HashSet[string]
    if callee.callsCrossingCalleeTransitively(procs, seen):
      return (refuseAt(fn, "inlining " & callee.name & " reaches a crossing call cycle, its pointer parameter cannot cross a WGSL signature"), false)
    # every argument binds once to a fresh local and the body substitutes onto
    # those aliases: an argument with a call evaluates once, like v1's expansion
    var bindings: seq[CirStmt]
    var subst = initTable[string, CirExpr]()
    for j, p in callee.params:
      var alias = "inl" & $fresh & "_" & p.name
      fresh.inc
      while alias in used:
        alias.add "_"
      used.incl(alias)
      bindings.add CirStmt(kind: sVar, name: alias, declType: p.typ, init: s.callArgs[j])
      subst[p.name] = sym(alias, p.typ)
    body.substituteArgsStmts(subst)
    ss = ss[0 ..< i] & bindings & body & ss[i + 1 .. ^1]
    return ("", true)
  for a in s.callArgs:
    let aerr = refuseValuePositionCalls(fn, a, procs)
    if aerr.len > 0:
      return (aerr, false)
  return ("", false)

proc walkValuePositions(fn: CirProc, s: CirStmt, procs: Table[string, CirProc], used: var HashSet[string], fresh: var int): string =
  ## One non-call statement: its subexpressions refuse value-position calls,
  ## nested statement lists recurse through `inlineInStmts`.
  case s.kind
  of sVar, sConst:
    let err = refuseValuePositionCalls(fn, s.init, procs)
    if err.len > 0:
      return err
  of sAsgn:
    let lerr = refuseValuePositionCalls(fn, s.lhs, procs)
    if lerr.len > 0:
      return lerr
    let rerr = refuseValuePositionCalls(fn, s.rhs, procs)
    if rerr.len > 0:
      return rerr
  of sIte:
    let cerr = refuseValuePositionCalls(fn, s.cond, procs)
    if cerr.len > 0:
      return cerr
    let terr = inlineInStmts(fn, s.thenBody, procs, used, fresh)
    if terr.len > 0:
      return terr
    let eerr = inlineInStmts(fn, s.elseBody, procs, used, fresh)
    if eerr.len > 0:
      return eerr
  of sWhile:
    let cerr = refuseValuePositionCalls(fn, s.whileCond, procs)
    if cerr.len > 0:
      return cerr
    let werr = inlineInStmts(fn, s.whileBody, procs, used, fresh)
    if werr.len > 0:
      return werr
  of sRet:
    let rerr = refuseValuePositionCalls(fn, s.retVal, procs)
    if rerr.len > 0:
      return rerr
  of sDiscard:
    let derr = refuseValuePositionCalls(fn, s.discardVal, procs)
    if derr.len > 0:
      return derr
  of sCall, sAsm, sEmit:
    discard

proc inlineInStmts(fn: CirProc, ss: var seq[CirStmt], procs: Table[string, CirProc], used: var HashSet[string], fresh: var int): string =
  var i = 0
  while i < ss.len:
    let s = ss[i]
    if s.kind == sCall and s.callCallee.kind == eSym:
      let (r, spliced) = inlineCallSite(fn, ss, i, procs, used, fresh)
      if r.len > 0:
        return r
      if not spliced:
        i.inc
      continue
    let r = walkValuePositions(fn, s, procs, used, fresh)
    if r.len > 0:
      return r
    i.inc
  return ""

proc inlinePointerParamCalls(m: CirModule): string =
  ## Placement contract:
  ## - the pass runs after the pointer-local explosion pass, spliced bodies
  ##   therefore carry no pointer locals
  ## - the per-address-space callee instantiation runs first, the pass refuses
  ##   a formal re-spaced by its call sites exactly when the re-spaced
  ##   signature could not print
  var procs = initTable[string, CirProc]()
  for fn in m.procs:
    procs[fn.name] = fn
  for fn in mitems(m.procs):
    var used = initHashSet[string]()
    var locals: seq[tuple[name: string, typ: CirType]]
    fn.body.collectDeclaredLocals(locals)
    for l in locals:
      used.incl(l.name)
    for p in fn.params:
      used.incl(p.name)
    var fresh = 0
    while true:
      let err = inlineInStmts(fn, fn.body, procs, used, fresh)
      if err.len > 0:
        return err
      var changed = false
      var targets: HashSet[string]
      fn.body.collectCallTargets(targets)
      for t in targets:
        let callee = procs.getOrDefault(t)
        if callee != nil and not callee.isKernel and callee.hasCrossingPointerParam:
          changed = true
      if not changed:
        break
  var kept: seq[CirProc]
  for fn in m.procs:
    if not fn.isKernel and fn.hasCrossingPointerParam:
      # every call site inlined, the signature has no WGSL form
      continue
    kept.add fn
  m.procs = kept
  return ""

