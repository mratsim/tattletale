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
    newExplodeCtx, explodeVarDecl, explodeAsgn,
    explodeDropsVar, explodeDropsAsgn, rewritePointerExpr)

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

proc containsCall(e: CirExpr): bool =
  if e.isNil:
    return false
  if e.kind == eCall:
    return true
  for ch in getExprChildren(e):
    if ch.containsCall:
      return true
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

proc inlineInStmts(fn: CirProc, ss: var seq[CirStmt], procs: Table[string, CirProc]): string
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

proc inlineCallSite(fn: CirProc, ss: var seq[CirStmt], i: int, procs: Table[string, CirProc]): tuple[err: string, spliced: bool] =
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
    for a in s.callArgs:
      if a.containsCall:
        return (refuseAt(fn, "call argument of " & callee.name & " contains a call, inlining would re-evaluate it"), false)
    var seen: HashSet[string]
    if callee.callsCrossingCalleeTransitively(procs, seen):
      return (refuseAt(fn, "inlining " & callee.name & " reaches another crossing callee, its pointer parameter cannot cross a WGSL signature"), false)
    var subst = initTable[string, CirExpr]()
    for j, p in callee.params:
      subst[p.name] = s.callArgs[j]
    body.substituteArgsStmts(subst)
    ss = ss[0 ..< i] & body & ss[i + 1 .. ^1]
    return ("", true)
  for a in s.callArgs:
    let aerr = refuseValuePositionCalls(fn, a, procs)
    if aerr.len > 0:
      return (aerr, false)
  return ("", false)

proc walkValuePositions(fn: CirProc, s: CirStmt, procs: Table[string, CirProc]): string =
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
    let terr = inlineInStmts(fn, s.thenBody, procs)
    if terr.len > 0:
      return terr
    let eerr = inlineInStmts(fn, s.elseBody, procs)
    if eerr.len > 0:
      return eerr
  of sWhile:
    let cerr = refuseValuePositionCalls(fn, s.whileCond, procs)
    if cerr.len > 0:
      return cerr
    let werr = inlineInStmts(fn, s.whileBody, procs)
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

proc inlineInStmts(fn: CirProc, ss: var seq[CirStmt], procs: Table[string, CirProc]): string =
  var i = 0
  while i < ss.len:
    let s = ss[i]
    if s.kind == sCall and s.callCallee.kind == eSym:
      let (r, spliced) = inlineCallSite(fn, ss, i, procs)
      if r.len > 0:
        return r
      if not spliced:
        i.inc
      continue
    let r = walkValuePositions(fn, s, procs)
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
    while true:
      let err = inlineInStmts(fn, fn.body, procs)
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

# ═══════════════════════════════════════════════════════════════════════
#  Pointer-local explosion
# ═══════════════════════════════════════════════════════════════════════

proc rewritePointerExpr(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string
  ## Forward declaration, expressions recurse into themselves.

proc wgslPointerArm(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  case e.kind
  of eDot:
    let err = rewritePointerExpr(fn, ctx, e.dotBase)
    if err.len > 0:
      return err
    return ""
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym:
      if op.name in ctx.pendingPtrs:
        return refuseAt(fn, "pointer local " & op.name & " reads ahead of its address-of binding")
      if op.name in ctx.structLocals:
        return refuseAt(fn, "deref of exploded struct local " & op.name & ", WGSL has no pointer struct members")
    return rewritePointerExpr(fn, ctx, op)
  else:
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""

proc rewritePointerExpr(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  rewritePointerExprShared(fn, ctx, e, "WGSL", wgslPointerArm)

proc explodeVarDecl(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt): string =
  if s.declType.kind in {ctPtr, ctAptr}:
    if s.init == nil:
      ctx.pendingPtrs.incl(s.name)
    elif s.init.kind == eAddr and s.init.addrOperand.kind == eSym:
      ctx.aliasOf[s.name] = s.init.addrOperand.name
    else:
      return refuseAt(fn, "pointer local " & s.name & " binds no address, WGSL locals are values")
    return ""
  if s.declType.kind == ctStruct and s.declType.containsPtrDeep:
    return explodeStructVarDecl(fn, ctx, s, "WGSL")
  return rewritePointerExpr(fn, ctx, s.init)

proc explodeAsgn(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt): string =
  if s.lhs.kind == eSym and s.lhs.typ != nil and
      s.lhs.typ.kind in {ctPtr, ctAptr}:
    if s.lhs.name notin ctx.pendingPtrs and s.lhs.name notin ctx.aliasOf:
      # a pointer parameter of the proc in flight, never touched by the pass
      let rerr = rewritePointerExpr(fn, ctx, s.rhs)
      if rerr.len > 0:
        return rerr
      return ""
    if s.lhs.name in ctx.aliasOf:
      return refuseAt(fn, "pointer local " & s.lhs.name & " rebinds, WGSL keeps one binding")
    if s.rhs.kind != eAddr or s.rhs.addrOperand.kind != eSym:
      return refuseAt(fn, "pointer local " & s.lhs.name & " rebinds to a non-address value, WGSL keeps one binding")
    ctx.pendingPtrs.excl(s.lhs.name)
    ctx.aliasOf[s.lhs.name] = s.rhs.addrOperand.name
    return ""
  let lerr = rewritePointerExpr(fn, ctx, s.lhs)
  if lerr.len > 0:
    return lerr
  return rewritePointerExpr(fn, ctx, s.rhs)

proc explodeDropsVar(s: CirStmt, ctx: ExplodeCtx): bool =
  s.declType.kind in {ctPtr, ctAptr} or
    (s.declType.kind == ctStruct and s.declType.containsPtrDeep)

proc explodeDropsAsgn(s: CirStmt, ctx: ExplodeCtx): bool =
  s.lhs.kind == eSym and s.lhs.typ != nil and
    s.lhs.typ.kind in {ctPtr, ctAptr} and
    (s.lhs.name in ctx.pendingPtrs or s.lhs.name in ctx.aliasOf)

proc newExplodeCtx(m: CirModule, fn: CirProc): ExplodeCtx =
  # WGSL keeps pointer parameters untouched: the pass never emits a by-reference
  # context and no callee table
  ExplodeCtx()
