# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## WGSL lowering: per-target transforms for the WGSL storage model,
## registered by the WebGPU pipeline and run before printing.
##
##   kernel body → explodeGlobalStructWithPtrFields → wgslExplodePointerLocals → wgslHoistProgramScopeLocals → wgslInlinePointerParamCalls
##   → printed WGSL source
##
## - `explodeGlobalStructWithPtrFields`: kernel parameters of struct type
##   carrying pointer fields expand into flat ptr/scalar leaf parameters
##   (WebGPU is a flat resource target)
## - `wgslExplodePointerLocals`: pointer locals rewrite to their pointee,
##   pointer-holding struct locals into per-field leaf rewrites,
##   reads through an alias rewrite to the pointee or leaf symbol
## - `wgslHoistProgramScopeLocals`: `{.smem.}` and `{.const_mem.}` kernel
##   locals relocate into `CirModule.globals` in kernel-body order,
##   the printer emits them at module scope, top-level kernel-body locals only
##
## `wgslInlinePointerParamCalls` runs last:
##
## - a statement call to a device function inlines the callee at the call site
## - the trigger is a callee parameter of buffer-reference type, or a pointer
##   parameter outside the function address space
## - the inlined callee drops from the module
##
## Kernel and device-function pointer parameters are untouched in the pass,
## the printer emits `ptr<function, T>` parameters as-is.
##
## Refusal homes:
## - a `caDevice`-space local refuses at the printer, the pass does not touch
##   device-space locals (WGSL locals are function or workgroup)
## - a threadgroup/constant local surviving in a nested scope or device
##   function refuses at the printer, the relocation covers top-level
##   kernel-body locals only
## - a const_mem local without an initializer refuses in the pass,
##   the printer keeps the shape guard on the relocated declarations
## - a pointer local read as a value or before its address-of binding,
##   a pointer local rebound to a second pointee, an exploded struct local
##   used whole or read through an unbound field path
##   refuse in the pass

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./passes_lowering_cfamily
import ./passes_low_pointers
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
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
# Pointer-param call inlining
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
  ## One substitution pass covers every parameter: the spliced argument is not
  ## re-walked, its symbols belong to the caller.
  ## Expressions are refs, the rewrite applies to the node every sharer sees.
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
  ## A value-position call to a callee carrying a crossing pointer has no
  ## statement form to splice into, so the call refuses.
  if e.isNil:
    return ""
  if e.kind == eCall and e.callee.kind == eSym:
    let callee = procs.getOrDefault(e.callee.name)
    if callee != nil and not callee.isKernel and callee.hasCrossingPointerParam:
      return refuseAt(fn, "device function " & callee.name &
        " takes a pointer that cannot cross a WGSL function signature," &
        " a value-position call has no statement form to inline into")
  for ch in getExprChildren(e):
    let err = refuseValuePositionCalls(fn, ch, procs)
    if err.len > 0:
      return err
  return ""

proc inlineInStmts(fn: CirProc, ss: var seq[CirStmt], procs: Table[string, CirProc]): string =
  var i = 0
  while i < ss.len:
    let s = ss[i]
    if s.kind == sCall and s.callCallee.kind == eSym:
      let callee = procs.getOrDefault(s.callCallee.name)
      if callee != nil and not callee.isKernel and callee.hasCrossingPointerParam:
        if callee.ret.kind != ctVoid:
          return refuseAt(fn, "device function " & callee.name &
            " takes a pointer that cannot cross a WGSL function signature" &
            " and returns a value, only void device functions inline")
        var body = cloneStmts(callee.body)
        if body.len > 0 and body[^1].kind == sRet and body[^1].retVal == nil:
          # an implicit fall-through return of the frontend, dropping it is
          # semantics-preserving at the splice position
          body.setLen(body.len - 1)
        if body.containsReturn:
          return refuseAt(fn, "device function " & callee.name &
            " takes a pointer that cannot cross a WGSL function signature," &
            " its body must not return, a spliced return would exit the caller")
        for a in s.callArgs:
          if a.containsCall:
            return refuseAt(fn, "call argument of " & callee.name &
              " contains a call, inlining would re-evaluate it")
        var seen: HashSet[string]
        if callee.callsCrossingCalleeTransitively(procs, seen):
          return refuseAt(fn, "device function " & callee.name &
            " takes a pointer that cannot cross a WGSL function signature" &
            " and reaches one through its calls, inlining would not terminate")
        var subst = initTable[string, CirExpr]()
        for j, p in callee.params:
          subst[p.name] = s.callArgs[j]
        body.substituteArgsStmts(subst)
        ss = ss[0 ..< i] & body & ss[i + 1 .. ^1]
        continue
      for a in s.callArgs:
        let aerr = refuseValuePositionCalls(fn, a, procs)
        if aerr.len > 0:
          return aerr
    else:
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
    i.inc
  return ""

proc inlinePointerParamCalls(m: CirModule): string =
  ## Placement contract:
  ## - the pass runs after the pointer-local explosion pass, spliced bodies
  ##   therefore carry no pointer locals
  ## - the pass runs after the per-address-space callee instantiation, a formal
  ##   re-spaced by its call sites refuses here exactly when the re-spaced
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
# Pointer-local explosion
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
        return refuseAt(fn, "pointer local " & op.name &
          " read before its address-of binding")
      if op.name in ctx.structLocals:
        return refuseAt(fn, "deref of exploded struct local " & op.name &
          ", WGSL has no pointer struct members")
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
      return refuseAt(fn, "pointer local " & s.name &
        " without an address-of binding, WGSL has no pointer locals")
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
      return refuseAt(fn, "pointer local " & s.lhs.name &
        " rebound, WGSL has no pointer values")
    if s.rhs.kind != eAddr or s.rhs.addrOperand.kind != eSym:
      return refuseAt(fn, "pointer local " & s.lhs.name &
        " rebound to a non-address value, WGSL has no pointer values")
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
  # WGSL keeps pointer parameters untouched: the pass needs no by-reference
  # context and no callee table
  ExplodeCtx()
