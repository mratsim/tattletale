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
##   kernel body → explodeGlobalStructWithPtrFields → wgslExplodePointerLocals → wgslRewriteBoolKernelParamReads → wgslHoistProgramScopeLocals → wgslInlinePointerParamCalls
##   → wgslScanStructRefs → printed WGSL source
##
## - `explodeGlobalStructWithPtrFields`: kernel parameters of struct type
##   carrying pointer fields expand into flat ptr/scalar leaf parameters
##   (WebGPU is a flat resource target)
## - `wgslExplodePointerLocals`: pointer locals rewrite to their pointee,
##   pointer-holding struct locals into per-field leaf rewrites,
##   reads through an alias rewrite to the pointee or leaf symbol
## - `wgslRewriteBoolKernelParamReads`: reads of the kernel's bool parameters
##   rewrite to `name != 0u` comparisons, the kernel body only
##   (bool is not host-shareable, the storage binding declares u32)
## - `wgslHoistProgramScopeLocals`: `{.smem.}` and `{.const_mem.}` kernel
##   locals relocate into `CirModule.globals` in kernel-body order,
##   the printer emits them at module scope, top-level kernel-body locals only
##
## `wgslScanStructRefs` fills `CirModule.structRefs` with the struct names
## that the printer declares: parameter type trees, declaration type trees,
## struct names the emitted WGSL text names, and the struct-member closure.
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
## - a smem local with an initializer or a const_mem local without one
##   refuses in the pass (WGSL workgroup variables are zero-initialized),
##   the printer keeps both shape guards on the relocated declarations
## - a pointer local read as a value or before its address-of binding,
##   a pointer local rebound to a second pointee, an exploded struct local
##   used whole or read through an unbound field path
##   refuse in the pass
##
## Shapes that reach the printer untouched keep only the type-based
## pointer-local refusals there.

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
  ## Registers the WGSL pointer-local explosion pass over the shared walk.
  registerExplodePointerLocals(reg, "wgslExplodePointerLocals",
    "Rewrites pointer locals to their pointees and pointer-holding struct" &
    " locals to their leaf symbols",
    newExplodeCtx, explodeVarDecl, explodeAsgn,
    explodeDropsVar, explodeDropsAsgn, rewritePointerExpr)

func registerWgslRewriteBoolKernelParamReads*(reg: var PassRegistry) =
  ## Registers the bool kernel-parameter read rewrite: a const_mem initializer
  ## reading a bool kernel parameter is patched before the hoist relocates it.
  reg.register("wgslRewriteBoolKernelParamReads", pkTransform,
    "Rewrites reads of the kernel's bool parameters to name != 0u comparisons",
    run = rewriteBoolKernelParamReads
  )

func registerWgslHoistProgramScopeLocals*(reg: var PassRegistry) =
  ## Registers the WGSL program-scope hoist: the rewritten aliases plus
  ## the relocated declarations use the renamed names that their body already references.
  registerCfamilyHoistProgramScopeLocals(reg, "wgslHoistProgramScopeLocals",
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope", smemInitRefuse,
    "WGSL workgroup variables are zero-initialized")

func registerWgslInlinePointerParamCalls*(reg: var PassRegistry) =
  ## Registers the pointer-param call inlining pass.
  reg.register("wgslInlinePointerParamCalls", pkTransform,
    "Inlines calls whose pointer arguments cannot cross a WGSL function" &
    " signature",
    run = inlinePointerParamCalls)

# ═══════════════════════════════════════════════════════════════════════
# Pointer-param call inlining
# ═══════════════════════════════════════════════════════════════════════

proc crossesWgslSignature(t: CirType): bool =
  ## Returns true when the parameter type has no WGSL function-parameter form.
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
  ## Returns true when a `return` statement appears in the statement list:
  ##
  ## - nested bodies count
  ## - the frontend's trailing `return` of a void body does not count, the caller drops it before splicing
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
  ## One substitution pass covers every parameter: the spliced argument is not re-walked, its symbols belong to the caller.
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
  ## Statement-walk overload of `substituteArgsExpr`, nested bodies included.
  ##
  ## The walk stays hand-rolled. A substitution splices the call argument
  ## into the symbol node and must not re-walk the spliced content: its symbols belong to the caller.
  ## A shared walk descends into the rewritten node and would substitute a second parameter inside the argument.
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
  ## statement form to splice into, so the call refuses. Returns "" when clean.
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
  ## Splices one inlinable call into the statement list at its position, then recurses into the nested bodies.
  ## Returns "" when clean, else the refusal.
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
  ## Inlines the statement calls whose callee takes a non-crossable pointer:
  ##
  ## - a non-crossable pointer is a buffer reference, or a pointer whose address space is not the function space
  ## - the callee body splices over the call with every argument substituted
  ##   at its parameter's uses
  ## - the inlined callee drops from the module
  ## - calls whose pointer arguments all live in the function space keep real calls
  ##
  ## Returns "" when clean, else the first refusal.
  ##
  ## Placement contract:
  ## - the pass runs after the pointer-local explosion pass
  ## - spliced bodies therefore carry no pointer locals
  ## - the pass runs after the per-address-space callee instantiation
  ## - a formal re-spaced by its call sites refuses here exactly when the re-spaced signature could not print
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

proc rewritePointerExpr(fn: CirProc, ctx: var ExplodeCtx, e: CirExpr): string =
  ## Rewrites one expression in place for the WGSL pointer model.
  ## Returns "" when clean, else the first refusal.
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in ctx.structLocals:
      return refuseAt(fn, "struct local " & e.name &
        " used whole, WGSL has no pointer struct members, read the fields")
    if e.name in ctx.pendingPtrs:
      return refuseAt(fn, "pointer local " & e.name &
        " read before its address-of binding")
    if e.name in ctx.aliasOf:
      return refuseAt(fn, "pointer local " & e.name &
        " used as a value, WGSL has no pointer values")
    return ""
  of eDot:
    # an exact bound path splices to its leaf symbol here, one node rewrite
    # replaces the whole dotted read
    let root = structRoot(e)
    if root != nil and root.name in ctx.structLocals:
      let path = dottedPath(e)
      if ctx.leafOf.hasKey(path):
        let leaf = ctx.leafOf[path]
        e.spliceSym(leaf.sym, leaf.typ)
        return ""
    let err = rewritePointerExpr(fn, ctx, e.dotBase)
    if err.len > 0:
      return err
    return ""
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym:
      if op.name in ctx.aliasOf:
        # pointee stands in for the pointer, the deref unwraps to it
        e.spliceSym(ctx.aliasOf[op.name], op.typ.pElem)
        return ""
      if op.name in ctx.pendingPtrs:
        return refuseAt(fn, "pointer local " & op.name &
          " read before its address-of binding")
      if op.name in ctx.structLocals:
        return refuseAt(fn, "deref of exploded struct local " & op.name &
          ", WGSL has no pointer struct members")
    return rewritePointerExpr(fn, ctx, op)
  of eAddr:
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""
  else:
    for ch in getExprChildren(e):
      let err = rewritePointerExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""

proc explodeVarDecl(fn: CirProc, ctx: var ExplodeCtx, s: CirStmt): string =
  ## `sVar` case: a pointer local pends for its address-of binding or binds its pointee,
  ## a pointer-holding struct local registers its field rewrites,
  ## anything else rewrites the initializer.
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
  ## `sAsgn` case: a pointer local binds exactly once, a pointer parameter
  ## of the proc in flight keeps its assignment untouched.
  ## Anything else rewrites both operands.
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
  ## A pointer local and an exploded struct local drop from their list.
  s.declType.kind in {ctPtr, ctAptr} or
    (s.declType.kind == ctStruct and s.declType.containsPtrDeep)

proc explodeDropsAsgn(s: CirStmt, ctx: ExplodeCtx): bool =
  ## A pointer-local binding drops from its list, a pointer parameter's
  ## assignment stays in place.
  s.lhs.kind == eSym and s.lhs.typ != nil and
    s.lhs.typ.kind in {ctPtr, ctAptr} and
    (s.lhs.name in ctx.pendingPtrs or s.lhs.name in ctx.aliasOf)

proc newExplodeCtx(m: CirModule, fn: CirProc): ExplodeCtx =
  ExplodeCtx()


# ═══════════════════════════════════════════════════════════════════════
# Bool kernel parameter reads
# ═══════════════════════════════════════════════════════════════════════

proc rewriteBoolReadsExpr(e: CirExpr, boolParams, boolPointees: HashSet[string]) =
  ## Replaces every read of a bool kernel parameter in place with the `name != 0u` comparison in its type-preserving bool form.
  ##
  ## Expressions are refs, so the rewrite applies to every sharer and checks
  ## the symbol's own type: a proc-typed callee position never fires
  ## and a bool-typed symbol is never a callee. The comparison's operand is
  ## a fresh symbol. The rewrite returns before the walk re-enters it.
  ##
  ## `boolPointees` stays empty, WGSL collects no bool-pointee storage.
  if e.isNil:
    return
  if e.kind == eSym and e.typ != nil and e.typ.kind == ctBool and
      e.name in boolParams:
    let paramType = e.typ
    let param = sym(e.name, paramType)
    e.kind = eNeq
    e.cmpType = paramType
    e.a = param
    e.b = intLit(0, tUint(32))
    return
  for ch in getExprChildren(e):
    rewriteBoolReadsExpr(ch, boolParams, boolPointees)

proc collectBoolKernelParams(fn: CirProc, vals, ptrs: var HashSet[string]) =
  ## Collects the kernel's bool parameter names. The storage keeps its type. `ptrs` stays empty, WGSL collects no bool-pointee storage.
  for p in fn.params:
    if p.typ.kind == ctBool:
      vals.incl(p.name)

proc rewriteBoolKernelParamReads(m: CirModule): string =
  ## Rewrites every read of a bool kernel parameter in the kernel body into
  ## a `name != 0u` comparison. Device functions are untouched, their own bool
  ## parameters and locals stay real bools. Returns "" when clean.
  ##
  ## WGSL keeps bool storage: no retype, no store rewrite, only the kernel bodies walk.
  m.patchBoolStorage(BoolPatchPolicy(
    walks: proc(fn: CirProc): bool = fn.isKernel,
    collectParams: collectBoolKernelParams,
    rewriteRead: rewriteBoolReadsExpr))

# ═══════════════════════════════════════════════════════════════════════
# Struct-reference registration
# ═══════════════════════════════════════════════════════════════════════

proc registerTypeStructs(needed: var HashSet[string], t: CirType) =
  ## Registers the struct names reachable from `t`: object fields, array elements, vector elements, pointer targets.
  var refs = needed
  proc inclStruct(tt: CirType) =
    if tt.kind == ctStruct:
      refs.incl(tt.name)
  walkTypeTree(t, inclStruct)
  needed = refs

proc scanBareTypeStructs(needed: var HashSet[string], t: CirType) =
  ## Registers the struct names one emitted WGSL type name covers:
  ##
  ## - an array's element name recurses
  ## - a struct name registers
  ## - pointer kinds print no type name themselves
  case t.kind
  of ctArray:
    scanBareTypeStructs(needed, t.aElem)
  of ctStruct:
    needed.incl(t.name)
  else:
    discard

proc scanExprStructs(needed: var HashSet[string], e: CirExpr) =
  ## Registers the struct names one expression node's printed form names:
  ## a cast or conversion target type, a constructor type.
  ##
  ## The scanner walks the children with `walkExprs`.
  if e.isNil:
    return
  case e.kind
  of eCast, eConv:
    scanBareTypeStructs(needed, e.castType)
  of eOconstr:
    scanBareTypeStructs(needed, e.ocType)
  of eAconstr:
    scanBareTypeStructs(needed, e.acType)
  else:
    discard

proc scanStmtStructs(needed: var HashSet[string], ss: seq[CirStmt]) =
  ## Registers the struct names the printed forms of one statement list name.
  ##
  ## - a `var` declaration registers its type only where the printer emits
  ##   the declaration: a relocated space survivor or a pointer-shaped
  ##   local refuses before any type name prints. The shared walk skips
  ##   its initializer with the declaration
  var refs = needed
  proc onDecl(s: CirStmt): bool =
    if s.kind == sVar:
      if s.space in {caThreadgroup, caConstant, caDevice} or
          s.declType.kind in {ctPtr, ctAptr} or
          (s.declType.kind == ctStruct and s.declType.containsPtrDeep):
        return false
    scanBareTypeStructs(refs, s.declType)
    true
  proc onExpr(e: CirExpr) =
    scanExprStructs(refs, e)
  ss.scanStmtRefs(onDecl, onExpr)
  needed = refs

proc closeStructRefs(byName: Table[string, CirType], needed: var HashSet[string], name: string) =
  ## Adds the struct-typed members of `name`'s declaration, recursively:
  ##
  ## - array dimensions unwrap, a struct member behind an array registers
  ## - a name without a declaration in the module expands to nothing
  let t = byName.getOrDefault(name, nil)
  if t == nil:
    return
  for f in t.fields:
    var ft = f.typ
    while ft != nil and ft.kind == ctArray:
      ft = ft.aElem
    if ft != nil and ft.kind == ctStruct and ft.name notin needed:
      needed.incl(ft.name)
      closeStructRefs(byName, needed, ft.name)

proc wgslScanStructRefs(m: CirModule): string =
  ## Registers the struct names the printer declares into `m.structRefs`:
  ##
  ## The registered names:
  ##
  ## - the type trees of the kernel's parameters and the relocated
  ##   program-scope declarations
  ## - the closure over struct-typed members, array dimensions unwrapped
  ##
  ## The emitted WGSL text names these sites:
  ##
  ## - local declaration types
  ## - cast and conversion target types
  ## - constructor types
  ## - signature types: device-function parameters and returns, pointer-parameter pointees
  ##
  ## The pass runs after the last transform: the walked declarations and bodies are final.
  ## The pass registry never invalidates a stored record. A module that lacks exactly one
  ## kernel contributes no kernel-side registrations. Such a module refuses at the printer.
  let kernel = m.singleKernel()
  if kernel != nil:
    for p in kernel.params:
      registerTypeStructs(m.structRefs, p.typ)
  for s in m.globals:
    if s.kind in {sVar, sConst}:
      registerTypeStructs(m.structRefs, s.declType)
  for fn in m.procs:
    if fn.isKernel:
      continue
    scanStmtStructs(m.structRefs, fn.body)
    for p in fn.params:
      case p.typ.kind
      of ctPtr:
        scanBareTypeStructs(m.structRefs, p.typ.pElem)
      of ctAptr:
        discard
      else:
        scanBareTypeStructs(m.structRefs, p.typ)
    if fn.ret.kind != ctVoid:
      scanBareTypeStructs(m.structRefs, fn.ret)
  if kernel != nil:
    scanStmtStructs(m.structRefs, kernel.body)
  var byName: Table[string, CirType]
  for t in m.types:
    if t.kind == ctStruct:
      byName[t.name] = t
  var pending: seq[string]
  for name in m.structRefs:
    pending.add name
  for name in pending:
    closeStructRefs(byName, m.structRefs, name)
  return ""

proc registerWgslStructRefPass*(reg: var PassRegistry) =
  ## Registers the WGSL struct-reference analysis pass:
  ##
  ## - the pass fills `CirModule.structRefs` with the struct names the printer declares, closure included
  ## - the printer declares exactly the stored names, it never scans for struct references itself
  reg.register("wgslScanStructRefs", pkAnalysis,
    "Registers the struct names the printer declares, the closure over" &
    " struct-typed members included",
    run = wgslScanStructRefs
  )
