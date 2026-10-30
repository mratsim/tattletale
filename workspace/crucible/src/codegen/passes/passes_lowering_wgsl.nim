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
##   kernel body → explodeKernelStructs → wgslDissolvePointerLocals → wgslRewriteBoolKernelParamReads → wgslHoistProgramScopeLocals → printed WGSL source
##
## - `explodeKernelStructs`: kernel parameters of struct type with pointer fields
##   expand into flat ptr/scalar leaf parameters, WebGPU is a flat resource target
## - `wgslDissolvePointerLocals`: pointer locals dissolve into their pointee,
##   pointer-holding struct locals into per-field leaf rewrites,
##   reads through an alias rewrite to the pointee or leaf symbol
## - `wgslRewriteBoolKernelParamReads`: reads of the kernel's bool parameters
##   rewrite to `name != 0u` comparisons, the kernel body only
##   (bool is not host-shareable, the storage binding declares u32)
## - `wgslHoistProgramScopeLocals`: `{.smem.}` and `{.const_mem.}` kernel
##   locals relocate into `CirModule.globals` in kernel-body order,
##   the printer emits them at module scope, top-level kernel-body locals only
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
##   a pointer local rebound to a second pointee, a dissolved struct local
##   used whole or read through an unbound field path
##   refuse in the pass
##
## Shapes that reach the printer untouched keep only the type-based
## pointer-local refusals there.

import std/[sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./passes_low_kernel_params
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerWgslLoweringPasses*(reg: var PassRegistry) =
  ## Registers the WGSL per-target lowering passes:
  ##
  ## - the kernel struct-argument explosion, WebGPU is a flat resource target
  ## - `wgslDissolvePointerLocals`, `wgslRewriteBoolKernelParamReads`, `wgslHoistProgramScopeLocals`, in that order, before printing
  ##
  ## Registration contract: the passes sit after the reserved-word rename.
  ## The dissolved aliases and the relocated declarations use the renamed names
  ## that their body already references, and a const_mem initializer reading a bool
  ## kernel parameter is patched before the relocation moves it out of the body.
  registerExplodeKernelStructs(reg)
  reg.register("wgslDissolvePointerLocals", pkTransform, phaseMain,
    "Dissolves pointer locals and pointer-holding struct locals into their" &
    " pointees and leaf symbols",
    run = dissolvePointerLocals
  )
  reg.register("wgslRewriteBoolKernelParamReads", pkTransform, phaseMain,
    "Rewrites reads of the kernel's bool parameters to name != 0u comparisons",
    run = rewriteBoolKernelParamReads
  )
  reg.register("wgslHoistProgramScopeLocals", pkTransform, phaseMain,
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope",
    run = hoistProgramScopeLocals
  )

# ═══════════════════════════════════════════════════════════════════════
# Pointer dissolution
# ═══════════════════════════════════════════════════════════════════════

type
  DissolveLeaf = tuple[sym: string, typ: CirType]
    ## One dissolved field read: the leaf symbol standing in for it plus
    ## the leaf type.

  DissolveCtx = object
    pendingPtrs: HashSet[string]
    ## Pointer locals awaiting their address-of binding, a read refuses.
    aliasOf: Table[string, string]
    ## Pointer local -> pointee local. One binding dissolves the pair, a rebinding refuses.
    structLocals: HashSet[string]
    ## Pointer-holding struct locals dissolved into field rewrites, a whole-value use refuses.
    leafOf: Table[string, DissolveLeaf]
    ## Dotted field path of a dissolved struct local -> the leaf symbol bound
    ## by the object construction: `v.data` -> the symbol bound to `data`,
    ## `v.inner.f` -> the symbol bound to the nested `f`.

proc dissolveFail(fn: CirProc, detail: string): string =
  "at " & fn.name & ": " & detail

proc toPlainSym(name: string, typ: CirType): CirExpr =
  ## Every dissolved read rewrites to the symbol this proc returns.
  result = CirExpr(kind: eSym, name: name, typ: typ, builtin: false,
    coordBuiltin: cbkNone, synchroBuiltin: sbkNone,
    reductionBuiltin: rbkNone, atomicsBuiltin: abkNone)

proc spliceSym(e: CirExpr, leaf: DissolveLeaf) =
  ## Splices one expression node into a plain symbol reference, in place.
  ## Expressions are refs, the rewrite applies to the node every sharer sees.
  e.kind = eSym
  e.name = leaf.sym
  e.typ = leaf.typ
  e.builtin = false
  e.coordBuiltin = cbkNone
  e.synchroBuiltin = sbkNone
  e.reductionBuiltin = rbkNone
  e.atomicsBuiltin = abkNone

proc structRoot(e: CirExpr): CirExpr =
  ## Symbol at the root of a dotted chain, nil when the expression is
  ## neither a symbol nor a dotted chain over one.
  result = e
  while result != nil and result.kind == eDot:
    result = result.dotBase
  if result != nil and result.kind != eSym:
    result = nil

proc dottedPath(e: CirExpr): string =
  ## Dotted field path of a chain rooted at a symbol, "" otherwise:
  ## `v` -> `v`, `v.inner.f` -> `v.inner.f`.
  if e.kind == eSym:
    return e.name
  if e.kind == eDot:
    let base = dottedPath(e.dotBase)
    if base.len > 0:
      return base & "." & e.fieldName
  return ""

proc registerDissolvedStruct(fn: CirProc, ctx: var DissolveCtx, path: string, t: CirType, c: CirExpr): string =
  ## Records the field rewrites of one pointer-holding struct local whose
  ## object construction binds leaf symbols, nested constructions recurse.
  ## Returns "" when clean, else the refusal.
  if c.kind != eOconstr:
    return dissolveFail(fn, "struct local " & path &
      " initializer is not a construction, WGSL has no pointer struct members")
  for kvv in c.kvFields:
    var ft: CirType = nil
    for f in t.fields:
      if f.name == kvv.name:
        ft = f.typ
    if ft.isNil:
      return dissolveFail(fn, "field " & kvv.name & " is not on " & t.name)
    if kvv.value.kind == eOconstr:
      let err = registerDissolvedStruct(fn, ctx, path & "." & kvv.name, ft, kvv.value)
      if err.len > 0:
        return err
    elif kvv.value.kind == eSym:
      ctx.leafOf[path & "." & kvv.name] = (kvv.value.name, kvv.value.typ)
    else:
      return dissolveFail(fn, "field " & path & "." & kvv.name &
        " initializer is not a leaf symbol, WGSL has no pointer struct members")
  return ""

proc rewriteDissolveExpr(fn: CirProc, ctx: var DissolveCtx, e: CirExpr): string =
  ## Rewrites one expression in place for the WGSL pointer model.
  ## Returns "" when clean, else the first refusal.
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in ctx.structLocals:
      return dissolveFail(fn, "struct local " & e.name &
        " used whole, WGSL has no pointer struct members, read the fields")
    if e.name in ctx.pendingPtrs:
      return dissolveFail(fn, "pointer local " & e.name &
        " read before its address-of binding")
    if e.name in ctx.aliasOf:
      return dissolveFail(fn, "pointer local " & e.name &
        " used as a value, WGSL has no pointer values")
    return ""
  of eDot:
    # an exact bound path splices to its leaf symbol here, one node rewrite
    # replaces the whole dotted read
    let root = structRoot(e)
    if root != nil and root.name in ctx.structLocals:
      let path = dottedPath(e)
      if ctx.leafOf.hasKey(path):
        e.spliceSym(ctx.leafOf[path])
        return ""
    let err = rewriteDissolveExpr(fn, ctx, e.dotBase)
    if err.len > 0:
      return err
    return ""
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym:
      if op.name in ctx.aliasOf:
        # pointee stands in for the pointer, the deref unwraps to it
        e.spliceSym((ctx.aliasOf[op.name], op.typ.pElem))
        return ""
      if op.name in ctx.pendingPtrs:
        return dissolveFail(fn, "pointer local " & op.name &
          " read before its address-of binding")
      if op.name in ctx.structLocals:
        return dissolveFail(fn, "deref of dissolved struct local " & op.name &
          ", WGSL has no pointer struct members")
    return rewriteDissolveExpr(fn, ctx, op)
  of eAddr:
    for ch in getExprChildren(e):
      let err = rewriteDissolveExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""
  else:
    for ch in getExprChildren(e):
      let err = rewriteDissolveExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""

proc rewriteDissolveStmts(fn: CirProc, ctx: var DissolveCtx, ss: var seq[CirStmt]): string
  ## Forward declaration, the structured statement bodies recurse.

proc rewriteDissolveStmt(fn: CirProc, ctx: var DissolveCtx, s: CirStmt): string =
  ## Rewrites one statement in place. Returns "" when clean, else the refusal.
  case s.kind
  of sVar:
    if s.declType.kind in {ctPtr, ctAptr}:
      if s.init == nil:
        ctx.pendingPtrs.incl(s.name)
      elif s.init.kind == eAddr and s.init.addrOperand.kind == eSym:
        ctx.aliasOf[s.name] = s.init.addrOperand.name
      else:
        return dissolveFail(fn, "pointer local " & s.name &
          " without an address-of binding, WGSL has no pointer locals")
      return ""
    if s.declType.kind == ctStruct and s.declType.containsPtrDeep:
      if s.init == nil or s.init.kind != eOconstr:
        return dissolveFail(fn, "struct local " & s.name &
          " without a construction, WGSL has no pointer struct members")
      let err = registerDissolvedStruct(fn, ctx, s.name, s.declType, s.init)
      if err.len > 0:
        return err
      ctx.structLocals.incl(s.name)
      return ""
    return rewriteDissolveExpr(fn, ctx, s.init)
  of sConst:
    return rewriteDissolveExpr(fn, ctx, s.init)
  of sAsgn:
    if s.lhs.kind == eSym and s.lhs.typ != nil and
        s.lhs.typ.kind in {ctPtr, ctAptr}:
      if s.lhs.name notin ctx.pendingPtrs and s.lhs.name notin ctx.aliasOf:
        # a pointer parameter of the proc in flight, the pass never touches it
        let rerr = rewriteDissolveExpr(fn, ctx, s.rhs)
        if rerr.len > 0:
          return rerr
        return ""
      if s.lhs.name in ctx.aliasOf:
        return dissolveFail(fn, "pointer local " & s.lhs.name &
          " rebound, WGSL has no pointer values")
      if s.rhs.kind != eAddr or s.rhs.addrOperand.kind != eSym:
        return dissolveFail(fn, "pointer local " & s.lhs.name &
          " rebound to a non-address value, WGSL has no pointer values")
      ctx.pendingPtrs.excl(s.lhs.name)
      ctx.aliasOf[s.lhs.name] = s.rhs.addrOperand.name
      return ""
    let lerr = rewriteDissolveExpr(fn, ctx, s.lhs)
    if lerr.len > 0:
      return lerr
    return rewriteDissolveExpr(fn, ctx, s.rhs)
  of sIte:
    let cerr = rewriteDissolveExpr(fn, ctx, s.cond)
    if cerr.len > 0:
      return cerr
    let terr = rewriteDissolveStmts(fn, ctx, s.thenBody)
    if terr.len > 0:
      return terr
    return rewriteDissolveStmts(fn, ctx, s.elseBody)
  of sLoop:
    return rewriteDissolveStmts(fn, ctx, s.loopBody)
  of sScope:
    return rewriteDissolveStmts(fn, ctx, s.scopeBody)
  of sAssume:
    return rewriteDissolveExpr(fn, ctx, s.assumeCond)
  of sRet:
    return rewriteDissolveExpr(fn, ctx, s.retVal)
  of sDiscard:
    return rewriteDissolveExpr(fn, ctx, s.discardVal)
  of sCall:
    let tmp = CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
    let err = rewriteDissolveExpr(fn, ctx, tmp)
    s.callCallee = tmp.callee
    s.callArgs = tmp.args
    return err
  of sJmp, sLab, sCase, sAsm, sEmit:
    return ""

proc rewriteDissolveStmts(fn: CirProc, ctx: var DissolveCtx, ss: var seq[CirStmt]): string =
  ## Rewrites one statement list in place, pointer-local declarations,
  ## their bindings and dissolved struct-local constructions drop out.
  ## Everything else rewrites where it stands, "" comes back when clean.
  var i = 0
  while i < ss.len:
    let s = ss[i]
    var drops = false
    if s.kind == sVar:
      if s.declType.kind in {ctPtr, ctAptr} or
          (s.declType.kind == ctStruct and s.declType.containsPtrDeep):
        let err = rewriteDissolveStmt(fn, ctx, s)
        if err.len > 0:
          return err
        drops = true
    elif s.kind == sAsgn and s.lhs.kind == eSym and s.lhs.typ != nil and
        s.lhs.typ.kind in {ctPtr, ctAptr} and
        (s.lhs.name in ctx.pendingPtrs or s.lhs.name in ctx.aliasOf):
      let err = rewriteDissolveStmt(fn, ctx, s)
      if err.len > 0:
        return err
      drops = true
    if drops:
      ss.delete(i)
      continue
    let err = rewriteDissolveStmt(fn, ctx, s)
    if err.len > 0:
      return err
    i.inc
  return ""

proc dissolvePointerLocals(m: CirModule): string =
  ## Dissolves the pointer locals of every proc into their pointees, in place.
  ## The walk covers proc bodies only, pointer parameters stay untouched.
  ## Returns "" when clean, else the first refusal.
  for fn in m.procs:
    var ctx = DissolveCtx()
    let err = rewriteDissolveStmts(fn, ctx, fn.body)
    if err.len > 0:
      return err
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Bool kernel parameter reads
# ═══════════════════════════════════════════════════════════════════════

proc rewriteBoolReadsExpr(e: CirExpr, boolParams: HashSet[string]) =
  ## Replaces every read of a bool kernel parameter in place with the `name != 0u` comparison,
  ## the type-preserving bool form.
  ##
  ## Expressions are refs, the rewrite applies to every sharer.
  if e.isNil:
    return
  if e.kind == eSym and e.name in boolParams:
    let paramType = e.typ
    let param = toPlainSym(e.name, paramType)
    e.kind = eNeq
    e.cmpType = paramType
    e.a = param
    e.b = intLit(0, tUint(32))
    return
  for ch in getExprChildren(e):
    rewriteBoolReadsExpr(ch, boolParams)

proc rewriteBoolReadsStmts(ss: var seq[CirStmt], boolParams: HashSet[string]) =
  ## Statement-walk overload of `rewriteBoolReadsExpr`.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      rewriteBoolReadsExpr(s.init, boolParams)
    of sAsgn:
      rewriteBoolReadsExpr(s.lhs, boolParams)
      rewriteBoolReadsExpr(s.rhs, boolParams)
    of sIte:
      rewriteBoolReadsExpr(s.cond, boolParams)
      rewriteBoolReadsStmts(s.thenBody, boolParams)
      rewriteBoolReadsStmts(s.elseBody, boolParams)
    of sLoop:
      rewriteBoolReadsStmts(s.loopBody, boolParams)
    of sScope:
      rewriteBoolReadsStmts(s.scopeBody, boolParams)
    of sAssume:
      rewriteBoolReadsExpr(s.assumeCond, boolParams)
    of sRet:
      rewriteBoolReadsExpr(s.retVal, boolParams)
    of sDiscard:
      rewriteBoolReadsExpr(s.discardVal, boolParams)
    of sCall:
      rewriteBoolReadsExpr(s.callCallee, boolParams)
      for a in mitems(s.callArgs):
        rewriteBoolReadsExpr(a, boolParams)
    of sJmp, sLab, sCase, sAsm, sEmit:
      discard

proc rewriteBoolKernelParamReads(m: CirModule): string =
  ## Rewrites every read of a bool kernel parameter in the kernel body into
  ## a `name != 0u` comparison. Device functions are untouched, their own bool
  ## parameters and locals stay real bools. Returns "" when clean.
  for fn in m.procs:
    if not fn.isKernel:
      continue
    var boolParams: HashSet[string]
    for p in fn.params:
      if p.typ.kind == ctBool:
        boolParams.incl(p.name)
    if boolParams.len == 0:
      continue
    fn.body.rewriteBoolReadsStmts(boolParams)
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Program-scope hoisting
# ═══════════════════════════════════════════════════════════════════════

proc hoistProgramScopeLocals(m: CirModule): string =
  ## Relocates the `{.smem.}` and `{.const_mem.}` locals of every proc into
  ## `m.globals`, in encounter order, dropping them from their bodies.
  ##
  ## Contract:
  ## - the smem relocation covers top-level kernel-body locals only, a nested
  ##   or device-function one stays in place and refuses at the printer
  ## - the const_mem relocation covers every proc and scope depth, the shared
  ##   const_mem walk recurses into the nested scopes at their encounter position,
  ##   the initializer is required and the printer emits it from the relocated node
  ## - a smem local with an initializer refuses, WGSL workgroup variables are zero-initialized
  ## - runs after the reserved-word rename, relocated declaration names match the renamed body references
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.space == caThreadgroup:
        if not fn.isKernel:
          # a device-function smem local stays, WGSL workgroup storage belongs
          # to the kernel, the printer refuses the survivor
          i.inc
          continue
        if s.init != nil:
          return "at " & fn.name & ": smem local " & s.name &
            " with an initializer, WGSL workgroup variables are zero-initialized"
        m.globals.add s
        fn.body.delete i
      elif s.kind == sVar and s.space == caConstant:
        if s.init == nil:
          return "at " & fn.name & ": const_mem local " & s.name &
            " without an initializer"
        m.globals.add s
        fn.body.delete i
      else:
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
