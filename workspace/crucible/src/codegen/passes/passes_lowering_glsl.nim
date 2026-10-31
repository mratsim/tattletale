# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## GLSL lowering: per-target transforms for the GLSL storage model,
## registered by the GLSL pipeline and run before printing.
##
##   kernel body → glslQualifyVarParams → glslDissolvePointerLocals → glslHoistProgramScopeLocals → printed GLSL source
##
## GLSL has no function-storage addressing, a local cannot be addressed.
## Threadgroup and constant-storage locals must live at program scope:
##
## - `glslQualifyVarParams`: `var T` device-function parameters bind as `inout`
## - the qualify pass retypes each marked parameter to its pointee, unwraps
##   the body derefs and rewrites the call sites to pass the plain lvalue
## - `glslDissolvePointerLocals`: pointer locals dissolve into their pointee,
##   reads through the alias rewrite to the pointee symbol
## - `glslHoistProgramScopeLocals`: `{.smem.}` kernel locals and `{.const_mem.}`
##   locals relocate into `CirModule.globals` in encounter order
##
## Kernel and device-function pointer parameters stay for the printer to refuse.
##
## Program-scope `const` storage is visible to all functions, a device function
## reads the constants of the kernels that it serves.
##
## Refusal homes:
## - a pointer read before its address-of binding, a value read of a pointer,
##   or a rebinding refuses in the dissolve pass
## - a device-space local and a nested-scope or device-function threadgroup local
##   refuse at the printer, GLSL locals are private
## - a const_mem local without an initializer refuses in the hoist pass and again
##   at the printer as a shape guard

import std/[sets, tables]
import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers
import ./passes_low_address_spaces

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerGlslLoweringPasses*(reg: var PassRegistry) =
  ## Registers the GLSL per-target lowering passes:
  ## - `glslDissolvePointerLocals` then `glslHoistProgramScopeLocals`, in that order, before printing
  ##
  ## Registration contract: the passes sit after the reserved-word rename, the relocated declarations
  ## use the renamed names that their body already references.
  reg.register("glslQualifyVarParams", pkTransform, phaseMain,
    "Retypes var parameters to their pointee, unwraps their body derefs and" &
    " the call-site address arguments",
    run = qualifyVarParams
  )
  reg.register("glslDissolvePointerLocals", pkTransform, phaseMain,
    "Dissolves pointer locals into their pointee locals",
    run = dissolvePointerLocals
  )
  reg.register("glslHoistProgramScopeLocals", pkTransform, phaseMain,
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope",
    run = hoistProgramScopeLocals
  )
  reg.registerInstantiateAddressSpaceGenerics()

# ═══════════════════════════════════════════════════════════════════════
# var-parameter qualification
# ═══════════════════════════════════════════════════════════════════════

proc splicePlainSym(e: CirExpr, name: string, typ: CirType) =
  ## Splices one expression node into a plain symbol reference, in place.
  ## Expressions are refs, the rewrite applies to the node every sharer sees.
  e.kind = eSym
  e.name = name
  e.typ = typ
  e.builtin = false
  e.coordBuiltin = cbkNone
  e.synchroBuiltin = sbkNone
  e.reductionBuiltin = rbkNone
  e.atomicsBuiltin = abkNone
  e.shuffleBuiltin = shkNone

proc unwrapVarDerefsExpr(e: CirExpr, varParams: HashSet[string]) =
  ## Rewrites one expression tree. Every deref of a `var` parameter
  ## splices to the parameter itself, typed at the pointee the deref read.
  ## Other derefs survive. Nested bodies recurse through the statement overload.
  if e.isNil:
    return
  case e.kind
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym and op.name in varParams and
        op.typ != nil and op.typ.kind == ctPtr:
      e.splicePlainSym(op.name, op.typ.pElem)
    else:
      op.unwrapVarDerefsExpr(varParams)
  else:
    for ch in getExprChildren(e):
      ch.unwrapVarDerefsExpr(varParams)

proc unwrapVarDerefsStmts(ss: var seq[CirStmt], varParams: HashSet[string]) =
  ## Statement-walk overload of `unwrapVarDerefsExpr`, nested bodies included.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      s.init.unwrapVarDerefsExpr(varParams)
    of sAsgn:
      s.lhs.unwrapVarDerefsExpr(varParams)
      s.rhs.unwrapVarDerefsExpr(varParams)
    of sIte:
      s.cond.unwrapVarDerefsExpr(varParams)
      s.thenBody.unwrapVarDerefsStmts(varParams)
      s.elseBody.unwrapVarDerefsStmts(varParams)
    of sLoop:
      s.loopBody.unwrapVarDerefsStmts(varParams)
    of sCase:
      s.selector.unwrapVarDerefsExpr(varParams)
      for br in mitems(s.branches):
        br.body.unwrapVarDerefsStmts(varParams)
      s.caseElse.unwrapVarDerefsStmts(varParams)
    of sScope:
      s.scopeBody.unwrapVarDerefsStmts(varParams)
    of sAssume:
      s.assumeCond.unwrapVarDerefsExpr(varParams)
    of sRet:
      s.retVal.unwrapVarDerefsExpr(varParams)
    of sDiscard:
      s.discardVal.unwrapVarDerefsExpr(varParams)
    of sCall:
      s.callCallee.unwrapVarDerefsExpr(varParams)
      for a in mitems(s.callArgs):
        a.unwrapVarDerefsExpr(varParams)
    of sJmp, sLab, sAsm, sEmit:
      discard

proc rewriteVarCallArgsExpr(fn: CirProc, e: CirExpr, varParams: HashSet[string], procs: Table[string, CirProc]): string =
  ## Rewrites the call arguments of one expression tree:
  ##
  ## - an argument at a `var` parameter position drops its address-of wrapper
  ## - an argument naming the enclosing proc's own `var` parameter re-types to the pointee
  ## - anything else at a `var` position must be an lvalue
  ##
  ## Returns "" when clean.
  if e.isNil:
    return ""
  if e.kind == eCall and e.callee.kind == eSym:
    let callee = procs.getOrDefault(e.callee.name)
    if callee != nil and e.args.len == callee.params.len:
      var hasVarParams = false
      for p in callee.params:
        hasVarParams = hasVarParams or p.varParam
      for i, arg in mpairs(e.args):
        if not callee.params[i].varParam:
          continue
        if arg.kind == eAddr:
          e.args[i] = arg.addrOperand
        elif arg.kind == eSym and arg.name in varParams and
            arg.typ != nil and arg.typ.kind == ctPtr:
          arg.typ = arg.typ.pElem
        elif arg.kind notin {eSym, eAt, ePat, eDot}:
          return dissolveFail(fn, "var parameter call argument must be an" &
            " lvalue, got " & $arg.kind)
      if hasVarParams:
        # a call site's callee type still states the pre-lowering signature, so the rewrite re-types it to the retyped parameters
        var ps: seq[CirType]
        for p in callee.params:
          ps.add p.typ
        e.callee.typ = freshProcType(ps, callee.ret)
    for ch in getExprChildren(e):
      let err = rewriteVarCallArgsExpr(fn, ch, varParams, procs)
      if err.len > 0:
        return err
    return ""
  for ch in getExprChildren(e):
    let err = rewriteVarCallArgsExpr(fn, ch, varParams, procs)
    if err.len > 0:
      return err
  return ""

proc rewriteVarCallArgsStmts(fn: CirProc, ss: var seq[CirStmt], varParams: HashSet[string], procs: Table[string, CirProc]): string =
  ## Statement-walk overload of `rewriteVarCallArgsExpr`, nested bodies included.
  for s in mitems(ss):
    case s.kind
    of sVar, sConst:
      let err = rewriteVarCallArgsExpr(fn, s.init, varParams, procs)
      if err.len > 0:
        return err
    of sAsgn:
      let lerr = rewriteVarCallArgsExpr(fn, s.lhs, varParams, procs)
      if lerr.len > 0:
        return lerr
      let rerr = rewriteVarCallArgsExpr(fn, s.rhs, varParams, procs)
      if rerr.len > 0:
        return rerr
    of sIte:
      let cerr = rewriteVarCallArgsExpr(fn, s.cond, varParams, procs)
      if cerr.len > 0:
        return cerr
      let terr = rewriteVarCallArgsStmts(fn, s.thenBody, varParams, procs)
      if terr.len > 0:
        return terr
      let eerr = rewriteVarCallArgsStmts(fn, s.elseBody, varParams, procs)
      if eerr.len > 0:
        return eerr
    of sLoop:
      let lerr = rewriteVarCallArgsStmts(fn, s.loopBody, varParams, procs)
      if lerr.len > 0:
        return lerr
    of sCase:
      let serr = rewriteVarCallArgsExpr(fn, s.selector, varParams, procs)
      if serr.len > 0:
        return serr
      for br in mitems(s.branches):
        let berr = rewriteVarCallArgsStmts(fn, br.body, varParams, procs)
        if berr.len > 0:
          return berr
      let eerr = rewriteVarCallArgsStmts(fn, s.caseElse, varParams, procs)
      if eerr.len > 0:
        return eerr
    of sScope:
      let serr = rewriteVarCallArgsStmts(fn, s.scopeBody, varParams, procs)
      if serr.len > 0:
        return serr
    of sAssume:
      let aerr = rewriteVarCallArgsExpr(fn, s.assumeCond, varParams, procs)
      if aerr.len > 0:
        return aerr
    of sRet:
      let rerr = rewriteVarCallArgsExpr(fn, s.retVal, varParams, procs)
      if rerr.len > 0:
        return rerr
    of sDiscard:
      let derr = rewriteVarCallArgsExpr(fn, s.discardVal, varParams, procs)
      if derr.len > 0:
        return derr
    of sCall:
      let tmp = CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
      let err = rewriteVarCallArgsExpr(fn, tmp, varParams, procs)
      s.callCallee = tmp.callee
      s.callArgs = tmp.args
      if err.len > 0:
        return err
    of sJmp, sLab, sAsm, sEmit:
      discard
  return ""

proc qualifyVarParams(m: CirModule): string =
  ## Lowers the Nim `var T` parameters of every proc to GLSL inout bindings:
  ##
  ## - each marked parameter retypes to its pointee
  ## - its body derefs unwrap to the parameter
  ## - its call sites pass the plain lvalue
  ## - the printer emits the `inout` qualifier on the marked parameters
  ##
  ## Returns "" when clean, else the first refusal.
  var procs = initTable[string, CirProc]()
  for fn in m.procs:
    procs[fn.name] = fn
  # every parameter retypes before any body walks: a call site in a proc
  # preceding its callee must see the retyped signature
  for fn in m.procs:
    for p in mitems(fn.params):
      if p.varParam:
        if p.typ != nil and p.typ.kind == ctPtr:
          p.typ = p.typ.pElem
        else:
          return dissolveFail(fn, "var parameter " & p.name &
            " without a pointer type")
  for fn in m.procs:
    var varParams: HashSet[string]
    for p in fn.params:
      if p.varParam:
        varParams.incl(p.name)
    fn.body.unwrapVarDerefsStmts(varParams)
    let err = rewriteVarCallArgsStmts(fn, fn.body, varParams, procs)
    if err.len > 0:
      return err
  return ""

# ═══════════════════════════════════════════════════════════════════════
# Pointer dissolution
# ═══════════════════════════════════════════════════════════════════════

type
  DissolveCtx = object
    pendingPtrs: HashSet[string]
    ## Pointer locals awaiting their address-of binding, a read refuses.
    aliasOf: Table[string, string]
    ## Pointer local -> pointee local. One binding dissolves the pair, a rebinding refuses.

proc dissolveFail(fn: CirProc, detail: string): string =
  "at " & fn.name & ": " & detail

proc toPlainSym(name: string, typ: CirType): CirExpr =
  ## Every dissolved read rewrites to the symbol this proc returns.
  result = CirExpr(kind: eSym, name: name, typ: typ, builtin: false,
    coordBuiltin: cbkNone, synchroBuiltin: sbkNone,
    reductionBuiltin: rbkNone, atomicsBuiltin: abkNone)

proc isBindShape(e: CirExpr): bool =
  ## Address-of binding shape: the address of a struct or array local.
  ## The pointee's name stands in for the pointer, only a composite local
  ## offers the reads that the alias resolves.
  e.kind == eAddr and e.addrOperand.kind == eSym and
    e.addrOperand.typ != nil and
    e.addrOperand.typ.kind in {ctStruct, ctArray}

proc bindPointee(e: CirExpr): string =
  ## Pointee local name of an address-of binding, `isBindShape` checked by the caller.
  e.addrOperand.name

proc rewriteDissolveExpr(fn: CirProc, ctx: var DissolveCtx, e: CirExpr): string =
  ## Rewrites one expression in place for the GLSL storage model.
  ## Returns "" when clean, else the first refusal.
  if e.isNil:
    return ""
  case e.kind
  of eSym:
    if e.name in ctx.pendingPtrs:
      return dissolveFail(fn, "pointer local " & e.name &
        " read before its address-of binding, GLSL has no function-storage" &
        " addressing")
    if e.name in ctx.aliasOf:
      return dissolveFail(fn, "pointer local " & e.name &
        " used as a value, GLSL has no pointer values")
    return ""
  of eDeref:
    let op = e.derefOperand
    if op.kind == eSym and op.name in ctx.aliasOf:
      # pointee stands in for the pointer, the deref unwraps to it
      e.kind = eSym
      e.name = ctx.aliasOf[op.name]
      e.typ = op.typ.pElem
      e.builtin = false
      e.coordBuiltin = cbkNone
      e.synchroBuiltin = sbkNone
      e.reductionBuiltin = rbkNone
      e.atomicsBuiltin = abkNone
      return ""
    for ch in getExprChildren(e):
      let err = rewriteDissolveExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    let ot = getExprType(e.derefOperand)
    if ot != nil and ot.kind == ctAptr:
      # a buffer-reference deref, the printer emits the indexed access
      return ""
    if e.derefOperand.kind == eSym and e.derefOperand.name in ctx.aliasOf:
      # a child rewrite spliced the operand to an aliased pointer's pointee
      let op2 = e.derefOperand
      e.kind = eSym
      e.name = ctx.aliasOf[op2.name]
      e.typ = op2.typ.pElem
      e.builtin = false
      e.coordBuiltin = cbkNone
      e.synchroBuiltin = sbkNone
      e.reductionBuiltin = rbkNone
      e.atomicsBuiltin = abkNone
      return ""
    return dissolveFail(fn, "deref of a non-buffer-reference operand, no" &
      " pointee: GLSL has no function-storage addressing, the GLSL lowering" &
      " pass dissolves pointer locals")
  of eAt, ePat:
    let base = (if e.kind == eAt: e.atBase else: e.patBase)
    if base.kind == eSym:
      if base.name in ctx.pendingPtrs:
        return dissolveFail(fn, "pointer local " & base.name &
          " read before its address-of binding, GLSL has no function-storage" &
          " addressing")
      if base.name in ctx.aliasOf:
        # aliased local stands in for the pointer, the index goes to the pointee
        let pointee = toPlainSym(ctx.aliasOf[base.name], base.typ.pElem)
        if e.kind == eAt:
          e.atBase = pointee
        else:
          e.patBase = pointee
    for ch in getExprChildren(e):
      let err = rewriteDissolveExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""
  of eAddr:
    for ch in getExprChildren(e):
      let err = rewriteDissolveExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    if e.addrOperand.kind == ePat:
      # a buffer element's address is the indexed reference access
      return ""
    return dissolveFail(fn, "address-of has no GLSL form outside a buffer" &
      " element, GLSL has no function-storage addressing")
  else:
    for ch in getExprChildren(e):
      let err = rewriteDissolveExpr(fn, ctx, ch)
      if err.len > 0:
        return err
    return ""

proc rewriteDissolveStmts(fn: CirProc, ctx: var DissolveCtx, ss: var seq[CirStmt]): string
  ## Forward declaration, statement lists recurse into themselves.

proc rewriteDissolveStmt(fn: CirProc, ctx: var DissolveCtx, s: CirStmt): string =
  ## Rewrites one statement in place. Returns "" when clean, else the refusal.
  case s.kind
  of sVar:
    if s.declType.kind == ctPtr:
      if s.init == nil:
        ctx.pendingPtrs.incl(s.name)
      elif isBindShape(s.init):
        ctx.aliasOf[s.name] = s.init.bindPointee
      else:
        return dissolveFail(fn, "pointer local " & s.name &
          " must bind the address of a local, GLSL has no function-storage" &
          " addressing")
      return ""
    return rewriteDissolveExpr(fn, ctx, s.init)
  of sConst:
    return rewriteDissolveExpr(fn, ctx, s.init)
  of sAsgn:
    if s.lhs.kind == eSym and s.lhs.typ != nil and s.lhs.typ.kind == ctPtr:
      if not isBindShape(s.rhs):
        return dissolveFail(fn, "pointer local " & s.lhs.name &
          " must bind the address of a local, GLSL has no function-storage" &
          " addressing")
      if s.lhs.name in ctx.aliasOf:
        return dissolveFail(fn, "pointer local " & s.lhs.name &
          " rebound, GLSL has no pointer values")
      ctx.pendingPtrs.excl(s.lhs.name)
      ctx.aliasOf[s.lhs.name] = s.rhs.bindPointee
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
  ## Rewrites one statement list in place. Pointer-local declarations drop out
  ## with their address-of bindings, everything else rewrites where it stands,
  ## "" comes back when clean.
  var i = 0
  while i < ss.len:
    let s = ss[i]
    var drops = false
    if s.kind == sVar and s.declType.kind == ctPtr:
      let err = rewriteDissolveStmt(fn, ctx, s)
      if err.len > 0:
        return err
      drops = true
    elif s.kind == sAsgn and s.lhs.kind == eSym and s.lhs.typ != nil and
        s.lhs.typ.kind == ctPtr:
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
  ## Returns "" when clean, else the first refusal.
  for fn in m.procs:
    var ctx = DissolveCtx()
    let err = rewriteDissolveStmts(fn, ctx, fn.body)
    if err.len > 0:
      return err
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
  ##   const_mem walk recurses into the nested scopes at their encounter position
  ## - the const_mem initializer is required, the printer emits it from the relocated node
  ## - runs after the reserved-word rename, the relocated declarations use the renamed names that their body already references
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.space == caThreadgroup:
        if not fn.isKernel:
          # a device-function smem local stays, GLSL program-scope `shared`
          # storage belongs to the kernel, the printer refuses the survivor
          i.inc
          continue
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
