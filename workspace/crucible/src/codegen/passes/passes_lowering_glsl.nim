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
##   kernel body → glslDissolvePointerLocals → glslHoistProgramScopeLocals → printed GLSL source
##
## GLSL has no function-storage addressing, so a local cannot be addressed,
## and threadgroup/constant-storage locals must live at program scope:
##
## - `glslDissolvePointerLocals`: pointer locals dissolve into their pointee,
##   reads through the alias rewrite to the pointee symbol, a pointer read before
##   its address-of binding or read as a value refuses. Kernel and device-function
##   pointer parameters are untouched, the printer refuses them.
## - `glslHoistProgramScopeLocals`: `{.smem.}` kernel locals and `{.const_mem.}`
##   locals relocate into `CirModule.globals` in encounter order,
##   the printer emits them at program scope
## - the smem relocation covers top-level kernel-body locals only, the const_mem
##   relocation covers every proc and scope depth, program-scope `const` storage
##   is visible to all functions
## - an initializer-less const_mem local refuses in the pass and again at the printer as a shape guard
##
## Refusal homes:
## - a `caDevice`-space local refuses at the printer, the pass does not touch
##   device-space locals (GLSL locals are private)
## - a threadgroup local surviving in a nested scope or device function refuses
##   at the printer, the smem relocation covers only top-level kernel-body locals
## - a const_mem local surviving outside the relocation refuses at the printer

import std/[sets, tables]
import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerGlslLoweringPasses*(reg: var PassRegistry) =
  ## Registers the GLSL per-target lowering passes:
  ## - `glslDissolvePointerLocals` then `glslHoistProgramScopeLocals`, in that order, before printing
  ##
  ## Registration contract: the passes sit after the reserved-word rename, the relocated declarations
  ## use the renamed names their body already references.
  reg.register("glslDissolvePointerLocals", pkTransform, phaseMain,
    "Dissolves pointer locals into their pointee locals",
    run = dissolvePointerLocals
  )
  reg.register("glslHoistProgramScopeLocals", pkTransform, phaseMain,
    "Relocates smem and const_mem kernel locals into the module's" &
    " program scope",
    run = hoistProgramScopeLocals
  )

# ═══════════════════════════════════════════════════════════════════════
# Pointer dissolution
# ═══════════════════════════════════════════════════════════════════════

type
  DissolveCtx = object
    pendingPtrs: HashSet[string]
    ## Pointer locals awaiting their address-of binding, a read refuses.
    aliasOf: Table[string, string]
    ## Pointer local -> pointee local, the binding dissolved the pair.

proc dissolveFail(fn: CirProc, detail: string): string =
  "at " & fn.name & ": " & detail

proc toPlainSym(name: string, typ: CirType): CirExpr =
  ## Plain symbol node without a builtin kind: the rewrite target of every dissolved read.
  result = CirExpr(kind: eSym, name: name, typ: typ, builtin: false,
    coordBuiltin: cbkNone, synchroBuiltin: sbkNone,
    reductionBuiltin: rbkNone, atomicsBuiltin: abkNone)

proc isBindShape(e: CirExpr): bool =
  ## Address-of binding shape: the address of a struct or array local.
  ## The pointee's name stands in for the pointer, only a composite local
  ## offers the reads the alias resolves.
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
  ## - runs after the reserved-word rename, the relocated declarations use the renamed names their body already references
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
