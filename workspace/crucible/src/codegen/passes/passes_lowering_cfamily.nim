# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Shared C-family lowering: pointer-local dissolution plus program-scope hoisting.
## One registration template per pass, one registration call per target.
##
## - `registerCfamilyDissolvePointerLocals`: registers the shared statement walk
##   over the pointer dissolution. The instantiating module expands
##   the template with its dissolve context type and its six module-local procs,
##   every refusal is raised there.
## - `registerCfamilyHoistProgramScopeLocals`: registers the smem and const_mem relocation.
##   `smemInitReason` is the target's refusal reason for a smem local with an initializer,
##   "" when the target accepts one.

import ../ir/cirv2_lang
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Program-scope hoisting
# ═══════════════════════════════════════════════════════════════════════

proc hoistProgramScopeLocalsCore(m: CirModule, smemInitReason: string): string =
  ## Relocates the `{.smem.}` and `{.const_mem.}` locals of every proc into
  ## `m.globals`, in encounter order, dropping them from their bodies.
  ##
  ## Contract:
  ## - the smem relocation covers top-level kernel-body locals only, a nested
  ##   or device-function one stays in place and refuses at the printer
  ## - a smem local with an initializer refuses when `smemInitReason` is
  ##   non-empty, it names the target's zero-initialization rule
  ## - the const_mem relocation covers every proc and scope depth, the shared
  ##   const_mem walk recurses into the nested scopes at their encounter position,
  ##   a const_mem initializer is required, the printer emits it
  ##   from the relocated node
  ## - runs after the reserved-word rename, the relocated declarations use
  ##   the renamed names that their body already references
  for fn in m.procs:
    var i = 0
    while i < fn.body.len:
      let s = fn.body[i]
      if s.kind == sVar and s.space == caThreadgroup:
        if not fn.isKernel:
          # a device-function smem local stays in place and the printer refuses it,
          # program-scope workgroup storage belongs to the kernel
          i.inc
          continue
        if s.init != nil and smemInitReason.len > 0:
          return "at " & fn.name & ": smem local " & s.name &
            " with an initializer, " & smemInitReason
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
        else:
          discard
        i.inc
  return ""

template registerCfamilyHoistProgramScopeLocals*(reg: var PassRegistry, passName: string, description, smemInitReason: string) =
  ## Registers the program-scope hoist for one target.
  ##
  ## Args:
  ## - `passName`: the registry key, one of `glslHoistProgramScopeLocals`
  ##   and `wgslHoistProgramScopeLocals`
  ## - `description`: the registry description
  ## - `smemInitReason`: the refusal reason for a smem local with an initializer,
  ##   "" when the target's workgroup storage accepts one
  reg.register(passName, pkTransform, phaseMain, description,
    run = proc(m: CirModule): string {.nimcall.} =
      hoistProgramScopeLocalsCore(m, smemInitReason)
  )

# ═══════════════════════════════════════════════════════════════════════
# Pointer dissolution
# ═══════════════════════════════════════════════════════════════════════

template registerCfamilyDissolvePointerLocals*(reg: var PassRegistry, passName: string, description: string,
    DissolveCtx: typedesc, newDissolveCtx, dissolveVarDecl, dissolveAsgn,
    dissolveDropsVar, dissolveDropsAsgn, rewriteDissolveExpr: untyped) =
  ## Registers the pointer-local dissolution pass over the shared statement walk.
  ## Every refusal is raised by the instantiating module's own procs,
  ## the walk only recurses, rewrites and drops.
  ##
  ## Args:
  ## - `reg`, `passName`, `description`: the registry entry
  ## - `DissolveCtx`: the target module's dissolve context type
  ## - `newDissolveCtx(m, fn)`: the context of one proc, the `var T` parameter
  ##   and callee-table setup included
  ## - `dissolveVarDecl` / `dissolveAsgn`: the `sVar` and `sAsgn` variants,
  ##   binding and rebinding refusals included, plus the drop predicates
  ##   `dissolveDropsVar` / `dissolveDropsAsgn` and the expression rewrite
  ##   `rewriteDissolveExpr(fn, ctx, e)`
  # Each target module expands the template through a file-private forwarding
  # proc placed after the machinery, one that passes the six module-local procs
  # as explicit template arguments.
  # codeReordering keeps the registration call at the module top.
  # Pass order: the dissolved aliases use the renamed names their body already references, the pass sits after the reserved-word rename.
  proc dissolveWalk(fn: CirProc, ctx: var DissolveCtx, ss: var seq[CirStmt]): string =
    ## Rewrites one statement list in place and drops the dissolved declarations.
    ## Returns "" when clean, else the first refusal.
    var i = 0
    while i < ss.len:
      let s = ss[i]
      var drops = false
      case s.kind
      of sVar:
        let err = dissolveVarDecl(fn, ctx, s)
        if err.len > 0:
          return err
        drops = dissolveDropsVar(s, ctx)
      of sAsgn:
        let err = dissolveAsgn(fn, ctx, s)
        if err.len > 0:
          return err
        drops = dissolveDropsAsgn(s, ctx)
      of sConst:
        let err = rewriteDissolveExpr(fn, ctx, s.init)
        if err.len > 0:
          return err
      of sIte:
        let cerr = rewriteDissolveExpr(fn, ctx, s.cond)
        if cerr.len > 0:
          return cerr
        let terr = dissolveWalk(fn, ctx, s.thenBody)
        if terr.len > 0:
          return terr
        let eerr = dissolveWalk(fn, ctx, s.elseBody)
        if eerr.len > 0:
          return eerr
      of sLoop:
        let cerr = rewriteDissolveExpr(fn, ctx, s.loopCond)
        if cerr.len > 0:
          return cerr
        let lerr = dissolveWalk(fn, ctx, s.loopBody)
        if lerr.len > 0:
          return lerr
      of sRet:
        let rerr = rewriteDissolveExpr(fn, ctx, s.retVal)
        if rerr.len > 0:
          return rerr
      of sDiscard:
        let derr = rewriteDissolveExpr(fn, ctx, s.discardVal)
        if derr.len > 0:
          return derr
      of sCall:
        let tmp = CirExpr(kind: eCall, callee: s.callCallee, args: s.callArgs)
        let err = rewriteDissolveExpr(fn, ctx, tmp)
        s.callCallee = tmp.callee
        s.callArgs = tmp.args
        if err.len > 0:
          return err
      of sAsm, sEmit:
        discard
      if drops:
        ss.delete(i)
        continue
      i.inc
    return ""
  proc dissolvePointerLocals(m: CirModule): string {.nimcall.} =
    ## Dissolves the pointer locals of every proc into their pointees, in place.
    ## Returns "" when clean, else the first refusal.
    for fn in m.procs:
      var ctx = newDissolveCtx(m, fn)
      let err = dissolveWalk(fn, ctx, fn.body)
      if err.len > 0:
        return err
    return ""
  reg.register(passName, pkTransform, phaseMain, description,
    run = dissolvePointerLocals)
