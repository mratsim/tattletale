# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

{.experimental: "codeReordering".}

## Lowering: per-address-space callee instantiation.
##
##   call sites → instantiateAddressSpaceGenerics → clones after their base callee → printed target source
##
## A pointer parameter of a module proc is generic over its address space.
## The call site picks the space, and the printers emit both spaces.
## Metal and OpenCL C reject a call whose argument names another space.
## Every distinct combination of per-argument spaces gets its own callee instantiation:
##
## - the clone's pointer formals re-type to the argument's space
## - the clone's body reads of those formals retype with them
## - one clone serves every call site with the same combination
##
## The clone name uses the instantiation-name convention: the callee name
## plus a base58 discriminator of the combination.
##
## Two shapes never specialize:
##
## - a formal declaring threadgroup or constant storage: its space names
##   the storage the callee declares, never a provenance the call site picks
## - a callee outside the module: a builtin such as `__builtin_memcpy` carries
##   a nominal formal space. The printers re-space it at the call site
##
## Every C-family bundle registers the pass. The SPIR-V encoder derives
## storage classes without pointer text and registers none.
##
## Clone bodies walk through the same driver: a call inside a clone
## instantiates against the callee's declared formals, so a self-recursive
## call resolves to the clone already keyed for its argument spaces.

import std/tables
import ../ir/cirv2_lang
import ../ir/cirv2_build
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerInstantiateAddressSpaceGenerics*(reg: var PassRegistry) =
  ## Registration position: after the target's other lowering transforms,
  ## whose rewrites append parameters and call-site arguments by callee name
  ## over the declared call graph.
  reg.register("instantiateAddressSpaceGenerics", pkTransform,
    "Instantiates module callees whose pointer arguments name another" &
    " address space than the formal",
    run = instantiateAddressSpaceGenericsPass)

# ═══════════════════════════════════════════════════════════════════════
# Instantiation
# ═══════════════════════════════════════════════════════════════════════

proc resolveWantedParamTypes(callee: CirProc, args: seq[CirExpr], wanted: var seq[CirType]): bool =
  # Precondition: `args` covers every formal, `args.len == callee.params.len`.
  wanted.setLen(callee.params.len)
  for i in 0 ..< callee.params.len:
    wanted[i] = callee.params[i].typ
    let formal = callee.params[i].typ
    let argT = getExprType(args[i])
    if formal != nil and formal.kind in {ctPtr, ctAptr} and
        formal.space in {caDevice, caThread} and
        argT != nil and argT.kind == formal.kind and
        sameType(argT.pElem, formal.pElem) and argT.space != formal.space:
      wanted[i] = withAddressSpace(formal, argT.space)
      result = true

proc buildInstKey(callee: CirProc, wanted: seq[CirType]): string =
  result = callee.name
  for i in 0 ..< callee.params.len:
    if not sameType(wanted[i], callee.params[i].typ):
      result.add "\x1f" & $i & "\x1f" & $wanted[i].space

proc retypeSymsExpr(e: CirExpr, name: string, typ: CirType) =
  ## Expressions are refs, so the retyping applies to every sharer of the symbol.
  if e.kind == eSym and e.name == name:
    e.typ = typ

proc cloneCallee(callee: CirProc, wanted: seq[CirType], name: string): CirProc =
  result = CirProc(name: name, ret: callee.ret, isKernel: callee.isKernel,
                   inline: callee.inline, exportName: callee.exportName,
                   workgroup: callee.workgroup,
                   body: cloneStmts(callee.body))
  for i in 0 ..< callee.params.len:
    let p = callee.params[i]
    if sameType(wanted[i], p.typ):
      result.params.add p
    else:
      result.params.add param(p.name, wanted[i])
      result.body.walkStmts(nil, proc(e: CirExpr) =
        e.retypeSymsExpr(p.name, wanted[i]))

type
  InstantiateCtx = ref object
    ## Shared state of the per-address-space instantiation walk.
    ##
    ## - `procs` maps every proc name to its body, clones included as they appear
    ## - `clones` keys one instantiation per argument-space combination
    ## - `byBase` groups clones under their base callee for the placement loop
    ## - `queue` lists the procs whose bodies still need the call-site walk
    ##
    ## `err` is the first refusal, empty while the walk is clean.
    procs: Table[string, CirProc]
    clones: Table[string, CirProc]
    byBase: Table[string, seq[CirProc]]
    queue: seq[CirProc]
    err: string

proc rewriteCallSite(calleeExpr: CirExpr, args: seq[CirExpr], calleeProc: CirProc, ctx: InstantiateCtx) =
  ## Rewrites one call site onto the instantiation its argument spaces select.
  ## The rewrite touches the `calleeExpr` and the `ctx` only, nothing returned.
  ## The pass refuses a name collision, `ctx.err` receives the refusal text and the call site remains untouched.
  var wanted: seq[CirType]
  var instantiation: CirProc
  if not calleeProc.resolveWantedParamTypes(args, wanted):
    return
  let key = calleeProc.buildInstKey(wanted)
  if ctx.clones.hasKey(key):
    instantiation = ctx.clones[key]
  else:
    var name = calleeProc.name & "_" & base58Short(typeIdentityHash(key))
    if ctx.procs.hasKey(name):
      ctx.err = "instantiation name '" & name & "' is already taken by another proc of the module"
      return
    instantiation = cloneCallee(calleeProc, wanted, name)
    ctx.procs[name] = instantiation
    ctx.clones[key] = instantiation
    ctx.byBase.mgetOrPut(calleeProc.name, @[]).add instantiation
    ctx.queue.add instantiation
  calleeExpr.name = instantiation.name
  calleeExpr.typ = freshProcType(wanted, instantiation.ret)

proc instantiateCallSitesStmts(ss: seq[CirStmt], ctx: InstantiateCtx) =
  ## Unspecialized call sites stay untouched: the printers judge whatever survives.
  ## Unspecialized means a callee outside the module or an arity mismatch against the module callee.
  let onExpr = proc(e: CirExpr) =
    if ctx.err.len > 0 or e.kind != eCall or e.callee.kind != eSym:
      return
    let calleeProc = ctx.procs.getOrDefault(e.callee.name)
    if calleeProc.isNil or e.args.len != calleeProc.params.len:
      return
    e.callee.rewriteCallSite(e.args, calleeProc, ctx)
  let onStmt = proc(s: CirStmt) =
    if ctx.err.len > 0 or s.kind != sCall:
      return
    let calleeProc = ctx.procs.getOrDefault(s.callCallee.name)
    if calleeProc != nil and s.callArgs.len == calleeProc.params.len:
      s.callCallee.rewriteCallSite(s.callArgs, calleeProc, ctx)
  ss.walkStmts(onStmt, onExpr)

proc instantiateAddressSpaceGenericsPass(m: CirModule): string =
  ## Clones join the walk, their bodies scanned like any other proc's body.
  var ctx = InstantiateCtx()
  for fn in m.procs:
    ctx.procs[fn.name] = fn
    ctx.queue.add fn
  var i = 0
  while i < ctx.queue.len and ctx.err.len == 0:
    ctx.queue[i].body.instantiateCallSitesStmts(ctx)
    i.inc
  if ctx.err.len > 0:
    return ctx.err
  # Clones insert after their base callee so the emitted proc order compiles declaration before use
  var placed: seq[CirProc]
  for fn in m.procs:
    placed.add fn
    for c in ctx.byBase.getOrDefault(fn.name):
      placed.add c
  m.procs = placed
