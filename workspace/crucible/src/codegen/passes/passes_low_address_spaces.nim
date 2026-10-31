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
## - a callee outside the module: a builtin such as `__builtin_memcpy` states
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
import ../frontend/nim_type_resolvers
import ./pass_datatypes
import ./pass_helpers

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerInstantiateAddressSpaceGenerics*(reg: var PassRegistry) =
  ## Registers the per-address-space callee instantiation pass of the lowering stage.
  ## Registration position: after the target's other lowering transforms,
  ## whose rewrites append parameters and call-site arguments by callee name
  ## over the declared call graph.
  reg.register("instantiateAddressSpaceGenerics", pkTransform, phaseMain,
    "Instantiates module callees whose pointer arguments name another" &
    " address space than the formal",
    run = instantiateAddressSpaceGenericsPass)

# ═══════════════════════════════════════════════════════════════════════
# Instantiation
# ═══════════════════════════════════════════════════════════════════════

proc wantedParamTypes(callee: CirProc, args: seq[CirExpr], wanted: var seq[CirType]): bool =
  ## Computes the per-parameter types one call site selects.
  ## Each `wanted[i]` re-types the formal's pointee to the argument's space whenever the argument names another space.
  ##
  ## Returns true when at least one pointer formal specializes.
  # Callers pre-check the arity, `args` always covers every formal.
  wanted.setLen(callee.params.len)
  for i in 0 ..< callee.params.len:
    wanted[i] = callee.params[i].typ
    let formal = callee.params[i].typ
    let argT = getExprType(args[i])
    if formal != nil and formal.kind in {ctPtr, ctAptr} and
        formal.space in {caDevice, caThread} and
        argT != nil and argT.kind == formal.kind and
        sameType(argT.pElem, formal.pElem) and argT.space != formal.space:
      wanted[i] = withSpace(formal, argT.space)
      result = true

proc instKey(callee: CirProc, wanted: seq[CirType]): string =
  ## Instantiation key: the callee name plus one position/space pair per re-spaced parameter.
  result = callee.name
  for i in 0 ..< callee.params.len:
    if not sameType(wanted[i], callee.params[i].typ):
      result.add "\x1f" & $i & "\x1f" & $wanted[i].space

proc freshProcType(params: seq[CirType], ret: CirType): CirType =
  ## A fresh proc type over `params`.
  ##
  ## Call-site callee types are interned: structurally identical signatures
  ## share one `CirType` ref. A call site that re-types its callee replaces
  ## the symbol type with this fresh copy, leaving the shared ref untouched.
  result = CirType(kind: ctProc)
  result.params = params
  result.ret = ret

proc retypeSymsExpr(e: CirExpr, name: string, typ: CirType) =
  ## Retypes the symbol named `name` in place.
  ## Expressions are refs: the rewrite applies to every sharer.
  if e.kind == eSym and e.name == name:
    e.typ = typ

proc retypeSymsStmts(ss: seq[CirStmt], name: string, typ: CirType) =
  ## Retypes every symbol named `name` across one statement list, nested bodies included.
  ss.walkStmts(nil, proc(e: CirExpr) =
    e.retypeSymsExpr(name, typ))

proc cloneCallee(callee: CirProc, wanted: seq[CirType], name: string): CirProc =
  ## One instantiation of `callee` over the re-spaced formals `wanted`.
  ## Each re-spaced parameter retypes every body read of the formal.
  ## Other parameters and the body copy pass through unchanged.
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
      retypeSymsStmts(result.body, p.name, wanted[i])

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

proc rewriteCallSite(calleeExpr: CirExpr, args: seq[CirExpr], calleeProc: CirProc, ctx: InstantiateCtx): CirProc =
  ## Rewrites one call site onto the instantiation its argument spaces select.
  ## A first-use combination creates the clone and queues it for the driver's body walk.
  ## Returns the instantiation, nil when the argument spaces match the formals
  ## or when a name collision refuses the pass and fills `ctx.err`.
  var wanted: seq[CirType]
  if not calleeProc.wantedParamTypes(args, wanted):
    return nil
  let key = calleeProc.instKey(wanted)
  if ctx.clones.hasKey(key):
    result = ctx.clones[key]
  else:
    var name = calleeProc.name & "_" & base58Short(typeIdentityHash(key))
    if ctx.procs.hasKey(name):
      ctx.err = "instantiation name '" & name & "' is already taken by another proc of the module"
      return nil
    result = cloneCallee(calleeProc, wanted, name)
    ctx.procs[name] = result
    ctx.clones[key] = result
    ctx.byBase.mgetOrPut(calleeProc.name, @[]).add result
    ctx.queue.add result
  calleeExpr.name = result.name
  calleeExpr.typ = freshProcType(wanted, result.ret)

proc instantiateCallSitesStmts(ss: seq[CirStmt], ctx: InstantiateCtx) =
  ## Instantiates every call site in one statement list, calls nested inside expressions included.
  ## The walk stops on the first refusal in `ctx.err`.
  ##
  ## Unspecialized shapes leave the call site untouched:
  ##
  ## - a callee outside the module
  ## - an arity mismatch with the module callee
  ##
  ## The printers and the verifier judge whatever survives.
  let onExpr = proc(e: CirExpr) =
    if ctx.err.len > 0 or e.kind != eCall or e.callee.kind != eSym:
      return
    let calleeProc = ctx.procs.getOrDefault(e.callee.name)
    if calleeProc.isNil or e.args.len != calleeProc.params.len:
      return
    discard e.callee.rewriteCallSite(e.args, calleeProc, ctx)
  let onStmt = proc(s: CirStmt) =
    if ctx.err.len > 0 or s.kind != sCall:
      return
    let calleeProc = ctx.procs.getOrDefault(s.callCallee.name)
    if calleeProc != nil and s.callArgs.len == calleeProc.params.len:
      discard s.callCallee.rewriteCallSite(s.callArgs, calleeProc, ctx)
  ss.walkStmts(onStmt, onExpr)

proc instantiateAddressSpaceGenericsPass(m: CirModule): string =
  ## Instantiates the module callees whose pointer arguments name another
  ## address space than the formal.
  ##
  ## Clones append to `m.procs` after the walk, their bodies walked like any other proc's.
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
  result = ""
