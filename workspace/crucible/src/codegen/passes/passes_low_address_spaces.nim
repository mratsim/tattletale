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
## The key is the callee name plus one position/space pair per re-spaced formal.
## The clone name uses the instantiation-name convention: callee name plus a base58 discriminator of the combination.
##
## Two shapes never specialize:
##
## - a formal declaring threadgroup or constant storage: its space names the storage the callee declares
## - a callee outside the module: a builtin such as `__builtin_memcpy` states
##   a nominal formal space. The printers re-space it at the call site
##
## Every C-family bundle registers the pass. The SPIR-V encoder derives
## storage classes without pointer text and registers none.
##
## Clone bodies walk through the same driver: a call inside a clone
## instantiates against the callee's declared formals, so a self-recursive
## call resolves to the clone already keyed for its argument spaces.

import std/[hashes, sets, tables]
import ../ir/cirv2_lang
import ../ir/cirv2_build
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

proc base58Discriminator(key: string): string =
  ## Base58 discriminator of an instantiation key: the suffix convention the frontend's instantiation names use.
  var h = hash(key)
  h = !$h
  var x = uint64(h) and 0xFFFFFFFFFFFFFFFF'u64
  const alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
  if x == 0:
    return "1"
  while x > 0:
    result = alphabet[int(x mod 58)] & result
    x = x div 58

proc wantedParamTypes(callee: CirProc, args: seq[CirExpr], wanted: var seq[CirType]): bool =
  ## Computes the per-parameter types one call site selects.
  ## Each `wanted[i]` re-types the formal's pointee to the argument's space whenever the argument names another space.
  ##
  ## Returns true when at least one pointer formal specializes.
  ##
  ## A formal declaring threadgroup or constant storage never specializes:
  ## that space names the storage the callee declares, never a provenance the call site picks.
  wanted.setLen(callee.params.len)
  for i in 0 ..< callee.params.len:
    wanted[i] = callee.params[i].typ
    if i >= args.len:
      continue
    let formal = callee.params[i].typ
    let argT = getExprType(args[i])
    if formal != nil and formal.kind in {ctPtr, ctAptr} and
        formal.space in {caDevice, caThread} and
        argT != nil and argT.kind == formal.kind and
        sameType(argT.pElem, formal.pElem) and argT.space != formal.space:
      wanted[i] = withSpace(formal, argT.space)
      result = true

proc instKey(callee: CirProc, wanted: seq[CirType]): string =
  ## Instantiation key: the callee name plus one position/space pair per
  ## re-spaced parameter. Two call sites with the same per-argument spaces
  ## produce the same key.
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
  ## Retypes every symbol named `name` in one expression tree, in place.
  ## Expressions are refs: the rewrite applies to every sharer.
  if e.isNil:
    return
  if e.kind == eSym:
    if e.name == name:
      e.typ = typ
    return
  for ch in getExprChildren(e):
    ch.retypeSymsExpr(name, typ)

proc retypeSymsStmts(ss: seq[CirStmt], name: string, typ: CirType) =
  ## Statement-walk overload of `retypeSymsExpr`, nested bodies included.
  for s in ss:
    case s.kind
    of sVar, sConst:
      retypeSymsExpr(s.init, name, typ)
    of sAsgn:
      retypeSymsExpr(s.lhs, name, typ)
      retypeSymsExpr(s.rhs, name, typ)
    of sIte:
      retypeSymsExpr(s.cond, name, typ)
      retypeSymsStmts(s.thenBody, name, typ)
      retypeSymsStmts(s.elseBody, name, typ)
    of sLoop:
      retypeSymsStmts(s.loopBody, name, typ)
    of sCase:
      retypeSymsExpr(s.selector, name, typ)
      for br in s.branches:
        retypeSymsStmts(br.body, name, typ)
      retypeSymsStmts(s.caseElse, name, typ)
    of sScope:
      retypeSymsStmts(s.scopeBody, name, typ)
    of sAssume:
      retypeSymsExpr(s.assumeCond, name, typ)
    of sRet:
      retypeSymsExpr(s.retVal, name, typ)
    of sDiscard:
      retypeSymsExpr(s.discardVal, name, typ)
    of sCall:
      retypeSymsExpr(s.callCallee, name, typ)
      for a in s.callArgs:
        retypeSymsExpr(a, name, typ)
    of sLab, sJmp, sAsm, sEmit:
      discard

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

proc rewriteCallSite(calleeExpr: CirExpr, args: seq[CirExpr],
                     calleeProc: CirProc, taken: var HashSet[string],
                     procs: var Table[string, CirProc],
                     clones: var Table[string, CirProc],
                     byBase: var Table[string, seq[CirProc]],
                     queue: var seq[CirProc]): CirProc =
  ## Rewrites one call site onto the instantiation its argument spaces select.
  ## A first-use combination creates the clone and queues it for the driver's body walk.
  ## Returns the instantiation, nil when the argument spaces match the formals.
  var wanted: seq[CirType]
  if not calleeProc.wantedParamTypes(args, wanted):
    return nil
  let key = calleeProc.instKey(wanted)
  if clones.hasKey(key):
    result = clones[key]
  else:
    var name = calleeProc.name & "_" & base58Discriminator(key)
    while name in taken:
      # a discriminator collision over an existing proc name: rehash with a salt.
      # every rehash salts the key, so the loop stays bounded
      name = calleeProc.name & "_" & base58Discriminator(key & "\x1f" & name)
    result = cloneCallee(calleeProc, wanted, name)
    taken.incl name
    procs[name] = result
    clones[key] = result
    byBase.mgetOrPut(calleeProc.name, @[]).add result
    queue.add result
  calleeExpr.name = result.name
  calleeExpr.typ = freshProcType(wanted, result.ret)

proc instantiateCallSitesStmts(ss: seq[CirStmt],
                               procs: var Table[string, CirProc],
                               taken: var HashSet[string],
                               clones: var Table[string, CirProc],
                               byBase: var Table[string, seq[CirProc]],
                               queue: var seq[CirProc]): string
  ## Forward declaration: nested statement lists recurse into each other.

proc instantiateCallSitesExpr(e: CirExpr, procs: var Table[string, CirProc],
                              taken: var HashSet[string],
                              clones: var Table[string, CirProc],
                              byBase: var Table[string, seq[CirProc]],
                              queue: var seq[CirProc]): string =
  ## Instantiates every call nested in one expression, outermost last.
  ## Returns "" when clean, else the first refusal.
  if e.isNil:
    return ""
  if e.kind == eCall and e.callee.kind == eSym:
    for a in e.args.mitems:
      let err = instantiateCallSitesExpr(a, procs, taken, clones, byBase, queue)
      if err.len > 0:
        return err
    let calleeProc = procs.getOrDefault(e.callee.name)
    if calleeProc.isNil or e.args.len != calleeProc.params.len:
      return ""
    discard e.callee.rewriteCallSite(e.args, calleeProc, taken, procs, clones, byBase, queue)
    return ""
  for ch in getExprChildren(e):
    let err = ch.instantiateCallSitesExpr(procs, taken, clones, byBase, queue)
    if err.len > 0:
      return err
  return ""

proc instantiateCallSitesStmts(ss: seq[CirStmt],
                               procs: var Table[string, CirProc],
                               taken: var HashSet[string],
                               clones: var Table[string, CirProc],
                               byBase: var Table[string, seq[CirProc]],
                               queue: var seq[CirProc]): string =
  ## Statement-walk overload of `instantiateCallSitesExpr`, nested bodies included.
  ## Returns "" when clean, else the first refusal.
  for s in ss:
    case s.kind
    of sVar, sConst:
      let err = instantiateCallSitesExpr(s.init, procs, taken, clones, byBase, queue)
      if err.len > 0:
        return err
    of sAsgn:
      let lerr = instantiateCallSitesExpr(s.lhs, procs, taken, clones, byBase, queue)
      if lerr.len > 0:
        return lerr
      let rerr = instantiateCallSitesExpr(s.rhs, procs, taken, clones, byBase, queue)
      if rerr.len > 0:
        return rerr
    of sIte:
      let cerr = instantiateCallSitesExpr(s.cond, procs, taken, clones, byBase, queue)
      if cerr.len > 0:
        return cerr
      let thenErr = instantiateCallSitesStmts(s.thenBody, procs, taken, clones, byBase, queue)
      if thenErr.len > 0:
        return thenErr
      let elseErr = instantiateCallSitesStmts(s.elseBody, procs, taken, clones, byBase, queue)
      if elseErr.len > 0:
        return elseErr
    of sLoop:
      let loopErr = instantiateCallSitesStmts(s.loopBody, procs, taken, clones, byBase, queue)
      if loopErr.len > 0:
        return loopErr
    of sCase:
      let selErr = instantiateCallSitesExpr(s.selector, procs, taken, clones, byBase, queue)
      if selErr.len > 0:
        return selErr
      for br in s.branches:
        let brErr = instantiateCallSitesStmts(br.body, procs, taken, clones, byBase, queue)
        if brErr.len > 0:
          return brErr
      let elseErr = instantiateCallSitesStmts(s.caseElse, procs, taken, clones, byBase, queue)
      if elseErr.len > 0:
        return elseErr
    of sScope:
      let scopeErr = instantiateCallSitesStmts(s.scopeBody, procs, taken, clones, byBase, queue)
      if scopeErr.len > 0:
        return scopeErr
    of sAssume:
      let aerr = instantiateCallSitesExpr(s.assumeCond, procs, taken, clones, byBase, queue)
      if aerr.len > 0:
        return aerr
    of sRet:
      let rerr = instantiateCallSitesExpr(s.retVal, procs, taken, clones, byBase, queue)
      if rerr.len > 0:
        return rerr
    of sDiscard:
      let derr = instantiateCallSitesExpr(s.discardVal, procs, taken, clones, byBase, queue)
      if derr.len > 0:
        return derr
    of sCall:
      for a in s.callArgs.mitems:
        let aerr = instantiateCallSitesExpr(a, procs, taken, clones, byBase, queue)
        if aerr.len > 0:
          return aerr
      let calleeProc = procs.getOrDefault(s.callCallee.name)
      if calleeProc != nil and s.callArgs.len == calleeProc.params.len:
        discard s.callCallee.rewriteCallSite(s.callArgs, calleeProc, taken,
          procs, clones, byBase, queue)
    of sLab, sJmp, sAsm, sEmit:
      discard
  return ""

proc instantiateAddressSpaceGenericsPass(m: CirModule): string =
  ## Instantiates the module callees whose pointer arguments name another
  ## address space than the formal.
  ##
  ## Clones append to `m.procs` after the walk, their bodies walked like any other proc's.
  var procs = initTable[string, CirProc]()
  var clones = initTable[string, CirProc]()
  var byBase = initTable[string, seq[CirProc]]()
  var taken: HashSet[string]
  for fn in m.procs:
    procs[fn.name] = fn
    taken.incl fn.name
  var queue: seq[CirProc]
  for fn in m.procs:
    queue.add fn
  var i = 0
  while i < queue.len:
    let err = instantiateCallSitesStmts(queue[i].body, procs, taken, clones,
      byBase, queue)
    if err.len > 0:
      return err
    i.inc
  # Clones insert after their base callee so the emitted proc order compiles declaration before use
  var placed: seq[CirProc]
  for fn in m.procs:
    placed.add fn
    for c in byBase.getOrDefault(fn.name):
      placed.add c
  m.procs = placed
  return ""
