# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            MSL lowering passes
#
# ############################################################
#
# Project the crucible IR onto Metal's calling convention:
# device-function `{.smem.}` vars hoist to threadgroup params,
# coordinate builtins materialize as appended params,
# pointer-carrying structs expand to ptr/scalar leaves

import std / [sequtils, sets, tables, strutils]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_validations

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerMetalLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = true)
  reg.registerSharedLoweringPasses()
  reg.registerExpandStructKernelParams()
  reg.registerSmemArena()
  reg.registerMaterializeIndexBuiltinParams()
  reg.registerCollectValueAddressSpaces()

func registerSmemArena(reg: var PassRegistry) =
  reg.register("smemArena", pkTransform, phaseMain,
    "Hoists device functions' {.smem.} vars to threadgroup params as Metal doesn't support threadgroup allocation in device functions",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      smemArenaImpl(ctx)
  )

func registerMaterializeIndexBuiltinParams(reg: var PassRegistry) =
  reg.register("materializeIndexBuiltinParams", pkTransform, phaseMain,
    "Materializes coordinate-builtin binding: builtin params appended to pParams, call-site forwarding args",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      materializeIndexBuiltinParamsImpl(ctx)
  )

# ═══════════════════════════════════════════════════════════════════════
# `{.smem.}` promotion to overseeing kernel
# ═══════════════════════════════════════════════════════════════════════

proc hoistSmemVar(fn: var GpuAst, stmt: GpuAst) =
  doAssert stmt.vType.kind in {gtArray, gtObject, gtGenericInst},
    "{.smem.} hoisting takes arrays or objects, got: " & $stmt.vType.kind
  doAssert stmt.vInit.kind == gpuDiscard,
    "a hoisted {.smem.} var takes no initializer: " & stmt.vName.symbol.name
  let arena = GpuType(kind: gtPtr, to: stmt.vType,
                      implicit: false, mutable: true)
  fn.pParams.add(GpuParam(ident: stmt.vName, typ: arena,
                          addressSpace: asSMEM, passByRef: false))

proc hoistSmemVars(fn: var GpuAst) =
  var kept: seq[GpuAst]
  for stmt in fn.pBody.statements:
    if stmt.kind == gpuVar and stmt.addressSpace == asSMEM:
      fn.hoistSmemVar(stmt)
    elif stmt.kind == gpuBlock:
      var inner: seq[GpuAst]
      for sub in stmt.statements:
        if sub.kind == gpuVar and sub.addressSpace == asSMEM:
          fn.hoistSmemVar(sub)
        else:
          inner.add sub
      stmt.statements = inner
      kept.add stmt
    else:
      kept.add stmt
  fn.pBody.statements = kept

proc hasArgNamed(call: GpuAst, name: string): bool =
  for a in call.cArgs:
    if a.kind == gpuIdent and a.symbol.name == name:
      return true
  false

proc markCalleeSmemParam(ctx: var GpuContext, cName: GpuAst, idx: int) =
  ## Marks every table's copy, the tables can hold the same body under several keys.
  for tab in [ctx.fnTab.addr, ctx.allFnTab.addr, ctx.genericInsts.addr]:
    for fnKey, fn in tab[].pairs:
      if fn.kind == gpuProc and fn.pName.symbol.iSym == cName.symbol.iSym:
        if idx < fn.pParams.len:
          fn.pParams[idx].addressSpace = asSMEM
          stripSmemDerefs(fn)

proc hasOwnSmemArena(fn: GpuAst, name: string): bool =
  for p in fn.pParams:
    if p.addressSpace == asSMEM and p.ident.symbol.name == name:
      return true
  for stmt in fn.pBody.statements:
    if stmt.kind == gpuVar and stmt.addressSpace == asSMEM and
        stmt.vName.symbol.name == name:
      return true
    if stmt.kind == gpuBlock:
      for sub in stmt.statements:
        if sub.kind == gpuVar and sub.addressSpace == asSMEM and
            sub.vName.symbol.name == name:
          return true
  false

proc stripSmemDerefs(fn: GpuAst) =
  ## `{.smem.}`-bound params are references and MSL rejects `(*ref)`.
  var names: seq[string]
  for p in fn.pParams:
    if p.addressSpace == asSMEM:
      names.add p.ident.symbol.name
  if names.len == 0:
    return
  fn.pBody.walk(proc(n: var GpuAst) =
    if n.kind == gpuDeref and n.dOf.kind == gpuIdent and
        n.dOf.symbol != nil and n.dOf.symbol.name in names:
      n = n.dOf)

proc bindSmemCallSites(ctx: var GpuContext, fn: var GpuAst) =
  ## Callers own the `{.smem.}` arena, the callee receives it as a threadgroup reference.
  ## Kernel entries without an arena get one synthesized.
  var calls: seq[GpuAst]
  fn.pBody.walk(proc(n: var GpuAst) =
    if n.kind == gpuCall:
      calls.add n)
  for call in calls:
    for i, arg in call.cArgs.mpairs:
      # A `var T` argument arrives as an implicit address-of, the arena
      # name hides behind it. An array arena arrives as `stage[thrID]`.
      let bare = if arg.kind == gpuAddr: arg.aOf else: arg
      let isSmemArg = (bare.kind == gpuIdent and bare.symbol != nil and
          hasOwnSmemArena(fn, bare.symbol.name)) or
        (bare.kind == gpuIndex and bare.iArr.kind == gpuIdent and
          bare.iArr.symbol != nil and hasOwnSmemArena(fn, bare.iArr.symbol.name))
      if isSmemArg:
        markCalleeSmemParam(ctx, call.cName, i)
        # The bound param is a threadgroup reference, the call passes the lvalue.
        call.cArgs[i] = bare
    for pi, p in ctx.getFnParams(call.cName).pairs:
      if p.addressSpace != asSMEM:
        continue
      # Hoisted params have no declared argument yet, a param already
      # carrying its arena stays put.
      if pi < call.cArgs.len:
        continue
      let pname = p.ident.symbol.name
      if hasOwnSmemArena(fn, pname):
        # The caller owns the arena, bind it by name.
        let arg = newGpuIdent()
        arg.symbol.name = pname
        arg.symbol.typ = p.typ
        call.cArgs.add(arg)
      elif attGlobal in fn.pAttributes:
        let arrType = p.typ.to
        doAssert arrType.kind in {gtArray, gtObject, gtGenericInst},
          "smem arena synthesis needs the hoisted type, got: " & $arrType.kind
        let arena = GpuAst(kind: gpuVar)
        arena.vName = newGpuIdent()
        arena.vName.symbol.name = pname
        arena.vName.symbol.typ = arrType
        arena.vType = arrType
        arena.addressSpace = asSMEM
        arena.vInit = GpuAst(kind: gpuDiscard)
        arena.vMutable = true
        fn.pBody.statements.insert(arena, 0)
        let arg = newGpuIdent()
        arg.symbol.name = pname
        arg.symbol.typ = p.typ
        call.cArgs.add(arg)
      else:
        doAssert false, "device fn '" & $call.cName.symbol.iSym &
          "' hoists its {.smem.} var '" & pname &
          "' to a threadgroup param, the caller '" & $fn.pName.symbol.iSym &
          "' must declare a matching {.smem.} var"

proc smemArenaImpl(ctx: var GpuContext) =
  # Sweep 1: the tables share GpuAst refs, hoisting is idempotent.
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc and attGlobal notin fn.pAttributes:
      fn.hoistSmemVars()
      ctx.allFnTab[fnKey] = fn
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc and attGlobal notin fn.pAttributes:
      fn.hoistSmemVars()
      ctx.genericInsts[fnKey] = fn

  # Sweep 2: bind call sites, each function once, the tables can hold
  # the same body under several keys.
  var done: seq[string]
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      let key = fn.pName.symbol.iSym
      if key notin done:
        done.add key
        bindSmemCallSites(ctx, fn)
      ctx.allFnTab[fnKey] = fn
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      let key = fn.pName.symbol.iSym
      if key notin done:
        done.add key
        bindSmemCallSites(ctx, fn)
      ctx.genericInsts[fnKey] = fn

# ═══════════════════════════════════════════════════════════════════════
# Coordinate-builtin param materialization
# ═══════════════════════════════════════════════════════════════════════

proc collectCoordBuiltinIdents(n: GpuAst, acc: var seq[(string, GpuCoordBuiltinKind)]) =
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.coordBuiltin != gbkNone:
      let name = n.ident()
      if not acc.anyIt(it[0] == name):
        acc.add (name, n.symbol.coordBuiltin)
  else:
    for ch in n:
      collectCoordBuiltinIdents(ch, acc)

proc collectCallees(n: GpuAst, callees: var seq[GpuAst]) =
  case n.kind
  of gpuCall:
    if n.cName.symbol.synchroBuiltin == gbkNone and not callees.anyIt(it == n.cName):
      callees.add n.cName
    for ch in n:
      collectCallees(ch, callees)
  else:
    for ch in n:
      collectCallees(ch, callees)

proc fnBuiltinNeedsImpl(ctx: GpuContext, fn: GpuAst, acc: var seq[(string, GpuCoordBuiltinKind)], visited: var seq[string]) =
  let key = fn.pName.symbol.iSym
  if key in visited:
    return
  visited.add key
  collectCoordBuiltinIdents(fn.pBody, acc)
  var callees: seq[GpuAst]
  collectCallees(fn.pBody, callees)
  for calleeIdent in callees:
    let callee = ctx.allFnTab.getOrDefault(calleeIdent,
                                           ctx.genericInsts.getOrDefault(calleeIdent))
    if not callee.isNil:
      if callee.kind == gpuProc:
        fnBuiltinNeedsImpl(ctx, callee, acc, visited)

proc fnBuiltinNeeds(ctx: GpuContext, fn: GpuAst): seq[(string, GpuCoordBuiltinKind)] =
  var visited: seq[string]
  fnBuiltinNeedsImpl(ctx, fn, result, visited)

proc builtinParamType(kind: GpuCoordBuiltinKind): GpuType =
  if kind == gbkThreadIndexInThreadgroup:
    GpuType(kind: gtUint32)
  else:
    GpuType(kind: gtGenericInst, gName: "uint3")

proc appendBuiltinParams(fn: GpuAst, needs: seq[(string, GpuCoordBuiltinKind)]) =
  for (name, kind) in needs:
    let typ = builtinParamType(kind)
    let sym = newSymbol(name, iSym = name & "_builtin", symKind = gsDeviceKernelParam)
    sym.coordBuiltin = coordBuiltinKind(name)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    fn.pParams.add GpuParam(ident: ident, typ: typ,
                            addressSpace: asRMEM, passByRef: false)

proc appendBuiltinForwardingArgs(n: var GpuAst, needs: seq[(string, GpuCoordBuiltinKind)]) =
  for (name, _) in needs:
    let sym = newSymbol(name, iSym = name & "_builtin", symKind = gsLocal)
    n.cArgs.add GpuAst(kind: gpuIdent, symbol: sym)

proc materializeCallArgsImpl(ctx: GpuContext, n: var GpuAst, needs: Table[string, seq[(string, GpuCoordBuiltinKind)]]) =
  case n.kind
  of gpuCall:
    let callee = ctx.allFnTab.getOrDefault(n.cName,
                                           ctx.genericInsts.getOrDefault(n.cName))
    var calleeNeeds: seq[(string, GpuCoordBuiltinKind)]
    if not callee.isNil:
      if callee.kind == gpuProc:
        calleeNeeds = needs.getOrDefault(callee.pName.symbol.iSym, @[])
    for ch in mitems(n):
      materializeCallArgsImpl(ctx, ch, needs)
    appendBuiltinForwardingArgs(n, calleeNeeds)
  else:
    for ch in mitems(n):
      materializeCallArgsImpl(ctx, ch, needs)

proc materializeIndexBuiltinParamsImpl(ctx: var GpuContext) =
  var needs = initTable[string, seq[(string, GpuCoordBuiltinKind)]]()
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      needs[fn.pName.symbol.iSym] = fnBuiltinNeeds(ctx, fn)
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      needs[fn.pName.symbol.iSym] = fnBuiltinNeeds(ctx, fn)

  var done = initHashSet[string]()
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      let key = fn.pName.symbol.iSym
      if key in done:
        continue
      done.incl key
      appendBuiltinParams(fn, needs[key])
      materializeCallArgsImpl(ctx, fn.pBody, needs)
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      let key = fn.pName.symbol.iSym
      if key in done:
        continue
      done.incl key
      appendBuiltinParams(fn, needs[key])
      materializeCallArgsImpl(ctx, fn.pBody, needs)
