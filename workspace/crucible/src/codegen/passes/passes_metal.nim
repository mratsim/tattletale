# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [sequtils, sets, tables]

import ../ir/gpu_types
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations
import ./passes_validations
import ./passes_preprocessing

proc materializeIndexBuiltinParamsImpl*(ctx: var GpuContext)

proc registerMetalPasses*(reg: var PassRegistry) =
  ## Register Metal-specific passes: MSL keyword rejection, coordinate-builtin param materialization.
  reg.register("rejectMetalKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved MSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["kernel", "device", "constant", "threadgroup"], "MSL")
  )
  # The runtime path fills `genericInsts`, not `allFnTab`, so the keyword pass
  # above iterates nothing there. The printer's `checkReservedIdent` rejects
  # reserved identifiers at emission, guarding params, locals and function names.
  # MSL kernels take exploded ptr/scalar/value params, never whole structs.
  reg.registerExplodePointerStructParams()
  reg.register("materializeIndexBuiltinParams", pkTransform, phaseMain,
    "Materializes coordinate-builtin binding: builtin params appended to pParams, call-site forwarding args",
    dependsOn = @["emitFunctionSignatures"],
    run = proc(ctx: var GpuContext): void =
      materializeIndexBuiltinParamsImpl(ctx)
  )

proc collectCoordBuiltinIdents(n: GpuAst, acc: var seq[(string, GpuCoordBuiltinKind)]) =
  ## Records coordinate-builtin identifiers in first-use order, deduped by name.
  ## The symbol's `coordBuiltin` decides, never the name: a local or declared
  ## param shadowing a canonical name carries `gbkNone` and is skipped.
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
  ## Records the functions a body calls, in first-use order, deduped.
  ## Barrier calls are skipped, they lower to a native statement, never a function call.
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
  ## Appends the coordinate builtins `fn` must bind, transitively: the builtins
  ## its own body references, then the needs of every device function it calls,
  ## each name first-seen once. A visited set guards recursive calls.
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
  ## Transitive coordinate-builtin needs of `fn`: builtins in the body,
  ## then builtins in every device function it calls.
  ## Every rewrite site reads the same first-seen order, so identity-named identifiers bind to the params.
  var visited: seq[string]
  fnBuiltinNeedsImpl(ctx, fn, result, visited)

proc builtinParamType(kind: GpuCoordBuiltinKind): GpuType =
  ## IR type of a coordinate builtin bound as a param. The flat thread index
  ## binds as scalar `uint32`, everything else as the MSL `uint3` vector spelling.
  ## `uint3` is a synthetic generic name carrying the printer's native spelling,
  ## no struct is ever registered for it.
  if kind == gbkThreadIndexInThreadgroup:
    GpuType(kind: gtUint32)
  else:
    GpuType(kind: gtGenericInst, gName: "uint3")

proc appendBuiltinParams(fn: GpuAst, needs: seq[(string, GpuCoordBuiltinKind)]) =
  ## Appends one plain param per transitive coordinate-builtin need, after the declared params, for kernels and device functions alike.
  ##
  ## The param's symbol carries the builtin kind, `coordBuiltin`, resolved
  ## from the canonical name via the builtin catalog, the same marking
  ## the frontend applies to builtin identifiers.
  ##
  ## The printer emits the attribute form for kernels and the plain form
  ## for device functions by the symbol alone.
  for (name, kind) in needs:
    let typ = builtinParamType(kind)
    let sym = newSymbol(name, iSym = name & "_builtin", symKind = gsDeviceKernelParam)
    sym.coordBuiltin = coordBuiltinKind(name)
    let ident = GpuAst(kind: gpuIdent, symbol: sym)
    fn.pParams.add GpuParam(ident: ident, typ: typ,
                            addressSpace: asRMEM, passByRef: false)

proc appendBuiltinForwardingArgs(n: var GpuAst, needs: seq[(string, GpuCoordBuiltinKind)]) =
  ## Appends one forwarding arg per callee need, in the callee's needs order.
  ## The identity-named identifiers bind to the caller's own params
  ## in the emitted source (kernel attribute params or device-fn hidden params).
  for (name, _) in needs:
    let sym = newSymbol(name, iSym = name & "_builtin", symKind = gsLocal)
    n.cArgs.add GpuAst(kind: gpuIdent, symbol: sym)

proc materializeCallArgsImpl(ctx: GpuContext, n: var GpuAst, needs: Table[string, seq[(string, GpuCoordBuiltinKind)]]) =
  ## Appends the forwarding args to every `gpuCall` whose callee has
  ## coordinate-builtin needs, in the callee's needs order. The walk happens
  ## before the append so the argument list is not mutated mid-iteration.
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

proc materializeIndexBuiltinParamsImpl*(ctx: var GpuContext) =
  ## Materializes coordinate-builtin binding for the Metal backend.
  ##
  ## Contract:
  ## - every function receives its transitive needs as params appended
  ##   after the declared params, the symbol carrying the builtin kind
  ## - every call site forwards the callee's needs as trailing args
  ## - the closure analysis runs first, once per function, so every rewrite
  ##   reads the same memoized needs in the same first-seen order
  var needs = initTable[string, seq[(string, GpuCoordBuiltinKind)]]()
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    if fn.kind == gpuProc:
      needs[fn.pName.symbol.iSym] = fnBuiltinNeeds(ctx, fn)
  for fnKey in ctx.genericInsts.keys:
    var fn = ctx.genericInsts[fnKey]
    if fn.kind == gpuProc:
      needs[fn.pName.symbol.iSym] = fnBuiltinNeeds(ctx, fn)

  # A pulled-in device function registers in both tables under the same
  # symbol (`allFnTab`, `genericInsts`), rewrite each function once.
  # Kernels and device functions alike receive the needs as params.
  # The printer's attribute form vs plain form is decided by the symbol's
  # `coordBuiltin` at emission time.
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
