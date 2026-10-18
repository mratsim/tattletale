# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [sequtils, tables, options]

import ../ir/gpu_types
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations
import ./passes_validations
import ./passes_preprocessing


# WGSL pipeline steps consumed by wgsl_lang.preprocess: address-of injection
# plus const_mem var pulling, struct pointer field stripping, code validation
# and the generic instantiation scan.

proc getStructType*(n: GpuAst): GpuType
proc patchTypeImpl*(t: GpuType): GpuType
proc patchSymbolImpl*(n: GpuAst): GpuAst
proc genGenericName*(ctx: GpuContext, n: GpuAst, params: seq[GpuParam], callerParams: Table[string, GpuParam]): string
proc makeFnGeneric*(ctx: var GpuContext, fn: GpuAst, gi: GenericInst): GpuAst
proc getGenericArguments*(ctx: GpuContext, args: seq[GpuAst], params: seq[GpuParam], callerParams: Table[string, GpuParam]): seq[GenericArg]

proc injectAddressOfImpl*(ctx: var GpuContext, n: var GpuAst) =
  ## Replaces storage-buffer idents with `gpuAddr(ident)` in global fns.
  ## Also handles bool patching: bool globals are wrapped in gpuConv(expr, gtBool).
  case n.kind
  of gpuIdent:
    if n.symbol.iSym in ctx.globals and (let p = ctx.globals[n.symbol.iSym]; p.typ.kind == gtPtr):
      n = GpuAst(kind: gpuAddr, aOf: n)
    elif n.symbol.iSym in ctx.globals and (let p = ctx.globals[n.symbol.iSym]; p.typ.kind == gtBool):
      n = GpuAst(kind: gpuConv, convTo: GpuType(kind: gtBool), convExpr: n)
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol.iSym in ctx.globals and
       (let p = ctx.globals[n.dOf.symbol.iSym]; p.typ.kind == gtPtr):
      n = n.dOf
  else:
    for ch in n.mitems:
      ctx.injectAddressOfImpl(ch)

proc pullConstantPragmaVarsImpl*(ctx: var GpuContext, blk: var GpuAst) =
  ## Filters out `var foo {.const_mem.}: dtype` from global blocks and adds to globals.
  doAssert blk.kind == gpuBlock
  var i = 0
  while i < blk.len:
    let g = blk.statements[i]
    if g.kind == gpuVar and g.addressSpace == asConstant:
      doAssert g.vInit.kind == gpuDiscard, "{.const_mem.} var must not have init!"
      let param = GpuParam(ident: g.vName, typ: g.vType, addressSpace: asDevice)
      ctx.globals[param.ident.symbol.iSym] = param
      blk.statements.delete(i)
    else:
      inc i

proc removeStructPointerFieldsImpl*(blk: var GpuAst) =
  ## Removes ptr fields from struct definitions (WGSL limitation).
  doAssert blk.kind == gpuBlock
  for typ in blk.mitems:
    if typ.kind == gpuAlias: continue
    doAssert typ.kind == gpuTypeDef
    var i = 0
    while i < typ.tFields.len:
      let f = typ.tFields[i]
      if f.typ.kind == gtPtr:
        typ.tFields.delete(i)
      else:
        inc i

proc makeCodeValidImpl*(ctx: var GpuContext, n: var GpuAst, inGlobal: bool) =
  ## Rewrites WGSL-invalid AST patterns: struct pointer field replacement,
  ## object constructor stripping, call signature updates, var type updates.
  case n.kind
  of gpuBinOp:
    n = rewriteCompoundAssignmentImpl(n)
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
  of gpuObjConstr:
    let t = n.ocType
    var i = 0
    while i < n.ocFields.len:
      let f = n.ocFields[i]
      if (t, f.name) in ctx.structsWithPtrs:
        if f.typ.kind == gtPtr:
          n.ocFields.delete(i)
        else:
          inc i
      else:
        inc i
  of gpuDot:
    var p = n.dParent
    if p.kind notin [gpuIdent, gpuDeref]:
      for ch in n.mitems:
        ctx.makeCodeValidImpl(ch, inGlobal)
    else:
      let id = getStructType(p)
      doAssert n.dField.kind == gpuIdent
      let field = n.dField.ident()
      if id.kind != gtVoid and (id, field) in ctx.structsWithPtrs:
        let v = ctx.structsWithPtrs[(id, field)]
        if inGlobal:
          n = GpuAst(kind: gpuAddr, aOf: v)
        else:
          n = v
  of gpuAssign:
    if n.aLeft.kind == gpuDot and n.aLeft.dParent.kind in [gpuIdent, gpuDeref]:
      let dot = n.aLeft
      let id = getStructType(dot.dParent)
      if id.kind != gtVoid:
        doAssert dot.dField.kind == gpuIdent
        let field = dot.dField.ident()
        if (id, field) in ctx.structsWithPtrs:
          raiseAssert "Assignment of a struct pointer field is not supported"
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
  of gpuCall:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
    let fnName = n.cName
    if fnName in ctx.fnTab:
      let fn = ctx.fnTab[fnName]
      let params = fn.pParams
      for i, arg in n:
        let argId = arg.determineIdent()
        if argId.kind != gpuDiscard and argId.ident().len > 0:
          var p = params[i]
          let argSpace = ctx.resolveIdentAddressSpace(argId)
          if p.addressSpace != argSpace:
            p.addressSpace = argSpace
            ctx.varAddressSpaces[p.ident.symbol.iSym] = argSpace
            fn.pParams[i] = p
  of gpuVar:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
    if n.vType.kind == gtPtr:
      let rightId = n.vInit.determineIdent()
      let space = ctx.resolveIdentAddressSpace(rightId)
      ctx.varAddressSpaces[n.vName.symbol.iSym] = space
      # The RHS ident's own type may be a non-pointer such as `addr arr[0]`: only a pointer RHS carries a mutability flag.
      let rightTyp = if rightId.symbol != nil: rightId.symbol.typ else: nil
      let rhsIsPtr = not rightTyp.isNil and rightTyp.kind == gtPtr
      n.vType.mutable = rhsIsPtr and rightTyp.mutable
      n.vName.symbol.typ.mutable = rhsIsPtr and rightTyp.mutable
  else:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)

proc checkCodeValidImpl*(ctx: var GpuContext, n: GpuAst) =
  ## Checks WGSL validity constraints.
  case n.kind
  of gpuVar:
    if n.vType.kind == gtPtr and n.vMutable:
      let code = "<WGSL>"
      raiseAssert "var to pointer type invalid in WGSL"
  else:
    for ch in n:
      ctx.checkCodeValidImpl(ch)

proc updateSymsInGlobalsImpl*(ctx: var GpuContext, n: GpuAst) =
  ## Update symbols in global functions to reflect symbol kind/mutability.
  case n.kind
  of gpuIdent:
    if n.symbol.iSym in ctx.globals:
      n.symbol.symKind = gsGlobalKernelParam
      if n.symbol.typ.kind == gtPtr:
        let g = ctx.globals[n.symbol.iSym]
        n.symbol.typ.mutable = g.typ.kind == gtPtr
  else:
    for ch in n:
      ctx.updateSymsInGlobalsImpl(ch)

proc scanGenericsImpl*(ctx: var GpuContext, n: GpuAst, callerParams: Table[string, GpuParam]) =
  ## Scans for gpuCall nodes, generates generic instantiations for pointer args.
  case n.kind
  of gpuCall:
    let fn = n.cName
    if fn in ctx.allFnTab:
      let params = ctx.allFnTab[fn].pParams
      let gi = GenericInst(name: genGenericName(ctx, n, params, callerParams),
                           args: getGenericArguments(ctx, n.cArgs, params, callerParams))
      let anyPointers = params.anyIt(it.typ.kind == gtPtr)
      if anyPointers:
        n.cName = GpuAst(kind: gpuIdent,
                         symbol: newSymbol(gi.name, symKind = gsProc, iSym = gi.name))
        let gName = n.cName
        if gName notin ctx.fnTab:
          let fnCalled = ctx.allFnTab[fn].clone()
          let fnGen = makeFnGeneric(ctx, fnCalled, gi)
          ctx.fnTab[gName] = fnGen
          ctx.allFnTab[gName] = fnGen
          var callParams = initTable[string, GpuParam]()
          for p in fnGen.pParams:
            callParams[p.ident.symbol.iSym] = p
          ctx.scanGenericsImpl(fnGen, callParams)
      elif fn notin ctx.fnTab:
        let fnCalled = ctx.allFnTab[fn]
        ctx.fnTab[fn] = fnCalled
        for ch in fnCalled:
          ctx.scanGenericsImpl(ch, callerParams)
      for arg in n.cArgs:
        ctx.scanGenericsImpl(arg, callerParams)
    else:
      for ch in n:
        ctx.scanGenericsImpl(ch, callerParams)
  of gpuObjConstr:
    for f in n.ocFields:
      if f.typ.kind == gtPtr:
        doAssert f.value.kind in [gpuAddr, gpuIdent]
        let id = f.value.determineIdent()
        doAssert id.symbol.symKind == gsGlobalKernelParam
        ctx.structsWithPtrs[(n.ocType, f.name)] = id
  else:
    for ch in n:
      ctx.scanGenericsImpl(ch, callerParams)

proc getStructType*(n: GpuAst): GpuType =
  ## Given an ident, return the struct type or void if not a struct.
  doAssert n.kind in [gpuIdent, gpuDeref]
  var p = n
  if p.kind == gpuDeref:
    p = n.dOf
  result = if p.symbol.typ.kind == gtPtr and p.symbol.typ.to.kind == gtObject:
             p.symbol.typ.to
           elif p.symbol.typ.kind == gtObject:
             p.symbol.typ
           else: GpuType(kind: gtVoid)

proc genGenericName*(ctx: GpuContext, n: GpuAst, params: seq[GpuParam], callerParams: Table[string, GpuParam]): string =
  ## Generates unique name for a generic WGSL function instantiation.
  doAssert n.kind == gpuCall
  result = n.cName.ident() & '_'
  for i, arg in n.cArgs:
    let p = params[i]
    var s: string
    if p.typ.kind != gtPtr:
      s = "l"
    else:
      let argIdent = arg.determineIdent()
      var lArg: GpuAst = arg
      if argIdent.kind != gpuDiscard and argIdent.symbol.iSym in callerParams:
        lArg = callerParams[argIdent.symbol.iSym].ident
      let addrSpace = ctx.resolveIdentAddressSpace(lArg)
      let mutable = lArg.determineMutability()
      let m = if mutable: "mut" else: ""
      s = shortAddrSpace(addrSpace) & m
    result.add s
    if i < n.cArgs.high:
      result.add '_'

proc makeFnGeneric*(ctx: var GpuContext, fn: GpuAst, gi: GenericInst): GpuAst =
  ## Returns a cloned function with params updated for the generic instantiation.
  result = fn
  let pnSym = newSymbol(gi.name, symKind = gsProc)
  result.pName = GpuAst(kind: gpuIdent, symbol: pnSym)
  for i, p in result.pParams.mpairs:
    let arg = gi.args[i]
    p.addressSpace = arg.addrSpace
    ctx.varAddressSpaces[p.ident.symbol.iSym] = arg.addrSpace
    if p.ident.symbol.typ.kind == gtPtr:
      p.ident.symbol.typ.mutable = arg.mutable
    p.ident = patchSymbolImpl(p.ident)
    p.typ = p.ident.symbol.typ
  proc getIf(params: seq[GpuParam], n: GpuAst): Option[GpuParam] =
    doAssert n.kind == gpuIdent
    for p in params:
      if p.ident.symbol.iSym == n.symbol.iSym: return some(p)
  proc updateSyms(n: var GpuAst, params: seq[GpuParam]) =
    case n.kind
    of gpuIdent:
      let pOpt = params.getIf(n)
      if pOpt.isSome:
        let p = pOpt.get
        n.symbol.symKind = p.ident.symbol.symKind
        n.symbol.typ = p.typ
    else:
      for ch in n.mitems:
        updateSyms(ch, params)
  updateSyms(result.pBody, result.pParams)

proc patchSymbolImpl*(n: GpuAst): GpuAst =
  doAssert n.kind == gpuIdent
  result = n
  if n.symbol != nil and n.symbol.symKind == gsGlobalKernelParam:
    result.symbol.typ = patchTypeImpl(result.symbol.typ)

proc patchTypeImpl*(t: GpuType): GpuType =
  result = t
  if result.kind == gtBool:
    result.kind = gtInt32
  elif result.kind == gtPtr and result.to.kind == gtBool:
    result.to.kind = gtInt32

proc getGenericArguments*(ctx: GpuContext, args: seq[GpuAst], params: seq[GpuParam], callerParams: Table[string, GpuParam]): seq[GenericArg] =
  for i, arg in args:
    let p = params[i]
    if p.typ.kind != gtPtr:
      result.add GenericArg(addrSpace: asRMEM, mutable: false)
    else:
      let argIdent = arg.determineIdent()
      var lArg: GpuAst = arg
      if argIdent.kind != gpuDiscard and argIdent.symbol.iSym in callerParams:
        lArg = callerParams[argIdent.symbol.iSym].ident
      let addrSpace = ctx.resolveIdentAddressSpace(lArg)
      let mutable = lArg.determineMutability()
      result.add GenericArg(addrSpace: addrSpace, mutable: mutable)




proc registerWgslPasses*(reg: var PassRegistry) =
  ## Register WGSL-specific passes: keyword rejection, global-fn flattening, code validation.
  reg.register("rejectWGSLKeywords", pkValidation, phaseEarly,
    "Rejects identifiers that are reserved WGSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["override", "storage", "uniform", "workgroup"], "WGSL")
  )
  # Struct params carrying pointers explode to leaves before the passes below reshape pointer code.
  reg.registerExpandStructKernelParams()
  # Explode before the passes below reshape code carrying pointers.
  reg.register("injectAddressOf", pkTransform, phaseMain,
    "Replaces storage-buffer idents with &ident in global fns",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      # Apply pre-WGSL-standard preprocessing
      # 1. pullConstantPragmaVars on globalBlocks[0]
      if ctx.globalBlocks.len > 0:
        pullConstantPragmaVarsImpl(ctx, ctx.globalBlocks[0])
      # 2. removeStructPointerFields on globalBlocks[1]
      if ctx.globalBlocks.len > 1:
        removeStructPointerFieldsImpl(ctx.globalBlocks[1])
      # 3. Remove args from global functions, update syms
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        if fn.isGlobalFn():
          for p in fn.pParams:
            ctx.globals[p.ident.symbol.iSym] = p
          fn.pParams.setLen(0)
          updateSymsInGlobalsImpl(ctx, fn)
      # 4. Scan generics
      let fns = toSeq(ctx.fnTab.pairs)
      for (fnIdent, fn) in fns:
        let fnOrig = ctx.allFnTab[fnIdent]
        var callParams = initTable[string, GpuParam]()
        for p in fnOrig.pParams:
          callParams[p.ident.symbol.iSym] = p
        ctx.scanGenericsImpl(fn, callParams)
  )
  reg.register("injectAddressOfApply", pkTransform, phaseMain,
    "Applies gpuAddr wrapping on global fn idents",
    dependsOn = @["injectAddressOf"],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        if fn.isGlobalFn():
          ctx.injectAddressOfImpl(fn)
  )
  reg.register("makeCodeValid", pkTransform, phaseMain,
    "Addresses WGSL AST patterns (compound assign, struct ptr fields)",
    dependsOn = @["injectAddressOf"],
    run = proc(ctx: var GpuContext): void =
      # Collect value address spaces before call args and pointer aliases
      # are resolved from the authoritative map.
      ctx.collectValueAddressSpaces()
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        ctx.makeCodeValidImpl(fn, inGlobal = fn.isGlobalFn())
  )
  reg.register("checkCodeValidWgsl", pkTransform, phaseMain,
    "Validates WGSL constraints after transformations",
    dependsOn = @["makeCodeValid"],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.pairs:
        ctx.checkCodeValidImpl(fn)
  )
