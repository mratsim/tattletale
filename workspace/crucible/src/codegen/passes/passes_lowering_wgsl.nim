# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            WGSL lowering passes
#
# ############################################################
#
# Shape the IR into WGSL-legal form, registered by the WebGPU pipeline:
# - global fns lose their params, they become storage bindings
# - pointer args are address-of'd, WGSL-forbidden AST patterns rewritten
# - structs carrying pointers expand to ptr/scalar leaves

import std / [macros, strformat, strutils, sugar, sequtils, sets, tables, options]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_registry
import ./pass_helpers
import ./passes_low_address_spaces
import ./passes_lowering
import ./passes_low_pointers
import ./passes_low_views
import ./passes_legalizations
import ./passes_validations
import ./passes_optimizations

{.experimental: "codeReordering".}

const WgslReserved = [
  "layout", "storage", "uniform", "workgroup", "override", "diagnostic",
  "enable", "requires", "alias", "bitcast", "discard", "push_constant",
  "readonly", "read_write", "function", "private", "compute", "vertex",
  "fragment", "type"
].toHashSet()

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerWgslLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = false)
  # Array copies surface unlowered, WGSL has no memcpy builtin and the printer
  # rejects the requiresMemcpy node
  reg.registerSharedLoweringPasses()
  # Struct params carrying pointers explode to leaves before the passes below reshape pointer code.
  reg.registerRenameReservedIdentifiers("wgslReservedIdentifiers", WgslReserved)
  reg.registerExpandStructKernelParams()
  reg.registerInstantiateImplicitAddressSpaceGenerics()
  reg.registerRecordPtrFieldBindings()
  reg.registerLiftKernelParamsToGlobals()
  reg.registerPullConstantPragmaVars()
  reg.registerRemoveStructPointerFields()
  reg.registerAddressOfInjection()
  reg.registerInlinePointerParamCalls()
  reg.registerMakeCodeValid()
  reg.registerPatchBoolToI32()
  reg.registerMarkWrittenGlobals()
  reg.registerStripGlobalPointerTypes()
  reg.registerCheckCodeValidWgsl()

func registerPullConstantPragmaVars(reg: var PassRegistry) =
  reg.register("pullConstantPragmaVars", pkTransform, phaseMain,
    "Hoists {.const.} pragma vars out of the top-level var block",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      if ctx.globalBlocks.len > 0:
        pullConstantPragmaVarsImpl(ctx, ctx.globalBlocks[0])
  )

func registerRemoveStructPointerFields(reg: var PassRegistry) =
  reg.register("removeStructPointerFields", pkTransform, phaseMain,
    "Drops pointer fields from struct types: WGSL has no pointer struct fields",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      if ctx.globalBlocks.len > 1:
        removeStructPointerFieldsImpl(ctx.globalBlocks[1])
  )

func registerAddressOfInjection(reg: var PassRegistry) =
  reg.register("injectAddressOf", pkTransform, phaseMain,
    "Replaces storage-buffer idents with &ident in global fns",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        if fn.isGlobal():
          ctx.injectAddressOfImpl(fn)
  )

func registerRecordPtrFieldBindings(reg: var PassRegistry) =
  reg.register("recordPtrFieldBindings", pkTransform, phaseMain,
    "Records where struct pointer fields bind: the printer rewrites field accesses to the recorded kernel-param ident",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      let fns = toSeq(ctx.fnTab.pairs)
      for (fnIdent, fn) in fns:
        ctx.recordPtrFieldBindingsImpl(fn)
  )

func registerLiftKernelParamsToGlobals(reg: var PassRegistry) =
  reg.register("liftKernelParamsToGlobals", pkTransform, phaseMain,
    "Moves global fns' params to ctx.globals and rewrites their uses",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        if fn.isGlobal():
          for p in fn.pParams:
            ctx.globals[p.ident.symbol.iSym] = p
          fn.pParams.setLen(0)
          updateSymsInGlobalsImpl(ctx, fn)
  )

func registerInlinePointerParamCalls(reg: var PassRegistry) =
  reg.register("inlinePointerParamCalls", pkTransform, phaseMain,
    "Inlines calls whose pointer arguments cannot cross a WGSL function signature",
    dependsOn = @["injectAddressOf"],
    run = proc(ctx: var GpuContext): void =
      ctx.inlinePointerParamCalls()
  )

func registerMakeCodeValid(reg: var PassRegistry) =
  reg.register("makeCodeValid", pkTransform, phaseMain,
    "Addresses WGSL AST patterns (compound assign, struct ptr fields)",
    dependsOn = @["inlinePointerParamCalls"],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.mpairs:
        ctx.makeCodeValidImpl(fn, inGlobal = fn.isGlobal())
  )

func registerMarkWrittenGlobals(reg: var PassRegistry) =
  reg.register("markWrittenGlobals", pkTransform, phaseMain,
    "Upgrades globals written through the kernel body to the kVar flavor, the printer derives read_write from it",
    dependsOn = @["makeCodeValid"],
    run = proc(ctx: var GpuContext): void =
      for (_, fn) in ctx.fnTab.mpairs:
        markWrittenGlobalsImpl(ctx, fn.pBody)
  )

func registerStripGlobalPointerTypes(reg: var PassRegistry) =
  reg.register("stripGlobalPointerTypes", pkTransform, phaseMain,
    "Strips the pointer indirection off globals: WGSL storage bindings cannot be ptr types",
    dependsOn = @["patchBoolToI32"],
    run = proc(ctx: var GpuContext): void =
      for iSym, g in ctx.globals.mpairs:
        if g.typ.kind == gtPtr:
          ctx.globals[iSym].typ = g.typ.to
  )

func registerCheckCodeValidWgsl(reg: var PassRegistry) =
  reg.register("checkCodeValidWgsl", pkTransform, phaseMain,
    "Validates WGSL constraints after transformations",
    dependsOn = @["makeCodeValid"],
    run = proc(ctx: var GpuContext): void =
      for (fnIdent, fn) in ctx.fnTab.pairs:
        ctx.checkCodeValidImpl(fn)
  )

# ═══════════════════════════════════════════════════════════════════════
# WGSL printer helpers
# ═══════════════════════════════════════════════════════════════════════

func shortAddrSpace(addrSpace: AddressSpace): string =
  case addrSpace
  of asDevice: "s"
  of asConstant: "u"
  of asSMEM: "w"
  of asRMEM: "l"

proc determineMutability(arg: GpuAst): bool =
  case arg.kind
  of gpuIdent: (not arg.symbol.isNil) and arg.symbol.typ != nil and arg.symbol.typ.kind == gtPtr
  of gpuAddr: true
  of gpuDeref: true
  of gpuCall: false
  of gpuIndex: arg.iArr.determineMutability()
  of gpuDot: arg.dParent.determineMutability()
  of gpuLit: false
  of gpuBinOp: false
  of gpuBlock: arg.statements[^1].determineMutability()
  of gpuPrefix: false
  of gpuConv: false
  of gpuCast: arg.cExpr.determineMutability()
  else:
    error "Cannot determine mutability from: " & $arg

proc determineIdent(arg: GpuAst): GpuAst =
  template dfl(): untyped = GpuAst(kind: gpuDiscard)
  case arg.kind
  of gpuIdent: arg
  of gpuAddr: arg.aOf.determineIdent()
  of gpuDeref: arg.dOf.determineIdent()
  of gpuCall: dfl()
  of gpuIndex: arg.iArr.determineIdent()
  of gpuDot: arg.dParent.determineIdent()
  of gpuLit: dfl()
  of gpuBinOp: dfl()
  of gpuBlock: arg.statements[^1].determineIdent()
  of gpuPrefix: dfl()
  of gpuConv: dfl()
  of gpuCast: arg.cExpr.determineIdent()
  of gpuObjConstr: dfl()
  of gpuMaterialize: arg.mExpr.determineIdent()
  else:
    error "Cannot determine ident from: " & $arg

# ═══════════════════════════════════════════════════════════════════════
# Written globals
# ═══════════════════════════════════════════════════════════════════════

proc markWrittenGlobalsImpl(ctx: var GpuContext, n: GpuAst) =
  ## Marks globals assigned anywhere in `n` as kVar.
  case n.kind
  of gpuAssign:
    collectWrittenGlobals(ctx, n.aLeft)
  else:
    for ch in n:
      ctx.markWrittenGlobalsImpl(ch)

proc collectWrittenGlobals(ctx: var GpuContext, n: GpuAst) =
  if n.kind == gpuIndex:
    # a[i] = x writes the base a, the index expression i is only evaluated
    # (a[b[j]] = x must not mark b written)
    ctx.collectWrittenGlobals(n.iArr)
    return
  if n.kind == gpuIdent and n.symbol != nil and
     n.symbol.iSym in ctx.globals:
    ctx.globals[n.symbol.iSym].flavor = kVar
  for ch in n:
    ctx.collectWrittenGlobals(ch)

# ═══════════════════════════════════════════════════════════════════════
# Forward declarations consumed by wgsl_lang.preprocess
# ═══════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════
# Address-of injection
# ═══════════════════════════════════════════════════════════════════════

func injectAddressOfImpl(ctx: var GpuContext, n: var GpuAst) =
  ## Replaces storage-buffer idents with `gpuAddr(ident)` in global fns.
  ## Also handles bool patching: bool globals are wrapped in gpuConv(expr, gtBool).
  case n.kind
  of gpuIdent:
    if n.symbol.iSym in ctx.globals and (let p = ctx.globals[n.symbol.iSym]; p.typ.kind == gtPtr):
      n = GpuAst(kind: gpuAddr, aOf: n)
    # bool load convs are patchBoolToI32's job
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol.iSym in ctx.globals and
       (let p = ctx.globals[n.dOf.symbol.iSym]; p.typ.kind == gtPtr):
      n = n.dOf
  else:
    for ch in n.mitems:
      ctx.injectAddressOfImpl(ch)

# ═══════════════════════════════════════════════════════════════════════
# Constant-var pulling
# ═══════════════════════════════════════════════════════════════════════

func pullConstantPragmaVarsImpl(ctx: var GpuContext, blk: var GpuAst) =
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

# ═══════════════════════════════════════════════════════════════════════
# Struct pointer-field stripping
# ═══════════════════════════════════════════════════════════════════════

func removeStructPointerFieldsImpl(blk: var GpuAst) =
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

# ═══════════════════════════════════════════════════════════════════════
# Code validation
# ═══════════════════════════════════════════════════════════════════════

proc makeCodeValidImpl(ctx: var GpuContext, n: var GpuAst, inGlobal: bool) =
  ## Rewrites WGSL-invalid AST patterns: struct pointer field replacement,
  ## object constructor stripping, call signature updates, var type updates.
  case n.kind
  of gpuBinOp:
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
        var v = ctx.structsWithPtrs[(id, field)]
        # Prefer the variable-keyed entry:
        # the type-keyed table cannot tell two variables of one struct type apart.
        let owner = if p.kind == gpuIdent: p.ident() else: ""
        if owner.len > 0 and (owner, field) in ctx.ptrFieldBindings:
          v = ctx.ptrFieldBindings[(owner, field)]
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
          error "Assignment of a struct pointer field is not supported"
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
  of gpuCall:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
  of gpuVar:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)
    if n.vType.kind == gtPtr:
      let rightId = n.vInit.determineIdent()
      # The RHS ident's own type may be a non-pointer such as `addr arr[0]`: only a pointer RHS has a mutability flag.
      let rightTyp = if rightId.symbol != nil: rightId.symbol.typ else: nil
      let rhsIsPtr = not rightTyp.isNil and rightTyp.kind == gtPtr
      n.vType.mutable = rhsIsPtr and rightTyp.mutable
      n.vName.symbol.typ.mutable = rhsIsPtr and rightTyp.mutable
  else:
    for ch in n.mitems:
      ctx.makeCodeValidImpl(ch, inGlobal)

func checkCodeValidImpl(ctx: var GpuContext, n: GpuAst) =
  ## Checks WGSL validity constraints.
  case n.kind
  of gpuVar:
    if n.vType.kind == gtPtr and n.vMutable:
      let code = "<WGSL>"
      error "var to pointer type invalid in WGSL"
  else:
    for ch in n:
      ctx.checkCodeValidImpl(ch)

func updateSymsInGlobalsImpl(ctx: var GpuContext, n: GpuAst) =
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

# ═══════════════════════════════════════════════════════════════════════
# Generic instantiation
# ═══════════════════════════════════════════════════════════════════════

proc recordPtrFieldBindingsImpl(ctx: var GpuContext, n: GpuAst, owner = "") =
  ## Records where struct pointer fields bind: a construction's pointer-typed
  ## field init is a kernel-param ident, the printer rewrites `foo.field`
  ## accesses to it
  ## Keys:
  ## - (type, field)
  ## - (variable, field), the type-keyed entry cannot separate same-type variables
  case n.kind
  of gpuVar:
    ctx.recordPtrFieldBindingsImpl(n.vInit, n.vName.ident())
  of gpuObjConstr:
    for f in n.ocFields:
      if f.typ.kind == gtPtr:
        doAssert f.value.kind in [gpuAddr, gpuIdent]
        let id = f.value.determineIdent()
        doAssert id.symbol.symKind == gsGlobalKernelParam
        ctx.structsWithPtrs[(n.ocType, f.name)] = id
        if owner.len > 0:
          ctx.ptrFieldBindings[(owner, f.name)] = id
  # call arguments leave the owner's scope, the old scan reset the owner there
  of gpuCall:
    for ch in n:
      ctx.recordPtrFieldBindingsImpl(ch)
  else:
    for ch in n:
      ctx.recordPtrFieldBindingsImpl(ch, owner)

# ═══════════════════════════════════════════════════════════════════════
# Pointer-param call inlining
# ═══════════════════════════════════════════════════════════════════════

proc substituteIdents(n: var GpuAst, subst: Table[string, GpuAst]) =
  ## Rewrites idents onto their call-site bindings.
  ## The substitution step of call inlining.
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in subst:
      n = subst[n.symbol.iSym]
  else:
    for ch in mitems(n):
      substituteIdents(ch, subst)

proc hasNonFnPtrParam(fn: GpuAst): bool =
  ## True when a pointer parameter sits outside the `function` address space.
  ## WGSL rejects storage, uniform and workgroup pointers as user-function parameter types.
  for p in fn.pParams:
    if p.typ.kind == gtPtr and p.typ.addressSpace != asRMEM:
      return true

proc hasPtrStructParam(fn: GpuAst, ctx: GpuContext): bool =
  for p in fn.pParams:
    if p.typ != nil and p.typ.kind == gtObject:
      for f in p.typ.fields:
        if f.typ.kind == gtPtr and (p.typ, f.name) in ctx.structsWithPtrs:
          return true

proc inlinePointerParamCalls(ctx: var GpuContext) =
  ## Inlines device-proc calls whose pointer parameters have no WGSL
  ## parameter type. WGSL rejects storage, uniform and workgroup pointers
  ## as user-function parameter types. Runtime-sized arrays cannot cross a call by value.
  ##
  ## Per call:
  ## - every argument binds to a fresh `let` alias holding the pointer at function scope
  ## - the callee body substitutes onto the aliases and splices over the call
  ## - the inlined callee drops from the function table
  ##
  ## Calls whose pointer arguments all live in `function` space keep real calls.
  ##
  ## Contracts:
  ## - callees return void, an inlined call produces no expression value
  ## - callee bodies hold no `return`, a spliced `return` exits the caller
  var inlCount = 0
  var usedNames = initHashSet[string]()
  for fn in ctx.fnTab.values:
    fn.pBody.collectIdentNames(usedNames)
    for p in fn.pParams:
      usedNames.incl p.ident.ident()
  for g in ctx.globals.values:
    g.ident.collectIdentNames(usedNames)

  proc inlineCall(ctx: var GpuContext, call: GpuAst, usedNames: HashSet[string]): seq[GpuAst] =
    ## Expands one call into parameter `let` bindings plus the substituted callee body.
    let callee = ctx.allFnTab[call.cName]
    doAssert callee.pRetType.kind == gtVoid,
      "WGSL: device proc `" & call.cName.ident() &
      "` takes a non-function-space pointer parameter, only void procs inline"
    doAssert not callee.pBody.containsReturnStmt(),
      "WGSL: device proc `" & call.cName.ident() &
      "` takes a non-function-space pointer parameter, its body must not `return`"
    var subst = initTable[string, GpuAst]()
    for i, p in callee.pParams:
      let arg = call.cArgs[i]
      var fresh: string
      while true:
        fresh = "inl" & $inlCount & '_' & p.ident.ident()
        inc inlCount
        if fresh notin usedNames:
          break
      var typ = p.typ
      if typ.kind == gtPtr:
        # `mutable` selects `read_write` in the WGSL pointer type.
        typ.mutable = arg.determineMutability()
      if typ.kind == gtObject and arg.kind == gpuIdent:
        # A struct value does not carry its pointer field: the pointer lives
        # in a global binding. Without rebinding, the callee body reads
        # whatever same-named buffer it finds.
        for f in typ.fields:
          if f.typ.kind == gtPtr and (arg.ident(), f.name) in ctx.ptrFieldBindings:
            ctx.ptrFieldBindings[(fresh, f.name)] = ctx.ptrFieldBindings[(arg.ident(), f.name)]
      let sym = newSymbol(fresh, iSym = fresh, typ = typ, symKind = gsLocal)
      let binding = GpuAst(kind: gpuIdent, symbol: sym)
      subst[p.ident.symbol.iSym] = binding
      result.add GpuAst(kind: gpuVar, vMutable: false, vName: binding,
                        vType: typ, vInit: arg, vRequiresMemcpy: false,
                        addressSpace: asRMEM)
    var body = callee.pBody.clone()
    body.substituteIdents(subst)
    result.add body.statements

  proc inlineCandidate(ctx: GpuContext, call: GpuAst): bool =
    ## True when a call must inline.
    ## The callee takes a pointer outside `function` space, or a struct
    ## holding one: neither crosses a WGSL function signature.
    if call.cName.symbol == nil:
      return false
    if call.cName notin ctx.allFnTab:
      return false
    let callee = ctx.allFnTab[call.cName]
    if callee.kind != gpuProc or callee.isGlobal():
      return false
    result = callee.hasNonFnPtrParam() or callee.hasPtrStructParam(ctx)

  proc inlineIn(ctx: var GpuContext, n: var GpuAst): bool =
    ## Splices inlinable calls in `n`, depth-first. Returns true when `n` changed.
    case n.kind
    of gpuBlock:
      var changed = false
      var i = 0
      while i < n.statements.len:
        var s = n.statements[i]
        if s.kind == gpuCall and ctx.inlineCandidate(s):
          let repl = ctx.inlineCall(s, usedNames)
          n.statements = n.statements[0 ..< i] & repl & n.statements[i + 1 .. ^1]
          changed = true
          # the spliced statements are visited in place, `i` stays
        else:
          changed = ctx.inlineIn(s) or changed
          n.statements[i] = s
          inc i
      result = changed
    else:
      var changed = false
      for ch in mitems(n):
        changed = ctx.inlineIn(ch) or changed
      result = changed

  for key in toSeq(ctx.fnTab.keys):
    var body = ctx.fnTab[key].pBody
    var changed = true
    while changed:
      changed = ctx.inlineIn(body)
    ctx.fnTab[key].pBody = body

  # Drop the inlined callees: their parameters cannot cross a WGSL function signature.
  for key in toSeq(ctx.fnTab.keys):
    let fn = ctx.fnTab[key]
    if not fn.isGlobal() and (fn.hasNonFnPtrParam() or fn.hasPtrStructParam(ctx)):
      ctx.fnTab.del(key)
