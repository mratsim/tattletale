## Shared lowering impls.
##
## - registerSharedLoweringPasses registers the ones every backend wants,
##   per-backend passes register in passes_lowering_<backend>.nim
## - the structural impls moved to passes_low_pointers, passes_low_views,
##   passes_low_array_assign, renaming, pruning and reserved-keyword handling
## - each replaces an IR construct with its composition at the target ABI,
##   a policy differing per target takes a flag
## - patchBoolToI32 lives here too: only pipelines whose target stores bools
##   as machine words register it (WGSL, SPIR-V)

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

import std / [tables, sequtils, sets, strutils]
import std/macros # for the `warning` builtin (compile-time diagnostics)
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ../ir/ast_type_constructors
import ./pass_datatypes
import ./pass_registry
import ./passes_low_pointers
import ./passes_legalizations

# ############################################################
#
# Lowering
#
# ############################################################
#
# Lowering replaces IR constructs with their composition at the target
# calling convention: spans become ptr + len, compile-time params are erased.

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerSharedLoweringPasses*(reg: var PassRegistry) =
  reg.registerTagPointerLocals()
  reg.registerPruneCompileTimeParams()

func registerPatchBoolToI32*(reg: var PassRegistry) =
  ## Patches bool storage to i32 while bool uses stay bool.
  ## - `{.global.}`-kernel parameters declare i32 in the push-constant
  ##   interface (SPIR-V) and bool globals declare i32 storage (WGSL)
  ## - loads wrap in conv(gtBool): any nonzero i32 reads true, the WGSL
  ##   bool(i32) truthiness rule
  ## - assignment right-hand sides wrap in conv(gtInt32)
  reg.register("patchBoolToI32", pkTransform, phaseMain,
    "Patches bool globals and kernel params to i32 storage, conv nodes translate loads and stores",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.patchBoolToI32()
  )

func registerRenameReservedIdentifiers*(reg: var PassRegistry, name: static string, reserved: static HashSet[string]) =
  reg.register(name, pkTransform, phaseMain,
    "Renames identifiers colliding with backend reserved words: fn names, call sites, params, globals, struct fields, field accesses",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      ctx.renameReservedIdentifiers(reserved)
  )

# ═══════════════════════════════════════════════════════════════════════
# Reserved keyword renaming
# ═══════════════════════════════════════════════════════════════════════

proc renameReservedKeywords(name: string, reserved: HashSet[string]): string =
  if name in reserved:
    name & "_nim"
  else:
    name

proc renameIdents(n: var GpuAst, symToRename: Table[string, string]) =
  ## Substitutes the names of identifiers whose iSym is in the table
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in symToRename:
      n.symbol.name = symToRename[n.symbol.iSym]
  of gpuCall:
    # the child iterator yields cArgs only, the callee ident is walked here
    renameIdents(n.cName, symToRename)
    for ch in mitems(n):
      renameIdents(ch, symToRename)
  else:
    for ch in mitems(n):
      renameIdents(ch, symToRename)

proc renameFieldAccesses(n: var GpuAst, reserved: HashSet[string]) =
  ## Renames field accesses whose name is a target keyword, to match `renameReservedIdentifiers`
  case n.kind
  of gpuDot:
    if n.dField.kind == gpuIdent:
      n.dField.symbol.name = renameReservedKeywords(n.dField.symbol.name, reserved)
    renameFieldAccesses(n.dParent, reserved)
  else:
    for ch in mitems(n):
      renameFieldAccesses(ch, reserved)

proc renameReservedIdentifiers(ctx: var GpuContext, reserved: HashSet[string]) =
  ## Renames identifiers colliding with target keywords at every emitted site:
  ## - fn names, call sites and fn params
  ## - globals, where WGSL lifts kernel params
  ## - struct fields, including block-wrapped type defs
  ## - field accesses
  var fnRenames = initTable[string, string]()
  for fn in ctx.allFnTab.values:
    let oldName = fn.pName.ident()
    let safeName = renameReservedKeywords(oldName, reserved)
    if oldName != safeName:
      fnRenames[fn.pName.symbol.iSym] = safeName
      fn.pName.symbol.name = safeName
  for fn in ctx.genericInsts.values:
    let oldName = fn.pName.ident()
    let safeName = renameReservedKeywords(oldName, reserved)
    if oldName != safeName:
      fnRenames[fn.pName.symbol.iSym] = safeName
      fn.pName.symbol.name = safeName
  for g in mvalues(ctx.globals):
    g.ident.symbol.name = renameReservedKeywords(g.ident.ident(), reserved)
  for tab in [ctx.allFnTab, ctx.genericInsts, ctx.fnTab]:
    for fn in tab.values:
      var renames = initTable[string, string]()
      for p in mitems(fn.pParams):
        let oldName = p.ident.ident()
        let safeName = renameReservedKeywords(oldName, reserved)
        if oldName != safeName:
          renames[p.ident.symbol.iSym] = safeName
          p.ident.symbol.name = safeName
      if renames.len > 0:
        renameIdents(fn.pBody, renames)
  for fn in ctx.allFnTab.values:
    renameIdents(fn.pBody, fnRenames)
  for fn in ctx.genericInsts.values:
    renameIdents(fn.pBody, fnRenames)
  for fn in ctx.fnTab.values:
    renameIdents(fn.pBody, fnRenames)
  proc renameFieldDecls(n: var GpuAst) =
    case n.kind
    of gpuTypeDef:
      for f in mitems(n.tFields):
        f.name = renameReservedKeywords(f.name, reserved)
    else:
      for ch in mitems(n):
        renameFieldDecls(ch)
  for blk in mitems(ctx.globalBlocks):
    renameFieldDecls(blk)
  for _, typ in mpairs(ctx.types):
    # field decls live in the type registry before the target preprocess
    # farms them into globalBlocks
    renameFieldDecls(typ)
  for tab in [ctx.allFnTab, ctx.genericInsts, ctx.fnTab]:
    for fn in tab.values:
      renameFieldAccesses(fn.pBody, reserved)
  for blk in mitems(ctx.globalBlocks):
    renameFieldAccesses(blk, reserved)

func registerPruneCompileTimeParams(reg: var PassRegistry) =
  reg.register("pruneCompileTimeParams", pkTransform, phaseMain,
    "Drop compile-time-only parameters (static records, dead scalar statics, typedescs, phantom values) and erase their call arguments",
    run = pruneCompileTimeParams
  )

# ═══════════════════════════════════════════════════════════════════════
# Compile-time parameter pruning
# ═══════════════════════════════════════════════════════════════════════

proc eraseCallArgs(n: var GpuAst, callee: string, dropped: seq[int]) =
  ## Removes the arguments at `dropped` positions from every call to `callee`
  if n.kind == gpuCall and n.cName.symbol.name == callee:
    var kept: seq[GpuAst]
    for i, arg in n.cArgs:
      if i notin dropped:
        kept.add arg
    n.cArgs = kept
  for child in n.mitems:
    child.eraseCallArgs(callee, dropped)

proc pruneCompileTimeParams(ctx: var GpuContext) =
  ## Drops parameters with no runtime bytes: compile-time arguments and phantoms,
  ## the value lives in the type
  ## Erases the matching call arguments.
  var drops: Table[string, seq[int]]
  for fnKey in ctx.allFnTab.keys:
    let fn = ctx.allFnTab[fnKey]
    if fn.kind != gpuProc:
      continue
    var dropped: seq[int]
    for i, p in fn.pParams:
      if size(p.typ) == 0:
        # no runtime representation: compile-time args and phantoms
        dropped.add i
    if dropped.len > 0:
      drops[fn.pName.symbol.name] = dropped
  for fnKey in ctx.allFnTab.keys:
    var fn = ctx.allFnTab[fnKey]
    for callee, dropped in drops:
      fn.pBody.eraseCallArgs(callee, dropped)
    if fn.pName.symbol.name in drops:
      let dropped = drops[fn.pName.symbol.name]
      var kept: seq[GpuParam]
      for i, p in fn.pParams:
        if i notin dropped:
          kept.add p
      fn.pParams = kept

# ═══════════════════════════════════════════════════════════════════════
# Bool storage lowering
# ═══════════════════════════════════════════════════════════════════════

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

proc patchBoolToI32Impl(n: var GpuAst, boolVals, boolPtrs: HashSet[string]) =
  ## Symbols keep their bool type, storage is i32: loads wrap in conv(gtBool),
  ## assignment right-hand sides in conv(gtInt32), the target stays bare.
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in boolVals:
      n = GpuAst(kind: gpuConv, convTo: GpuType(kind: gtBool), convExpr: n)
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol != nil and n.dOf.symbol.iSym in boolPtrs:
      n = GpuAst(kind: gpuConv, convTo: GpuType(kind: gtBool), convExpr: n)
  of gpuAssign:
    let lhs = n.aLeft.determineIdent()
    if lhs.kind == gpuIdent and lhs.symbol != nil and
       (lhs.symbol.iSym in boolVals or lhs.symbol.iSym in boolPtrs):
      n.aRight = GpuAst(kind: gpuConv, convTo: GpuType(kind: gtInt32), convExpr: n.aRight)
    for ch in n.aLeft.mitems:
      ch.patchBoolToI32Impl(boolVals, boolPtrs)
    n.aRight.patchBoolToI32Impl(boolVals, boolPtrs)
  else:
    for ch in n.mitems:
      ch.patchBoolToI32Impl(boolVals, boolPtrs)

proc patchBoolToI32(ctx: var GpuContext) =
  var boolVals, boolPtrs: HashSet[string]
  for iSym, g in ctx.globals.mpairs:
    if g.typ.kind == gtBool:
      ctx.globals[iSym].typ = GpuType(kind: gtInt32)
      boolVals.incl iSym
    elif g.typ.kind == gtPtr and g.typ.to.kind == gtBool:
      # Mutate in place: a fresh gtPtr drops mutable and implicit
      g.typ.to.kind = gtInt32
      boolPtrs.incl iSym
  for _, fn in ctx.fnTab.mpairs:
    if fn.isGlobal():
      for p in fn.pParams.mitems:
        if p.typ.kind == gtBool:
          p.typ = GpuType(kind: gtInt32)
          p.ident.symbol.typ = GpuType(kind: gtInt32)
          boolVals.incl p.ident.symbol.iSym
    fn.patchBoolToI32Impl(boolVals, boolPtrs)
  for blk in ctx.globalBlocks.mitems:
    blk.patchBoolToI32Impl(boolVals, boolPtrs)
