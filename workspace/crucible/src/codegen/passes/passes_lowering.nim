## Shared lowering impls.
##
## - registerSharedLoweringPasses registers the ones every backend wants,
##   per-backend passes register in passes_lowering_<backend>.nim
## - the structural impls moved to passes_low_pointers, passes_low_views,
##   passes_low_array_assign, renaming, pruning and reserved-keyword handling
## - each replaces an IR construct with its composition at the target ABI,
##   a policy differing per target takes a flag

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
