# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#       Implicit address-space generic instantiation
#
# ############################################################
#
# Instantiate functions and struct types whose pointer/span arguments
# specialize the declared address space:
#
# - call sites: one callee instantiation per space combination
# - struct constructions: one type def per field-address-space combination
# - identity: callee key + the type hash of each parameter's address space,
#   reusing the existing instantiation hashing
#
# A pointer/span parameter is implicitly generic over its space,
# exactly like a struct's ptr field.

import std / [tables, sequtils]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ../ir/ast_type_constructors
import ./pass_datatypes
import ./pass_registry

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerInstantiateImplicitAddressSpaceGenerics*(reg: var PassRegistry) =
  reg.register("instantiateImplicitAddressSpaceGenerics", pkTransform, phaseMain,
    "Instantiates callees and struct types whose pointer/span arguments specialize the declared address space",
    dependsOn = @[],
    run = instantiateGenericsImpl
  )

# ═══════════════════════════════════════════════════════════════════════
# Function instantiation
# ═══════════════════════════════════════════════════════════════════════

# a call site's instantiation nests further instantiations in the cloned body
proc instantiateCall(ctx: var GpuContext, call: var GpuAst): void
proc instantiateCallsIn(ctx: var GpuContext, n: var GpuAst): void
proc instantiateStructTypesIn(ctx: var GpuContext, n: var GpuAst): void

proc instantiateCall(ctx: var GpuContext, call: var GpuAst) =
  ## Rewrites `call` onto the instantiation its argument address spaces select:
  ## the declared callee, or a space-keyed clone.
  if call.cName.symbol.isNil:
    return
  let callee = ctx.allFnTab.getOrDefault(call.cName, nil)
  if callee.isNil or callee.kind != gpuProc or callee.pParams.len != call.cArgs.len:
    return

  var wanted: seq[GpuType]
  var specializes = false
  for i, p in callee.pParams:
    # Storage-tagged params ({.smem.}/{.const_mem.}) and implicit `var T`
    # params bind by their own space, never specialize
    if not p.typ.isNil and p.typ.kind in {gtPtr, gtSpan} and
        not p.typ.implicit and p.addressSpace in {asDevice, asRMEM}:
      let want = p.typ.withAddressSpace(ctx.resolveAddressSpace(call.cArgs[i]))
      if want != p.typ:
        specializes = true
      wanted.add want
      continue
    wanted.add p.typ
  if not specializes:
    return

  # Nim-style generic identity, the position separates same-type rewrites at different parameters
  var key = call.cName.symbol.iSym
  for i, p in callee.pParams:
    if wanted[i] != p.typ:
      key.add "_" & $i & "_" & $hash(wanted[i])
  let instIdent = GpuAst(kind: gpuIdent,
                         symbol: newSymbol(mangledHashName(callee.pName.symbol.name, key),
                                           iSym = key, symKind = gsProc))
  call.cName = instIdent
  if instIdent in ctx.fnTab:
    return

  var inst = callee.clone()
  inst.pName = instIdent
  # The clone shares the original's Symbols (a gpuIdent clone shares the ref),
  # re-keyed params get fresh symbols and the body rebinds onto them
  var fresh = initTable[string, Symbol]()
  for i, p in inst.pParams.mpairs:
    if wanted[i] != p.typ:
      let old = p.ident.symbol
      let sym = newSymbol(old.name, iSym = old.iSym, typ = wanted[i], symKind = old.symKind)
      sym.coordBuiltin = old.coordBuiltin
      sym.synchroBuiltin = old.synchroBuiltin
      sym.reductionBuiltin = old.reductionBuiltin
      sym.atomicsBuiltin = old.atomicsBuiltin
      fresh[old.iSym] = sym
      p.ident = GpuAst(kind: gpuIdent, symbol: sym)
      p.typ = wanted[i]
      p.addressSpace = getTypeAccessedSpace(wanted[i])
  rebindIdents(inst.pBody, fresh)
  ctx.fnTab[instIdent] = inst
  ctx.allFnTab[instIdent] = inst
  ctx.instantiateStructTypesIn(inst.pBody)
  ctx.instantiateCallsIn(inst.pBody)

proc rebindIdents(n: var GpuAst, fresh: Table[string, Symbol]) =
  ## Points every ident whose iSym was re-keyed onto its fresh Symbol.
  if n.isNil:
    return
  if n.kind == gpuIdent:
    if not n.symbol.isNil and n.symbol.iSym in fresh:
      n.symbol = fresh[n.symbol.iSym]
  else:
    for ch in n.mitems:
      ch.rebindIdents(fresh)

proc instantiateCallsIn(ctx: var GpuContext, n: var GpuAst) =
  ## Instantiates every call in `n` whose argument address spaces mismatch
  ## the callee's pointer/span parameter address spaces
  if n.isNil:
    return
  if n.kind == gpuCall:
    ctx.instantiateCall(n)
  for ch in n.mitems:
    ctx.instantiateCallsIn(ch)

# ═══════════════════════════════════════════════════════════════════════
# Struct instantiation
# ═══════════════════════════════════════════════════════════════════════

proc instantiatedStructType(ocType: GpuType, ocFields: seq[GpuFieldInit]): GpuType =
  ## Instantiated struct type of a construction, nil when it matches the declared field types
  var differs = false
  var fields: seq[GpuTypeField]
  for f in ocType.fields:
    var fTyp = f.typ
    for cf in ocFields:
      if cf.name == f.name and not cf.typ.isNil and cf.typ.kind != gtVoid:
        fTyp = cf.typ
        break
    differs = differs or fTyp != f.typ
    fields.add GpuTypeField(name: f.name, typ: fTyp)
  if not differs:
    return nil

  result = ocType.clone()
  result.fields = fields
  # gtObject prints its bare name, the hash keeps the instantiated def distinct
  # from the declared def; gtGenericInst names are hash-derived already
  if result.kind == gtObject:
    result.name = mangledHashName(ocType.name, $hash(result))

proc registerStructTypeDef(ctx: var GpuContext, t: GpuType) =
  ## Registers one `gpuTypeDef` per distinct instantiated struct type
  ## next to the declared defs in globalBlocks
  for blk in ctx.globalBlocks.mitems:
    if blk.kind == gpuTypeDef:
      if blk.tTyp == t:
        return
    elif blk.kind == gpuBlock:
      for stmt in blk.statements:
        if stmt.kind == gpuTypeDef and stmt.tTyp == t:
          return
  # the def slots in after the last declared def, the var block stays last
  var def = GpuAst(kind: gpuTypeDef, tTyp: t)
  for f in t.fields:
    def.tFields.add GpuTypeField(name: f.name, typ: f.typ)
  var insertIdx = 0
  for i, blk in ctx.globalBlocks:
    if blk.kind == gpuTypeDef:
      insertIdx = i + 1
    elif blk.kind == gpuBlock and
        blk.statements.anyIt(it.kind == gpuTypeDef):
      insertIdx = i + 1
  ctx.globalBlocks.insert(def, insertIdx)

proc instantiateStructConstr(ctx: var GpuContext, constr: var GpuAst): bool =
  ## Rewrites a construction onto its instantiated struct type
  ## Returns true on specialization, the caller retypes the variable holding it
  if constr.ocType.kind notin {gtObject, gtGenericInst}:
    return false
  for f in constr.ocFields.mitems:
    let valueTyp = ctx.getOperandType(f.value)
    if not valueTyp.isNil:
      f.typ = valueTyp
  let inst = instantiatedStructType(constr.ocType, constr.ocFields)
  if inst.isNil:
    return false
  ctx.registerStructTypeDef(inst)
  constr.ocType = inst
  result = true

proc instantiateStructTypesIn(ctx: var GpuContext, n: var GpuAst) =
  ## Instantiates every struct construction in `n` whose field inits
  ## differ from the declared field address spaces.
  if n.isNil:
    return
  case n.kind
  of gpuVar:
    if n.vInit.kind == gpuObjConstr and ctx.instantiateStructConstr(n.vInit):
      n.vType = n.vInit.ocType
      n.vName.symbol.typ = n.vInit.ocType
    ctx.instantiateStructTypesIn(n.vInit)
  of gpuObjConstr:
    discard ctx.instantiateStructConstr(n)
    for f in n.ocFields.mitems:
      ctx.instantiateStructTypesIn(f.value)
  else:
    for ch in n.mitems:
      ctx.instantiateStructTypesIn(ch)

# ═══════════════════════════════════════════════════════════════════════
# Pass entry
# ═══════════════════════════════════════════════════════════════════════

proc instantiateGenericsImpl(ctx: var GpuContext) =
  for fnKey in toSeq(ctx.fnTab.keys):
    let fn = ctx.fnTab[fnKey]
    if not fn.isNil and fn.kind == gpuProc:
      ctx.instantiateCallsIn(fn.pBody)
      ctx.instantiateStructTypesIn(fn.pBody)
  for blk in ctx.globalBlocks.mitems:
    ctx.instantiateStructTypesIn(blk)
