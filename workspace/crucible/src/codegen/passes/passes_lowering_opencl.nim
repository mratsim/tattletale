# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            OpenCL lowering passes
#
# ############################################################
#
# Project the crucible IR onto OpenCL C's calling convention:
# passByRef structs pass as hidden const pointers and structs
# carrying pointers expand to ptr/scalar leaves, since OpenCL
# params cross by value

import std / [tables, sets]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_low_pointers
import ./passes_low_views
import ./passes_low_array_assign
import ./passes_low_address_spaces
import ./passes_legalizations
import ./passes_validations
import ./passes_optimizations

{.experimental: "codeReordering".}

const OclReserved = [
  "kernel", "__kernel", "global", "__global", "local", "__local",
  "constant", "__constant", "read_only", "write_only", "read_write"
].toHashSet()

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerOpenclLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = true)
  reg.registerSharedLoweringPasses()
  reg.registerDecomposeArrayAssign()
  reg.registerMaterializePassByRefArgs()
  reg.registerExplodeStructWithPtrFields()
  reg.registerRenameReservedIdentifiers("openclReservedIdentifiers", OclReserved)
  reg.registerInstantiateImplicitAddressSpaceGenerics()
  reg.registerLowerByrefParams()
  reg.register("insertByrefAddrs", pkTransform, phaseMain,
    "Wraps by-ref call args in gpuAddr",
    dependsOn = @["lowerByrefParams"],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          ctx.insertByrefAddrsImpl(fn)
  )

func registerLowerByrefParams(reg: var PassRegistry) =
  reg.register("lowerByrefParams", pkTransform, phaseMain,
    "Rewrites passByRef params into const Type* ptr params",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      for fnKey in ctx.allFnTab.keys:
        var fn = ctx.allFnTab[fnKey]
        if fn.kind == gpuProc:
          ctx.lowerByrefParamsImpl(fn)
  )

# ═══════════════════════════════════════════════════════════════════════
# By-ref parameter lowering
# ═══════════════════════════════════════════════════════════════════════

proc lowerByrefParamsImpl(ctx: var GpuContext, n: var GpuAst) =
  ## Rewrites passByRef params into `const Type*` ptr params for OpenCL.
  ## CUDA keeps `const Type&` natively, no body changes there.
  ##
  ## Contract:
  ## - structs of 24 bytes or more pass as hidden const reference
  ##   (isLargeStruct), no call-site copy
  ## - the param symbol renames from "t" to "_p_t", the symbol is shared with
  ##   body idents so they rename too
  ## - each renamed body ident wraps in gpuDeref, `t.data[...]` becomes
  ##   `(*_p_t).data[...]`, valid C for pointer-to-struct member access
  ## - no local copy is prepended, body refs resolve through the pointer,
  ##   a `Type t = *_p_t;` init would be dead code
  ##
  # Rename and deref-wrap helper over the shared param symbol.
  proc wrapInDeref(body: var GpuAst, renamedSym: Symbol) =
    case body.kind
    of gpuIdent:
      if body.symbol == renamedSym:
        body = GpuAst(kind: gpuDeref, dOf: body)
    else:
      for ch in body.mitems:
        wrapInDeref(ch, renamedSym)
  case n.kind
  of gpuProc:
    let isKernel = attGlobal in n.pAttributes
    if not isKernel:
      var renamedSyms: seq[Symbol]
      for p in n.pParams:
        if p.passByRef:
          let oldSym = p.ident.symbol
          p.ident.symbol.name = "_p_" & p.ident.ident()
          renamedSyms.add oldSym
      # Walk body: wrap renamed idents in gpuDeref (so t → (*_p_t))
      for s in renamedSyms:
        wrapInDeref(n.pBody, s)
    for ch in n.mitems:
      ctx.lowerByrefParamsImpl(ch)
  else:
    for ch in n.mitems:
      ctx.lowerByrefParamsImpl(ch)

proc insertByrefAddrsImpl(ctx: var GpuContext, n: var GpuAst) =
  ## Wraps byref args in `gpuAddr` nodes at call sites.
  case n.kind
  of gpuCall:
    let fnParams = ctx.getFnParams(n.cName)
    for i, arg in n.cArgs:
      if i < fnParams.len and fnParams[i].passByRef:
        if arg.kind != gpuMaterialize and arg.kind in {gpuIdent, gpuIndex, gpuDeref}:
          n.cArgs[i] = GpuAst(kind: gpuAddr, aOf: arg)
    for ch in n.mitems:
      ctx.insertByrefAddrsImpl(ch)
  else:
    for ch in n.mitems:
      ctx.insertByrefAddrsImpl(ch)
