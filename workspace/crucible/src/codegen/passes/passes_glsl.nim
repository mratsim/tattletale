# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[macros, sequtils, sets, strutils, tables]
import ../ir/gpu_ast_datatypes
import ../ir/gpu_ast_type_resolvers
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations
import ./passes_validations
import ./passes_preprocessing
import ./passes_lowering # insertPointerDeref

# ############################################################
#
# GLSL (Vulkan) validations and pass impls
#
# ############################################################
#
# Validations reject what GLSL cannot express: reserved words, divergent
# push-constant layouts. The GLSL lowering impls (kVar inout params)
# are registered by passes_lowering_glsl.


{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

proc registerVulkanPasses*(reg: var PassRegistry) =
  ## Register Vulkan-specific validation passes: GLSL keyword rejection,
  ## shared push-constant layout consistency.
  reg.register("rejectVulkanKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved GLSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["extern", "interface", "buffer"], "GLSL")
  )
  reg.register("validateSharedPushConstLayout", pkValidation, phaseMain,
    "Rejects multi-kernel sources whose push-constant layouts differ",
    proc(ctx: var GpuContext): void =
      ctx.validateSharedPushConstLayout()
  )


# kVar parameter legalization (GLSL inout)
# ═══════════════════════════════════════════════════════════════════════

proc unwrapVarDerefs(ctx: var GpuContext, n: var GpuAst, kvarSyms: HashSet[string], kvarNames: HashSet[string]) =
  ## Replace `gpuDeref(gpuIdent(s))` with the plain ident when `s` is a kVar
  ## param of the enclosing function. Symbols are per-body, match by iSym
  ## then name. Other derefs (device pointers) survive.
  case n.kind
  of gpuDeref:
    if n.dOf.kind == gpuIdent and n.dOf.symbol != nil and
        (n.dOf.symbol.iSym in kvarSyms or n.dOf.symbol.name in kvarNames):
      n = n.dOf
    else:
      ctx.unwrapVarDerefs(n.dOf, kvarSyms, kvarNames)
  else:
    for ch in n.mitems:
      ctx.unwrapVarDerefs(ch, kvarSyms, kvarNames)

proc qualifyVarParamsImpl(ctx: var GpuContext, n: var GpuAst) =
  ## Lower Nim `var T` params (kVar flavor) to GLSL inout bindings:
  ## retype each param to its pointee, unwrap body derefs, unwrap
  ## gpuAddr args at call sites.
  case n.kind
  of gpuProc:
    var kvarSyms: HashSet[string]
    var kvarNames: HashSet[string]
    for p in mitems(n.pParams):
      if p.flavor == kVar:
        if p.typ.kind == gtPtr and p.typ.implicit:
          p.typ = p.typ.to
        else:
          raiseAssert "kVar param without an implicit pointer type: " & p.ident.ident()
        kvarSyms.incl p.ident.symbol.iSym
        kvarNames.incl p.ident.symbol.name
    ctx.unwrapVarDerefs(n.pBody, kvarSyms, kvarNames)
    for ch in n.mitems:
      ctx.qualifyVarParamsImpl(ch)
  of gpuCall:
    let fnParams = ctx.getFnParams(n.cName)
    for i, arg in mpairs(n.cArgs):
      if i < fnParams.len and fnParams[i].flavor == kVar:
        if arg.kind == gpuAddr:
          n.cArgs[i] = arg.aOf
        elif arg.kind notin {gpuIdent, gpuIndex, gpuDot}:
          raiseAssert "kVar call arg must be an lvalue, got " & $arg.kind
    for ch in n.mitems:
      ctx.qualifyVarParamsImpl(ch)
  else:
    for ch in n.mitems:
      ctx.qualifyVarParamsImpl(ch)

proc qualifyVarParams*(ctx: var GpuContext) =
  for fn in ctx.allFnTab.mvalues():
    ctx.qualifyVarParamsImpl(fn)
  for fn in ctx.fnTab.mvalues():
    ctx.qualifyVarParamsImpl(fn)

# Shared push-constant layout validation
# ═══════════════════════════════════════════════════════════════════════

proc sameLayout(a, b: seq[(string, GpuType, AddressSpace)]): bool =
  ## Element-wise compare, direct calls only: the generic seq/tuple
  ## `==` cannot lift the side-effecting `GpuType ==`
  if a.len != b.len:
    return false
  for i in 0 ..< a.len:
    if a[i][0] != b[i][0] or a[i][1] != b[i][1] or a[i][2] != b[i][2]:
      return false
  result = true

proc validateSharedPushConstLayout(ctx: var GpuContext) =
  ## Vulkan kernels in one source share one push-constant block: every
  ## kernel's ordered non-smem params must agree, else a launch serializes
  ## its args at offsets another kernel reads as different members
  var reference: seq[(string, GpuType, AddressSpace)]
  for _, fn in ctx.allFnTab:
    if not fn.isGlobalFn():
      continue
    var layout: seq[(string, GpuType, AddressSpace)]
    for p in fn.pParams:
      if p.addressSpace != asSMEM:
        layout.add (p.ident.ident(), p.typ, p.addressSpace)
    if reference.len == 0:
      reference = layout
    elif not sameLayout(layout, reference):
      let names = layout.mapIt(it[0]).join(", ")
      error "Multi-kernel Vulkan sources share one push-constant block, all kernels " &
            "must declare the same ordered params, got: " & names