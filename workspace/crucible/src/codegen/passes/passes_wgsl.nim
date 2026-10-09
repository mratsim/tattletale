# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [sequtils, tables]

import ../ir/gpu_types
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations
import ./passes_validations
import ./passes_preprocessing

proc registerWgslPasses*(reg: var PassRegistry) =
  ## Register WGSL-specific passes: keyword rejection, global-fn flattening, code validation.
  reg.register("rejectWGSLKeywords", pkValidation, phaseEarly,
    "Rejects identifiers that are reserved WGSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["override", "storage", "uniform", "workgroup"], "WGSL")
  )
  # Explode before the passes below reshape pointer-bearing code.
  reg.registerExplodePointerStructParams()
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
