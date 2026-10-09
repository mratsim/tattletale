# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [sequtils, strformat, tables]

import ../ir/gpu_types
import ./pass_datatypes
import ./pass_registry
import ./passes_legalizations
import ./passes_validations
import ./passes_preprocessing

proc compareGpuTypeShallow*(a, b: GpuType): bool
proc lowerSsboParamsImpl*(ctx: var GpuContext)
proc lowerPushConstantsImpl*(ctx: var GpuContext)

proc registerVulkanPasses*(reg: var PassRegistry) =
  ## Register Vulkan-specific passes: SSBO/push-constant lowering, GLSL keyword rejection.
  # Explode before the SSBO/push-constant lowering below scans the params.
  reg.registerExplodePointerStructParams()
  reg.register("lowerSsboParams", pkTransform, phaseMain,
    "Scans kernels, builds canonical SSBO list, normalizes param names",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      lowerSsboParamsImpl(ctx)
  )
  reg.register("lowerPushConstants", pkTransform, phaseMain,
    "Lifts non-ptr non-workspace params into push-constant block metadata",
    dependsOn = @[],
    run = proc(ctx: var GpuContext): void =
      lowerPushConstantsImpl(ctx)
  )
  reg.register("rejectVulkanKeywords", pkValidation, phaseMain,
    "Rejects identifiers that are reserved GLSL keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["extern", "interface", "buffer"], "GLSL")
  )

proc renameIdentRefsImpl*(n: var GpuAst; symToRename: Table[string, string])

proc compareGpuTypeShallow*(a, b: GpuType): bool =
  ## Shallow comparison of GpuType for SSBO validation.
  ## Compares kind and immediate fields (not deep recursion).
  if a.isNil or b.isNil: return false
  if a.kind != b.kind: return false
  case a.kind
  of gtPtr:
    if a.to.isNil and b.to.isNil: return true
    if a.to.isNil or b.to.isNil: return false
    result = a.to.kind == b.to.kind
  else:
    result = true

proc lowerSsboParamsImpl*(ctx: var GpuContext) =
  ## Scans all kernels, builds canonical SSBO list (deduped by position),
  ## validates type consistency, normalizes parameter names across kernels.
  for (fnIdent, fn) in ctx.fnTab.mpairs:
    if fn.isGlobalFn():
      var ssboIdx = 0
      for p in fn.pParams:
        if p.typ.kind == gtPtr:
          if ssboIdx < ctx.ssboCanonicalInfo.len:
            let (canonName, canonInner) = ctx.ssboCanonicalInfo[ssboIdx]
            if not canonInner.compareGpuTypeShallow(p.typ.to):
              raiseAssert &"Type mismatch at SSBO pos {ssboIdx}"
            if p.ident.ident() != canonName:
              var renames = initTable[string, string]()
              renames[p.ident.symbol.iSym] = canonName
              renameIdentRefsImpl(fn.pBody, renames)
              p.ident.symbol.name = canonName
          else:
            ctx.ssboCanonicalInfo.add (p.ident.ident(), p.typ.to.clone())
          inc ssboIdx

proc renameIdentRefsImpl*(n: var GpuAst, symToRename: Table[string, string]) =
  case n.kind
  of gpuIdent:
    if n.symbol != nil and n.symbol.iSym in symToRename:
      n.symbol.name = symToRename[n.symbol.iSym]
  else:
    for ch in n.mitems:
      renameIdentRefsImpl(ch, symToRename)

proc lowerPushConstantsImpl*(ctx: var GpuContext) =
  ## Lifts non-pointer, non-workspace params into a uniform push-constant block.
  ## Marks them in the context for codegen to emit.
  var pushConstParams: seq[GpuParam]
  for (fnIdent, fn) in ctx.fnTab.mpairs:
    if fn.isGlobalFn():
      for p in fn.pParams:
        if p.typ.kind != gtPtr and p.addressSpace != asSMEM:
          # Check if already added
          let alreadyAdded = pushConstParams.anyIt(
            it.ident.ident() == p.ident.ident() and it.typ.kind == p.typ.kind)
          if not alreadyAdded:
            pushConstParams.add p
  if pushConstParams.len > 0:
    var pcBlock = GpuAst(kind: gpuBlock)
    for p in pushConstParams:
      let comment = GpuAst(kind: gpuComment,
        comment: &"__push_const:{p.ident.ident()}:{$p.typ.kind}")
      pcBlock.statements.add comment
    ctx.globalBlocks.add pcBlock


# ═══════════════════════════════════════════════════════════════════════════
