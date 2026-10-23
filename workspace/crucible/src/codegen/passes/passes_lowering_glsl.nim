# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            GLSL lowering passes
#
# ############################################################
#
# Lower the crucible IR onto GLSL's storage model, registered by
# the GLSL pipeline only:
# - buffer references cannot be indexed directly, pointer parents
#   get an explicit deref the renderer composes as `.x`
# - kVar params bind as `inout`, retyped to their pointee

import std / [macros, sequtils, sets, strutils, tables]
import ../ir/ast_datatypes
import ../ir/ast_type_resolvers
import ./pass_datatypes
import ./pass_registry
import ./passes_lowering
import ./passes_low_address_spaces
import ./passes_legalizations
import ./passes_validations

{.experimental: "codeReordering".}

const GlslReserved = [
  "output", "input", "in", "out", "attribute", "uniform", "varying",
  "buffer", "shared", "layout", "main", "void", "return",
  "if", "else", "for", "while", "break", "continue", "struct",
  "const", "true", "false", "bool", "int", "uint", "float", "double",
  "vec2", "vec3", "vec4", "mat2", "mat3", "mat4",
  "sampler", "texture", "image", "subroutine", "discard", "precise"
].toHashSet()

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

func registerQualifyVarParams*(reg: var PassRegistry) =
  reg.register("qualifyVarParams", pkTransform, phaseMain,
    "Retypes kVar params to their pointee, unwraps their body derefs and call-site gpuAddr args",
    dependsOn = @[],
    run = qualifyVarParams
  )

func registerGlslLoweringPasses*(reg: var PassRegistry) =
  reg.registerLowerArraysAndViews(decayArrays = false)
  reg.registerSharedLoweringPasses()
  reg.registerRenameReservedIdentifiers("glslReservedIdentifiers", GlslReserved)
  reg.registerInstantiateImplicitAddressSpaceGenerics()
  reg.registerInsertPointerDeref()
  reg.registerQualifyVarParams()

# ═══════════════════════════════════════════════════════════════════════
# kVar parameter qualification
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
          error "kVar param without an implicit pointer type: " & p.ident.ident() &
            " in " & (if n.pName != nil and n.pName.kind == gpuIdent: n.pName.ident() else: "?")
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
          error "kVar call arg must be an lvalue, got " & $arg.kind
    for ch in n.mitems:
      ctx.qualifyVarParamsImpl(ch)
  else:
    for ch in n.mitems:
      ctx.qualifyVarParamsImpl(ch)

proc qualifyVarParams(ctx: var GpuContext) =
  for fn in ctx.allFnTab.mvalues():
    ctx.qualifyVarParamsImpl(fn)
