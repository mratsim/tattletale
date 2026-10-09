# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std / [tables]

import ../ir/gpu_types
import ./pass_datatypes
import ./pass_registry
import ./passes_validations
import ./passes_optimizations
import ./passes_preprocessing

proc lowerByrefParamsImpl*(ctx: var GpuContext; n: var GpuAst)
proc insertByrefAddrsImpl*(ctx: var GpuContext; n: var GpuAst)

proc registerOpenclPasses*(reg: var PassRegistry) =
  ## Register OpenCL-specific passes: byref-arg materialization, keyword rejection.
  reg.register("materializePassByRefArgs", pkTransform, phaseMain,
    "Wraps non-lvalue passByRef args in gpuMaterialize nodes",
    dependsOn = @["ensureBlock"],
    run = materializePassByRefArgs
  )
  reg.register("rejectOpenCLKeywords", pkValidation, phaseEarly,
    "Rejects identifiers that are reserved OpenCL C keywords",
    proc(ctx: var GpuContext): void =
      ctx.checkReservedKeywords(["kernel", "__kernel", "global", "__global",
        "local", "__local", "constant", "__constant",
        "read_only", "write_only", "read_write"], "OpenCL C")
  )

proc lowerByrefParamsImpl*(ctx: var GpuContext, n: var GpuAst) =
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

proc insertByrefAddrsImpl*(ctx: var GpuContext, n: var GpuAst) =
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
# ═══════════════════════════════════════════════════════════════════════════
