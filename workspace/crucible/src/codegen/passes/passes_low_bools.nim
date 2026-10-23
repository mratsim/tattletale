# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ═══════════════════════════════════════════════════════════════════════
#  Bool storage lowering
# ═══════════════════════════════════════════════════════════════════════

import std/sets
import std/tables
import std/macros # for the `error` builtin (compile-time diagnostics)
import ../ir/ast_datatypes
import ./pass_datatypes
import ./pass_registry

{.experimental: "codeReordering".}

# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
# Bool globals and kernel params to i32 storage
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
