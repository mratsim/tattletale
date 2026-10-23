# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## scalar-cotangent reduce kernel emission
##
## Test that a scalar gradient generates exactly one axis-reduce kernel
## that sums the backward kernel's per-index partials into the scalar's
## 1-element gradient buffer:
##
## - one reduce kernel per scalar cotangent, named `<kernel>_bwd_d_<s>_reduce`
## - signature `(partials, n, d_<s>)`, `d_<s>` carries the scalar's NAME, type buffer(1)
## - body: zero accumulator, serial sum over the partials, single store
##
## Run:
##   nim c --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/autodiff/t_ir_axpy_scalar_reduce.nim

import std/[sequtils, strutils]
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_autodiff
import workspace/crucible/tests/codegen/autodiff/ad_ir_testutils

static:
  let procs = adKernels:
    proc axpy(y: ptr UncheckedArray[float32], alpha: float32,
              a: ptr UncheckedArray[float32], b: ptr UncheckedArray[float32],
              n: int32) {.global.} =
      for i in 0 ..< n:
        y[i] = alpha * a[i] + b[i]

  let fwd = procs.procNamed("axpy")
  doAssert not fwd.isNil, "the forward kernel must survive the pass"
  let alphaParam = fwd.pParams[1]
  doAssert alphaParam.ident.ident() == "alpha", "scalar param position"

  let reduces: seq[GpuAst] = procs.filterIt(
    it.kind == gpuProc and it.pName.ident().endsWith("_bwd_d_alpha_reduce"))
  doAssert reduces.len == 1,
    "exactly one reduce kernel per scalar cotangent, got " & $reduces.len
  let reduce = reduces[0]

  # ── expected reduce kernel, from the fixture's own symbols ──────────────
  let f32 = GpuType(kind: gtFloat32)
  let i32 = GpuType(kind: gtInt32)
  let bufTyp = GpuType(kind: gtPtr,
                       to: GpuType(kind: gtUncheckedArray, uaTo: f32),
                       addressSpace: asDevice)

  let alphaISym = alphaParam.ident.symbol.iSym
  let partialsIdent = GpuAst(kind: gpuIdent, symbol: newSymbol(
    "d_alpha_partials", iSym = "d_" & alphaISym & "_partials",
    typ = bufTyp, symKind = gsDeviceKernelParam))
  let outIdent = GpuAst(kind: gpuIdent, symbol: newSymbol(
    "d_alpha", iSym = "d_" & alphaISym, typ = bufTyp, symKind = gsDeviceKernelParam))
  let nIdent = GpuAst(kind: gpuIdent, symbol: newSymbol(
    "n", iSym = alphaISym & "_reduce_n", typ = i32, symKind = gsDeviceKernelParam))
  let iIdent = GpuAst(kind: gpuIdent, symbol: newSymbol(
    "i", iSym = alphaISym & "_reduce_i", typ = i32, symKind = gsLocal))
  let accIdent = GpuAst(kind: gpuIdent, symbol: newSymbol(
    "acc", iSym = alphaISym & "_reduce_acc", typ = f32, symKind = gsLocal))

  func bin(l, r: GpuAst, t: GpuType): GpuAst =
    GpuAst(kind: gpuBinOp, bOp: newGpuIdent("+"), bLeft: l, bRight: r,
           bIsOverloaded: false, bType: t)

  let expected = GpuAst(kind: gpuProc,
    pName: GpuAst(kind: gpuIdent,
                  symbol: newSymbol(fwd.pName.ident() & "_bwd_d_alpha_reduce",
                                    iSym = fwd.pName.symbol.iSym & "_bwd_d_" &
                                           alphaISym & "_reduce",
                                    symKind = gsProc)),
    pRetType: GpuType(kind: gtVoid),
    pParams: @[
      GpuParam(ident: partialsIdent, typ: bufTyp, addressSpace: asDevice,
               passByRef: false, flavor: kIn),
      GpuParam(ident: nIdent, typ: i32, addressSpace: asRMEM,
               passByRef: false, flavor: kIn),
      GpuParam(ident: outIdent, typ: bufTyp, addressSpace: asDevice,
               passByRef: false, flavor: kIn)
    ],
    pBody: GpuAst(kind: gpuBlock, statements: @[
      GpuAst(kind: gpuVar, vName: accIdent, vType: f32,
             vInit: GpuAst(kind: gpuLit, lValue: "0", lType: f32),
             vRequiresMemcpy: false, vMutable: true, addressSpace: asRMEM),
      GpuAst(kind: gpuFor,
             fVar: iIdent,
             fStart: GpuAst(kind: gpuLit, lValue: "0", lType: i32),
             fEnd: nIdent,
             fStep: GpuAst(kind: gpuLit, lValue: "1", lType: i32),
             fBody: GpuAst(kind: gpuBlock, statements: @[
               GpuAst(kind: gpuAssign,
                      aLeft: accIdent,
                      aRight: bin(accIdent,
                                  GpuAst(kind: gpuIndex, iArr: partialsIdent,
                                         iIndex: iIdent), f32),
                      aOp: "=")
             ]),
             fRangeKind: rkExclusive),
      GpuAst(kind: gpuAssign,
             aLeft: GpuAst(kind: gpuIndex, iArr: outIdent,
                           iIndex: GpuAst(kind: gpuLit, lValue: "0", lType: i32)),
             aRight: accIdent,
             aOp: "=")
    ]),
    pAttributes: {attGlobal})

  doAssert reduce.pretty() == expected.pretty(),
    "reduce kernel mismatch:\n--- generated ---\n" & reduce.pretty() &
    "\n--- expected ---\n" & expected.pretty()

  # no atomics anywhere in the generated kernels
  for fn in procs:
    if fn.kind == gpuProc and fn.pName.ident().startsWith("axpy"):
      for ch in fn.items:
        doAssert ch.kind != gpuCall,
          "generated kernels contain no calls, got one in " & fn.pName.ident()

echo "  OK — scalar cotangent reduced by one generated axis-reduce kernel"
