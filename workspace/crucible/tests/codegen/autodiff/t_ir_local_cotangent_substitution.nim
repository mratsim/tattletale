# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Local substitution in cotangent expressions at IR level
##
## Test that the transposed backward is self-contained when a mid-kernel
## local appears inside a differentiated expression:
##
## - every cotangent expression re-derives the local, no backward
##   statement names a forward local
## - the exact structure matches the hand-written closed form
##
## Run:
##   nim c --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/autodiff/t_ir_local_cotangent_substitution.nim

import std/strutils
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_autodiff
import workspace/crucible/tests/codegen/autodiff/ad_ir_testutils


# ── Expected backward construction, from the fixture's own symbols ───────

proc bufferIdent(name, iSym: string, typ: GpuType): GpuAst =
  ## A shadow-adjoint parameter ident over the same buffer type.
  GpuAst(kind: gpuIdent,
         symbol: newSymbol(name, iSym = iSym, typ = typ,
                           symKind = gsDeviceKernelParam))

func bin(op: string, l, r: GpuAst, t: GpuType): GpuAst =
  GpuAst(kind: gpuBinOp, bOp: newGpuIdent(op), bLeft: l, bRight: r,
         bIsOverloaded: false, bType: t)

func idx(n: GpuAst, i: GpuAst): GpuAst =
  GpuAst(kind: gpuIndex, iArr: n, iIndex: i)


static:
  let procs = adKernels:
    proc locSub(y, a: ptr UncheckedArray[float32]) {.global.} =
      let s = a[0] * a[1]
      y[0] = s * a[2]

  let fwd = procs.procNamed("locSub")
  doAssert not fwd.isNil, "the forward kernel must survive the pass"

  let bwd = procs.procNamed("locSub", "_bwd")
  doAssert not bwd.isNil, "the backward kernel must be generated"

  # The backward declares no locals of the forward, so a raw local ident
  # would print an undefined identifier in the kernel source.
  var fwdLocalISyms: seq[string]
  fwd.pBody.walk(proc(n: var GpuAst) =
    if n.kind == gpuVar and n.vName.symbol != nil:
      fwdLocalISyms.add n.vName.symbol.iSym)
  proc referencesLocal(n: GpuAst, iSyms: seq[string]): bool =
    if n.kind == gpuIdent and n.symbol != nil and n.symbol.iSym in iSyms:
      return true
    for ch in n.items:
      if ch.referencesLocal(iSyms):
        return true
    result = false
  doAssert not bwd.pBody.referencesLocal(fwdLocalISyms),
    "the backward references a forward local by name:\n" & bwd.pretty()

  # Exact structure: the s cotangent re-derives s as (a[0]*a[1]).
  let params = fwd.pParams
  let yTyp = params[0].typ
  let f32 = yTyp.to.uaTo
  let dY = bufferIdent("d_y", "d_" & params[0].ident.symbol.iSym, yTyp)
  let dA = bufferIdent("d_a", "d_" & params[1].ident.symbol.iSym, params[1].typ)
  let aIdent = params[1].ident

  proc lit(i: int): GpuAst =
    GpuAst(kind: gpuLit, lValue: $i, lType: GpuType(kind: gtInt32))

  let idxY0 = idx(dY, lit(0))
  let sDef = bin("*", idx(aIdent, lit(0)), idx(aIdent, lit(1)), f32)

  let stmt0 = GpuAst(kind: gpuAssign, aLeft: idx(dA, lit(0)),
                     aRight: bin("+", idx(dA, lit(0)),
                       bin("*", bin("*", idxY0, idx(aIdent, lit(2)), f32),
                           idx(aIdent, lit(1)), f32), f32),
                     aOp: "=")
  let stmt1 = GpuAst(kind: gpuAssign, aLeft: idx(dA, lit(1)),
                     aRight: bin("+", idx(dA, lit(1)),
                       bin("*", bin("*", idxY0, idx(aIdent, lit(2)), f32),
                           idx(aIdent, lit(0)), f32), f32),
                     aOp: "=")
  let stmt2 = GpuAst(kind: gpuAssign, aLeft: idx(dA, lit(2)),
                     aRight: bin("+", idx(dA, lit(2)),
                       bin("*", idxY0, sDef, f32), f32),
                     aOp: "=")

  let expected = GpuAst(kind: gpuProc,
                        pName: GpuAst(kind: gpuIdent,
                                      symbol: newSymbol(fwd.pName.ident() & "_bwd",
                                                        iSym = fwd.pName.symbol.iSym & "_bwd",
                                                        symKind = gsProc)),
                        pRetType: GpuType(kind: gtVoid),
                        pParams: @[GpuParam(ident: dY, typ: yTyp,
                                            addressSpace: params[0].addressSpace,
                                            passByRef: params[0].passByRef, flavor: kIn),
                                   GpuParam(ident: dA, typ: params[1].typ,
                                            addressSpace: params[1].addressSpace,
                                            passByRef: params[1].passByRef, flavor: kIn),
                                   params[1]],
                        pBody: GpuAst(kind: gpuBlock, statements: @[stmt0, stmt1, stmt2]),
                        pAttributes: fwd.pAttributes)

  doAssert bwd.pretty() == expected.pretty(),
    "local-substitution backward mismatch:\n--- generated ---\n" & bwd.pretty() &
    "\n--- expected ---\n" & expected.pretty()

echo "  OK — cotangent expressions re-derive mid-kernel locals, backward self-contained"
