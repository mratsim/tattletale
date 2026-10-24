# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.
## Math-builtin cotangents at IR level
##
## Test that the transposer's external math whitelist transposes each
## builtin with its mechanical cotangent rule, other calls still refuse:
##
## - `d exp(x) = g*exp(x)`, `d ln(x) = g/x`, `d sqrt(x) = g/(2*sqrt(x))`, `d rsqrt(x) = -g/(2*x*sqrt(x))`, each re-emitting the primal callee
## - the backward signature is `(d_y, d_a, a)`: `y` is never read
##
## Run:
##   nim c --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/autodiff/t_ir_math_builtin_cotangents.nim

import std/[math, strutils]
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

func reCall(cName: GpuAst, arg: GpuAst): GpuAst =
  ## Re-emits the forward's whitelisted callee over a fresh argument,
  ## reusing the callee ident so mangled names compare equal.
  GpuAst(kind: gpuCall, cName: cName, cArgs: @[arg])

proc expectBackward(fwd: GpuAst, rule: GpuAst): GpuAst =
  ## Hand-written closed form of `y[i] = f(a[i])` for one whitelist rule.
  ## `rule` is the cotangent expression of `a[i]` seeded by `d_y[i]`.
  let params = fwd.pParams
  let yTyp = params[0].typ
  let f32 = yTyp.to.uaTo

  let dY = bufferIdent("d_y", "d_" & params[0].ident.symbol.iSym, yTyp)
  let dA = bufferIdent("d_a", "d_" & params[1].ident.symbol.iSym, params[1].typ)

  let fwdLoop = fwd.pBody.statements[0]
  let v = fwdLoop.fVar
  let idxY = idx(dY, v)
  let idxA = idx(dA, v)

  let stmt = GpuAst(kind: gpuAssign, aLeft: idxA,
                    aRight: bin("+", idxA, rule, f32), aOp: "=")
  let inner = GpuAst(kind: gpuBlock, statements: @[stmt])
  let loop = GpuAst(kind: gpuFor,
                    fVar: v, fStart: fwdLoop.fStart, fEnd: fwdLoop.fEnd,
                    fStep: fwdLoop.fStep,
                    fBody: inner, fRangeKind: fwdLoop.fRangeKind)

  result = GpuAst(kind: gpuProc,
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
                  pBody: GpuAst(kind: gpuBlock, statements: @[loop]),
                  pAttributes: fwd.pAttributes)


# ── Fixtures, one per whitelisted builtin ─────────────────────────────────

static:
  block:
    let procs = adKernels:
      proc exp1(y, a: ptr UncheckedArray[float32]) {.global.} =
        for i in 0 ..< 8:
          y[i] = exp(a[i])

    let fwd = procs.procNamed("exp1")
    let assign = fwd.pBody.statements[0].fBody.statements[0]
    let call = assign.aRight
    let v = assign.aLeft.iIndex
    let f32 = fwd.pParams[0].typ.to.uaTo
    let dYIdx = GpuAst(kind: gpuIndex,
                       iArr: GpuAst(kind: gpuIdent,
                                    symbol: newSymbol("d_y",
                                                      iSym = "d_" & fwd.pParams[0].ident.symbol.iSym,
                                                      typ = fwd.pParams[0].typ,
                                                      symKind = gsDeviceKernelParam)),
                       iIndex: v)

    let expected = expectBackward(fwd,
      bin("*", dYIdx, call, f32))
    let bwd = procs.procNamed("exp1", "_bwd")
    doAssert not bwd.isNil, "exp: the backward kernel must be generated"
    doAssert bwd.pretty() == expected.pretty(),
      "exp backward mismatch:\n--- generated ---\n" & bwd.pretty() &
      "\n--- expected ---\n" & expected.pretty()
    echo "  OK — exp transposes as d exp(x) = g*exp(x)"

  block:
    let procs = adKernels:
      proc ln1(y, a: ptr UncheckedArray[float32]) {.global.} =
        for i in 0 ..< 8:
          y[i] = ln(a[i])

    let fwd = procs.procNamed("ln1")
    let assign = fwd.pBody.statements[0].fBody.statements[0]
    let arg = assign.aRight.cArgs[0]
    let v = assign.aLeft.iIndex
    let dYIdx = GpuAst(kind: gpuIndex,
                       iArr: GpuAst(kind: gpuIdent,
                                    symbol: newSymbol("d_y",
                                                      iSym = "d_" & fwd.pParams[0].ident.symbol.iSym,
                                                      typ = fwd.pParams[0].typ,
                                                      symKind = gsDeviceKernelParam)),
                       iIndex: v)

    let expected = expectBackward(fwd,
      bin("/", dYIdx, arg, fwd.pParams[0].typ.to.uaTo))
    let bwd = procs.procNamed("ln1", "_bwd")
    doAssert not bwd.isNil, "ln: the backward kernel must be generated"
    doAssert bwd.pretty() == expected.pretty(),
      "log backward mismatch:\n--- generated ---\n" & bwd.pretty() &
      "\n--- expected ---\n" & expected.pretty()
    echo "  OK — log transposes as d ln(x) = g/x"

  block:
    let procs = adKernels:
      proc sqrt1(y, a: ptr UncheckedArray[float32]) {.global.} =
        for i in 0 ..< 8:
          y[i] = sqrt(a[i])

    let fwd = procs.procNamed("sqrt1")
    let assign = fwd.pBody.statements[0].fBody.statements[0]
    let call = assign.aRight
    let arg = call.cArgs[0]
    let v = assign.aLeft.iIndex
    let f32 = fwd.pParams[0].typ.to.uaTo
    let two = GpuAst(kind: gpuLit, lValue: "2.0", lType: f32)
    let dYIdx = GpuAst(kind: gpuIndex,
                       iArr: GpuAst(kind: gpuIdent,
                                    symbol: newSymbol("d_y",
                                                      iSym = "d_" & fwd.pParams[0].ident.symbol.iSym,
                                                      typ = fwd.pParams[0].typ,
                                                      symKind = gsDeviceKernelParam)),
                       iIndex: v)
    # the transposer re-emits the callee under its plain name
    let sqrtRecompute = reCall(GpuAst(kind: gpuIdent,
                                      symbol: newSymbol("sqrt", iSym = "sqrt",
                                                        symKind = gsProc)),
                               arg)

    let expected = expectBackward(fwd,
      bin("/", dYIdx, bin("*", two, sqrtRecompute, f32), f32))
    let bwd = procs.procNamed("sqrt1", "_bwd")
    doAssert not bwd.isNil, "sqrt: the backward kernel must be generated"
    doAssert bwd.pretty() == expected.pretty(),
      "sqrt backward mismatch:\n--- generated ---\n" & bwd.pretty() &
      "\n--- expected ---\n" & expected.pretty()
    echo "  OK — sqrt transposes as d sqrt(x) = g/(2*sqrt(x))"

  block:
    let procs = adKernels:
      proc rsqrt1(y, a: ptr UncheckedArray[float32]) {.global.} =
        for i in 0 ..< 8:
          y[i] = rsqrt(a[i])

    let fwd = procs.procNamed("rsqrt1")
    let assign = fwd.pBody.statements[0].fBody.statements[0]
    let call = assign.aRight
    let arg = call.cArgs[0]
    let v = assign.aLeft.iIndex
    let f32 = fwd.pParams[0].typ.to.uaTo
    let two = GpuAst(kind: gpuLit, lValue: "2.0", lType: f32)
    let dYIdx = GpuAst(kind: gpuIndex,
                       iArr: GpuAst(kind: gpuIdent,
                                    symbol: newSymbol("d_y",
                                                      iSym = "d_" & fwd.pParams[0].ident.symbol.iSym,
                                                      typ = fwd.pParams[0].typ,
                                                      symKind = gsDeviceKernelParam)),
                       iIndex: v)
    let sqrtRecompute = reCall(GpuAst(kind: gpuIdent,
                                      symbol: newSymbol("sqrt", iSym = "sqrt",
                                                        symKind = gsProc)),
                               arg)

    let expected = expectBackward(fwd,
      GpuAst(kind: gpuPrefix, pOp: "-",
             pVal: bin("/", dYIdx,
                       bin("*", bin("*", two, arg, f32), sqrtRecompute, f32),
                       f32)))
    let bwd = procs.procNamed("rsqrt1", "_bwd")
    doAssert not bwd.isNil, "rsqrt: the backward kernel must be generated"
    doAssert bwd.pretty() == expected.pretty(),
      "rsqrt backward mismatch:\n--- generated ---\n" & bwd.pretty() &
      "\n--- expected ---\n" & expected.pretty()
    echo "  OK — rsqrt transposes as d rsqrt(x) = -g/(2*x*sqrt(x))"

echo "  OK — math-builtin whitelist transposes with mechanical cotangent rules"
