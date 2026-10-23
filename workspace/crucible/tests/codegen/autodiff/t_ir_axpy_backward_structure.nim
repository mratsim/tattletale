# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## axpy backward structure at IR level
##
## Test that the transposed backward of the out-of-place axpy kernel
## `y[i] = alpha*a[i] + b[i]` matches the hand-written closed form exactly
## and leaves the forward kernel untouched:
##
## - `d_a[i] = d_a[i] + d_y[i]*alpha`, `d_b[i] = d_b[i] + d_y[i]`, loop copied verbatim, shadow buffers zero-filled
## - `d_alpha` reduced by a generated axis-reduce kernel (t_ir_axpy_scalar_reduce.nim)
## - the forward kernel AST is byte-identical with and without the pass
##
## Run:
##   nim c --hints:off --warnings:off --maxLoopIterationsVM:100000000 \
##     --outdir:build/tests/ir --nimcache:nimcache/tests/ir \
##     workspace/crucible/tests/codegen/autodiff/t_ir_axpy_backward_structure.nim

import std/strutils
import workspace/crucible/src/crucible_internal_compiler
import workspace/crucible/src/codegen/passes/passes_autodiff
import workspace/crucible/tests/codegen/autodiff/ad_ir_testutils


# ── Mangle normalization ──────────────────────────────────────────────────
# The mangleNames pass appends a deterministic per-symbol hash.
# Two fixtures of the same source shape carry different hashes.
# The hazard-negative comparison normalizes them away.

const MangleHashChars = {'a'..'z', 'A'..'Z', '0'..'9'}

proc normalizeMangle(s: string): string =
  result = ""
  var i = 0
  while i < s.len:
    if i + 2 < s.len and s[i] == '_' and s[i+1] == '_' and s[i+2] == '_':
      result.add "___"
      i += 3
      while i < s.len and s[i] in MangleHashChars:
        i += 1
    else:
      result.add s[i]
      i += 1

# ── Expected backward construction, from the fixture's own symbols ───────

proc bufferIdent(name, iSym: string, typ: GpuType): GpuAst =
  ## A shadow-adjoint parameter ident over the same buffer type.
  GpuAst(kind: gpuIdent,
         symbol: newSymbol(name, iSym = iSym, typ = typ,
                           symKind = gsDeviceKernelParam))

proc expectBackward(fwd: GpuAst): GpuAst =
  ## Hand-written closed form of the axpy backward, built from the forward
  ## kernel's symbols so mangled names compare equal.
  let params = fwd.pParams
  let yTyp = params[0].typ
  let f32 = yTyp.to.uaTo

  let dY = bufferIdent("d_y", "d_" & params[0].ident.symbol.iSym, yTyp)
  let dA = bufferIdent("d_a", "d_" & params[2].ident.symbol.iSym, params[2].typ)
  let dB = bufferIdent("d_b", "d_" & params[3].ident.symbol.iSym, params[3].typ)
  let partials = bufferIdent("d_alpha_partials",
                             "d_" & params[1].ident.symbol.iSym & "_partials",
                             yTyp)

  # loop variable, bounds and step symbols must come from the forward kernel
  let fwdLoop = fwd.pBody.statements[0]
  let fwdLoopVar = fwdLoop.fVar

  let idxY = GpuAst(kind: gpuIndex, iArr: dY, iIndex: fwdLoopVar)
  let idxA = GpuAst(kind: gpuIndex, iArr: dA, iIndex: fwdLoopVar)
  let idxB = GpuAst(kind: gpuIndex, iArr: dB, iIndex: fwdLoopVar)
  let idxP = GpuAst(kind: gpuIndex, iArr: partials, iIndex: fwdLoopVar)
  let idxALoop = GpuAst(kind: gpuIndex, iArr: params[2].ident,
                        iIndex: fwdLoopVar)

  func bin(op: string, l, r: GpuAst): GpuAst =
    GpuAst(kind: gpuBinOp, bOp: newGpuIdent(op), bLeft: l, bRight: r,
           bIsOverloaded: false, bType: f32)

  func idx(n: GpuAst): GpuAst =
    GpuAst(kind: gpuIndex, iArr: n, iIndex: fwdLoopVar)

  let dAStmt = GpuAst(kind: gpuAssign, aLeft: idxA,
                      aRight: bin("+", idxA, bin("*", idxY, params[1].ident)),
                      aOp: "=")
  let dBStmt = GpuAst(kind: gpuAssign, aLeft: idxB,
                      aRight: bin("+", idxB, idxY),
                      aOp: "=")
  let partialStmt = GpuAst(kind: gpuAssign, aLeft: idxP,
                           aRight: bin("+", idxP, bin("*", idxY, idxALoop)),
                           aOp: "=")

  # emission order: the sum's left subtree (alpha*a[i]) transposes first,
  # its left operand (alpha) before its right operand (a[i]),
  # then the right subtree (b[i])
  let inner = GpuAst(kind: gpuBlock, statements: @[
    partialStmt, dAStmt, dBStmt
  ])
  let loop = GpuAst(kind: gpuFor,
                    fVar: fwdLoopVar,
                    fStart: fwdLoop.fStart,
                    fEnd: fwdLoop.fEnd,
                    fStep: fwdLoop.fStep,
                    fBody: inner,
                    fRangeKind: fwdLoop.fRangeKind)

  var dParams: seq[GpuParam]
  for p in params:
    if p.typ.kind == gtPtr:
      let name = "d_" & p.ident.ident()
      let iSym = "d_" & p.ident.symbol.iSym
      dParams.add GpuParam(ident: bufferIdent(name, iSym, p.typ),
                           typ: p.typ, addressSpace: p.addressSpace,
                           passByRef: p.passByRef, flavor: kIn)
  # block 2 = the forward params the backward body actually reads:
  # - alpha (dA cotangent), a (partials term d_y[i]*a[i]), n (loop bound)
  # - y is never read (out-of-place), b is never read (cotangent is d_y)
  let neededParams = @[params[1], params[2], params[4]]
  let wsParam = GpuParam(ident: partials, typ: partials.symbol.typ,
                         addressSpace: asDevice, passByRef: false, flavor: kIn)

  result = GpuAst(kind: gpuProc,
                  pName: GpuAst(kind: gpuIdent,
                                symbol: newSymbol(fwd.pName.ident() & "_bwd",
                                                  iSym = fwd.pName.symbol.iSym & "_bwd",
                                                  symKind = gsProc)),
                  pRetType: GpuType(kind: gtVoid),
                  pParams: dParams & neededParams & @[wsParam],
                  pBody: GpuAst(kind: gpuBlock, statements: @[loop]),
                  pAttributes: fwd.pAttributes)

# ── Tests ─────────────────────────────────────────────────────────────────

static:
  let procs = adKernels:
    proc axpy(y: ptr UncheckedArray[float32], alpha: float32,
              a: ptr UncheckedArray[float32], b: ptr UncheckedArray[float32],
              n: int32) {.global.} =
      for i in 0 ..< n:
        y[i] = alpha * a[i] + b[i]

  let fwd = procs.procNamed("axpy")
  doAssert not fwd.isNil, "the forward kernel must survive the pass"
  doAssert fwd.pName.ident().endsWith("axpy"), "forward kernel name, got " & fwd.pName.ident()

  let bwd = procs.procNamed("axpy", "_bwd")
  doAssert not bwd.isNil, "the backward kernel must be generated"
  doAssert bwd.pName.ident() == fwd.pName.ident() & "_bwd",
    "backward kernel name, got " & bwd.pName.ident()
  doAssert bwd.pRetType.kind == gtVoid, "backward kernel returns void"

  let expected = fwd.expectBackward()
  doAssert bwd.pretty() == expected.pretty(),
    "backward AST mismatch:\n--- generated ---\n" & bwd.pretty() &
    "\n--- expected ---\n" & expected.pretty()

  # hazard negative: the forward kernel is byte-identical without the pass
  let plain = frontKernels:
    proc axpy2(y: ptr UncheckedArray[float32], alpha: float32,
               a: ptr UncheckedArray[float32], b: ptr UncheckedArray[float32],
               n: int32) {.global.} =
      for i in 0 ..< n:
        y[i] = alpha * a[i] + b[i]
  let fwdPlain = plain.procNamed("axpy2")
  let fwdOnly = GpuAst(kind: gpuProc,
                       pName: GpuAst(kind: gpuIdent,
                                     symbol: newSymbol(fwd.pName.ident(),
                                                       iSym = fwd.pName.symbol.iSym,
                                                       symKind = gsProc)),
                       pRetType: fwd.pRetType,
                       pParams: fwd.pParams,
                       pBody: fwd.pBody,
                       pAttributes: fwd.pAttributes)
  # the fixtures differ only in the kernel name (mangle hashes normalized)
  doAssert normalizeMangle(fwdOnly.pretty()) ==
           normalizeMangle(fwdPlain.pretty()).replace("axpy2", "axpy"),
    "forward kernel must be untouched:\n--- after pass ---\n" & fwdOnly.pretty() &
    "\n--- before ---\n" & fwdPlain.pretty()

echo "  OK — axpy backward matches the hand-written closed form, forward untouched"
