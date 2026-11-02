# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## autodiffBwdDerive on CIRv2: the axpy backward kernel pair
##
## Checks the derived axpy backward kernel pair: the d-block-first signature,
## the exact gradient algebra, one single-thread reduce kernel, plus the pair
## metadata for the module, and a `bwd_diff:` refusal on a data-dependent loop
##
## Run:
##   nim c --hints:off --warnings:off --outdir:build/tests workspace/crucible/tests/ir/autodiff/t_ir_ad_axpy_derive.nim

import std/[macros, sequtils, sets, strutils, tables]
import workspace/crucible
import workspace/crucible/src/codegen/frontend/nim_to_crucible
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_normalizations
import workspace/crucible/src/codegen/passes/passes_autodiff
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_pretty
import workspace/crucible/src/codegen/autodiff_types
import workspace/crucible/src/builtins/builtins_compilermagic

macro checkAutodiffBwdDerive(body: typed): string =
  ## Lowers the kernel body through the frontend, runs the registered
  ## normalization and backward-derivation passes, and asserts the structural
  ## delta on the derived module. A failing doAssert surfaces as the compile
  ## error naming the check, a clean compile is the pass signal.
  crucibleCompileTarget = ctMetal
  var reg = PassRegistry.new()
  registerNormalizationPasses(reg)
  registerAutodiffPasses(reg)
  let (m, lerr) = lowerCirv2(body)
  if lerr.len > 0:
    error(lerr)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    error("pipeline refusal: " & perr)

  block emittedKernels:
    doAssert m.procs.len == 3,
      "expected axpy, axpy_bwd and axpy_bwd_d_alpha_reduce, got " & $m.procs.len
    doAssert m.procs[0].name == "axpy", "the forward kernel stays first"
    doAssert m.procs[0].body.len == 4, "the forward body stays untouched"

  var bwd: CirProc = nil
  var reduce: CirProc = nil
  for fn in m.procs:
    if fn.name == "axpy_bwd":
      bwd = fn
    if fn.name == "axpy_bwd_d_alpha_reduce":
      reduce = fn
  doAssert bwd != nil, "no axpy_bwd kernel emitted"
  doAssert reduce != nil, "no axpy_bwd_d_alpha_reduce kernel emitted"
  doAssert bwd.isKernel, "the backward kernel is a compute entry point"
  doAssert reduce.isKernel, "the reduce kernel is a compute entry point"
  doAssert bwd.workgroup == [1, 1, 1], "the backward keeps the forward workgroup"
  doAssert bwd.ret.kind == ctVoid

  block backwardSignature:
    var names: seq[string]
    for p in bwd.params:
      names.add p.name
    doAssert names == @["d_dst", "d_a", "d_b", "a", "alpha",
                        "d_alpha_partials"],
      "backward param order, d-block then needed forward params then " &
      "workspaces; b's gradient reads only d_b and d_dst, so b is not " &
      "needed, got: " & $names
    let dDst = bwd.params[0].typ
    doAssert dDst.kind == ctAptr and dDst.space == caDevice and
      dDst.pElem.kind == ctFloat and dDst.pElem.bits == 32,
      "the d_dst shadow types like the forward buffer"
    doAssert bwd.params[5].typ.kind == ctAptr and
      bwd.params[5].typ.pElem.kind == ctFloat,
      "the partials workspace is a device buffer over the scalar's width"

  block backwardBody:
    doAssert bwd.body.len == 4,
      "preamble, start, while, fall-through return, got " & $bwd.body.len
    doAssert bwd.body[0].kind == sVar and bwd.body[0].name == "i"
    doAssert bwd.body[1].kind == sAsgn
    doAssert bwd.body[2].kind == sWhile
    doAssert bwd.body[3].kind == sRet
    let whl = bwd.body[2]
    doAssert renderExpr(whl.whileCond) == "(lt (i +32) i 8)",
      "the loop condition re-emits verbatim, got " & renderExpr(whl.whileCond)
    doAssert whl.whileBody.len == 4, "three gradient statements plus the increment"
    let txt = renderProc(bwd)
    const dAlphaPartial =
      "(asgn (pat d_alpha_partials i) (add (f +32) (pat d_alpha_partials i) (mul (f +32) (pat d_dst i) (pat a i))))"
    const dAScatter =
      "(asgn (pat d_a i) (add (f +32) (pat d_a i) (mul (f +32) (pat d_dst i) alpha)))"
    const dBScatter =
      "(asgn (pat d_b i) (add (f +32) (pat d_b i) (pat d_dst i)))"
    doAssert txt.contains(dAlphaPartial),
      "the alpha gradient accumulates d_dst[i] * a[i] into its partials " &
      "slot, got:\n" & txt
    doAssert txt.contains(dAScatter),
      "the a gradient scatter-adds d_dst[i] * alpha, got:\n" & txt
    doAssert txt.contains(dBScatter),
      "the b gradient scatter-adds d_dst[i], got:\n" & txt

  block reduceKernel:
    doAssert reduce.params.len == 3
    doAssert reduce.params[0].name == "d_alpha_partials"
    doAssert reduce.params[1].name == "n"
    doAssert reduce.params[2].name == "d_alpha"
    doAssert reduce.body.len == 6
    let txt = renderProc(reduce)
    const fold =
      "(asgn alpha_reduce_acc (add (f +32) alpha_reduce_acc (pat d_alpha_partials alpha_reduce_i)))"
    const writeback =
      "(asgn (pat d_alpha 0) (add (f +32) (pat d_alpha 0) alpha_reduce_acc))"
    doAssert txt.contains(fold), "the reduce folds the partials, got:\n" & txt
    doAssert txt.contains(writeback),
      "the reduce writes the sum into d_alpha[0], got:\n" & txt

  block metadata:
    doAssert m.adMetas.hasKey("axpy"), "no adMetas record for axpy"
    let metas = m.adMetas["axpy"]
    doAssert metas.params.len == 4, "one record per forward parameter"
    doAssert metas.params[0].name == "dst" and
      metas.params[0].kind == pkResidual and
      metas.params[0].gradKind == gkBuffer and
      metas.params[0].gradName == "d_dst" and
      metas.params[0].extent == xeWrapperField,
      "the written buffer records as a residual with a same-shape gradient"
    doAssert metas.params[1].name == "a" and metas.params[1].kind == pkBuffer and
      metas.params[1].gradKind == gkBuffer and metas.params[1].gradName == "d_a"
    doAssert metas.params[2].name == "b" and metas.params[2].kind == pkBuffer and
      metas.params[2].gradKind == gkBuffer and metas.params[2].gradName == "d_b"
    doAssert metas.params[3].name == "alpha" and
      metas.params[3].kind == pkScalar and
      metas.params[3].gradKind == gkBuffer1 and
      metas.params[3].gradName == "d_alpha"
    doAssert metas.workspaces.len == 1, "one workspace per scalar gradient"
    let ws = metas.workspaces[0]
    doAssert ws.gradName == "d_alpha" and ws.workspaceName == "d_alpha_partials"
    doAssert ws.tripCount == 8 and ws.tripCountParam == "" and
      not ws.tripCount64 and ws.threadCount == 1,
      "the static trip count comes off the loop bound"
    doAssert m.adBwdKernels.contains("axpy_bwd") and
      m.adBwdKernels.contains("axpy_bwd_d_alpha_reduce"),
      "the generated kernels never re-derive"

  block refusalChannel:
    # hazard negative: a data-dependent loop refuses and the module stays untouched
    let dstTyp = tAptr(caDevice, tFloat(32))
    let negFn = CirProc(
      name: "negk",
      params: @[param("dst", dstTyp)],
      ret: tVoid(),
      body: @[whileS(trueE(), @[asgnS(
        patE(sym("dst", dstTyp), intLit(0, tInt(32))),
        fltLit(1.0, tFloat(32)))])],
      isKernel: true)
    let neg = moduleD("neg", @[], @[negFn])
    var negReg = PassRegistry.new()
    registerAutodiffPasses(negReg)
    let nerr = runPasses(neg, negReg)
    doAssert nerr.startsWith("pass autodiffBwdDerive: bwd_diff: "),
      "expected a bwd_diff refusal, got: " & nerr
    doAssert nerr.contains("sWhile"), "the refusal names the construct: " & nerr
    doAssert neg.procs.len == 1, "the refused module stays untouched"
    doAssert neg.adBwdKernels.len == 0, "nothing derives from a refused kernel"

  result = newLit("ok")

discard checkAutodiffBwdDerive:
  proc axpy(dst, a, b: ptr UncheckedArray[float32], alpha: float32) {.global, workgroup: (1, 1).} =
    for i in 0 ..< 8:
      dst[i] = alpha * a[i] + b[i]
