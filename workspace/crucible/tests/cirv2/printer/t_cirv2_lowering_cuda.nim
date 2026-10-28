# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## CUDA lowering fixtures: hand-built CIRv2 modules run through the registered
## CUDA stage set (the registry the future CUDA printer wiring builds).
##
## Assertions inspect the module AST after the passes: the `cuda:` macro
## refuses loudly (no CUDA printer exists yet), so AST-level fixtures
## are operator-authorized for CUDA.
##
## The CUDA per-target stage set contains one transform:
## the reserved-word rename with `CudaReserved` as data.
##
## - rename fixture: qualifier-colliding names take the `_nim` suffix,
##   the bare-pipeline control asserts the surviving raw names
## - by-reference call arguments: evidence for the negative audit finding,
##   the frontend materializes every `var T` argument into an address-of
##   over the lvalue, the pipeline rewrites nothing
## - array parameters: negative-finding evidence, a `ctArray` parameter
##   passes through unchanged and the decay belongs to the printer
##
## - Run: `nim c -r --outdir:build/tests tests/cirv2/printer/t_cirv2_lowering_cuda.nim`

import std/[strutils, unittest]
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/ir/cirv2_verify
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_legalizations
import workspace/crucible/src/codegen/passes/passes_normalizations
import workspace/crucible/src/codegen/passes/passes_optimizations
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/passes/passes_low_reserved
import workspace/crucible/src/codegen/passes/passes_lowering_cuda
import workspace/crucible/src/codegen/passes/passes_validations

proc cudaPipeline(m: CirModule): string =
  ## Runs the registered CUDA stage set over one module in place:
  ## shared stages, the reserved-word rename, validation, in that order.
  ## Returns "" when clean, else the first refusal message.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerCudaLoweringPasses()
  reg.registerValidationPostPasses()
  return runPasses(m, reg)

proc barePipeline(m: CirModule): string =
  ## Runs the stage set with the CUDA rename registration skipped:
  ## the control that fails once the rename registration disappears.
  var reg = PassRegistry.new()
  reg.registerLegalizationPasses()
  reg.registerNormalizationPasses()
  reg.registerOptimizationPasses()
  reg.registerSharedLoweringPasses()
  reg.registerValidationPostPasses()
  return runPasses(m, reg)

# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════

proc buildRenameModule(): CirModule =
  ## Kernel plus device fn colliding with the execution-space qualifiers:
  ## a proc name, a parameter name and a local rename to `_nim`.
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let ptrT = tPtr(caThread, f32)
  let addTy = tProcT(ptrT, f32, tVoid())
  let helper = procD("__device__", @[
    param("__constant__", ptrT), param("v", f32)], tVoid(), @[
    asgnS(derefE(sym("__constant__", ptrT)),
      addE(f32, derefE(sym("__constant__", ptrT)), sym("v", f32))),
    retS(),
  ])
  let kern = procD("kern_rename", @[
    param("dst", aptrD)], tVoid(), @[
    varD("__shared__", f32, fltLit(1.5, f32)),
    callS(sym("__device__", addTy),
      @[addrE(ptrT, sym("__shared__", f32)), fltLit(2.0, f32)]),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("__shared__", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_rename", @[], @[helper, kern])

proc buildPlainModule(): CirModule =
  ## Kernel without a reserved name anywhere, the rename leaves it untouched.
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let kern = procD("kern_plain", @[
    param("dst", aptrD)], tVoid(), @[
    varD("acc", f32, fltLit(1.5, f32)),
    asgnS(patE(sym("dst", aptrD), intLit(0, i32)), sym("acc", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_plain", @[], @[kern])

proc buildByRefModule(): CirModule =
  ## Kernel plus device fn with a `var T` parameter, the call site passes
  ## the address-of over the local the frontend materializes.
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let ptrT = tPtr(caThread, f32)
  let addTy = tProcT(ptrT, f32, tVoid())
  let helper = procD("addinto", @[
    param("dst", ptrT), param("v", f32)], tVoid(), @[
    asgnS(derefE(sym("dst", ptrT)),
      addE(f32, derefE(sym("dst", ptrT)), sym("v", f32))),
    retS(),
  ])
  let kern = procD("kern_byref", @[
    param("outp", aptrD)], tVoid(), @[
    varD("acc", f32, fltLit(0.0, f32)),
    callS(sym("addinto", addTy),
      @[addrE(ptrT, sym("acc", f32)), fltLit(2.0, f32)]),
    asgnS(patE(sym("outp", aptrD), intLit(0, i32)), sym("acc", f32)),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_byref", @[], @[helper, kern])

proc buildArrayParamModule(): CirModule =
  ## Kernel plus device fn with a by-value `ctArray` parameter, the shape the v1 emitter decayed to `const T name[]` at emission.
  let f32 = tFloat(32)
  let i32 = tInt(32)
  let aptrD = tAptr(caDevice, f32)
  let arr4 = tArray(4, f32)
  let sumTy = tProcT(arr4, f32)
  let helper = procD("sum4", @[param("a", arr4)], f32, @[
    varD("result", f32, nil),
    asgnS(sym("result", f32),
      addE(f32, atE(sym("a", arr4), intLit(0, i32)),
                atE(sym("a", arr4), intLit(3, i32)))),
    retS(sym("result", f32)),
  ])
  let kern = procD("kern_arrayparam", @[
    param("outp", aptrD)], tVoid(), @[
    varD("w", arr4, aconstrE(arr4, fltLit(1.0, f32), fltLit(2.0, f32),
                             fltLit(3.0, f32), fltLit(4.0, f32))),
    asgnS(patE(sym("outp", aptrD), intLit(0, i32)),
      callE(sym("sum4", sumTy), @[sym("w", arr4)])),
    retS(),
  ])
  kern.isKernel = true
  moduleD("lower_arrayparam", @[], @[helper, kern])

# ═══════════════════════════════════════════════════════════════════════
#  Pipeline runs for the consts
# ═══════════════════════════════════════════════════════════════════════

proc runRename(build: proc(): CirModule, pipe: proc(m: CirModule): string):
    tuple[err, fnName, paramName, localName, argName: string] =
  ## One pipeline run over one fixture module, the renamed names read back from the module AST afterwards.
  let m = build()
  result.err = pipe(m)
  let helper = m.procs[0]
  let kern = m.procs[1]
  result.fnName = helper.name
  result.paramName = helper.params[0].name
  for s in kern.body:
    if s.kind == sVar:
      result.localName = s.name
    if s.kind == sCall:
      for a in s.callArgs:
        if a.kind == eAddr:
          result.argName = a.addrOperand.name

proc runByRefShape(build: proc(): CirModule): tuple[err, argShape: string] =
  ## One full-pipeline run over the by-reference fixture, reading the call-site argument shape back from the AST.
  let m = build()
  result.err = cudaPipeline(m)
  let kern = m.procs[1]
  for s in kern.body:
    if s.kind == sCall:
      let a = s.callArgs[0]
      result.argShape = $a.kind & " over " & $a.addrOperand.kind & " " &
        a.addrOperand.name

proc runArrayParamShape(build: proc(): CirModule): tuple[err, paramType: string] =
  ## One full-pipeline run over the array-parameter fixture, reading the parameter type back from the AST.
  let m = build()
  result.err = cudaPipeline(m)
  result.paramType = $m.procs[0].params[0].typ.kind

proc runPlainNames(build: proc(): CirModule, pipe: proc(m: CirModule): string):
    tuple[err, fnName, localName: string] =
  ## One pipeline run over the plain fixture, reading the untouched names back from the AST.
  let m = build()
  result.err = pipe(m)
  result.fnName = m.procs[0].name
  for s in m.procs[0].body:
    if s.kind == sVar:
      result.localName = s.name

const
  renameFull = runRename(buildRenameModule, cudaPipeline)
  renameBare = runRename(buildRenameModule, barePipeline)
  plainFull = runPlainNames(buildPlainModule, cudaPipeline)
  byRef = runByRefShape(buildByRefModule)
  arrayParam = runArrayParamShape(buildArrayParamModule)

# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

suite "CUDA lowering fixtures":

  test "reserved-word rename: qualifier-colliding names take the _nim suffix":
    check renameFull.err == ""
    check renameFull.fnName == "__device___nim"
    check renameFull.paramName == "__constant___nim"
    check renameFull.localName == "__shared___nim"
    # call site argument keeps the renamed local name, decl and use agree
    check renameFull.argName == "__shared___nim"

  test "bare pipeline control: without the rename the reserved names survive":
    check renameBare.err == ""
    check renameBare.fnName == "__device__"
    check renameBare.paramName == "__constant__"
    check renameBare.localName == "__shared__"
    check renameBare.argName == "__shared__"

  test "plain module passes through untouched":
    check plainFull.err == ""
    check plainFull.fnName == "kern_plain"
    check plainFull.localName == "acc"

  test "by-reference call arguments are materialized already, nothing to transform":
    ## Negative-finding evidence for the v1 materializePassByRefArgs transform:
    ## the call-site argument is the address-of over the local, and the pipeline verifies that shape clean and rewrites nothing.
    check byRef.err == ""
    check byRef.argShape == "eAddr over eSym acc"

  test "array parameters pass through, the decay is the printer's table":
    ## Negative-finding evidence for the v1 by-value array-parameter decay:
    ## the shared pipeline passes the ctArray parameter through unchanged
    ## and the module verifies clean, no transform waits for extraction.
    check arrayParam.err == ""
    check arrayParam.paramType == "ctArray"
