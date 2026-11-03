## CIRv2 pass-boundary scope verification
##
## Test that runPasses refuses a kernel body that uses an undeclared symbol.
## runPasses names the proc and the symbol in the refusal.
## A kernel body that declares the symbol before use passes clean.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:/tmp/c17d06-m2-test-out --nimcache:/tmp/c17d06-m2-test-cache \
##     workspace/crucible/tests/verify/t_cirv2_verify_scope.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_datatypes

proc runPipeline(m: CirModule): string =
  ## One identity transform leaves the module untouched. Verification runs after it.
  var reg = PassRegistry.new()
  reg.register("identity", pkTransform, "Leaves the module untouched",
    run = proc(m: CirModule): string = "")
  result = runPasses(m, reg)

proc kernelModule(body: seq[CirStmt]): CirModule =
  let outParam = param("out", tPtr(caDevice, tInt(32)))
  let fn = procD("k", @[outParam], tVoid(), body)
  fn.isKernel = true
  result = moduleD("verify_scope", @[], @[fn])
  result.verifyIR = true

proc runTest() =
  block undeclared:
    let m = kernelModule(@[asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("ghost", tInt(32)))])
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal, got: " & err
    doAssert "k" in err and "ghost" in err, "expected the refusal to name the proc and the symbol, got: " & err

  block declaredBeforeUse:
    let m = kernelModule(@[
      varD("ghost", tInt(32), intLit(0, tInt(32))),
      asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("ghost", tInt(32)))])
    let err = runPipeline(m)
    doAssert err.len == 0, "expected a clean pass over declared symbols, got: " & err

  block declaredAfterUse:
    let m = kernelModule(@[
      asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("ghost", tInt(32))),
      varD("ghost", tInt(32), intLit(0, tInt(32)))])
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a use before its declaration, got: " & err
    doAssert "ghost" in err, "expected the refusal to name the symbol, got: " & err

  block branchScoped:
    let m = kernelModule(@[
      iteS(trueE(), @[varD("local", tInt(32), intLit(0, tInt(32)))], @[]),
      asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("local", tInt(32)))])
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a use outside its branch, got: " & err
    doAssert "local" in err, "expected the refusal to name the symbol, got: " & err

  block verifyIRDisabled:
    let m = kernelModule(@[asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("ghost", tInt(32)))])
    m.verifyIR = false
    let err = runPipeline(m)
    doAssert err.len == 0, "expected no refusal with verifyIR off, got: " & err

when isMainModule:
  runTest()
