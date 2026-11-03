## CIRv2 pass-boundary verification: scope and name uniqueness
##
## Test that runPasses verifies at the final pass boundary and renders the offending proc
## into each refusal. Undeclared symbols, duplicate param names, and shared proc names refuse.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/wip_c17d06fix --nimcache:nimcache/wip_c17d06fix \
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
    doAssert "at k:" in err and "ghost" in err, "expected the refusal to name the proc and the symbol, got: " & err
    doAssert "(proc k (param out" in err, "expected the refusal to render the offending proc, got: " & err

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
    doAssert "(proc k (param out" in err, "expected the refusal to render the offending proc, got: " & err

  block branchScoped:
    let m = kernelModule(@[
      iteS(trueE(), @[varD("local", tInt(32), intLit(0, tInt(32)))], @[]),
      asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("local", tInt(32)))])
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a use outside its branch, got: " & err
    doAssert "local" in err, "expected the refusal to name the symbol, got: " & err
    doAssert "(proc k (param out" in err, "expected the refusal to render the offending proc, got: " & err

  block duplicateProcs:
    let fn = procD("k", @[], tVoid(), @[])
    fn.isKernel = true
    let m = moduleD("verify_scope", @[], @[fn, procD("k", @[], tVoid(), @[])])
    m.verifyIR = true
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal, got: " & err
    doAssert "'k'" in err, "expected the refusal to name the duplicated proc, got: " & err
    doAssert "(proc k (void)" in err, "expected the refusal to render a duplicated proc, got: " & err

  block duplicateParams:
    let fn = procD("k",
      @[param("out", tPtr(caDevice, tInt(32))), param("a", tInt(32)), param("a", tInt(32))],
      tVoid(),
      @[asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("a", tInt(32)))])
    fn.isKernel = true
    let m = moduleD("verify_scope", @[], @[fn])
    m.verifyIR = true
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal, got: " & err
    doAssert "at k:" in err and "'a'" in err, "expected the refusal to name the proc and the duplicated parameter, got: " & err
    doAssert "(proc k (param out" in err, "expected the refusal to render the offending proc, got: " & err

  block duplicateLocals:
    # A verbatim callee-body inliner emits this shape when one callee is inlined at two
    # call sites: the callee's locals are cloned into the caller's statement list.
    # Two same-name declarations in one list print into one brace pair.
    let m = kernelModule(@[
      varD("tmp", tInt(32), intLit(0, tInt(32))),
      varD("tmp", tInt(32), intLit(1, tInt(32)))])
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for duplicate locals, got: " & err
    doAssert "'tmp'" in err, "expected the refusal to name the duplicated local, got: " & err
    doAssert "(proc k (param out" in err, "expected the refusal to render the offending proc, got: " & err

  block duplicateLocalsInBranch:
    let m = kernelModule(@[
      iteS(trueE(), @[
        varD("tmp", tInt(32), intLit(0, tInt(32))),
        varD("tmp", tInt(32), intLit(1, tInt(32)))], @[])])
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for duplicate locals in one branch, got: " & err
    doAssert "'tmp'" in err, "expected the refusal to name the duplicated local, got: " & err

  block localShadowsParam:
    let fn = procD("k",
      @[param("out", tPtr(caDevice, tInt(32))), param("a", tInt(32))],
      tVoid(),
      @[varD("a", tInt(32), intLit(0, tInt(32)))])
    fn.isKernel = true
    let m = moduleD("verify_scope", @[], @[fn])
    m.verifyIR = true
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a local shadowing a parameter, got: " & err
    doAssert "'a'" in err, "expected the refusal to name the shadowed parameter, got: " & err

  block branchShadowStaysLegal:
    # Branch-scoped shadowing stays legal: the frontend deliberately emits branch-local
    # declarations reusing an outer name.
    let m = kernelModule(@[
      iteS(trueE(), @[varD("a", tInt(32), intLit(0, tInt(32)))], @[]),
      asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("out", tPtr(caDevice, tInt(32))))])
    let err = runPipeline(m)
    doAssert err.len == 0, "expected a clean pass over branch-scoped shadowing, got: " & err

when isMainModule:
  runTest()
