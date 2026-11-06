## CIRv2 pass-boundary verification: scope and name uniqueness
##
## Test that runPasses verifies at the final pass boundary and renders the offending proc
## into each refusal. Undeclared symbols, duplicate param names, and shared proc names refuse.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/verify/t_cirv2_verify_scope.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_datatypes

proc runPipeline(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.register("identity", pkTransform, "Leaves the module untouched",
    run = proc(m: CirModule): string = "")
  result = reg.runPasses(m)

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

  block duplicateKernelExportNames:
    # Printers emit a kernel under its `exportName` when set: two kernels sharing one
    # exportName emit two file-scope declarations of one name.
    let fn1 = procD("k1", @[], tVoid(), @[])
    fn1.isKernel = true
    fn1.exportName = "emitted"
    let fn2 = procD("k2", @[], tVoid(), @[])
    fn2.isKernel = true
    fn2.exportName = "emitted"
    let m = moduleD("verify_scope", @[], @[fn1, fn2])
    m.verifyIR = true
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for duplicate emitted kernel names, got: " & err
    doAssert "duplicate emitted name 'emitted'" in err, "expected the refusal to name the colliding emitted name, got: " & err
    doAssert "(proc k2 (void)" in err, "expected the refusal to render the second kernel, got: " & err

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
    let m = kernelModule(@[
      varD("a", tInt(32), intLit(0, tInt(32))),
      iteS(trueE(), @[varD("a", tInt(32), intLit(1, tInt(32)))], @[]),
      iteS(trueE(), @[varD("out", tInt(32), intLit(2, tInt(32)))], @[]),
      asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("a", tInt(32)))])
    let err = runPipeline(m)
    doAssert err.len == 0, "expected a clean pass over branch-scoped shadowing, got: " & err

  block duplicateGlobals:
    # Hoisting passes append one program-scope declaration per kernel verbatim.
    let m = kernelModule(@[asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("g", tInt(32)))])
    m.globals = @[
      varD("g", tInt(32), intLit(0, tInt(32)), caThreadgroup),
      varD("g", tInt(32), intLit(1, tInt(32)), caThreadgroup)]
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for duplicate globals, got: " & err
    doAssert "'g'" in err, "expected the refusal to name the duplicated global, got: " & err

  block globalInitUndeclared:
    let m = kernelModule(@[])
    m.globals = @[varD("g", tInt(32), sym("nowhere", tInt(32)), caThreadgroup)]
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for an unresolvable global initializer, got: " & err
    doAssert "at global 'g':" in err and "nowhere" in err, "expected the refusal to name the global and the symbol, got: " & err

  block globalInitDeclaresBeforeUse:
    let m = kernelModule(@[])
    m.globals = @[
      varD("a", tInt(32), sym("b", tInt(32)), caThreadgroup),
      varD("b", tInt(32), intLit(0, tInt(32)), caThreadgroup)]
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a forward global reference, got: " & err
    doAssert "at global 'a':" in err and "'b'" in err, "expected the refusal to name the global and the symbol, got: " & err

  block globalInitSelfReference:
    let m = kernelModule(@[])
    m.globals = @[varD("g", tInt(32), sym("g", tInt(32)), caThreadgroup)]
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a self-referential global initializer, got: " & err
    doAssert "at global 'g': symbol 'g' resolves to no declaration in scope" in err, "expected the refusal to name the global and its own unresolved symbol, got: " & err

  block globalReadResolves:
    let m = kernelModule(@[asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("g", tInt(32)))])
    m.globals = @[
      varD("g0", tInt(32), intLit(0, tInt(32)), caThreadgroup),
      varD("g", tInt(32), sym("g0", tInt(32)), caThreadgroup)]
    let err = runPipeline(m)
    doAssert err.len == 0, "expected a clean pass over resolvable globals, got: " & err

  block nonSVarGlobalRefuses:
    # Lowering passes hoist `var` statements only, any other statement kind here is a transform bug.
    let m = kernelModule(@[])
    m.globals = @[asgnS(sym("x", tInt(32)), intLit(0, tInt(32)))]
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a non-var global, got: " & err
    doAssert "sAsgn" in err, "expected the refusal to name the statement kind, got: " & err

  block globalNameCollidesWithProc:
    let m = kernelModule(@[])
    m.globals = @[varD("k", tInt(32), intLit(0, tInt(32)), caThreadgroup)]
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a global named like a proc, got: " & err
    doAssert "'k'" in err, "expected the refusal to name the colliding name, got: " & err

  block globalNameCollidesWithKernelExportName:
    let fn = procD("k", @[], tVoid(), @[])
    fn.isKernel = true
    fn.exportName = "g"
    let m = moduleD("verify_scope", @[], @[fn])
    m.globals = @[varD("g", tInt(32), intLit(0, tInt(32)), caThreadgroup)]
    m.verifyIR = true
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a global named like a kernel's emitted name, got: " & err
    doAssert "global name 'g' collides with a proc name" in err, "expected the refusal to name the colliding name, got: " & err

  block zeroTransformRegistry:
    # An analysis-only registry has no transform boundary: builder-produced IR is
    # final by definition and still verifies.
    let m = kernelModule(@[asgnS(sym("out", tPtr(caDevice, tInt(32))), sym("ghost", tInt(32)))])
    var reg = PassRegistry.new()
    reg.register("scan", pkAnalysis, "Computes metadata",
      run = proc(m: CirModule): string = "")
    let err = reg.runPasses(m)
    doAssert "verify: " in err, "expected the verify refusal on an analysis-only registry, got: " & err
    doAssert "ghost" in err and "(proc k (param out" in err, "expected the refusal to name the symbol and render the proc, got: " & err

  block nilStatementRefuses:
    let m = kernelModule(@[
      asgnS(sym("out", tPtr(caDevice, tInt(32))), intLit(0, tInt(32))),
      CirStmt(nil)])
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for a nil statement, got: " & err
    doAssert "nil statement" in err and "(proc k (param out" in err, "expected the refusal to name the defect and render the proc, got: " & err

  block emptyParamNameRefuses:
    # checkUniqueNames reports "" for a clean scan, two empty names would otherwise
    # pass as the clean sentinel.
    let fn = procD("k", @[param("", tInt(32)), param("", tInt(32))], tVoid(), @[])
    fn.isKernel = true
    let m = moduleD("verify_scope", @[], @[fn])
    m.verifyIR = true
    let err = runPipeline(m)
    doAssert "verify: " in err, "expected the verify refusal for an empty parameter name, got: " & err
    doAssert "empty parameter name" in err and "(proc k" in err, "expected the refusal to name the defect and render the proc, got: " & err

when isMainModule:
  runTest()
