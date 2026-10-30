## Hand-built CIRv2: the feature scans cover every statement shape that can reach a call.
##
## Test that the CUDA scan fills `features.reductions` from a statement-position reduction call through the pass registry.
## Test that the GLSL scan fills `features.subgroup` from a reduction call under `discard` through the pass registry.
## The printers print both statement shapes, so the scans must cover them.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_feature_scan_stmt_coverage.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/scan_features

proc runCudaScan(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerCudaFeatureScan()
  result = m.runPasses(reg)

proc runGlslScan(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerGlslFeatureScan()
  result = m.runPasses(reg)

static:
  let f32 = tFloat(32)
  let reduceAddSym = sym("reduceAdd", tProcT(f32, f32), builtin = true,
    reductionBuiltin = rbkReduceAdd)

  # k1 calls reduceAdd in statement position. The printer prints that shape through warpReduceAdd(v)
  let callKernel = procD("k1", @[param("x", f32)], tVoid(), @[
    callS(reduceAddSym, sym("x", f32)),
    retS(),
  ])
  callKernel.isKernel = true
  let callModule = moduleD("scanCall", @[], @[callKernel])

  # k2 discards an expression holding the call: the value computes and drops
  let discardKernel = procD("k2", @[param("x", f32)], tVoid(), @[
    discardS(callE(reduceAddSym, sym("x", f32))),
    retS(),
  ])
  discardKernel.isKernel = true
  let discardModule = moduleD("scanDiscard", @[], @[discardKernel])

  block cudaStatementCall:
    let err = callModule.runCudaScan()
    doAssert err.len == 0, "the CUDA feature scan must succeed: " & err
    doAssert rbkReduceAdd in callModule.features.reductions,
      "a statement-position reduceAdd call must fill features.reductions, got: " &
        $callModule.features.reductions

  block glslCallUnderDiscard:
    let err = discardModule.runGlslScan()
    doAssert err.len == 0, "the GLSL feature scan must succeed: " & err
    doAssert discardModule.features.subgroup,
      "a reduceAdd call under discard must fill features.subgroup"

echo "  OK — feature scans cover statement-position calls and calls under discard"
