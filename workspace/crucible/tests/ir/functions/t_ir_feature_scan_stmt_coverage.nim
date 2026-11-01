## Hand-built CIRv2: the feature scans cover every statement shape that can reach a call.
##
## Test that the CUDA scan fills `features.reductions` from a statement-position reduction call through the pass registry.
## Test that the GLSL scan fills `features.subgroup` from a reduction call under `discard` through the pass registry.
## Test that the GLSL scan fills `features.fp16` from an `sAsm`/`sEmit` spliced operand through the pass registry.
## Test that the WGSL scan fills `features.fp16` from a call argument and from an operand under a condition through the pass registry.
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

proc runWgslScan(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerWgslFeatureScan()
  result = m.runPasses(reg)

static:
  let f32 = tFloat(32)
  let f16 = tFloat(16)
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

  # k3 splices an fp16 symbol through inline assembly: the spliced operand's
  # type is the only fp16 occurrence in the module
  let asmKernel = procD("k3", @[param("p", tPtr(caDevice, f32))], tVoid(), @[
    asmS(@["custom_asm $0"], @[sym("h", f16)]),
    retS(),
  ])
  asmKernel.isKernel = true
  let asmModule = moduleD("scanAsmOperand", @[], @[asmKernel])

  # k4 splices an fp16 symbol through an `{.emit.}` statement
  let emitKernel = procD("k4", @[param("p", tPtr(caDevice, f32))], tVoid(), @[
    emitS(@["custom_emit "], @[sym("h", f16)]),
    retS(),
  ])
  emitKernel.isKernel = true
  let emitModule = moduleD("scanEmitOperand", @[], @[emitKernel])

  # k5 passes an fp16 argument to an extension proc in statement position
  let extCallKernel = procD("k5", @[param("p", tPtr(caDevice, f32))], tVoid(), @[
    callS(sym("extConv", tProcT(f16, tVoid())), sym("h", f16)),
    retS(),
  ])
  extCallKernel.isKernel = true
  let extCallModule = moduleD("scanWgslCallArg", @[], @[extCallKernel])

  # k6 compares fp16 values under an `if`: the condition is bool-typed,
  # fp16 operands sit one level below it
  let condKernel = procD("k6", @[param("p", tPtr(caDevice, f32))], tVoid(), @[
    iteS(eqE(sym("h", f16), sym("g", f16)), @[retS()]),
    retS(),
  ])
  condKernel.isKernel = true
  let condModule = moduleD("scanWgslCond", @[], @[condKernel])

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

  block glslAsmOperand:
    let err = asmModule.runGlslScan()
    doAssert err.len == 0, "the GLSL feature scan must succeed: " & err
    doAssert asmModule.features.fp16,
      "an fp16-typed sAsm spliced operand must fill features.fp16, got: " &
        $asmModule.features

  block glslEmitOperand:
    let err = emitModule.runGlslScan()
    doAssert err.len == 0, "the GLSL feature scan must succeed: " & err
    doAssert emitModule.features.fp16,
      "an fp16-typed sEmit spliced operand must fill features.fp16, got: " &
        $emitModule.features

  block wgslCallArgument:
    let err = extCallModule.runWgslScan()
    doAssert err.len == 0, "the WGSL feature scan must succeed: " & err
    doAssert extCallModule.features.fp16,
      "an fp16-typed call argument must fill features.fp16 for WGSL, got: " &
        $extCallModule.features

  block wgslConditionOperand:
    let err = condModule.runWgslScan()
    doAssert err.len == 0, "the WGSL feature scan must succeed: " & err
    doAssert condModule.features.fp16,
      "fp16 operands under an sIte condition must fill features.fp16 for WGSL, got: " &
        $condModule.features

echo "  OK — feature scans cover statement-position calls and calls under discard"
