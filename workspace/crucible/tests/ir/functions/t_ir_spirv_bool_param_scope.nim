## Hand-built CIRv2: the SPIR-V bool retype pass scoping the kernel names.
##
## Test that a device function with its own bool local named like a kernel
## bool parameter keeps real bool storage: the kernel parameter name set
## never leaks into a device function body.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_spirv_bool_param_scope.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_aliases
import workspace/crucible/src/codegen/passes/passes_lowering_spirv

proc retype(m: CirModule): tuple[m: CirModule, err: string] =
  ## Runs the SPIR-V bool retype pass over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerBuiltinKindAssignment()
  reg.registerSpirvLoweringPasses()
  let err = m.runPasses(reg)
  result = (m, err)

static:
  # kernel: bool parameter `enabled`, read once into a bool local
  let kernelFn = procD("boolKernel", @[param("enabled", tBool())], tVoid(), @[
    varD("out", tBool(), sym("enabled", tBool())),
    retS(),
  ])
  kernelFn.isKernel = true
  # device function placed after the kernel, own bool local `enabled`
  let devFn = procD("dev", @[], tBool(), @[
    varD("enabled", tBool(), trueE()),
    retS(sym("enabled", tBool())),
  ])
  let (outM, err) = retype(moduleD("probe", @[], @[kernelFn, devFn]))
  doAssert err.len == 0, "bool retype must succeed: " & err

  # kernel parameter retypes to i32, its read wraps in conv
  doAssert outM.procs[0].params[0].typ.kind == ctInt,
    "the kernel bool parameter must retype to i32 storage"
  doAssert outM.procs[0].body[0].init.kind == eConv,
    "the kernel's read of the retyped parameter must wrap in conv"

  # device function keeps real bool storage end to end
  doAssert outM.procs[1].body[0].declType.kind == ctBool,
    "the device function's own bool local must stay a bool"
  doAssert outM.procs[1].body[0].init.kind == eTrue,
    "the device function's bool local init must stay a bool literal"
  doAssert outM.procs[1].body[1].retVal.kind == eSym,
    "the device function's bool read must stay a plain symbol, got " &
      $outM.procs[1].body[1].retVal.kind
