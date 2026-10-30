## Hand-built CIRv2: the CUDA reserved-identifier rename registration.
##
## Test that the registered CUDA lowering passes rename an identifier
## colliding with the execution-space qualifiers the CUDA-C++ emitter writes:
## a local named `__shared__` emits as `__shared___nim`.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_cuda_reserved_rename.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_cuda

proc runCudaLowering(m: CirModule): string =
  var reg = PassRegistry.new()
  reg.registerCudaLoweringPasses()
  result = m.runPasses(reg)

static:
  let u32 = tUint(32)
  # Nim's lexer refuses `_`-initial identifiers, so Nim source cannot name a local `__shared__` directly.
  # IR-level names face no such restriction: `__shared__` in the IR collides with the `__shared__` qualifier emitted for `{.smem.}` locals.
  let kernelFn = procD("k", @[param("output", tAptr(caDevice, u32))], tVoid(), @[
    constD("__shared__", u32, intLit(1, u32)),
    asgnS(atE(sym("output", tAptr(caDevice, u32)), intLit(0, u32)),
      sym("__shared__", u32)),
    retS(),
  ])
  kernelFn.isKernel = true
  let m = moduleD("probe", @[], @[kernelFn])

  let err = m.runCudaLowering()
  doAssert err.len == 0, "the CUDA reserved rename must succeed: " & err
  let body = m.procs[0].body
  doAssert body[0].name == "__shared___nim",
    "the local named `__shared__` must rename to `__shared___nim`, got: " &
      body[0].name
  doAssert body[1].rhs.name == "__shared___nim",
    "the reader of the renamed local must rename with it"

echo "  OK — the CUDA reserved-identifier rename registration renames `__shared__`"
