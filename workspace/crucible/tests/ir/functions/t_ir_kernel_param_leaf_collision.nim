## Hand-built CIRv2: the kernel struct-argument explosion's leaf-name collisions.
##
## Test that a leaf name colliding with a kernel parameter or a body local
## refuses through the registry's refusal string, and that a clean struct
## parameter still explodes into its leaves.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_kernel_param_leaf_collision.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_low_kernel_params

proc explode(m: CirModule): tuple[m: CirModule, err: string] =
  ## Runs the kernel struct-argument explosion over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerExplodeKernelStructs()
  let err = m.runPasses(reg)
  result = (m, err)

static:
  # AxpyView has one pointer field: the parameter explodes to the leaf `view_p`
  let viewTy = tStruct("AxpyView", [fld("p", tPtr(caDevice, tUint(32)))])

  # kernel: a body local named like the exploded leaf
  let localKernel = procD("localKernel", @[param("view", viewTy)], tVoid(), @[
    varD("view_p", tUint(32), intLit(0, tUint(32))),
    retS(),
  ])
  localKernel.isKernel = true
  let (_, localErr) = explode(moduleD("probe", @[], @[localKernel]))
  doAssert localErr.contains("collides"),
    "a body local named like an exploded leaf must refuse, got: " & localErr

  # kernel: a scalar parameter named like the exploded leaf
  let paramKernel = procD("paramKernel",
    @[param("view", viewTy), param("view_p", tUint(32))], tVoid(), @[retS()])
  paramKernel.isKernel = true
  let (_, paramErr) = explode(moduleD("probe", @[], @[paramKernel]))
  doAssert paramErr.contains("collides"),
    "a kernel parameter named like an exploded leaf must refuse, got: " & paramErr

  # clean kernel: the struct parameter explodes and the body opens with the rebuilt local
  let cleanKernel = procD("cleanKernel", @[param("view", viewTy)], tVoid(), @[retS()])
  cleanKernel.isKernel = true
  let (outM, cleanErr) = explode(moduleD("probe", @[], @[cleanKernel]))
  doAssert cleanErr.len == 0, "the clean kernel must pass the explosion: " & cleanErr
  doAssert outM.procs[0].params.len == 1 and
    outM.procs[0].params[0].name == "view_p" and
    outM.procs[0].params[0].typ.kind == ctPtr,
    "the struct parameter must explode to the pointer leaf view_p"
  doAssert outM.procs[0].body[0].kind == sVar and
    outM.procs[0].body[0].name == "view",
    "the kernel body must open with the rebuilt struct local"
