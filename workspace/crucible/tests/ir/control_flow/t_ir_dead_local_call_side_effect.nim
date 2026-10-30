## Hand-built CIRv2: the dead-local pass keeping call side effects.
##
## Test that an `asgn` whose right-hand side is a call survives dead-local
## elimination even when nothing reads the destination, the call's side
## effects must keep running.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/control_flow/t_ir_dead_local_call_side_effect.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_aliases
import workspace/crucible/src/codegen/passes/passes_optimizations

proc deadLocals(m: CirModule): tuple[m: CirModule, err: string] =
  ## Runs the optimization stage over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerBuiltinKindAssignment()
  reg.registerOptimizationPasses()
  let err = m.runPasses(reg)
  result = (m, err)

proc hasCallAsgn(body: seq[CirStmt]): bool =
  ## True when a statement list still binds a call result to a destination.
  for st in body:
    if st.kind == sAsgn and st.rhs.kind == eCall:
      return true

proc hasDecl(body: seq[CirStmt], name: string): bool =
  ## True when a statement list still declares `name`.
  for st in body:
    if st.kind in {sVar, sConst} and st.name == name:
      return true

static:
  # callee writes through its pointer argument, the discarded return value
  # binds to a temp nothing reads: the write is the call's side effect
  let writerFn = procD("writer",
    @[param("p", tPtr(caDevice, tUint(32)))], tUint(32), @[
    retS(intLit(1, tUint(32))),
  ])
  let kernelFn = procD("callKernel",
    @[param("out", tPtr(caDevice, tUint(32)))], tVoid(), @[
    varD("tmp", tUint(32)),
    asgnS(sym("tmp", tUint(32)),
      callE(sym("writer", tProcT([tPtr(caDevice, tUint(32))], tUint(32))),
        sym("out", tPtr(caDevice, tUint(32))))),
    retS(),
  ])
  kernelFn.isKernel = true
  let (outM, err) = deadLocals(moduleD("probe", @[], @[kernelFn, writerFn]))
  doAssert err.len == 0, "dead-local elimination must succeed: " & err
  doAssert hasCallAsgn(outM.procs[0].body),
    "the call binding must survive: the call's side effects must keep running"
  doAssert hasDecl(outM.procs[0].body, "tmp"),
    "the call destination's declaration must survive with the call binding"
