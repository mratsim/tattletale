## Hand-built CIRv2: the dead-local pass counting discard and assume reads.
##
## Test that `discard <local>` and `assume <local>` count as reads:
## a local read only by discard or assume keeps its declaration.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/control_flow/t_ir_dead_local_discard_read.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_aliases
import workspace/crucible/src/codegen/passes/passes_optimizations

proc deadLocals(m: CirModule): tuple[m: CirModule, err: string] =
  ## Runs the dead-local elimination pass over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerBuiltinKindAssignment()
  reg.registerOptimizationPasses()
  let err = m.runPasses(reg)
  result = (m, err)

proc hasDecl(body: seq[CirStmt], name: string): bool =
  ## True when a statement list still declares `name`.
  for st in body:
    if st.kind in {sVar, sConst} and st.name == name:
      return true

static:
  # kernel: `discard x` and `assume y` read their bare locals,
  # an unrelated local `dead` stays removable
  let fn = procD("discardKernel", @[], tVoid(), @[
    varD("x", tInt(32), intLit(1, tInt(32))),
    varD("y", tInt(32), intLit(2, tInt(32))),
    varD("dead", tInt(32), intLit(3, tInt(32))),
    discardS(sym("x", tInt(32))),
    assumeS(sym("y", tInt(32))),
    retS(),
  ])
  fn.isKernel = true
  let (outM, err) = deadLocals(moduleD("probe", @[], @[fn]))
  doAssert err.len == 0, "dead-local elimination must succeed: " & err
  doAssert hasDecl(outM.procs[0].body, "x"),
    "the discarded local's declaration must survive"
  doAssert hasDecl(outM.procs[0].body, "y"),
    "the assumed local's declaration must survive"
  doAssert not hasDecl(outM.procs[0].body, "dead"),
    "an unread local must still be removed"
