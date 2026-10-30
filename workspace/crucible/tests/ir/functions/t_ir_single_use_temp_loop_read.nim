## Hand-built CIRv2: the single-use temp fold refusing a loop-body read.
##
## Test that a temp read inside a loop keeps its declaration and its store,
## the loop must keep reading the stored value.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_single_use_temp_loop_read.nim

import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_optimizations

proc optimize(m: CirModule): tuple[m: CirModule, err: string] =
  ## Runs the optimization stage over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerOptimizationPasses()
  let err = m.runPasses(reg)
  result = (m, err)

proc hasDecl(body: seq[CirStmt], name: string): bool =
  ## True when a statement list still declares `name`.
  for st in body:
    if st.kind in {sVar, sConst} and st.name == name:
      return true

static:
  # a read sits in the loop body: folding would reevaluate the initializer
  # once per iteration, or hoist a store out of the loop's reach
  let loopRead = procD("loopRead", @[param("x", tInt(32))], tInt(32), @[
    varD("t", tInt(32), sym("x", tInt(32))),
    varD("acc", tInt(32), intLit(0, tInt(32))),
    loopS(@[
      asgnS(sym("acc", tInt(32)),
        addE(tInt(32), sym("acc", tInt(32)), sym("t", tInt(32)))),
    ]),
    retS(sym("acc", tInt(32))),
  ])
  let (m, err) = optimize(moduleD("probe", @[], @[loopRead]))
  doAssert err.len == 0, "the pass must succeed: " & err
  doAssert hasDecl(m.procs[0].body, "t"),
    "a loop-read temp must keep its declaration"
  var loopKeepsRead = false
  for st in m.procs[0].body:
    if st.kind == sLoop:
      for inner in st.loopBody:
        if inner.kind == sAsgn and inner.rhs.kind == eAdd and
            inner.rhs.right.kind == eSym and inner.rhs.right.name == "t":
          loopKeepsRead = true
  doAssert loopKeepsRead, "the loop body must keep reading the temp"
