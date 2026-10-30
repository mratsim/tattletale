## Hand-built CIRv2: the single-use temp fold refusing a double read.
##
## Test that a temp read twice keeps both its declaration and its store:
## the second reader must still observe the stored value.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_single_use_temp_double_read.nim

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
  # `t` feeds two destinations: folding would duplicate its initializer
  # or leave the second reader reading nothing
  # two reads share one statement, no auxiliary temp can absorb them
  let doubleRead = procD("doubleRead", @[param("x", tInt(32))], tInt(32), @[
    varD("t", tInt(32), sym("x", tInt(32))),
    varD("o", tInt(32)),
    asgnS(sym("o", tInt(32)),
      addE(tInt(32), sym("t", tInt(32)), sym("t", tInt(32)))),
    retS(sym("o", tInt(32))),
  ])
  let (m, err) = optimize(moduleD("probe", @[], @[doubleRead]))
  doAssert err.len == 0, "the pass must succeed: " & err
  doAssert hasDecl(m.procs[0].body, "t"),
    "a twice-read temp must keep its declaration"
  var retReadsT = 0
  for st in m.procs[0].body:
    if st.kind == sRet and st.retVal != nil and st.retVal.kind == eAdd:
      if st.retVal.left.kind == eSym and st.retVal.left.name == "t":
        retReadsT.inc
      if st.retVal.right.kind == eSym and st.retVal.right.name == "t":
        retReadsT.inc
  doAssert retReadsT == 2, "both readers must keep reading the temp"
