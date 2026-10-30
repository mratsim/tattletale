## Hand-built CIRv2: the single-use temp fold refusing an address-taken temp.
##
## Test that an address-taken temp keeps both its declaration and its store:
## the address must keep naming the temp's own storage.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_single_use_temp_addr_taken.nim

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
  # `addr t` names the temp's own storage. A fold would delete the declaration
  # that the address names, so the fold refuses.
  let addrTaken = procD("addrTaken", @[param("x", tInt(32))], tInt(32), @[
    varD("t", tInt(32), sym("x", tInt(32))),
    varD("o", tInt(32)),
    asgnS(sym("o", tInt(32)),
      derefE(addrE(tPtr(caThread, tInt(32)), sym("t", tInt(32))))),
    retS(sym("o", tInt(32))),
  ])
  let (m, err) = optimize(moduleD("probe", @[], @[addrTaken]))
  doAssert err.len == 0, "the pass must succeed: " & err
  doAssert hasDecl(m.procs[0].body, "t"),
    "an address-taken temp must keep its declaration"
  var addrStays = false
  for st in m.procs[0].body:
    if st.kind == sRet and st.retVal != nil and st.retVal.kind == eDeref and
        st.retVal.derefOperand.kind == eAddr and
        st.retVal.derefOperand.addrOperand.kind == eSym and
        st.retVal.derefOperand.addrOperand.name == "t":
      addrStays = true
  doAssert addrStays, "the address-of must keep naming the temp"
