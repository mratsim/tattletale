## Hand-built CIRv2: the single-use temp fold refusing a side-effecting call.
##
## Test that a pointer-writing call temp keeps its declaration and store:
## the write must keep running at the store site.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_single_use_temp_effectful_call.nim

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
  # `writer` writes through its pointer argument. A fold would move the call
  # next to the reader and reorder the write across the reader's statement
  let writerFn = procD("writer",
    @[param("p", tPtr(caDevice, tUint(32)))], tUint(32), @[
    asgnS(derefE(sym("p", tPtr(caDevice, tUint(32)))),
      intLit(1, tUint(32))),
    retS(intLit(1, tUint(32))),
  ])
  let callTemp = procD("effectfulCall",
    @[param("buf", tPtr(caDevice, tUint(32)))], tUint(32), @[
    varD("tmp", tUint(32)),
    asgnS(sym("tmp", tUint(32)),
      callE(sym("writer", tProcT([tPtr(caDevice, tUint(32))], tUint(32))),
        sym("buf", tPtr(caDevice, tUint(32))))),
    varD("o", tUint(32)),
    asgnS(sym("o", tUint(32)), sym("tmp", tUint(32))),
    retS(sym("o", tUint(32))),
  ])
  let (m, err) = optimize(moduleD("probe", @[], @[callTemp, writerFn]))
  doAssert err.len == 0, "the pass must succeed: " & err
  doAssert hasDecl(m.procs[0].body, "tmp"),
    "a side-effecting call temp must keep its declaration"
  var callStays = false
  for st in m.procs[0].body:
    if st.kind == sAsgn and st.rhs.kind == eCall and
        st.rhs.callee.kind == eSym and st.rhs.callee.name == "writer":
      callStays = true
  doAssert callStays, "the side-effecting call must stay at the store site"
