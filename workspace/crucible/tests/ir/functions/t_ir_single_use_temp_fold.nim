## Hand-built CIRv2: the single-use temp fold.
##
## Test that a temp assigned once and read once folds into its only reader,
## the declaration and the store delete, and a side-effect-free call folds
## into a whole destination read.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_single_use_temp_fold.nim

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

proc asgnRhs(body: seq[CirStmt], dest: string): CirExpr =
  ## Returns the right-hand side of the assignment to `dest`, nil when absent.
  for st in body:
    if st.kind == sAsgn and st.lhs.kind == eSym and st.lhs.name == dest:
      return st.rhs

static:
  # positive 1: `var t = x; o = t + 1`. The read sits nested in the use's
  # right-hand side, so the pure initializer splices into it
  let initTemp = procD("foldInit", @[param("x", tInt(32))], tInt(32), @[
    varD("t", tInt(32), sym("x", tInt(32))),
    asgnS(sym("out0", tInt(32)),
      addE(tInt(32), sym("t", tInt(32)), intLit(1, tInt(32)))),
    retS(),
  ])
  let (m1, err1) = optimize(moduleD("probe", @[], @[initTemp]))
  doAssert err1.len == 0, "the fold must succeed: " & err1
  doAssert not hasDecl(m1.procs[0].body, "t"),
    "the single-use temp's declaration must delete"
  let rhs1 = asgnRhs(m1.procs[0].body, "out0")
  doAssert rhs1 != nil and rhs1.kind == eAdd and
    rhs1.left.kind == eSym and rhs1.left.name == "x",
    "the read must fold to the temp's initializer"

  # positive 2: `tmp = pureFn(x); o = tmp`, a side-effect-free call folds
  # into a whole destination read
  let pureFn = procD("pureFn", @[param("a", tInt(32))], tInt(32), @[
    retS(addE(tInt(32), sym("a", tInt(32)), intLit(1, tInt(32)))),
  ])
  let callTemp = procD("foldCall", @[param("x", tInt(32))], tInt(32), @[
    varD("tmp", tInt(32)),
    asgnS(sym("tmp", tInt(32)),
      callE(sym("pureFn", tProcT([tInt(32)], tInt(32))), sym("x", tInt(32)))),
    asgnS(sym("o", tInt(32)), sym("tmp", tInt(32))),
    retS(sym("o", tInt(32))),
  ])
  let (m2, err2) = optimize(moduleD("probe", @[], @[callTemp, pureFn]))
  doAssert err2.len == 0, "the fold must succeed: " & err2
  doAssert not hasDecl(m2.procs[0].body, "tmp"),
    "the call temp's declaration must delete"
  let rhs2 = asgnRhs(m2.procs[0].body, "o")
  doAssert rhs2 != nil and rhs2.kind == eCall and
    rhs2.callee.kind == eSym and rhs2.callee.name == "pureFn",
    "the side-effect-free call must move to the destination"

  # positive 3: a two-temp chain folds to a fixpoint, `o` reads `x` directly
  let chain = procD("foldChain", @[param("x", tInt(32))], tInt(32), @[
    varD("a", tInt(32)),
    asgnS(sym("a", tInt(32)), sym("x", tInt(32))),
    varD("b", tInt(32)),
    asgnS(sym("b", tInt(32)), sym("a", tInt(32))),
    asgnS(sym("o", tInt(32)), sym("b", tInt(32))),
    retS(sym("o", tInt(32))),
  ])
  let (m3, err3) = optimize(moduleD("probe", @[], @[chain]))
  doAssert err3.len == 0, "the fold must succeed: " & err3
  doAssert not hasDecl(m3.procs[0].body, "a") and
    not hasDecl(m3.procs[0].body, "b"),
    "both chained temps must delete"
  let rhs3 = asgnRhs(m3.procs[0].body, "o")
  doAssert rhs3 != nil and rhs3.kind == eSym and rhs3.name == "x",
    "the chain must fold to the original value"
