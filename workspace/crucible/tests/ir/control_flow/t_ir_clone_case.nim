## Hand-built CIRv2: cloneStmts deep-copies case statements and discard values.
##
## Test that a cloned sCase copies its selector, branch ranges, branch bodies and else body,
## and that a cloned sDiscard keeps its value.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/control_flow/t_ir_clone_case.nim

import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_helpers

static:
  let u32 = tUint(32)
  let orig = @[
    caseS(sym("s", u32), @[
      branch(@[rng(0, 3)], @[asgnS(sym("a", u32), intLit(1, u32))]),
    ], @[asgnS(sym("a", u32), intLit(2, u32))]),
    discardS(addE(u32, sym("a", u32), intLit(1, u32))),
  ]

  let copy = cloneStmts(orig)
  doAssert copy.len == 2, "cloneStmts kept both statements"
  doAssert copy[0] != orig[0], "the clone is a fresh statement ref"

  let c = copy[0]
  doAssert c.kind == sCase
  doAssert c.selector.name == "s", "the selector is cloned"
  doAssert c.selector != orig[0].selector, "the selector is a fresh ref"
  doAssert c.branches.len == 1, "the branch list is cloned"
  doAssert c.branches[0].ranges == orig[0].branches[0].ranges,
    "the branch ranges carry over"
  doAssert c.branches[0].body.len == 1, "the branch body is cloned"
  doAssert c.branches[0].body[0] != orig[0].branches[0].body[0],
    "the branch body is a fresh ref"
  doAssert c.branches[0].body[0].lhs.name == "a", "the branch body survives"
  doAssert c.caseElse.len == 1, "the else body is cloned"
  doAssert c.caseElse[0].lhs.name == "a", "the else body survives"

  let d = copy[1]
  doAssert d.kind == sDiscard
  doAssert d.discardVal != nil, "the discarded value survives"
  doAssert d.discardVal.kind == eAdd, "the discarded value survives"
  doAssert d.discardVal != orig[1].discardVal, "the value is a fresh ref"
