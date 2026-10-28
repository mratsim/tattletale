## CIRv2 control flow: forward-only scoped lab/jmp and the conditionless loop
##
## The verifier rejects backward jumps, scope-entering jumps, unknown labels, and unused labels.
## A loop is conditionless by shape: guards are statements in the body, exits are forward jmps
## to labs outside the loop.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests/cirv2 --nimcache:nimcache/tests/cirv2 \
##     workspace/crucible/tests/cirv2/t_cirv2_controlflow.nim

import
  workspace/crucible/src/codegen/ir/cirv2_lang,
  workspace/crucible/src/codegen/ir/cirv2_build,
  workspace/crucible/src/codegen/ir/cirv2_verify,
  workspace/crucible/tests/cirv2/cirv2_testutils

static:
  let
    i32t = tInt(32)
    i = sym("i", i32t)

  # backward jmp: the lab appears textually before the jmp
  block:
    let fn = procD("backward_jmp", @[], tVoid(), @[
      labS("back"),
      jmpS("back"),
    ])
    fn.expectVerifyError("backward jump to back")

  # jmp entering a scope: the target lab lies inside a later sibling scope
  block:
    let fn = procD("jmp_enters_scope", @[], tVoid(), @[
      jmpS("inside"),
      scopeS(@[labS("inside")]),
    ])
    fn.expectVerifyError("target lab inside lies in a scope the jmp would enter")

  # unknown label: the jmp target never appears
  block:
    let fn = procD("unknown_label", @[], tVoid(), @[
      jmpS("nowhere"),
    ])
    fn.expectVerifyError("unknown label nowhere")

  # unused lab: declared but never targeted
  block:
    let fn = procD("unused_lab", @[], tVoid(), @[
      labS("orphan"),
    ])
    fn.expectVerifyError("label orphan is never targeted")

  # duplicate lab names: labels are proc-unique symbols
  block:
    let fn = procD("duplicate_lab", @[], tVoid(), @[
      labS("twice"),
      labS("twice"),
    ])
    fn.expectVerifyError("duplicate label twice")

  # loop shape: a conditionless loop with a leading guard. The guard is
  # an `ite` in the body. The false variant jmps forward to a lab after the loop.
  block:
    let fn = procD("guarded_loop", @[], tVoid(), @[
      varD("i", i32t, intLit(0, i32t)),
      loopS(@[
        iteS(ltE(i, intLit(10, i32t)),
          @[asgnS(i, addE(i32t, i, intLit(1, i32t)))],
          @[jmpS("done")]),
      ]),
      labS("done"),
    ])
    verify(fn)

  # shape assertion: a loop has no condition slot. The sLoop branch
  # has no cond field, so a loop with a condition is not expressible.
  doAssert not compiles(CirStmt(kind: sLoop, cond: trueE()))
