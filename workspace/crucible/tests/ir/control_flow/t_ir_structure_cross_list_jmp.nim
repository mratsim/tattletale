## Hand-built CIRv2: the control-flow structuring pass over cross-list jumps.
##
## Test that a `jmp` whose target lives in no statement list refuses loudly,
## while a one-sided `jmp` to a loop continue label stays accepted.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/control_flow/t_ir_structure_cross_list_jmp.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_structure

proc structure(m: CirModule): tuple[m: CirModule, err: string] =
  ## Runs the structuring pass over `m` through the registry.
  var reg = PassRegistry.new()
  reg.registerStructuring()
  let err = m.runPasses(reg)
  result = (m, err)

static:
  # one-sided continue jump from an ite inside a loop body
  # legitimate loop-continue shape, accepted unchanged
  let contFn = procD("contKernel", @[param("x", tInt(32))], tVoid(), @[
    loopS(@[
      iteS(eqE(sym("x", tInt(32)), intLit(3, tInt(32))), @[jmpS("lc1")], @[]),
      labS("lc1"),
      asgnS(sym("x", tInt(32)),
        addE(tInt(32), sym("x", tInt(32)), intLit(1, tInt(32)))),
    ]),
    labS("lx1"),
  ])
  contFn.isKernel = true
  let (contOut, contErr) = structure(moduleD("probe", @[], @[contFn]))
  doAssert contErr.len == 0, "continue shape must stay accepted: " & contErr
  doAssert contOut.procs[0].body.len == 2, "expected loop + exit label"

  # a `jmp` whose target lives in no statement list, mid-body inside a loop
  # this jump is the loop's only exit, dropping it loops forever
  let crossFn = procD("crossKernel", @[], tVoid(), @[
    loopS(@[
      jmpS("afterLoop"),
      asgnS(sym("x", tInt(32)), intLit(1, tInt(32))),
    ]),
    labS("lx1"),
    asgnS(sym("x", tInt(32)), intLit(2, tInt(32))),
  ])
  crossFn.isKernel = true
  let (crossOut, crossErr) = structure(moduleD("probe", @[], @[crossFn]))
  doAssert crossErr.len > 0, "cross-list jmp must refuse, got a clean pass over " &
    $crossOut.procs[0].body.len & " statements"
  doAssert "leaves the statement list" in crossErr, "unexpected refusal: " & crossErr
