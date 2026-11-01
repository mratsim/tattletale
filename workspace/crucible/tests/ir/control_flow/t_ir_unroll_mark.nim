## Hand-built CIRv2: the unroll-marking pass.
##
## Test that the `{.unroll.}` marker statement sets the unroll flag on the loop
## whose body states it, that an inner-loop marker marks the inner loop only,
## and that a stray marker drops without marking anything.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/control_flow/t_ir_unroll_mark.nim

import std/sequtils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_datatypes
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_norm_unroll

static:
  let u32 = tUint(32)
  let marker = emitS(@["{.unroll.}"], @[])
  let outT = tAptr(caDevice, u32)

  proc body(ss: seq[CirStmt]): CirProc =
    procD("k", @[param("output", outT)], tVoid(), ss & @[retS()])

  # a loop body stating the marker gets the loop unrolled and the marker dropped
  var m = moduleD("t", @[], @[
    body(@[
      varD("i", u32, intLit(0, u32)),
      whileS(ltE(sym("i", u32), intLit(4, u32)), @[
        marker,
        asgnS(sym("i", u32), addE(u32, sym("i", u32), intLit(1, u32))),
      ]),
    ]),
  ])
  var reg = PassRegistry.new()
  reg.registerMarkExplicitUnrolls()
  doAssert runPasses(m, reg) == ""

  let loops = m.procs[0].body.filterIt(it.kind == sLoop)
  doAssert loops.len == 1
  doAssert loops[0].unroll, "the marked loop sets unroll"
  doAssert loops[0].loopBody.len == 1, "the marker statement drops"

  # a marker nested in an inner loop's body marks the inner loop only
  m = moduleD("t", @[], @[
    body(@[
      varD("i", u32, intLit(0, u32)),
      whileS(ltE(sym("i", u32), intLit(4, u32)), @[
        varD("j", u32, intLit(0, u32)),
        whileS(ltE(sym("j", u32), intLit(2, u32)), @[
          marker,
          asgnS(sym("j", u32), addE(u32, sym("j", u32), intLit(1, u32))),
        ]),
        asgnS(sym("i", u32), addE(u32, sym("i", u32), intLit(1, u32))),
      ]),
    ]),
  ])
  reg = PassRegistry.new()
  reg.registerMarkExplicitUnrolls()
  doAssert runPasses(m, reg) == ""

  let outer = m.procs[0].body[1]
  doAssert outer.kind == sLoop
  doAssert not outer.unroll, "the outer loop stays unmarked"
  doAssert outer.loopBody.len == 3, "the inner marker drops from the inner body only"
  let inner = outer.loopBody[1]
  doAssert inner.kind == sLoop
  doAssert inner.unroll, "the inner loop takes the marker"
  doAssert inner.loopBody.len == 1, "the marker drops from the inner body"

  # a stray marker outside any loop drops without marking
  m = moduleD("t", @[], @[
    body(@[marker, asgnS(sym("i", u32), intLit(0, u32))]),
  ])
  reg = PassRegistry.new()
  reg.registerMarkExplicitUnrolls()
  doAssert runPasses(m, reg) == ""
  doAssert m.procs[0].body.len == 2, "the stray marker drops"
