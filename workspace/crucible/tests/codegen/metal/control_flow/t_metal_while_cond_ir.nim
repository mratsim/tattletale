## Metal: conditioned loop through hand-built CIRv2, launched and read back.
##
## Test that a conditioned loop and a ternary from hand-built IR compile, launch on Metal, and read back 100, 1, 2, 0.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/m2metal \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_while_cond_ir.nim

import std/unittest
import workspace/crucible
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/lang_msl

const condMsl = block:
  let u32 = tUint(32)
  let outT = tAptr(caDevice, u32)

  # device fn: limitOf(1) = 3, computed per iteration by the loop condition
  let limitOf = procD("limitOf", @[param("x", u32)], u32, @[
    retS(addE(u32, sym("x", u32), intLit(2, u32))),
  ])
  let kern = procD("condKernel", @[param("output", outT)], tVoid(), @[
    varD("i", u32, intLit(0, u32)),
    whileS(ltE(sym("i", u32), callE(sym("limitOf", u32), intLit(1, u32))), @[
      asgnS(patE(sym("output", outT), sym("i", u32)),
        ternaryE(u32, ltE(sym("i", u32), intLit(1, u32)),
          intLit(100, u32), sym("i", u32))),
      asgnS(sym("i", u32),
        addE(u32, sym("i", u32), intLit(1, u32))),
    ]),
    retS(),
  ])
  kern.isKernel = true

  let m = moduleD("probe", @[], @[limitOf, kern])
  printMslModule(m)

proc runTest() =
  suite "Metal - conditioned loop from hand-built IR":
    test "while i < limitOf(1) runs and the ternary writes 100, 1, 2":
      var engine = newEngine(runMetal)
      engine.ingest(condMsl)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("condKernel", oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 100'u32
      check res[1] == 1'u32
      check res[2] == 2'u32
      check res[3] == 0'u32

when isMainModule:
  runTest()
