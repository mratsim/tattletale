## Hand-built CIRv2: the instantiation-name collision refusal.
##
## Test that the per-address-space instantiation pass refuses a clone whose
## instantiation name is already taken by another module proc:
##
## - a clean module instantiates `f` once per distinct argument space
## - a module carrying a second proc under the clone's name refuses
##
## The decoy runs the compile-signal path: a name collision refuses lowering,
## so there is no kernel to launch.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/ir/functions/t_ir_inst_name_collision.nim

import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_lowering_msl

static:
  let f32 = tFloat(32)
  let devP = tPtr(caDevice, f32)
  let tgP = tPtr(caThreadgroup, f32)
  let arrT = tArray(8, f32)

  proc runMslLowering(m: CirModule): string =
    var reg = PassRegistry.new()
    reg.registerMslLoweringPasses()
    result = m.runPasses(reg)

  proc buildModule(decoyName: string): CirModule =
    ## One kernel calling the device proc `f` with a threadgroup argument,
    ## plus an optional decoy proc named `decoyName`.
    let fFn = procD("f", @[param("p", devP), param("i", tInt(32))], f32, @[
      retS(derefE(sym("p", devP))),
    ])
    let kern = procD("k", @[param("out", tAptr(caDevice, f32))], tVoid(), @[
      varD("scratch", arrT, nil, caThreadgroup),
      asgnS(patE(sym("out", tAptr(caDevice, f32)), intLit(0, tInt(32))),
        callE(sym("f", tProcT(tgP, tInt(32), f32)),
          addrE(tgP, atE(sym("scratch", arrT), intLit(0, tInt(32)))),
          intLit(0, tInt(32)))),
    ])
    kern.isKernel = true
    var procs = @[fFn, kern]
    if decoyName.len > 0:
      let decoy = procD(decoyName, @[param("p", devP), param("i", tInt(32))], f32, @[
        retS(derefE(sym("p", devP))),
      ])
      procs.add decoy
    result = moduleD("probe", @[], procs)

  let clean = buildModule("")
  let err = clean.runMslLowering()
  doAssert err.len == 0, "the clean module must lower: " & err

  var cloneName = ""
  for fn in clean.procs:
    if fn.name.startsWith("f_"):
      cloneName = fn.name
  doAssert cloneName.len > 0, "the threadgroup call site must create one f clone"

  let colliding = buildModule(cloneName)
  let refusal = colliding.runMslLowering()
  doAssert refusal.len > 0,
    "a clone name already taken by another module proc must refuse, got clean " &
    "lowering with the clone renamed to an unknown discriminator"
