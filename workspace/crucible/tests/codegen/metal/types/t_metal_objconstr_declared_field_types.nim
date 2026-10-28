## Metal: struct construction keeps the declared field types.
## Test that a value init leaves the field's declared type alone.
## The kernel passes the constructed struct to a device function.
## The function takes the declared struct, so a retyped construction fails to compile.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/types/t_metal_objconstr_declared_field_types.nim

import workspace/crucible

type Pack = object
  atom: uint32
  tag: int32

const packMsl = metal:
  proc readAtom(p: Pack): uint32 {.device.} =
    p.atom
  proc packKernel(outp: ptr UncheckedArray[uint32]) {.global.} =
    const p = Pack(atom: 3, tag: 7)
    outp[thread_position_in_grid.x] = readAtom(p) + uint32(p.tag)

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(packMsl)
  let oDev = engine.allocDevice(uint32, 1)
  engine.launchKernel("packKernel", oDev, ())
  var res: array[1, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 1)
  doAssert res[0] == 10'u32, "got " & $res[0]
  echo "  OK - t_metal_objconstr_declared_field_types"

when isMainModule:
  runTest()
