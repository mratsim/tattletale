## Metal: if-expression branches carry side effects.
## Test that an if-expression runs only the selected branch's statements
## and yields the selected branch's value, for both polarities.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_ifexpr_side_effects.nim

import workspace/crucible

const ifExprMsl = metal:
  proc ifExprKernel(output, scratch, selector: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if selector[0] != 0'u32:
      (block:
        scratch[0] = 1'u32
        10'u32)
    else:
      (block:
        scratch[1] = 1'u32
        20'u32)
    output[1] = scratch[0]
    output[2] = scratch[1]

proc runPolarity(selector: uint32, wantOut: uint32, writeSlot: int) =
  var engine = newEngine(runMetal)
  engine.ingest(ifExprMsl)
  let oDev = engine.allocDevice(uint32, 1)
  let sDev = engine.allocDevice(uint32, 2)
  var zeros: array[2, uint32] = [0'u32, 0'u32]
  engine.copyToDeviceFromHost(sDev, addr zeros[0], 2)
  let selDev = engine.allocDevice(uint32, 1)
  var sel: array[1, uint32] = [selector]
  engine.copyToDeviceFromHost(selDev, addr sel[0], 1)
  engine.launchKernel("ifExprKernel", oDev, (sDev, selDev))
  var res: array[3, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 3)
  doAssert res[0] == wantOut, "polarity " & $selector & " yielded " & $res[0]
  doAssert res[1] == (if writeSlot == 0: 1'u32 else: 0'u32) and
         res[2] == (if writeSlot == 1: 1'u32 else: 0'u32),
    "the selected branch must write its slot only, got [" &
    $res[1] & ", " & $res[2] & "]"

proc runTest() =
  runPolarity(1'u32, 10'u32, 0)
  runPolarity(0'u32, 20'u32, 1)
  echo "  OK - t_metal_ifexpr_side_effects"

when isMainModule:
  runTest()
