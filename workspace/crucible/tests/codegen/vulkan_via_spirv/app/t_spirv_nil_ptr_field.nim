## SPIR-V nil pointer field
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_nil_ptr_field.nim

# An omitted constructor field with pointer type reads as nil through isNil
# on the device.

import workspace/crucible

type Slot = object
  v: int32
  buf: ptr UncheckedArray[int32]

proc nilSlot(outBuf: ptr UncheckedArray[int32]) =
  let s = Slot(v: 7)
  if s.buf.isNil():
    outBuf[0] = 1
  else:
    outBuf[0] = 0

proc runTest() =
  var engine = newEngine(runVulkan)
  let oDev = engine.allocDevice(int32, 1)
  engine.launchDeviceFn nilSlot(oDev) do:
    lang = spirv
    grid = (1, 1)
    blk = (1, 1)
  var got: array[1, int32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 0), oDev)
  doAssert got[0] == 1,
    "isNil on the omitted pointer field: got " & $got[0] & ", want 1"

when isMainModule:
  runTest()
  echo "[OK] SPIR-V: the omitted pointer field reads as nil"
