## Vulkan (GLSL): the address route survives engine route autodetection.
##
## Test that a `glsl:` kernel with buffer and by-value scalar arguments keeps
## running through the address route under engine route autodetection:
##
## - two buffer inputs + one by-value scalar, packed as push constants
## - results read back and asserted exactly
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/app/t_vulkan_route_address.nim

import workspace/crucible

const addVk = glsl:
  proc addScalarKernel(output, a, b: ptr UncheckedArray[uint32], z: uint32) {.global.} =
    output[0] = a[0] + b[0] + z
    output[1] = a[1] + b[1] + z

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = newEngine(runVulkan)
  engine.ingest(addVk)

  var a: array[2, uint32] = [10'u32, 20'u32]
  var b: array[2, uint32] = [1'u32, 2'u32]
  let aDev = engine.allocDevice(uint32, a.len)
  let bDev = engine.allocDevice(uint32, b.len)
  let oDev = engine.allocDevice(uint32, 2)
  engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
  engine.copyToDeviceFromHost(bDev, addr b[0], b.len)

  engine.launchKernel("addScalarKernel", oDev, (aDev, bDev, 5'u32))

  var out32: array[2, uint32]
  engine.copyToHostFromDevice(addr out32[0], oDev, 2)
  doAssert out32[0] == 16, "slot 0 got " & $out32[0] & ", want 10 + 1 + 5 = 16"
  doAssert out32[1] == 27, "slot 1 got " & $out32[1] & ", want 20 + 2 + 5 = 27"
  echo "  address route: [", out32[0], ", ", out32[1], "] exact readback"

when isMainModule:
  runTest()
  echo "t_vulkan_route_address: all checks passed"
