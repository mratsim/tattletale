## Vulkan: one kernel mixing several scalars and two buffer pointers.
##
## A multi-arg kernel packs every parameter through the shared
## push-constant block. Pointers serialize as 8-byte device addresses,
## scalars as 4-byte values, read back at the offsets the block declares.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_multiarg_pushconst.nim

import std/strformat
import workspace/crucible

const code = glsl:
  proc mix(outp: ptr UncheckedArray[uint32],
           a: uint32, b: uint32, c: uint32,
           xf: ptr UncheckedArray[float32], yf: ptr UncheckedArray[float32]) {.global.} =
    outp[0] = a + b * c + uint32(xf[0]) + uint32(yf[0])

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo "=== Vulkan: multi-arg kernel, scalars and two pointers ===\n"

  var engine = newEngine(runVulkan)
  engine.ingest(code)
  let oDev = engine.allocDevice(uint32, 1)
  let xDev = engine.allocDevice(float32, 1)
  let yDev = engine.allocDevice(float32, 1)
  var xHost: array[1, float32] = [7.0'f32]
  var yHost: array[1, float32] = [20.0'f32]
  engine.copyToDeviceFromHost(xDev, addr xHost[0], 1)
  engine.copyToDeviceFromHost(yDev, addr yHost[0], 1)

  # 3 + 4*5 + 7 + 20 = 50, every arg in declaration order through the block
  engine.launchKernel("mix", oDev, (3'u32, 4'u32, 5'u32, xDev, yDev))
  var got: array[1, uint32]
  engine.copyToHostFromDevice(addr got[0], oDev, 1)
  doAssert got[0] == 50'u32, &"mix: got {got[0]}, expected 50"
  echo "  OK — mix(3, 4, 5, x, y) = ", got[0]

when isMainModule:
  runTest()
