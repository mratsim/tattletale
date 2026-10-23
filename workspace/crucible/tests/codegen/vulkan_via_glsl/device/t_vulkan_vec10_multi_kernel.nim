## Vulkan multi-kernel Vec10
##
## Test that one ingest compiles both kernels and both launch on the same engine against the shared push-constant block
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/device/t_vulkan_vec10_multi_kernel.nim

import workspace/crucible

const code = glsl:
  proc vec10_add(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] + b[i]
  proc vec10_mul(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] * b[i]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  var engine = newEngine(runVulkan)
  engine.ingest(code)
  let aDev = engine.allocDevice(uint32, a.len)
  let bDev = engine.allocDevice(uint32, b.len)
  let oDev = engine.allocDevice(uint32, 10)
  engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
  engine.copyToDeviceFromHost(bDev, addr b[0], b.len)

  engine.launchKernel("vec10_add", oDev, (aDev, bDev))
  var res: array[10, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 10)
  for i in 0 ..< 10:
    doAssert res[i] == a[i] + b[i]
  echo "  OK — vec10_add"

  engine.launchKernel("vec10_mul", oDev, (aDev, bDev))
  engine.copyToHostFromDevice(addr res[0], oDev, 10)
  for i in 0 ..< 10:
    doAssert res[i] == a[i] * b[i]
  echo "  OK — vec10_mul"

  echo "  OK — Multi-kernel vec10 (Vulkan), one engine, one ingest"

when isMainModule:
  runTest()
