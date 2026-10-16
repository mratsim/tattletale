## Multi-kernel Vec10 add/mul on the Vulkan (GLSL/SPIR-V) backend.
## Two pointer-only kernels in one source, each launched on its own engine.
##
## Run:
##   cd tattletale && \
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_vec10_multi_kernel.nim

import workspace/crucible

const code = vulkan:
  proc vec10_add(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] + b[i]
  proc vec10_mul(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] * b[i]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  echo code

  var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]

  echo "=== Running vec10_add ===\n"
  block:
    var engine = bkVulkan.init()
    engine.ingest(code)
    let aDev = engine.allocDevice(a.len * sizeof(uint32))
    let bDev = engine.allocDevice(b.len * sizeof(uint32))
    let oDev = engine.allocDevice(10 * sizeof(uint32))
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len * sizeof(uint32))
    engine.launchKernel("vec10_add", oDev, (aDev, bDev))
    var res: array[10, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 10 * sizeof(uint32))
    engine.freeDevice(aDev)
    engine.freeDevice(bDev)
    engine.freeDevice(oDev)
    for i in 0 ..< 10:
      doAssert res[i] == a[i] + b[i]
    echo "  OK — vec10_add"

  echo "=== Running vec10_mul ===\n"
  block:
    var engine = bkVulkan.init()
    engine.ingest(code)
    let aDev = engine.allocDevice(a.len * sizeof(uint32))
    let bDev = engine.allocDevice(b.len * sizeof(uint32))
    let oDev = engine.allocDevice(10 * sizeof(uint32))
    engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
    engine.copyToDeviceFromHost(bDev, addr b[0], b.len * sizeof(uint32))
    engine.launchKernel("vec10_mul", oDev, (aDev, bDev))
    var res: array[10, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 10 * sizeof(uint32))
    engine.freeDevice(aDev)
    engine.freeDevice(bDev)
    engine.freeDevice(oDev)
    for i in 0 ..< 10:
      doAssert res[i] == a[i] * b[i]
    echo "  OK — vec10_mul"

  echo "\n  OK — Multi-kernel vec10 (Vulkan)"

when isMainModule:
  runTest()
