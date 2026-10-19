## Metal: multi-kernel source, one `metal:` block emits two kernels
## (vec10_add, vec10_mul). One ingest compiles the whole library, each run
## gets its own pipeline state from the engine's level-2 cache.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_vec10_multi_kernel.nim

import std/unittest
import workspace/crucible

const code = metal:
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

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo code

  var a: array[10, uint32] = [1'u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  var b: array[10, uint32] = [10'u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]

  suite "Metal - multi-kernel source":
    test "vec10_add and vec10_mul from one library":
      var engine = bkMetal.newEngine()
      engine.ingest(code)
      let aDev = engine.allocDevice(uint32, a.len)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      let bDev = engine.allocDevice(uint32, b.len)
      engine.copyToDeviceFromHost(bDev, addr b[0], b.len)
      let oDev = engine.allocDevice(uint32, 10)
      var res: array[10, uint32]
      engine.launchKernel("vec10_add", oDev, (aDev, bDev))
      engine.copyToHostFromDevice(addr res[0], oDev, 10)
      for i in 0 ..< 10:
        check res[i] == a[i] + b[i]
      engine.launchKernel("vec10_mul", oDev, (aDev, bDev))
      engine.copyToHostFromDevice(addr res[0], oDev, 10)
      for i in 0 ..< 10:
        check res[i] == a[i] * b[i]

when isMainModule:
  runTest()
