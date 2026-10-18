## Metal: large structs passed to device functions by value. The struct is
## 32 bytes, above the passByRef threshold, so the printer emits the device-fn
## param as `thread const T&`. Two call-arg shapes run on the device.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_large_struct.nim

import std/unittest
import workspace/crucible

# Test 1: inline constructor arg to device function
const code1 = metal:
  type LargeStruct = object
    data: array[8, uint32]  # 32 bytes > 24 threshold
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0] + s.data[1]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = takeLarge(LargeStruct(data: [10'u32, 20, 30, 40, 50, 60, 70, 80]))

# Test 2: function return value arg
const code2 = metal:
  type LargeStruct = object
    data: array[8, uint32]
  proc makeLarge(val: uint32): LargeStruct {.device.} =
    result.data[0] = val
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = takeLarge(makeLarge(42'u32))

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  echo code1

  suite "Metal - large struct by value":
    test "inline constructor arg":
      var engine = bkMetal.init()
      engine.ingest(code1)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("kernelMain", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      engine.freeDevice(oDev)
      check res[0] == 30

    test "function return arg":
      var engine = bkMetal.init()
      engine.ingest(code2)
      echo code2
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("kernelMain", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      engine.freeDevice(oDev)
      check res[0] == 42

when isMainModule:
  runTest()
