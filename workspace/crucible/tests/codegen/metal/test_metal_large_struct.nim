## Metal: large structs passed to device functions by value. The struct
## is 32 bytes, above the passByRef threshold, so the printer emits the device-fn param
## as `thread const T&`. The gpuMaterialize pass wraps the call-site value.
## Two shapes execute on the device:
##   - an inline constructor as the call arg (code1)
##   - a device fn return value as the call arg (code2)
## Both kernels write a verifiable sum and run through `engine.run()`,
## confirming the by-value struct path is safe on AGX.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_large_struct.nim

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
      var res: array[1, uint32]
      engine.run("kernelMain", res, ())
      check res[0] == 30

    test "function return arg":
      var engine = bkMetal.init()
      engine.ingest(code2)
      echo code2
      var res: array[1, uint32]
      engine.run("kernelMain", res, ())
      check res[0] == 42

when isMainModule:
  runTest()
