## OpenCL: a non-lvalue argument to a passByRef parameter materializes a temp.
##
## Test that two non-lvalue argument shapes pass to a large-struct by-ref parameter:
##
## - an inline constructor in the call argument position
## - the return value of a device function
##
## The threshold is 24 bytes: the lowering copies each temporary into
## a fresh local, passes the local's address, and the kernel reads back the values.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/serialization/t_opencl_passbyref_nonlvalue.nim

import workspace/crucible

type
  LargeStruct = object
    data: array[8, uint32]  # 32 bytes, over the 24-byte by-ref threshold

# Inline constructor argument, not an lvalue
const codeCtor = opencl:
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0] + s.data[1]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    let v = takeLarge(LargeStruct(data: [10'u32, 20, 30, 40, 50, 60, 70, 80]))
    output[0] = v

# Device-function return value argument, not an lvalue
const codeCall = opencl:
  proc makeLarge(val: uint32): LargeStruct {.device.} =
    result.data[0] = val
    result.data[1] = val * 2'u32
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0] + s.data[1]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    let v = takeLarge(makeLarge(42'u32))
    output[0] = v

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block:  # inline constructor arg: 10 + 20
    var engine = newEngine(runOpenCL)
    engine.ingest(codeCtor)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("kernelMain", oDev, ())
    var out32: array[1, uint32]
    engine.copyToHostFromDevice(addr out32[0], oDev, 1)
    doAssert out32[0] == 30'u32,
      "inline constructor arg to a by-ref param: got " & $out32[0] & ", want 30"

  block:  # function-call return arg: 42 + 84
    var engine = newEngine(runOpenCL)
    engine.ingest(codeCall)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("kernelMain", oDev, ())
    var out32: array[1, uint32]
    engine.copyToHostFromDevice(addr out32[0], oDev, 1)
    doAssert out32[0] == 126'u32,
      "call-result arg to a by-ref param: got " & $out32[0] & ", want 126"

  echo "[OK] non-lvalue arguments to passByRef params materialize temps and read back"

when isMainModule:
  runTest()
