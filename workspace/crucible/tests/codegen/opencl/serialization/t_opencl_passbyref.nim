## OpenCL: passByRef call-site lvalue check for the `&` prefix.
##
## Test that a large struct argument (over 24 bytes) passes by reference:
## the call site takes an lvalue address and the kernel returns s.data[0]
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/serialization/t_opencl_passbyref.nim

import std/strutils
import workspace/crucible

type
  LargeStruct = object
    data: array[8, uint32]  # 32 bytes > 24

const passbyrefOpencl = opencl:
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    var s = LargeStruct(data: [1'u32, 2, 3, 4, 5, 6, 7, 8])
    let v = takeLarge(s)
    output[0] = v

proc runTest() =   # private: engines are destroyed at return
  # A single-use struct local folds into its only reader (the call argument).
  # Splicing leaves no named lvalue, so the pass-by-ref call site materializes
  # its own temp and takes its address under `&`
  doAssert passbyrefOpencl.contains("takeLarge((&"),
    "Expected the pass-by-ref call site to take an lvalue address:\n" & passbyrefOpencl

  var engine = newEngine(runOpenCL)
  engine.ingest(passbyrefOpencl)
  let oDev = engine.allocDevice(uint32, 1)
  engine.launchKernel("kernelMain", oDev, ())
  var out32: array[1, uint32]
  engine.copyToHostFromDevice(addr out32[0], oDev, 1)
  doAssert out32[0] == 1'u32,
    "kernelMain readback got " & $out32[0] & ", want 1 (s.data[0] through the by-ref call)"

when isMainModule:
  runTest()
  echo "t_opencl_passbyref: all checks passed"
