## compound-assign `a[i] += v` read-modify-write store-back (NVRTC roundtrip)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/control_flow/t_nvrtc_compound_assign.nim
import std/[unittest]
import workspace/crucible

const kernel = cuda:
  proc compoundKernel(C: ptr UncheckedArray[int32]) {.global.} =
    for i in 0 ..< 4:
      C[i] += 5
      C[i] *= 2

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "compound assignment (a[i] += v)":
    test "accumulates and stores back (read-modify-write)":
      # Host-initialized values are copied H->D before launch: the kernel's `+=`
      # must READ the stored value and write the accumulation back, otherwise
      # the result would not depend on the initialization at all.
      var output: array[4, int32] = [10'i32, 11, 12, 13]
      var engine = bkCuda.newEngine()
      engine.ingest(kernel)
      let oDev = engine.allocDevice(int32, output.len)
      engine.copyToDeviceFromHost(oDev, addr output[0], output.len)
      engine.launchKernel<<(1, 1)>>("compoundKernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, output.len)
      check output[0] == 30  # (10 + 5) * 2
      check output[1] == 32  # (11 + 5) * 2
      check output[2] == 34  # (12 + 5) * 2
      check output[3] == 36  # (13 + 5) * 2

when isMainModule:
  runTest()
