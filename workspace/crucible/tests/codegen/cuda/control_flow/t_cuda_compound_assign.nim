## CUDA: compound-assign `+=` / `*=` executes on the device, including side-effecting-index and var-return-call lvalues (NVRTC roundtrip)
##
## Test that the value accumulates in the stored element and the LHS expression is evaluated exactly once
##
## Run (requires a CUDA GPU):
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/control_flow/t_cuda_compound_assign.nim

import std/[unittest]
import workspace/crucible

const kernel = cuda:
  proc compoundKernel(C: ptr UncheckedArray[int32]) {.global.} =
    for i in 0 ..< 4:
      C[i] += 5
      C[i] *= 2

const hazardKernel = cuda:
  proc tick(x: ptr UncheckedArray[int32]): int32 {.device.} =
    result = x[0]
    x[0] = result + 1

  proc hazardKernel(x: ptr UncheckedArray[int32]) {.global.} =
    x[tick(x) + 1] += 7

const callLhsKernel = cuda:
  proc at(x: ptr UncheckedArray[int32], i: int32): var int32 {.device.} =
    return x[i]

  proc callLhsKernel(x: ptr UncheckedArray[int32]) {.global.} =
    for i in 0 ..< 2:
      x.at(i.int32) = 7
      x.at(i.int32) += 5

proc runTest() =   # private, engines are destroyed at proc return
  suite "compound assignment (a[i] += v) with C-like LHS semantics":
    test "accumulates and stores back (read-modify-write)":
      # Host-initialized values are copied H->D before launch: the kernel's `+=`
      # must READ the stored value and write the accumulation back, otherwise the result would not depend on the initialization at all.
      var output: array[4, int32] = [10'i32, 11, 12, 13]
      var engine = newEngine(runCuda)
      engine.ingest(kernel)
      let oDev = engine.allocDevice(int32, output.len)
      engine.copyToDeviceFromHost(oDev, addr output[0], output.len)
      engine.launchKernel<<(1, 1)>>("compoundKernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, output.len)
      check output[0] == 30  # (10 + 5) * 2
      check output[1] == 32  # (11 + 5) * 2
      check output[2] == 34  # (12 + 5) * 2
      check output[3] == 36  # (13 + 5) * 2

    test "x[tick(x) + 1] += 7 evaluates the index expression exactly once":
      var x: array[4, int32] = [0'i32, 0, 0, 0]
      var engine = newEngine(runCuda)
      engine.ingest(hazardKernel)
      let oDev = engine.allocDevice(int32, x.len)
      engine.copyToDeviceFromHost(oDev, addr x[0], x.len)
      engine.launchKernel<<(1, 1)>>("hazardKernel", oDev, ())
      engine.copyToHostFromDevice(addr x[0], oDev, x.len)
      check x[0] == 1  # the index side effect fired exactly once
      check x[1] == 7  # stored back into the element the single index selected
      check x[2] == 0  # a second index evaluation would have stored here

    test "x.at(i.int32) = 7; x.at(i.int32) += 5 goes through the var-return call lvalue":
      var x: array[4, int32] = [0'i32, 0, 0, 0]
      var engine = newEngine(runCuda)
      engine.ingest(callLhsKernel)
      let oDev = engine.allocDevice(int32, x.len)
      engine.copyToDeviceFromHost(oDev, addr x[0], x.len)
      engine.launchKernel<<(1, 1)>>("callLhsKernel", oDev, ())
      engine.copyToHostFromDevice(addr x[0], oDev, x.len)
      check x[0] == 12  # 7, then += 5 through the returned reference
      check x[1] == 12
      check x[2] == 0
      check x[3] == 0

when isMainModule:
  runTest()
