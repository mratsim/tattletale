## Metal: compound-assign `+=` / `*=` executes on the device, including side-effecting-index and block-expression lvalues
##
## Test that the value accumulates in the stored element and the LHS index expression is evaluated exactly once
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/control_flow/t_metal_compound_assign.nim

import std/unittest
import workspace/crucible

const rmwMsl = metal:
  proc rmwKernel(C: ptr UncheckedArray[int32]) {.global.} =
    for i in 0 ..< 4:
      C[i] += 5
      C[i] *= 2

const hazardMsl = metal:
  proc tick(x: ptr UncheckedArray[int32]): int32 {.device.} =
    result = x[0]
    x[0] = result + 1

  proc hazardKernel(x: ptr UncheckedArray[int32]) {.global.} =
    x[tick(x) + 1] += 7

const blitMsl = metal:
  proc blitKernel(x: ptr UncheckedArray[int32]) {.global.} =
    for i in 0 ..< 2:
      x[block:
          let j = i * 2
          j] += 3

proc runTest() =   # private, tests run in a proc so engines are destroyed at return
  suite "Metal - compound assignment (+=, *=) with C-like LHS semantics":
    test "read-modify-write accumulates over the stored element":
      # Host-initialized values are copied H->D before launch, so the kernel's
      # `+=` must READ the stored value and write the accumulation back:
      # otherwise the result would not depend on the initialization at all.
      var output: array[4, int32] = [10'i32, 11, 12, 13]
      var engine = newEngine(runMetal)
      engine.ingest(rmwMsl)
      let oDev = engine.allocDevice(int32, output.len)
      engine.copyToDeviceFromHost(oDev, addr output[0], output.len)
      engine.launchKernel("rmwKernel", oDev, ())
      engine.copyToHostFromDevice(addr output[0], oDev, output.len)
      check output[0] == 30  # (10 + 5) * 2
      check output[1] == 32  # (11 + 5) * 2
      check output[2] == 34  # (12 + 5) * 2
      check output[3] == 36  # (13 + 5) * 2

    test "x[tick(x) + 1] += 7 evaluates the index expression exactly once":
      var x: array[4, int32] = [0'i32, 0, 0, 0]
      var engine = newEngine(runMetal)
      engine.ingest(hazardMsl)
      let oDev = engine.allocDevice(int32, x.len)
      engine.copyToDeviceFromHost(oDev, addr x[0], x.len)
      engine.launchKernel("hazardKernel", oDev, ())
      engine.copyToHostFromDevice(addr x[0], oDev, x.len)
      check x[0] == 1  # the index side effect fired exactly once
      check x[1] == 7  # stored back into the element the single index selected
      check x[2] == 0  # a second index evaluation would have stored here

    test "x[block-expr] += 3 accumulates through the blitted lvalue":
      var x: array[4, int32] = [0'i32, 0, 0, 0]
      var engine = newEngine(runMetal)
      engine.ingest(blitMsl)
      let oDev = engine.allocDevice(int32, x.len)
      engine.copyToDeviceFromHost(oDev, addr x[0], x.len)
      engine.launchKernel("blitKernel", oDev, ())
      engine.copyToHostFromDevice(addr x[0], oDev, x.len)
      check x[0] == 3  # i = 0: x[0] += 3
      check x[1] == 0
      check x[2] == 3  # i = 1: x[2] += 3
      check x[3] == 0

when isMainModule:
  runTest()
