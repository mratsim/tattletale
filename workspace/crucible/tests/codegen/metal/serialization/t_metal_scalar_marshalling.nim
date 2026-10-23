## Metal: scalar by-value marshalling through the engine's packed constant buffer,
## each scalar kernel param emits as `constant T& name`, a Nim `bool` marshals as a 4-byte i32 `int` in MSL.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/t_metal_scalar_marshalling.nim


import workspace/crucible

import std/unittest

# ── Kernels: one scalar param each, echoed into the output ─────────────────

const kernelI32 = metal:
  proc scalarI32Kernel(output: ptr UncheckedArray[uint32]; x: int32) {.global.} =
    output[0] = uint32(x)

const kernelU32 = metal:
  proc scalarU32Kernel(output: ptr UncheckedArray[uint32]; x: uint32) {.global.} =
    output[0] = x

const kernelF32 = metal:
  proc scalarF32Kernel(output: ptr UncheckedArray[float32]; x: float32) {.global.} =
    output[0] = x

const kernelBool = metal:
  proc scalarBoolKernel(output: ptr UncheckedArray[uint32]; flag: bool) {.global.} =
    if flag:
      output[0] = 1'u32
    else:
      output[0] = 0'u32

# ── Host side ───────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - scalar by-value marshalling":

    test "int32 by value (4-byte blob)":
      var engine = newEngine(runMetal)
      engine.ingest(kernelI32)
      echo kernelI32
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("scalarI32Kernel", oDev, (-42'i32,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == uint32(-42'i32)

    test "uint32 by value (4-byte blob)":
      var engine = newEngine(runMetal)
      engine.ingest(kernelU32)
      echo kernelU32
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("scalarU32Kernel", oDev, (7'u32,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 7

    test "float32 by value (4-byte blob)":
      var engine = newEngine(runMetal)
      engine.ingest(kernelF32)
      echo kernelF32
      let oDev = engine.allocDevice(float32, 1)
      engine.launchKernel("scalarF32Kernel", oDev, (1.5'f32,))
      var res: array[1, float32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 1.5'f32

    test "bool by value (widened to i32 on the host)":
      var engine = newEngine(runMetal)
      engine.ingest(kernelBool)
      echo kernelBool
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("scalarBoolKernel", oDev, (true,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 1
      engine.launchKernel("scalarBoolKernel", oDev, (false,))
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 0

when isMainModule:
  runTest()
