## Metal: scalar by-value marshalling through the engine's packed constant buffer.
## The printer emits each scalar kernel param as the MSL constant-ref form
## `constant T& name [[buffer(n)]]`. The engine packs the 4-byte host values
## into one shared constant buffer at 16-byte slots, bound per index.
## A Nim `bool` arg marshals as a 4-byte i32 on the host, and the printer
## declares it `int` in MSL, matching every backend's buffer width.
## Every kernel runs through `engine.run()` with byte-exact asserts.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_scalar_marshalling.nim

import std/unittest
import workspace/crucible

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
      var engine = bkMetal.init()
      engine.ingest(kernelI32)
      echo kernelI32
      var res: array[1, uint32]
      engine.run("scalarI32Kernel", res, (-42'i32,))
      check res[0] == uint32(-42'i32)

    test "uint32 by value (4-byte blob)":
      var engine = bkMetal.init()
      engine.ingest(kernelU32)
      echo kernelU32
      var res: array[1, uint32]
      engine.run("scalarU32Kernel", res, (7'u32,))
      check res[0] == 7

    test "float32 by value (4-byte blob)":
      var engine = bkMetal.init()
      engine.ingest(kernelF32)
      echo kernelF32
      var res: array[1, float32]
      engine.run("scalarF32Kernel", res, (1.5'f32,))
      check res[0] == 1.5'f32

    test "bool by value (widened to i32 on the host)":
      var engine = bkMetal.init()
      engine.ingest(kernelBool)
      echo kernelBool
      var res: array[1, uint32]
      engine.run("scalarBoolKernel", res, (true,))
      check res[0] == 1
      engine.run("scalarBoolKernel", res, (false,))
      check res[0] == 0

when isMainModule:
  runTest()
