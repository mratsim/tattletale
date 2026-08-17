## Metal: bool buffer marshalling through the engine.
## Bool buffer elements marshal at their Nim width (1 byte/element).
## The printer declares buffer bool params `device bool*` / `device const bool*`
## (MSL bool is 1 byte in the device address space), matching the host blobs.
## Scalar bools stay 4-byte i32 on the host and are covered by
## test_metal_scalar_marshalling.nim.
## Reads and writes round-trip on-device with byte-exact asserts.
## A seq[bool] input exercises the same 1-byte blob path as array[bool].
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_bool_buffer.nim

import std/unittest
import workspace/crucible

# ── Kernels ─────────────────────────────────────────────────────────────────

const kernelBoolRead = metal:
  proc boolReadKernel(output: ptr UncheckedArray[uint32]; flags: ptr UncheckedArray[bool]) {.global.} =
    let i = thread_position_in_grid.x
    if flags[i]:
      output[i] = 1'u32
    else:
      output[i] = 0'u32

const kernelBoolWrite = metal:
  proc boolWriteKernel(output: ptr UncheckedArray[bool]; vals: ptr UncheckedArray[uint32]) {.global.} =
    let i = thread_position_in_grid.x
    output[i] = vals[i] != 0'u32

# ── Host side ───────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - bool buffer marshalling":

    test "array[bool] input reads 1-byte elements":
      var engine = bkMetal.init()
      engine.ingest(kernelBoolRead)
      let flags = [true, false, true, false]
      var res: array[4, uint32]
      engine.run<<(grid: (4, 1))>>("boolReadKernel", res, (flags,))
      check res == [1'u32, 0'u32, 1'u32, 0'u32]

    test "seq[bool] input reads 1-byte elements":
      var engine = bkMetal.init()
      engine.ingest(kernelBoolRead)
      let flags = @[true, false, true, false]
      var res: array[4, uint32]
      engine.run<<(grid: (4, 1))>>("boolReadKernel", res, (flags,))
      check res == [1'u32, 0'u32, 1'u32, 0'u32]

    test "array[bool] output writes 1-byte elements":
      var engine = bkMetal.init()
      engine.ingest(kernelBoolWrite)
      let vals = [1'u32, 0'u32, 1'u32, 0'u32]
      var outBools: array[4, bool]
      engine.run<<(grid: (4, 1))>>("boolWriteKernel", outBools, (vals,))
      check outBools == [true, false, true, false]

when isMainModule:
  runTest()
