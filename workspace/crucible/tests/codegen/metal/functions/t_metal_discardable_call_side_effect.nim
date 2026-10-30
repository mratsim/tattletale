## Metal: a device call kept for its side effects when the result is dropped.
##
## Test that a statement-position call to a discardable device proc
## still runs: the buffer write through the pointer argument reaches the output.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/functions/t_metal_discardable_call_side_effect.nim

import std/unittest
import workspace/crucible

func writeAndReturn(p: ptr UncheckedArray[uint32], v: uint32): uint32 {.device, discardable.} =
  # only the return value is dropped, buffer writes through the pointer stay
  p[0] = v
  result = v + 1'u32

const sideEffectMsl = metal:
  proc sideEffectKernel(o: ptr UncheckedArray[uint32]) {.global.} =
    writeAndReturn(o, 7'u32)

proc runTest() =
  suite "Metal - dropped result of a side-effecting device call":
    test "the call keeps running when nothing reads its result":
      var engine = newEngine(runMetal)
      engine.ingest(sideEffectMsl)
      let oDev = engine.allocDevice(uint32, 1)
      # Sentinel: nonzero init proves the write came from the kernel call
      let sentinel: uint32 = 41
      engine.copyToDeviceFromHost(oDev, addr sentinel, 1)
      engine.launchKernel("sideEffectKernel", oDev, ())
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 7'u32

when isMainModule:
  runTest()
