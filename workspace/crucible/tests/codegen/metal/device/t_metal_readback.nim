## Metal: output readback straight from the shared-storage buffer, the CPU and GPU
## share one allocation. The readback is a plain memcpy from `contents()` after `waitUntilCompleted`.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_readback.nim


import workspace/crucible

import std/unittest

# ── Kernel: elementwise double of a 4-element uint32 buffer ──────────────────

const kernelCode = metal:
  proc mapReadbackKernel(output: ptr UncheckedArray[uint32];
                         input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 2'u32
    output[1] = input[1] * 2'u32
    output[2] = input[2] * 2'u32
    output[3] = input[3] * 2'u32

# ── Host side ────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - direct contents() readback":

    test "multi-element uint32 output readback":
      var engine = newEngine(runMetal)
      engine.ingest(kernelCode)
      echo kernelCode
      var input = [1'u32, 2'u32, 3'u32, 4'u32]
      let iDev = engine.allocDevice(uint32, input.len)
      engine.copyToDeviceFromHost(iDev, addr input[0], input.len)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel("mapReadbackKernel", oDev, (iDev,))
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 2
      check res[1] == 4
      check res[2] == 6
      check res[3] == 8

when isMainModule:
  runTest()
