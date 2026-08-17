## Metal: output readback straight from the shared-storage buffer's `contents()`
## after `waitUntilCompleted`. The Metal engine has no staging buffer
## and no map path. The output buffer uses `ResourceStorageModeShared | HazardTrackingModeUntracked`, so the CPU
## and GPU share one allocation and the readback is a plain memcpy
## from `contents()`. This test is the rework of the webgpu `map_readback`
## test, which exercised the staging/map path this backend intentionally deletes.
## The kernel is the same elementwise double. Only the readback mechanism differs.
##
## Tested ABI (macOS 26.6.1, 2026-08-17): libobjc, Metal.framework, CLT SDK 26.5,
## Nim 2.2.10, MSL 4.0.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_readback.nim

import std/unittest
import workspace/crucible

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
      var engine = bkMetal.init()
      engine.ingest(kernelCode)
      echo kernelCode
      var input = [1'u32, 2'u32, 3'u32, 4'u32]
      var res: array[4, uint32]
      engine.run("mapReadbackKernel", res, (input,))
      check res[0] == 2
      check res[1] == 4
      check res[2] == 6
      check res[3] == 8

when isMainModule:
  runTest()
