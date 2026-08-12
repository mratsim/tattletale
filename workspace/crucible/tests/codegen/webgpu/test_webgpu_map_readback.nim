## Test: WebGPU output readback through the staging-buffer map path
##
## Exercises runImpl's section 9-10 (runtime/engines/wgpu.nim): the output
## buffer is copied to a MapRead staging buffer, wgpuBufferMapAsync fires the
## bufferMapCb callback into a heap MapDoneData, and the mapped bytes are
## copied back into the host output array. A multi-element uint32 output
## covers the full MapDoneData round trip (done, status, resultBytes).
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/webgpu/test_webgpu_map_readback.nim

import std/[unittest, strformat]
import workspace/crucible

# ── Kernel: elementwise double of a 4-element uint32 buffer ──────────────────

const kernelCode = webgpu:
  proc mapReadbackKernel(output: ptr UncheckedArray[uint32];
                         input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 2'u32
    output[1] = input[1] * 2'u32
    output[2] = input[2] * 2'u32
    output[3] = input[3] * 2'u32

# ── Host side ────────────────────────────────────────────────────────────────

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "WebGPU - staging-buffer map readback":

    test "multi-element uint32 output readback":
      var engine = bkWGSL.init()
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
