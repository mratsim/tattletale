## WebGPU: statement-position calls that do not inline terminate.
##
## Test that a kernel with a barrier statement and a void call statement compiles and executes, the inline pass advances.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/functions/t_webgpu_stmt_call_terminates.nim

import std/unittest
import workspace/crucible

const kernelTerminates = webgpu:
  proc bump(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 1).} =
    C[thread_index_in_threadgroup] = 1'u32
    syncthreads()
    C[4 + thread_index_in_threadgroup] = 2'u32

proc runTest() =
  suite "WebGPU - statement calls terminate":
    test "barrier statement compiles and executes":
      var engine = newEngine(runWebGPU)
      engine.ingest(kernelTerminates)
      var res: array[8, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel<<(grid: (1, 1, 1), blk: (4, 1))>>("bump", oDev, ())
      engine.copyToHostFromDevice(res, oDev)
      for i in 0 ..< 4:
        check res[i] == 1'u32
      for i in 4 ..< 8:
        check res[i] == 2'u32

when isMainModule:
  runTest()
