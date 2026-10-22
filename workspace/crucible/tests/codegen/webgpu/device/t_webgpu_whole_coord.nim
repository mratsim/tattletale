## WebGPU whole-value coordinates
##
## Test that `let p = thread_position_in_grid` constructs the declared signed
## tuple type from the u32 builtin components.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/t_webgpu_whole_coord.nim

import std/[unittest]
import workspace/crucible

const kernelWhole = webgpu:
  proc whole(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    let p = thread_position_in_grid
    let flat = uint32(p.y * 4 + p.x)
    C[flat] = flat

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - whole-value coordinate":

    test "let p = thread_position_in_grid keeps the i32 domain":
      var engine = bkWebGPU.newEngine()
      engine.ingest(kernelWhole)
      var res: array[8, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel<<(grid: (1, 1), blk: (4, 2))>>("whole", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      for y in 0 ..< 2:
        for x in 0 ..< 4:
          check res[y * 4 + x] == uint32(y * 4 + x)

when isMainModule:
  runTest()
