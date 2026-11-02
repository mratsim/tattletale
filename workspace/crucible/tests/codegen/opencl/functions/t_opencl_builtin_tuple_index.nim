## OpenCL: a tuple bracket access on a coordinate builtin reads that component.
##
## Test that a positional bracket access over the coordinate builtins compiles
## through the opencl: macro, launches, and returns the component the bracket
## names:
##
## - `blockIdx[1]` returns the workgroup id along y
## - `get_global_id(1)` / `get_global_id(2)` return the global id along y / z
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/functions/t_opencl_builtin_tuple_index.nim

import workspace/crucible

const tupleIndexCl = opencl:
  proc tupleIndexKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let gy = get_global_id(1)
    C[gy * 3 + 0] = uint32(blockIdx[1])
    C[gy * 3 + 1] = uint32(get_global_id(1))
    C[gy * 3 + 2] = uint32(get_global_id(2))

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # grid (1, 3) x blk (1, 2): six work-items, global y = 0..5, z = 0,
  # workgroup id y = global y div 2
  var engine = newEngine(runOpenCL)
  engine.ingest(tupleIndexCl)
  let oDev = engine.allocDevice(uint32, 18)
  engine.launchKernel<<(grid: (1, 3), blk: (1, 2))>>("tupleIndexKernel", oDev, ())
  var res: array[18, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 18)
  for gy in 0 ..< 6:
    doAssert res[gy * 3 + 0] == uint32(gy div 2),
      "blockIdx[1] at global y " & $gy & ": got " & $res[gy * 3 + 0]
    doAssert res[gy * 3 + 1] == uint32(gy),
      "get_global_id(1) at global y " & $gy & ": got " & $res[gy * 3 + 1]
    doAssert res[gy * 3 + 2] == 0'u32,
      "get_global_id(2) at global y " & $gy & ": got " & $res[gy * 3 + 2]
  echo "  OK — tuple bracket and native spellings read the right coordinate component"

when isMainModule:
  runTest()
