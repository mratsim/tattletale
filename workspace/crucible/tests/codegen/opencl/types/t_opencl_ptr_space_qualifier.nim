## OpenCL pointer qualifiers derive from the pointer type's address space.
##
## Test that a clone parameter specialized to the thread space dereferences
## an unqualified pointer, and that a pointer into `{.smem.}` scratch
## dereferences shared memory through `__local`.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/types/t_opencl_ptr_space_qualifier.nim

import std/strutils
import workspace/crucible

const kernelCode = opencl:
  proc read(p: ptr float32): float32 =
    p[]

  proc cloneKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    var tmp: float32 = 1
    C[0] = uint32(read(addr tmp))

  proc smemCastKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.smem.}: array[8, uint32]
    let q = cast[ptr UncheckedArray[uint32]](addr scratch[0])
    let tid = thread_index_in_threadgroup
    q[tid] = uint32(tid) * 3'u32
    C[tid] = scratch[tid] + 1

proc runTest() =
  var engine = newEngine(runOpenCL)
  engine.ingest(kernelCode)

  var cloneRes: array[1, uint32]
  let cDev = engine.allocDevice(uint32, 1)
  engine.launchKernel<<(grid: (1, 1), blk: (1, 1))>>("cloneKernel", cDev, ())
  engine.copyToHostFromDevice(addr cloneRes[0], cDev, 1)
  doAssert cloneRes[0] == 1,
    "thread-space clone read got " & $cloneRes[0] & ", expected 1"

  var smemRes: array[8, uint32]
  let sDev = engine.allocDevice(uint32, 8)
  engine.launchKernel<<(grid: (1, 1), blk: (8, 1))>>("smemCastKernel", sDev, ())
  engine.copyToHostFromDevice(addr smemRes[0], sDev, 8)
  for i in 0 ..< 8:
    doAssert smemRes[i] == uint32(i) * 3'u32 + 1,
      "slot " & $i & ": got " & $smemRes[i] & ", expected " &
      $(uint32(i) * 3'u32 + 1) & " (write through __local pointer)"

  echo "  OK - OpenCL pointer qualifiers follow the address space and compute"

when isMainModule:
  runTest()
