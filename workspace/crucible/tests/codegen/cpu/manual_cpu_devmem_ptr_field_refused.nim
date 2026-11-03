## launchCpuFn refuses a pointer argument whose pointee declares a DeviceMem field.
##
## Test that a compile-time refusal fires when a pointer parameter points
## at an aggregate with a DeviceMem field:
##
## - a DeviceMem check on the argument type alone misses that shape
## - the kernel reads the device handle through the pointer as a host
##   integer
## - the refusal names the parameter, the pointee type and the field
##
## A value aggregate holding DeviceMem never reaches the scan, the copy into
## the kernel fails on DeviceMem's move-only `=copy` operator.
##
## Run (the compile must fail, both thread modes):
##   nim c --hints:off --warnings:off --threads:off \
##     --path:. --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_devmem_ptr_field_refused.nim
##   nim c --hints:off --warnings:off --threads:on \
##     --path:. --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cpu/manual_cpu_devmem_ptr_field_refused.nim

import workspace/crucible

type
  FakeBuffer = object
    raw: int64
  DataSet = object
    buf: DeviceMem[FakeBuffer, int32]
    n: int32

proc devmemKernel(ds: ptr DataSet, outBuf: ptr UncheckedArray[int32]) =
  outBuf[0] = 1

proc runTest() =
  var engine = newEngine(runCpu)
  defer: engine.destroyCpuEngine()
  var buf: array[4, int32]
  engine.launchCpuFn devmemKernel(cast[ptr DataSet](nil),
                                  cast[ptr UncheckedArray[int32]](addr buf[0])) do:
    grid = (1, 1, 1)
    blk = (1, 1, 1)

when isMainModule:
  runTest()
