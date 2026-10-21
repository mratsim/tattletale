## CUDA launches reject raw pointer args at compile time: the unified kernel
## API passes DeviceMem handles, never raw pointers.
##
## Run:
##   cd tattletale
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/cuda/device/manual_cuda_ptr_arg_rejected.nim

import workspace/crucible

const code = cuda:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = 1'u32

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  var engine = bkCuda.newEngine()
  engine.ingest(code)
  let oDev = engine.allocDevice(uint32, 1)
  # a raw pointer arg must fail to compile: it would bind host pointer bits
  # where the kernel expects device memory
  engine.launchKernel("k", oDev, (cast[ptr UncheckedArray[uint32]](0),))
  echo "UNREACHABLE — raw pointer args must be rejected at compile time"

when isMainModule:
  runTest()
