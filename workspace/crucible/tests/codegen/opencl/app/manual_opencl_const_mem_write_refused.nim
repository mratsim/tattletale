## OpenCL: an assignment into `{.const_mem.}` storage refuses at compile time.
##
## Test that an assignment to a `{.const_mem.}` local in a kernel body
## dies in the constant-write validation before any shader is emitted, naming the kernel and the target:
##
##   Error: pass validateConstantWrites: at kern: assignment into constant storage: the target c lives in read-only program-scope memory.
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/app/manual_opencl_const_mem_write_refused.nim

import workspace/crucible

const constWriteOpencl = opencl:
  proc kern(output: ptr UncheckedArray[float32]) {.global.} =
    var c {.const_mem.}: float32 = 1.0'f32
    c = 2.0'f32
    output[0] = c

proc runTest() =
  var engine = newEngine(runOpenCL)
  engine.ingest(constWriteOpencl)
  echo "UNREACHABLE — the constant-storage write must be refused at compile time"

when isMainModule:
  runTest()
