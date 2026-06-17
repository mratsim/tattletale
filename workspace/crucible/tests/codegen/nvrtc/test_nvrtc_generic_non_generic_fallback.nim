## NVRTC: generic → non-generic fallback in initGpuGenericInst
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_nvrtc_generic_non_generic_fallback.nim
##
## Exercises the early return guard at nim_to_gpu.nim ~line 118:
##   if t.typeKind notin {ntyGenericInst}:
##     if t.typeKind != ntyNone:
##       return ctx.nimToGpuType(t.getTypeInst())
##
## This path is defensive fallback when a type expected to be generic
## resolves to a non-generic type. Uses a generic object with a
## static-int-sized array (triggers non-trivial type resolution).
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Sized[N: static int] = object
    data: array[N, uint32]

const kernelCode = cuda:
  proc sizedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = Sized[4](data: [1'u32, 2'u32, 3'u32, 4'u32])
    output[0] = x.data[0]
    output[1] = x.data[3]

var buf: array[2, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"

nv.execute("sizedKernel", buf, ())
doAssert buf[0] == 1, &"Sized[4].data[0]: got {buf[0]}, expected 1"
doAssert buf[1] == 4, &"Sized[4].data[3]: got {buf[1]}, expected 4"

echo "  OK"
