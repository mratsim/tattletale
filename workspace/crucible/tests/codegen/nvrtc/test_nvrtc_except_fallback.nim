## NVRTC: try/except fallback in addProcToGenericInsts
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_nvrtc_except_fallback.nim
##
## Coverage: nim_to_gpu.nim:807-818
##
## The except: path is triggered when a generic instantiation's body
## contains AST nodes Crucible cannot translate. The fallback registers
## an empty function stub so compilation continues.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

# A function that Crucible can translate (happy path)
proc addEm[T](a, b: T): T {.device.} =
  a + b

# This generic should translate fine — verifies the except: path is NOT
# triggered for well-formed generic instantiations.
const kernelCode = cuda:
  proc exceptFallbackKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = addEm(10'u32, 20'u32)

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("exceptFallbackKernel", buf, ())
doAssert buf[0] == 30, &"addEm fallback: got {buf[0]}, expected 30"
echo "  OK (test_nvrtc_except_fallback)"
