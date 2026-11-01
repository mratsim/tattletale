## HLSL: a buffer reaching a storing callee from a condition position classifies read-write.
##
## Test that a buffer reaching a storing callee from a condition position
## classifies read-write:
##
## - the emitted shader binds the buffer as `RWStructuredBuffer`
## - the module compiles clean through DXC and spirv-val
## - a read-only classification refuses the store inside the callee
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/t_hlsl_cond_call_rw_classify.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const condCallHlsl = hlsl:
  proc storeUntil(buf: ptr UncheckedArray[uint32]): uint32 {.device.} =
    buf[0] = buf[0] + 1'u32
    result = buf[0]

  proc condCallKernel(output: ptr UncheckedArray[uint32], a: ptr UncheckedArray[uint32]) {.global.} =
    var i: uint32 = 0'u32
    while storeUntil(a) < 3'u32:
      i = i + 1'u32
    output[0] = i
    output[1] = if storeUntil(a) < 10'u32: 11'u32 else: 22'u32

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order: `output` and `a` are both written.
  let bindings: seq[BindingExpectation] = @[("output", 0, false), ("a", 1, false)]
  discard compileCheck("cond_call_rw_classify", condCallHlsl, bindings)

when isMainModule:
  runTest()
