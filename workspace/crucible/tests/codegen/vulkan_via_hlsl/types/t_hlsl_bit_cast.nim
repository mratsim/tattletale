## HLSL: the 32-bit bit-reinterpret casts print as the asint/asuint/asfloat reinterprets.
##
## Test that `cast[uint32](float32)` and `cast[float32](uint32)` emit the HLSL
## reinterprets and compile clean through DXC in the DXIL and SPIR-V stages.
## HLSL has no other bit-cast form, a non-32-bit destination refuses at the printer.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/types/t_hlsl_bit_cast.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const bitsToUintHlsl = hlsl:
  proc bitsToUintKernel(output, inBuf: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = cast[uint32](cast[float32](inBuf[0])) + inBuf[1]

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order: `output` is written, `inBuf` is read-only
  let bindings: seq[BindingExpectation] =
    @[("output", 0, false), ("inBuf", 1, true)]
  discard compileCheck("bit_cast", bitsToUintHlsl, bindings)

when isMainModule:
  runTest()
