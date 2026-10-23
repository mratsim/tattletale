## HLSL: a buffer stored through `{.emit.}` binds read-write.
##
## The write-scan reads the emit's expression parts and scans the verbatim
## literal runs for the buffer name. The store classifies the buffer
## read-write, the declaration is RWStructuredBuffer.
##
## A StructuredBuffer declaration fails the DXC DXIL leg:
## no store through a read-only resource.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_emit_write.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const emitWriteHlsl = hlsl:
  proc emitWriteKernel(output: ptr UncheckedArray[uint32], z: uint32) {.global.} =
    {.emit: "output[0] = 42;".}
    {.emit: ["output[1] = ", z, " + 1;"].}

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  let bindings: seq[BindingExpectation] =
    @[("output", 0, false)]
  discard compileCheck("emit_write", emitWriteHlsl, bindings)

when isMainModule:
  runTest()
