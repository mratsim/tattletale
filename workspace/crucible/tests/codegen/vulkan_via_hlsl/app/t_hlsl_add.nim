## HLSL: add kernels through the real pass bundle, compile check via DXC and spirv-val.
##
## Test that the `hlsl:` macro emits HLSL for the plain and push-scalar add kernels
## compiling clean through DXC and spirv-val with the binding reflection asserted.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_add.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const addHlsl = hlsl:
  proc addKernel(output, a, b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const addHlslZ = hlsl:
  proc addKernelZ(output, a, b: ptr UncheckedArray[uint32], z: uint32) {.global.} =
    output[0] = a[0] + b[0] + z
    output[1] = a[1] + b[1] + z

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order: `output` is written, `a` and `b` are read-only
  let addBindings: seq[BindingExpectation] =
    @[("output", 0, false), ("a", 1, true), ("b", 2, true)]
  discard compileCheck("add", addHlsl, addBindings)
  discard compileCheck("add_push", addHlslZ, addBindings)

when isMainModule:
  runTest()
