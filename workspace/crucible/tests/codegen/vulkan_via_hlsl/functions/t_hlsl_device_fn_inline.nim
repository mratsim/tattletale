## HLSL: device functions splice into the kernel, buffer-resource-taking ones included.
##
## Test that a kernel calling a buffer-taking device fn twice keeps the splices' locals apart and a three-deep call chain binding
## a non-void result to a var initializer compiles clean through glslangValidator and spirv-val with the binding reflection asserted.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/t_hlsl_device_fn_inline.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

proc scale(buf: ptr UncheckedArray[uint32], n: uint32, factor: uint32) {.device.} =
  for i in 0 ..< n:
    buf[i] = buf[i] * factor

proc inner(x: uint32): uint32 {.device.} =
  result = x + 1

proc middle(x: uint32): uint32 {.device.} =
  result = inner(x) * 2

const chainHlsl = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    scale(output, 2, 3)
    scale(output, 2, 5)
    let y = middle(10)
    output[0] = y

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order, the splices write the single `output` buffer
  let kBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("device_fn_inline", chainHlsl, kBindings)

when isMainModule:
  runTest()
