## HLSL: device functions emit as native HLSL functions.
##
## Test that a kernel calling scalar device fns emits the function definitions
## with forward declarations and the calls survive into the shader:
##
## - an early return in a device fn emits natively
## - the call sites keep the callee names, no splicing into the kernel body
## - the source compiles clean through DXC and spirv-val with the binding reflection asserted
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/t_hlsl_device_fn_call.nim

import std/[re, strutils]
import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

proc inner(x: uint32): uint32 {.device.} =
  result = x + 1

proc middle(x: uint32): uint32 {.device.} =
  result = inner(x) * 2

proc clampHi(x: uint32): uint32 {.device.} =
  if x > 100u32:
    return 100u32
  result = x

const chainHlsl = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let y = middle(10)
    output[0] = clampHi(y)

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Fn names carry the mangler's hash suffix:
  # the definition, the forward declaration and the call site must agree on it
  doAssert find(chainHlsl, re"uint middle_\w+\(uint x\) \{") >= 0,
    "emitted HLSL is missing the `middle` device fn definition:\n" & chainHlsl
  doAssert find(chainHlsl, re"uint middle_\w+\(uint x\);") >= 0,
    "emitted HLSL is missing the `middle` forward declaration:\n" & chainHlsl
  doAssert find(chainHlsl, re"middle_\w+\(10\)") >= 0,
    "the device fn call did not survive into the kernel body:\n" & chainHlsl
  doAssert find(chainHlsl, re"uint inner_\w+\(uint x\) \{") >= 0,
    "emitted HLSL is missing the transitively called `inner` definition:\n" & chainHlsl
  doAssert find(chainHlsl, re"return 100u;") >= 0,
    "the early return in a device fn was not emitted natively:\n" & chainHlsl
  # Bindings take buffer parameter order, the device fns take no buffers
  let kBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("device_fn_call", chainHlsl, kBindings)

when isMainModule:
  runTest()
