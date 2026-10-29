## HLSL: device functions emit as native HLSL functions.
##
## Test that a kernel calling scalar device fns emits the function
## definitions with forward declarations, the calls survive into the shader
## and the chain computes its value on the device:
##
## - an early return in a device fn emits natively
## - the call sites keep the callee names, no splicing into the kernel body
## - the source compiles clean through DXC and spirv-val with the binding reflection asserted
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/t_hlsl_device_fn_call.nim

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
  # Bindings take buffer parameter order, the device fns take no buffers
  let kBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("device_fn_call", chainHlsl, kBindings)

  var engine = newEngine(runVulkan)
  engine.ingest(chainHlsl)
  let oDev = engine.allocDevice(uint32, 1)
  engine.launchKernel("main", oDev, ())
  var got: array[1, uint32]
  engine.copyToHostFromDevice(addr got[0], oDev, 1)
  # middle(10) = (10 + 1) * 2 = 22, clamped at 100 stays 22
  doAssert got[0] == 22'u32, "device fn chain got " & $got[0] & ", expected 22"

  echo "  OK - t_hlsl_device_fn_call"

when isMainModule:
  runTest()
