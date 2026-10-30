## HLSL: a buffer written only through a device fn call in a let initializer binds read-write.
##
## Test that let-initializer call edges reach the argument-flow scan.
## RWStructuredBuffer must appear for the kernel buffer and the device fn parameter.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/functions/t_hlsl_device_fn_buffer_let_init.nim

import std/strutils
import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

proc bump(counts: ptr UncheckedArray[uint32], i: uint32): uint32 {.device.} =
  counts[i] = counts[i] + 1u32
  result = counts[i]

const letInitHlsl = hlsl:
  proc k(counts: ptr UncheckedArray[uint32], outBuf: ptr UncheckedArray[uint32]) {.global.} =
    let old = bump(counts, 0u32)
    outBuf[0] = old

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # `counts` reaches the device fn only as a call argument inside a let initializer,
  # read-write classification flows to the kernel's buffer over the argument-flow edge.
  doAssert letInitHlsl.contains("RWStructuredBuffer<uint> counts : register(u0, space0);"),
    "the kernel buffer written only through the device fn call in the let" &
    " initializer must bind read-write:\n" & letInitHlsl
  let bindings: seq[BindingExpectation] =
    @[("counts", 0, false), ("outBuf", 1, false)]
  discard compileCheck("device_fn_buffer_let_init", letInitHlsl, bindings)

  echo "  OK - t_hlsl_device_fn_buffer_let_init"

when isMainModule:
  runTest()
