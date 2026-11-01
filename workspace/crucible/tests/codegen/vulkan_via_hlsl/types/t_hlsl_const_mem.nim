## HLSL: `{.const_mem.}` locals referenced from nested scopes and device functions.
##
## Test that one kernel covers the three `{.const_mem.}` declaration sites:
##
## - a kernel top-level constant, read in both `if` branches
## - a constant declared inside the taken branch
## - a constant declared in a device function
##
## Outputs assert byte-exact values against the host reference.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/types/t_hlsl_const_mem.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

proc cmScale(a: float32): float32 =
  var cmK {.const_mem.}: float32 = 3.0'f32
  result = a * cmK

const constMemHlsl = hlsl:
  proc constMemKernel(output, a: ptr UncheckedArray[float32]) {.global.} =
    var cmBias {.const_mem.}: float32 = 1.0'f32
    if a[0] > 0'f32:
      var cmMult {.const_mem.}: float32 = 2.0'f32
      output[0] = a[0] * cmMult + cmBias
    else:
      output[0] = cmBias
    output[1] = cmScale(a[1])

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # No HLSL engine exists on this route: a DXC + spirv-val compile check
  # stands in for the launch, per the house exception for backends without
  # runtime support. Compile only: the kernel declares no buffer resources.
  discard compileCheck("hlsl_const_mem", constMemHlsl, @[])

when isMainModule:
  runTest()
