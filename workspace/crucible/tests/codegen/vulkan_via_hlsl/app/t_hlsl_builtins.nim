## HLSL: coordinate-builtin reads and numeric builtins through the real pass bundle.
##
## Test that the `hlsl:` macro emits HLSL compiling clean through DXC
## and spirv-val with the binding reflection asserted:
##
## - a kernel reading thread_position_in_grid emits the SV_DispatchThreadID input
## - min/max materialize and pass through under their mangled names, magic `abs` prints plain
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_builtins.nim

import std/[re, strutils]
import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

const coordHlsl = hlsl:
  proc coordKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let i = thread_position_in_grid.x
    let tg = threadgroup_position_in_grid.x
    output[i] = i.uint32 + tg.uint32

const intrinsicsHlsl = hlsl:
  proc intrinsicKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = max(30'u32, 7'u32)
    output[1] = min(3'u32, 9'u32)
    let a = int32(-5)
    output[2] = abs(a).uint32
    output[3] = uint32(max(2.5'f32, 7.25'f32))
    # in-range operands: a negative float32-to-uint32 conversion is undefined per HLSL,
    # DXC constant-folds it to undef and the DXIL validator rejects the UAV store
    output[4] = uint32(min(0.5'f32, 1.5'f32))

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order, each kernel writes its single `output` buffer
  let coordBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("coord", coordHlsl, coordBindings, "uint3 gid : SV_DispatchThreadID")
  let intrinsicBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("intrinsics", intrinsicsHlsl, intrinsicBindings)
  # min/max materialize as wrapper functions under their mangled instantiation names
  # and the call sites use those names (same pass-through shape as every backend)
  doAssert intrinsicsHlsl.find(re"max_\w+\(30u, 7u\)") >= 0,
    "uint32 max must call its materialized wrapper:\n" & intrinsicsHlsl
  doAssert intrinsicsHlsl.find(re"min_\w+\(3u, 9u\)") >= 0,
    "uint32 min must call its materialized wrapper:\n" & intrinsicsHlsl
  doAssert intrinsicsHlsl.find(re"max_\w+\(2\.5f, 7\.25f\)") >= 0,
    "float32 max must call its materialized wrapper:\n" & intrinsicsHlsl
  doAssert intrinsicsHlsl.find(re"min_\w+\(0\.5f, 1\.5f\)") >= 0,
    "float32 min must call its materialized wrapper:\n" & intrinsicsHlsl
  doAssert intrinsicsHlsl.contains("abs(a)"),
    "int32 abs must print the plain intrinsic, Nim constant-folds abs of literals:\n" & intrinsicsHlsl

when isMainModule:
  runTest()
