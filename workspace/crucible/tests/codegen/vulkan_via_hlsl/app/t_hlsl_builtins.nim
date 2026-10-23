## HLSL: coordinate-builtin reads and max/min/abs intrinsic calls through the real pass bundle.
##
## Test that the `hlsl:` macro emits HLSL for a kernel reading thread_position_in_grid
## (SV_DispatchThreadID input) and for kernels calling the scalar numeric builtins,
## compiling clean through glslangValidator and spirv-val with the binding reflection asserted.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_builtins.nim

import std/strutils
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
    output[4] = uint32(min(0.5'f32, -1.5'f32))

proc runTest() =   # private: the compile checks run in a proc so build artifacts stay scoped
  # Bindings take buffer parameter order, each kernel writes its single `output` buffer
  let coordBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("coord", coordHlsl, coordBindings, "uint3 gid : SV_DispatchThreadID")
  let intrinsicBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("intrinsics", intrinsicsHlsl, intrinsicBindings, "max(30u, 7u)")
  doAssert intrinsicsHlsl.contains("min(3u, 9u)"),
    "uint32 min must print the plain intrinsic:\n" & intrinsicsHlsl
  doAssert intrinsicsHlsl.contains("abs(a)"),
    "int32 abs must print the plain intrinsic, Nim constant-folds abs of literals:\n" & intrinsicsHlsl
  doAssert intrinsicsHlsl.contains("max(2.5f, 7.25f)"),
    "float32 max must print the plain intrinsic:\n" & intrinsicsHlsl
  doAssert intrinsicsHlsl.contains("min(0.5f, -1.5f)"),
    "float32 min must print the plain intrinsic:\n" & intrinsicsHlsl

when isMainModule:
  runTest()
