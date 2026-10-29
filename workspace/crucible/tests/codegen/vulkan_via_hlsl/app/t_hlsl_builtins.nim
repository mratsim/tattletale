## HLSL: coordinate-builtin reads and numeric builtins through the real pass bundle.
##
## Test that the `hlsl:` macro emits HLSL compiling clean through DXC
## and spirv-val with the binding reflection asserted, and the kernels
## compute their values on the device:
##
## - a kernel reading thread_position_in_grid emits the SV_DispatchThreadID input
## - min/max/abs compute under their materialized wrappers
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_builtins.nim

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

  block:
    var engine = newEngine(runVulkan)
    engine.ingest(coordHlsl)
    let oDev = engine.allocDevice(uint32, 2)
    engine.launchKernel("main", oDev, ())
    var got: array[2, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, 2)
    # grid (1,1) x block (2,1): global id i in 0..1, group id 0
    for i in 0 ..< 2:
      doAssert got[i] == uint32(i), "coord got " & $got[i] & " at " & $i & ", expected " & $i

  block:
    var engine = newEngine(runVulkan)
    engine.ingest(intrinsicsHlsl)
    let oDev = engine.allocDevice(uint32, 5)
    engine.launchKernel("main", oDev, ())
    var got: array[5, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, 5)
    doAssert got[0] == 30'u32, "max(30u, 7u) got " & $got[0]
    doAssert got[1] == 3'u32, "min(3u, 9u) got " & $got[1]
    doAssert got[2] == 5'u32, "abs(-5) got " & $got[2]
    doAssert got[3] == 7'u32, "uint32(max(2.5f, 7.25f)) got " & $got[3] & ", expected 7"
    doAssert got[4] == 0'u32, "uint32(min(0.5f, 1.5f)) got " & $got[4] & ", expected 0"

  echo "  OK - t_hlsl_builtins"

when isMainModule:
  runTest()
