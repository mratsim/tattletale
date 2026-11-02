## HLSL: one module assembling the shader-tier sections, both flavors
##
## Test that the HLSL module assembly works for fvD3D12 and fvVulkan alike and the kernels compute:
##
## - a device function in the call chain
## - a struct push-constant parameter
## - the `[numthreads]` entry head over the body-read semantics
## - dxc compiles through spirv-val, launch readback exact on both flavors
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_shader_tier_assembly.nim

import workspace/crucible
import workspace/crucible/tests/codegen/vulkan_via_hlsl/hlsl_testutils

type Pair = object
  a: int32
  b: int32

proc widen(x: uint32): uint32 {.device.} =
  result = x + 1000u32

const structHlsl = hlsl:
  proc k(output: ptr UncheckedArray[uint32], p: Pair) {.global.} =
    let i = thread_position_in_grid.x
    if i == 0:
      output[0] = uint32(p.a + p.b) + widen(3)

const fnHlsl = hlsl:
  proc k(output: ptr UncheckedArray[uint32]) {.global.} =
    let i = thread_position_in_grid.x
    if i == 0:
      output[0] = widen(45)

const structVulkanHlsl = hlsl(fvVulkan):
  proc k(output: ptr UncheckedArray[uint32], p: Pair) {.global.} =
    let i = thread_position_in_grid.x
    if i == 0:
      output[0] = uint32(p.a + p.b) + widen(3)

proc runTest() =   # private: engines run in a proc so resources release at return
  # fvD3D12: dxc DXIL + SPIR-V + spirv-val over the struct kernel, the struct on the Push block behind the buffer.
  # descriptor route: the engine refuses scalar and struct kernel args (pre-existing at base), so the launch runs the struct-free kernel
  # struct-free kernel: the readback covers the device-function chain over the entry semantics.
  let structBindings: seq[BindingExpectation] = @[("output", 0, false)]
  discard compileCheck("assembly_d3d12_struct", structHlsl, structBindings)
  block:
    var engine = newEngine(runVulkan)
    engine.ingest(fnHlsl)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("main", oDev, ())
    var got: array[1, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, 1)
    doAssert got[0] == 1045'u32, "fvD3D12 device-fn chain got " & $got[0] & ", want 1045"
    echo "  fvD3D12 device-fn chain: [", got[0], "] exact readback (widen(45))"

  # fvVulkan: dxc -spirv (the vk:: constructs have no DXIL stage), spirv-val, then the push-block route launches with the struct member
  discard compileFvVulkanSpirV("assembly_fvulkan", structVulkanHlsl)
  block:
    var engine = newEngine(runVulkan)
    engine.ingest(structVulkanHlsl)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("main", oDev, (Pair(a: 20'i32, b: 22'i32),))
    var got: array[1, uint32]
    engine.copyToHostFromDevice(addr got[0], oDev, 1)
    doAssert got[0] == 1045'u32, "fvVulkan assembly got " & $got[0] & ", want 1045"
    echo "  fvVulkan assembly: [", got[0], "] exact readback (20 + 22 + widen(3))"

when isMainModule:
  runTest()
  echo "t_hlsl_shader_tier_assembly: all checks passed"
