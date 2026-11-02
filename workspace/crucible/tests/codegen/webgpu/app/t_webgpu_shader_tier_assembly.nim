## WebGPU: one module assembling every shader-tier section
##
## Test that the WGSL module assembly orders every section at once and the kernel computes:
##
## - a device function
## - a struct kernel parameter with a bool field
## - a smem workgroup variable with an initializer and a const_mem constant
## - the entry signature over the coordinate builtins the body reads
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/app/t_webgpu_shader_tier_assembly.nim

import std/unittest
import workspace/crucible

type Controls = object
  scale: uint32
  enable: bool

const assemblyWgsl = webgpu:
  proc triple(x: uint32): uint32 = x * 3'u32
  proc assemblyKernel(output: ptr UncheckedArray[uint32], c: Controls) {.global.} =
    var scratch {.smem.}: array[2, uint32] = [5'u32, 6]
    var cmBias {.const_mem.}: uint32 = 11'u32
    let i = thread_position_in_grid.x
    if i == 0:
      if c.enable:
        scratch[0] = triple(scratch[0])
      scratch[1] = scratch[1] + cmBias
      output[0] = scratch[0] + c.scale
      output[1] = scratch[1]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - shader-tier module assembly":

    test "device fn + bool-struct param + smem + const_mem readback, enabled branch":
      var engine = newEngine(runWebGPU)
      engine.ingest(assemblyWgsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("assemblyKernel", oDev,
        (Controls(scale: 100'u32, enable: true),))
      var got: array[2, uint32]
      engine.copyToHostFromDevice(addr got[0], oDev, 2)
      check got[0] == 115'u32   # triple(5) + 100 = 115
      check got[1] == 17'u32    # 6 + 11

    test "the disabled branch keeps the smem initializer and the const":
      var engine = newEngine(runWebGPU)
      engine.ingest(assemblyWgsl)
      let oDev = engine.allocDevice(uint32, 2)
      engine.launchKernel("assemblyKernel", oDev,
        (Controls(scale: 0'u32, enable: false),))
      var got: array[2, uint32]
      engine.copyToHostFromDevice(addr got[0], oDev, 2)
      check got[0] == 5'u32     # triple skipped, 5 + 0
      check got[1] == 17'u32    # 6 + 11

when isMainModule:
  runTest()
  echo "t_webgpu_shader_tier_assembly: all checks passed"
