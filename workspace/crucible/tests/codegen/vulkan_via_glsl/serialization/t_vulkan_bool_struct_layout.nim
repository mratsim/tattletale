## Vulkan/GLSL: bool members inside struct kernel params round-trip.
##
## Test that a bool inside a struct kernel param — scalar field or array
## field — launches and reads back with the same values: the host packs
## boundary bools at the 4-byte std430 width the GLSL block declares.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_glsl/serialization/t_vulkan_bool_struct_layout.nim

import std/strformat
import workspace/crucible

type
  BoolPair = object
    a: bool
    b: bool
    c: float32

  FlagPack = object
    flags: array[4, bool]
    x: float32

# One kernel per source: the codegen emits a single file-scope KernelParams
# push-constant block per source = the union of every kernel's by-value params.
const scalarCode = glsl:
  proc kernelBoolPair(outP: ptr UncheckedArray[uint32]; s: BoolPair) {.global.} =
    outP[0] = uint32(s.a)
    outP[1] = uint32(s.b)
    outP[2] = uint32(s.c == 1.0'f32)

const arrayCode = glsl:
  proc kernelFlagPack(outP: ptr UncheckedArray[uint32]; s: FlagPack) {.global.} =
    outP[0] = uint32(s.flags[0])
    outP[1] = uint32(s.flags[1])
    outP[2] = uint32(s.flags[2])
    outP[3] = uint32(s.flags[3])
    outP[4] = uint32(s.x == 2.5'f32)

proc runTest() =
  echo "=== Vulkan/GLSL: bool members inside struct kernel params ===\n"

  block:
    var engine = newEngine(runVulkan)
    engine.ingest(scalarCode)
    let oDev = engine.allocDevice(uint32, 3)
    var gpuOut: array[3, uint32] = [0xDEADBEEF'u32, 0xDEADBEEF'u32, 0xDEADBEEF'u32]
    engine.copyToDeviceFromHost(oDev, addr gpuOut[0], 3)
    engine.launchKernel("kernelBoolPair", oDev, (BoolPair(a: true, b: false, c: 1.0'f32),))
    engine.copyToHostFromDevice(addr gpuOut[0], oDev, 3)
    doAssert gpuOut[0] == 1, &"scalar bool field a: got {gpuOut[0]:#x}, expected 1"
    doAssert gpuOut[1] == 0, &"scalar bool field b: got {gpuOut[1]:#x}, expected 0"
    doAssert gpuOut[2] == 1, &"scalar float field c: got {gpuOut[2]:#x}, expected 1"
    echo "  OK — scalar bool fields pack at the block width"

  block:
    var engine = newEngine(runVulkan)
    engine.ingest(arrayCode)
    let oDev = engine.allocDevice(uint32, 5)
    var gpuOut: array[5, uint32] = [0xDEADBEEF'u32, 0xDEADBEEF'u32, 0xDEADBEEF'u32,
                                    0xDEADBEEF'u32, 0xDEADBEEF'u32]
    engine.copyToDeviceFromHost(oDev, addr gpuOut[0], 5)
    engine.launchKernel("kernelFlagPack", oDev,
      (FlagPack(flags: [true, false, true, false], x: 2.5'f32),))
    engine.copyToHostFromDevice(addr gpuOut[0], oDev, 5)
    doAssert gpuOut[0] == 1, &"bool array element 0: got {gpuOut[0]:#x}, expected 1"
    doAssert gpuOut[1] == 0, &"bool array element 1: got {gpuOut[1]:#x}, expected 0"
    doAssert gpuOut[2] == 1, &"bool array element 2: got {gpuOut[2]:#x}, expected 1"
    doAssert gpuOut[3] == 0, &"bool array element 3: got {gpuOut[3]:#x}, expected 0"
    doAssert gpuOut[4] == 1, &"float field after bool array: got {gpuOut[4]:#x}, expected 1"
    echo "  OK — bool array fields pack at the block width"

when isMainModule:
  runTest()
