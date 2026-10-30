## std430 vector alignment in the emitted push block
##
## Test that a push block over vec2/vec3/vec4 members and a vector array member matches the std430 offsets the Vulkan spec mandates:
## spirv-val tolerates many layouts, so the numbers come from the `spirv-dis` output
##
## - vec2 base alignment is 2x the component size, vec3/vec4 base alignment is 4x (Vulkan spec, "Offset and Stride")
## - an array's base alignment matches its element's base alignment, and its stride is a multiple of that alignment
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_push_block_vector_layout.nim

import std/os
import std/osproc
import std/strutils
import workspace/crucible/src/codegen/ir/cirv2_lang
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/backends/spirv_shader_lang

proc buildVecLayoutModule(): CirModule =
  ## Push members, in order: vec2<f32>, vec3<f32>, vec4<f32>, array[2] of vec3<f32>
  let kernel = procD("vecLayoutKernel", @[
    param("v2", tVec(2, tFloat(32))),
    param("v3", tVec(3, tFloat(32))),
    param("v4", tVec(4, tFloat(32))),
    param("arr", tArray(2, tVec(3, tFloat(32)))),
  ], tVoid(), @[retS()])
  kernel.isKernel = true
  moduleD("vecLayout", @[], @[kernel])

const vecEmission = static:
  printSpirvModule(buildVecLayoutModule())

# std430 member offsets (Vulkan spec, "Offset and Stride"):
# - vec2: align 8, offset 0
# - vec3: align 16, offset 16
# - vec4: align 16, offset 32
# - array of vec3: align 16 (the element's base alignment), offset 48, stride 16
const expectedOffsets = @[0, 16, 32, 48]
const expectedStrides = @[16]

proc runTest() =
  doAssert vecEmission.err.len == 0,
    "SPIR-V emission refused the vector push block: " & vecEmission.err
  doAssert vecEmission.words.len > 0 and vecEmission.words[0] == 0x07230203'u32,
    "SPIR-V magic number missing"

  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"

  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_push_block_vector_layout.spv"
  # files close at block exit, before the subprocesses read them
  let words = vecEmission.words
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr words[0], words.len * 4)

  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the emitted module:\n" & valOut

  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed"

  var offsets: seq[int]
  for line in disOut.splitLines():
    if "OpMemberDecorate" in line and "Offset" in line:
      offsets.add parseInt(line.strip().splitWhitespace()[^1])
  doAssert offsets == expectedOffsets,
    "push member offsets " & $offsets & " must be the std430 layout " &
    $expectedOffsets & " (vec2 -> 2N, vec3/vec4 -> 4N), disassembly:\n" & disOut

  var strides: seq[int]
  for line in disOut.splitLines():
    if "ArrayStride" in line:
      strides.add parseInt(line.strip().splitWhitespace()[^1])
  doAssert strides == expectedStrides,
    "array strides " & $strides & " must be the std430 stride " &
    $expectedStrides & " (element base alignment rounded), disassembly:\n" & disOut

when isMainModule:
  runTest()
  echo "t_spirv_push_block_vector_layout: std430 vector offsets and strides correct"
