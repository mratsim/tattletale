## SPIR-V binary emission: device-function and accumulator kernels through the `spirv:` macro
##
## Test that `spirv:` blocks emit binary SPIR-V accepted by `spirv-val --target-env vulkan1.4`:
## - a kernel over an external object type and `{.device.}` proc emits OpFunction,
##   OpFunctionCall, object construction and member stores
## - a kernel with a Function-storage accumulator over buffer-device-address loads
##   emits the phi-free loop and OpFAdd
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_kernels.nim

import std/os
import std/osproc
import std/strutils
import workspace/crucible

# External types and device fns live outside the `spirv:` blocks,
# the kernel body resolves them by symbol

type
  Vec2 = object
    x: uint32
    y: uint32

proc vec2Add(a, b: Vec2): Vec2 {.device.} =
  result.x = a.x + b.x
  result.y = a.y + b.y

const vec2Spv = spirv:
  proc vec2AddKernel(output: ptr UncheckedArray[uint32], a: ptr UncheckedArray[uint32], b: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const sumSpv = spirv:
  proc sumKernel(output: ptr UncheckedArray[float32], x: ptr UncheckedArray[float32], n: int32) {.global.} =
    var acc: float32 = 0.0
    for i in 0 ..< n:
      acc = acc + x[i]
    output[0] = acc

const pickSpv = spirv:
  proc pickKernel(output: ptr UncheckedArray[float32], flag: int32) {.global.} =
    if flag > 0:
      var acc: array[2, float32] = [1.0'f32, 2.0'f32]
      acc = [3.0'f32, 4.0'f32]
      output[0] = acc[0]
    else:
      var acc: array[2, float32] = [5.0'f32, 6.0'f32]
      acc = [7.0'f32, 8.0'f32]
      output[0] = acc[0]

proc validateSpv(words: seq[uint32], name: string, expectOpcodes: openArray[string]) =
  doAssert words.len > 0, "`spirv:` macro emitted no words for " & name
  doAssert words[0] == 0x07230203'u32, "SPIR-V magic number missing for " & name

  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let spirvDis = findExe("spirv-dis")

  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / (name & ".spv")
  # the file closes at block exit, before the subprocesses read it
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr words[0], words.len * 4)

  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  if valCode != 0:
    if spirvDis.len > 0:
      let (disOut, _) = execCmdEx(spirvDis & " " & spvPath)
      echo "spirv-dis dump:\n", disOut
    doAssert false, "spirv-val rejected the emitted module " & name & ":\n" & valOut
  # spirv-val checks structure, not the construct mix: the disassembly locks the emitted opcodes so a still-valid regression fails here
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on " & name
  for op in expectOpcodes:
    doAssert op in disOut, name & " does not emit " & op
  doAssert "OpPhi" notin disOut, name & " emits OpPhi, the emission is phi-free by construction"

proc runTest() =
  validateSpv(vec2Spv, "t_spirv_kernels_vec2",
              @["OpFunctionCall", "OpCompositeConstruct",
                "OpTypePointer PhysicalStorageBuffer"])
  validateSpv(sumSpv, "t_spirv_kernels_sum",
              @["OpLoopMerge", "OpFAdd", "OpVariable"])
  # sibling same-name array locals with composite assigns: the assign store
  # must address the branch-local variable, keyed by iSym first
  validateSpv(pickSpv, "t_spirv_kernels_pick",
              @["OpSelectionMerge", "OpCompositeConstruct", "OpStore"])

when isMainModule:
  runTest()
  echo "t_spirv_kernels: spirv-val accepted both emitted modules"
