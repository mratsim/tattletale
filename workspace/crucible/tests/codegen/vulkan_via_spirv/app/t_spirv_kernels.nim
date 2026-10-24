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

const atomicSpv = spirv:
  proc atomicKernel(counter: ptr UncheckedArray[float32]) {.global.} =
    discard atomicFetchAdd(addr counter[0], 1.0'f32)

const ternaryRangeSpv = spirv:
  proc ternaryRangeKernel(output: ptr UncheckedArray[float32], n: int32) {.global.} =
    for i in 0 ..< (if n > 0: 4 else: 8):
      output[i] = float32(i)

const boolEqSpv = spirv:
  proc boolEqKernel(output: ptr UncheckedArray[uint32], a: int32, b: int32) {.global.} =
    let eq = a == b
    if eq == true:
      output[0] = 1
    else:
      output[0] = 2

const fmaxSpv = spirv:
  proc fmaxKernel(output: ptr UncheckedArray[float32], x: ptr UncheckedArray[float32], n: int32) {.global.} =
    for i in 0 ..< n:
      output[i] = max(x[i], 0.0'f32)

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
  # atomic add through a buffer device address: the semantics word 72
  # (0x48, UniformMemory | AcquireRelease) carries the non-relaxed memory
  # order the Vulkan environment requires whenever a storage-class semantics
  # bit is set, over the GLSL450 memory model the glslang reference compiler
  # pairs with the atomic-float extension
  validateSpv(atomicSpv, "t_spirv_kernels_atomic",
              @["OpAtomicFAddEXT",
                "OpExtension \"SPV_EXT_shader_atomic_float_add\"",
                "%uint_72",
                "OpMemoryModel PhysicalStorageBuffer64 GLSL450"])
  # a selection expression as the loop range end: the bound must compute
  # in the pre-header block, a bound computed inside the header block
  # strands OpLoopMerge in a block with branches
  validateSpv(ternaryRangeSpv, "t_spirv_kernels_ternary_range",
              @["OpLoopMerge", "OpBranchConditional"])
  # bool equality: OpIEqual requires integer operands, booleans lower to the logical equality opcodes
  validateSpv(boolEqSpv, "t_spirv_kernels_bool_eq",
              @["OpLogicalEqual"])
  # float max/min: the integer compare opcodes are illegal over float operands, floats lower through ordered float comparisons
  validateSpv(fmaxSpv, "t_spirv_kernels_fmax",
              @["OpFOrdGreaterThan", "OpSelect"])

when isMainModule:
  runTest()
  echo "t_spirv_kernels: spirv-val accepted all emitted modules"
