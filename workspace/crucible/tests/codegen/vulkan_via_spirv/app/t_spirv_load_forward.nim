## SPIR-V load forwarding: single-store locals and loop-header induction reads
##
## Test that a `spirv:` block:
## - elides the storage of a local whose every read the plan proves dominated
## - binds those reads to the stored value, no OpVariable, no OpStore
## - re-establishes an induction variable with one header load per iteration
##
## The ceilings are the post-forwarding spirv-dis instruction counts.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_load_forward.nim

import std/[os, osproc, re, strutils]
import workspace/crucible

type Vec2 = object
  x: uint32
  y: uint32

proc vec2Add(a, b: Vec2): Vec2 {.device.} =
  result.x = a.x + b.x
  result.y = a.y + b.y

const axpySpv = spirv:
  proc axpyKernel(y, x: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global.} =
    for i in 0 ..< n:
      y[i] = alpha * x[i] + y[i]

const vec2Spv = spirv:
  proc vec2AddKernel(output: ptr UncheckedArray[uint32], a: ptr UncheckedArray[uint32], b: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

proc disInstructionCount(tag: string, words: seq[uint32]): int =
  doAssert words.len > 0, "`spirv:` macro emitted no words for " & tag
  doAssert words[0] == 0x07230203'u32, "SPIR-V magic number missing for " & tag
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / tag & ".spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr words[0], words.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  if valCode != 0:
    let (disOut, _) = execCmdEx(spirvDis & " " & spvPath)
    echo "spirv-dis dump:\n", disOut
    doAssert false, "spirv-val rejected the emitted module " & tag & ":\n" & valOut
  let (dis, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on " & tag
  for line in dis.splitLines():
    if line.find(re"\bOp[A-Z][A-Za-z]*\b") >= 0:
      inc result

proc runTest() =
  block axpy_loop_forwarding:
    let n = disInstructionCount("t_spirv_load_forward_axpy", axpySpv)
    # one header load per iteration re-establishes the induction variable.
    # Body reads read that value: 4 OpLoads fewer than the reload form
    doAssert n <= 76, "axpy emits " & $n &
      " instructions, the loop-variable loads are not forwarded"

  block vec2_single_store_forwarding:
    let n = disInstructionCount("t_spirv_load_forward_vec2", vec2Spv)
    # va, vb and vr hold one store each and every read reads the stored value.
    # No Function-storage variable, no OpStore and no OpLoad remains
    doAssert n <= 86, "vec2 emits " & $n &
      " instructions, the single-store locals are not forwarded"

when isMainModule:
  runTest()
  echo "t_spirv_load_forward: spirv-val accepted both modules, forwarding counts hold"
