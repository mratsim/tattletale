## SPIR-V binary emission: `{.unroll.}` sets LoopControlUnroll on OpLoopMerge
##
## Test that the `{.unroll.}`-marked OpLoopMerge has LoopControl Unroll,
## the unmarked one has None, both spirv-val checked
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_unroll_loop.nim

import std/os
import std/osproc
import std/strutils
import workspace/crucible

const unrollSpv = spirv:
  proc unrollKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 4:
      {.unroll.}
      output[i] = uint32(i)

const plainSpv = spirv:
  proc plainKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 4:
      output[i] = uint32(i)

proc runTest() =
  doAssert unrollSpv.len > 0 and plainSpv.len > 0, "`spirv:` macro emitted no words"

  block validation:
    let spirvVal = findExe("spirv-val")
    doAssert spirvVal.len > 0,
      "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
    let spirvDis = findExe("spirv-dis")

    let outDir = "build/tests"
    createDir(outDir)
    for pair in [(name: "t_spirv_unroll_loop", words: unrollSpv, control: "Unroll"),
                 (name: "t_spirv_unroll_loop_plain", words: plainSpv, control: "None")]:
      let spvPath = outDir / (pair.name & ".spv")
      block:
        var f = open(spvPath, fmWrite)
        defer: f.close()
        discard f.writeBuffer(addr pair.words[0], pair.words.len * 4)

      let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
      if valCode != 0:
        if spirvDis.len > 0:
          let (disOut, _) = execCmdEx(spirvDis & " " & spvPath)
          echo "spirv-dis dump:\n", disOut
        doAssert false, "spirv-val rejected the emitted module:\n" & valOut
      # the disassembly locks the loop control word so a silent regression fails here
      if spirvDis.len > 0:
        let (disOut, _) = execCmdEx(spirvDis & " " & spvPath)
        var mergeCount = 0
        for line in disOut.splitLines():
          if "OpLoopMerge" in line:
            inc mergeCount
            doAssert line.contains(pair.control),
              pair.name & ": expected LoopControl " & pair.control & ", got: " & line
        doAssert mergeCount > 0,
          pair.name & ": no OpLoopMerge in the disassembly:\n" & disOut

  echo "  OK - t_spirv_unroll_loop"

when isMainModule:
  runTest()
