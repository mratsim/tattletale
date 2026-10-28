## SPIR-V binary emission: `{.unroll.}` loops execute with LoopControlUnroll
##
## Test that the `{.unroll.}`-marked loop kernel and the unmarked kernel
## run on the Vulkan device with the expected output, the marked OpLoopMerge
## shows LoopControl Unroll and the unmarked one None
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
  proc unrollKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    for i in 0 ..< 4:
      {.unroll.}
      output[i] = uint32(i)

const plainSpv = spirv:
  proc plainKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    for i in 0 ..< 4:
      output[i] = uint32(i)

proc runTest() =
  doAssert unrollSpv.len > 0 and plainSpv.len > 0, "`spirv:` macro emitted no words"

  block execution:
    var engine = newEngine(runVulkan)
    for pair in [(name: "unrollKernel", words: unrollSpv),
                 (name: "plainKernel", words: plainSpv)]:
      engine.ingest(pair.words)
      let oDev = engine.allocDevice(uint32, 4)
      engine.launchKernel(pair.name, oDev, ())
      var res: array[4, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      for i in 0 ..< 4:
        doAssert res[i] == uint32(i),
          pair.name & ": output[" & $i & "] read " & $res[i] & ", expected " & $i

  block loopControl:
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
      # secondary lock: the disassembly locks the loop control word
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
