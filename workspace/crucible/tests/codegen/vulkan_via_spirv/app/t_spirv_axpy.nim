## SPIR-V binary emission: axpy kernel through the `spirv:` macro, spirv-val checked
##
## Test that a `spirv:` block with one `{.global.}` kernel emits binary SPIR-V
## accepted by `spirv-val --target-env vulkan1.4`. The kernel body loops
## over a scalar push-constant parameter and two buffer-device-address pointer parameters.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_axpy.nim

import std/os
import std/osproc
import std/strutils
import workspace/crucible

const axpySpv = spirv:
  proc axpyKernel(y, x: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global.} =
    for i in 0 ..< n:
      y[i] = alpha * x[i] + y[i]

proc runTest() =
  block emission:
    doAssert axpySpv.len > 0, "`spirv:` macro emitted no words"
    doAssert axpySpv[0] == 0x07230203'u32, "SPIR-V magic number missing"

  block validation:
    let spirvVal = findExe("spirv-val")
    doAssert spirvVal.len > 0,
      "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
    let spirvDis = findExe("spirv-dis")

    let outDir = "build/tests"
    createDir(outDir)
    let spvPath = outDir / "t_spirv_axpy.spv"
    # the file closes at block exit, before the subprocesses read it
    block:
      var f = open(spvPath, fmWrite)
      defer: f.close()
      discard f.writeBuffer(addr axpySpv[0], axpySpv.len * 4)

    let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
    if valCode != 0:
      if spirvDis.len > 0:
        let (disOut, _) = execCmdEx(spirvDis & " " & spvPath)
        echo "spirv-dis dump:\n", disOut
      doAssert false, "spirv-val rejected the emitted module:\n" & valOut
    # spirv-dis only feeds the failure dump above. The check is spirv-val itself.
    doAssert spirvDis.len > 0,
      "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"

when isMainModule:
  runTest()
  echo "t_spirv_axpy: spirv-val accepted the emitted module"
