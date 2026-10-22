## ccGetBackend detects the compile target inside a kernel body.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/device/t_vulkan_ccgetbackend.nim

import std/[os, osproc, strutils, tempfiles]
import workspace/crucible

const kernelCode = vulkan:
  proc backendProbe(output: ptr UncheckedArray[int]) {.global, workgroup: (32, 1, 1).} =
    when ccGetBackend() == ctOpenCL: output[0] = 1
    elif ccGetBackend() == ctVulkan: output[0] = 2
    elif ccGetBackend() == ctMetal: output[0] = 3
    elif ccGetBackend() == ctCuda: output[0] = 4
    else: output[0] = 5

proc runTest() =
  # GLSL kernel entry points must be named `main`, since glslang links against `main`.
  # The pin renames the kernel in its copy of the source.
  block:
    let exe = findExe("glslangValidator")
    doAssert exe.len > 0,
      "glslangValidator not found on PATH (the GLSL pin needs it)"
    let src = kernelCode.replace("void backendProbe()", "void main()")
    let (tmpFile, tmpPath) = createTempFile("vk_ccgetbackend", ".comp")
    defer: tmpFile.close()
    tmpFile.write(src)
    tmpFile.flushFile()
    let (outp, exitCode) = execCmdEx(
      quoteShell(exe) & " -V --target-env vulkan1.1 " & quoteShell(tmpPath) & " -o /dev/null")
    doAssert exitCode == 0,
      "glslangValidator rejected the ccGetBackend shader:\n" & outp & "\n--- shader ---\n" & src
    echo "  OK — GLSL ccGetBackend emission + glslangValidator compile (SPIR-V, vulkan1.1 target)"

when isMainModule:
  runTest()
