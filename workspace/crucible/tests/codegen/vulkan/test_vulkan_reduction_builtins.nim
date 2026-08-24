## Vulkan: reduction builtin emission check + real glslangValidator 16.5.0 run.
##
## Two parts:
## 1. Emission: the GLSL header carries the KHR subgroup shuffle extensions
##    (scan-driven) and the body spells `subgroupShuffleDown`/`subgroupShuffle`.
## 2. Real check: the emitted shader is compiled by glslangValidator with the
##    subgroup extensions injected at harness level and `--target-env vulkan1.1`
##    (the shuffles need SPIR-V 1.3, while the default -V target is SPIR-V 1.0).
##
## The umbrella `GL_KHR_shader_subgroup` is not a
## GLSL extension name (glslang rejects it). subgroupShuffleDown needs
## GL_KHR_shader_subgroup_shuffle_relative, subgroupShuffle needs
## GL_KHR_shader_subgroup_shuffle. Both are emitted when either kind is used.
##
## Run from the tattletale root:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan/test_vulkan_reduction_builtins.nim

import std/[os, osproc, strutils, tempfiles]
import workspace/crucible

const kernelCode = vulkan:
  proc reductionKernel(output: ptr UncheckedArray[float32]) {.global, workgroup: (32, 1, 1).} =
    let acc = output[0]
    let v = simdShuffleDown(acc, 1'u32)
    output[1] = v
    let w = simdShuffle(v, 0'u32)
    output[2] = w

proc runTest() =
  # ── emission part ──────────────────────────────────────────────────────────
  doAssert "#extension GL_KHR_shader_subgroup_shuffle_relative : enable" in kernelCode,
    "GLSL header missing the shuffle_relative extension:\n" & kernelCode
  doAssert "#extension GL_KHR_shader_subgroup_shuffle : enable" in kernelCode,
    "GLSL header missing the shuffle extension:\n" & kernelCode
  doAssert "subgroupShuffleDown(acc, 1U)" in kernelCode,
    "GLSL subgroupShuffleDown spelling missing:\n" & kernelCode
  doAssert "subgroupShuffle(v, 0U)" in kernelCode,
    "GLSL subgroupShuffle spelling missing:\n" & kernelCode

  # ── real glslangValidator check ──────────────────────────────────────────────
  block:
    let exe = findExe("glslangValidator")
    doAssert exe.len > 0,
      "glslangValidator not found on PATH (the GLSL check needs it)"
    # Inject the subgroup extensions at harness level (belt and suspenders
    # over the scan): the shader must compile with them regardless of where
    # they come from. Dedupe when the scan already emitted them. GLSL kernel
    # entry points must be named `main` (glslang links against `main`), so the
    # harness copy renames the kernel.
    let harnessExt =
      "#extension GL_KHR_shader_subgroup_shuffle_relative : enable\n" &
      "#extension GL_KHR_shader_subgroup_shuffle : enable\n"
    var src = kernelCode
    if harnessExt notin src:
      # After the #version line (extensions must follow the version).
      let pos = src.find("\n")
      src = src[0 .. pos] & harnessExt & src[pos + 1 .. ^1]
    src = src.replace("void reductionKernel()", "void main()")
    let (tmpFile, tmpPath) = createTempFile("vk_reduction_builtins", ".comp")
    defer: tmpFile.close()
    tmpFile.write(src)
    tmpFile.flushFile()
    let (outp, exitCode) = execCmdEx(
      quoteShell(exe) & " -V --target-env vulkan1.1 " & quoteShell(tmpPath) & " -o /dev/null")
    doAssert exitCode == 0,
      "glslangValidator rejected the shuffle kernel:\n" & outp & "\n--- shader ---\n" & src
    echo "  OK — GLSL reduction builtin emission + glslangValidator compile (SPIR-V, vulkan1.1 target)"

when isMainModule:
  runTest()
