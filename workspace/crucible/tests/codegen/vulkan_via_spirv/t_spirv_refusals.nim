## SPIR-V codegen refusals: kernel-shape and push-constant-type rejections
##
## Test that the spirv macro refuses unsupported kernel shapes at compile time.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/t_spirv_refusals.nim

import std/[os, osproc, strutils]
import workspace/crucible

# The spirv: macro's colon syntax does not parse inside compiles(...),
# the templates defer expansion to the compiles call site

template noGlobalKernelBlock() =
  let k = spirv:
    proc flagSet(output: ptr uint32, flag: bool) =
      if flag:
        output[] = 1'u32
      else:
        output[] = 0'u32
  discard k

const boolRefusalMessage =
  "push-constant member bool, OpTypeBool cannot appear in a push-constant block"

proc compileFromSource(src: string): tuple[output: string, exitCode: int] =
  ## Compiles a generated check file against the workspace/crucible
  ## import path, returns the compiler output for message assertions
  let testDir = currentSourcePath().parentDir()
  # tests/codegen/vulkan_via_spirv -> the repo root that carries workspace/
  let projectRoot = testDir / "../../../.."
  let tmpDir = testDir / "nimcache" / "refusal_checks"
  createDir(tmpDir)
  let checkPath = tmpDir / "check_bool_push_constant.nim"
  writeFile(checkPath, src)
  let cmd = "nim c --hints:off --warnings:off" &
            " --path:" & projectRoot.absolutePath &
            " --outdir:" & tmpDir & " --nimcache:" & tmpDir / "nimcache" &
            " " & checkPath
  result = execCmdEx(cmd)
  removeFile(checkPath)

proc runTest() =
  block noGlobalKernel:
    doAssert not compiles(noGlobalKernelBlock()),
      "a spirv: block without a {.global.} kernel must refuse to compile"

  block boolPushConstant:
    # compiles() cannot distinguish the documented refusal from any other
    # compile-time failure, the message is asserted on the compiler output
    let (outp, rc) = compileFromSource("""
import workspace/crucible
let k = spirv:
  proc flagSet(output: ptr uint32, flag: bool) {.global.} =
    if flag:
      output[] = 1'u32
    else:
      output[] = 0'u32
discard k
""")
    doAssert rc != 0, "a bool kernel parameter must refuse to compile, got: " & outp
    doAssert boolRefusalMessage in outp,
      "the bool push-constant refusal must name the OpTypeBool block defect, got: " & outp

when isMainModule:
  runTest()
  echo "t_spirv_refusals: all checks passed"
