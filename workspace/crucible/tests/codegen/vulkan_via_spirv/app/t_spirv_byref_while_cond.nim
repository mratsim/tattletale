## Direct SPIR-V: while conditions calling by-ref device functions
## Test that a while condition rebuilds its fresh struct argument per test
##
## The loop condition passes a struct built from the induction variable by reference.
## A per-test rebuild runs 2 iterations.
## A struct hoisted before the loop freezes the first iteration's value and runs 3.
## The last output cell separates the two.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_while_cond.nim

import std/[os, osproc]
import workspace/crucible

type
  Count = object
    n: int32
    pad: array[16, int32]

const byrefWhileCondSpv = spirv:
  proc inWindow(c: Count): bool {.device.} =
    c.n > 0'i32 and c.n < 3'i32

  proc windowKernel(output: ptr UncheckedArray[int32]) {.global.} =
    var i: int32 = 0'i32
    while inWindow(Count(n: i + 1'i32)) and i < 3'i32:
      output[i] = i + 10'i32
      i = i + 1'i32
    output[7] = i

proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_byref_while_cond.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr byrefWhileCondSpv[0], byrefWhileCondSpv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(byrefWhileCondSpv)
  let oDev = engine.allocDevice(int32, 8)

  engine.launchKernel("windowKernel", oDev, ())
  var got: array[8, int32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 7), oDev)
  doAssert got[7] == 2'i32,
    "the condition rebuilds its struct argument every test, got final i=" & $got[7] &
    ", want 2 (a pre-loop hoist freezes n=1 and runs 3 iterations)"
  doAssert got[0] == 10'i32 and got[1] == 11'i32 and got[2] == 0'i32,
    "the loop body writes once per admitted iteration, got " & $got[0] & ", " & $got[1] & ", " & $got[2]
  echo "[OK] direct SPIR-V: while condition rebuilds its by-ref argument per test"

when isMainModule:
  runTest()
  echo "t_spirv_byref_while_cond: all checks passed"
