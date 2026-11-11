## Direct SPIR-V: assignments whose target indexes through a by-ref device call
## Test that a by-ref call inside an assignment target passes a pointer
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_asgn_target.nim

import std/[os, osproc]
import workspace/crucible

type
  Slot = object
    base: int32
    pad: array[16, int32]

const byrefAsgnTargetSpv = spirv:
  proc chooseSlot(s: Slot): int32 {.device.} =
    s.base + 1'i32

  proc slotKernel(output: ptr UncheckedArray[int32]) {.global.} =
    var sel: Slot
    sel.base = 0'i32
    output[chooseSlot(sel)] = 42'i32
    output[0] = output[1]

proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_byref_asgn_target.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr byrefAsgnTargetSpv[0], byrefAsgnTargetSpv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(byrefAsgnTargetSpv)
  let oDev = engine.allocDevice(int32, 4)

  engine.launchKernel("slotKernel", oDev, ())
  var got: array[4, int32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 3), oDev)
  doAssert got[1] == 42'i32,
    "the call in the target's index expression picks slot 1, got " & $got[1]
  doAssert got[0] == 42'i32,
    "the copy reads the slot written through the indexed target, got " & $got[0]
  echo "[OK] direct SPIR-V: by-ref calls wrap inside assignment targets"

when isMainModule:
  runTest()
  echo "t_spirv_byref_asgn_target: all checks passed"
