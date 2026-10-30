## SPIR-V binary emission: `{.exportc.}` entry-point name override, end to end.
##
## Test that `{.exportc: "my_custom_entry".}` renames the OpEntryPoint Name
## operand of the emitted module, that the engine launches the renamed kernel
## under that name, and that a bare `{.exportc.}` kernel keeps its proc name.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_exportc_entry.nim

import std/[os, osproc, strutils]
import workspace/crucible

const renamedSpv = spirv:
  proc renamedKernel(o: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    o[thread_position_in_grid.x] = 42'u32

const bareSpv = spirv:
  proc bareKernel(o: ptr UncheckedArray[uint32]) {.global, exportc.} =
    o[thread_position_in_grid.x] = 43'u32

proc entryPointLine(spv: seq[uint32], name: string): string =
  # spirv-val the module before it reaches the engine: a mis-encoded word
  # sequence must fail here, not as a driver-side opaque launch error
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / (name & ".spv")
  # File handle closes at block exit, before the subprocesses read it
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr spv[0], spv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the emitted module:\n" & valOut
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on " & name
  for line in disOut.splitLines():
    if "OpEntryPoint" in line:
      return line
  doAssert false, "no OpEntryPoint in the " & name & " disassembly"

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  block renamedEntry:
    # OpEntryPoint Name is the operand the host launcher looks up
    let entry = entryPointLine(renamedSpv, "t_spirv_exportc_entry_renamed")
    doAssert "\"my_custom_entry\"" in entry,
      "OpEntryPoint does not carry the exportc name, got: " & entry
    doAssert "renamedKernel" notin entry,
      "OpEntryPoint still carries the proc name: " & entry

    var engine = newEngine(runVulkan)
    engine.ingest(renamedSpv)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("my_custom_entry", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    doAssert res[0] == 42, "renamed entry got " & $res[0] & ", want 42"
    echo "  renamed entry: OpEntryPoint \"my_custom_entry\", launch exact on MoltenVK"

  block bareEntry:
    let entry = entryPointLine(bareSpv, "t_spirv_exportc_entry_bare")
    doAssert "\"bareKernel\"" in entry,
      "OpEntryPoint does not carry the proc name, got: " & entry
    doAssert "\"my_custom_entry\"" notin entry,
      "OpEntryPoint carries the exportc name on a bare kernel: " & entry

    var engine = newEngine(runVulkan)
    engine.ingest(bareSpv)
    let oDev = engine.allocDevice(uint32, 1)
    engine.launchKernel("bareKernel", oDev, ())
    var res: array[1, uint32]
    engine.copyToHostFromDevice(addr res[0], oDev, 1)
    doAssert res[0] == 43, "bare entry got " & $res[0] & ", want 43"
    echo "  bare entry: OpEntryPoint \"bareKernel\", launch exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_exportc_entry: all checks passed"
