## SPIR-V direct: the address route survives engine route autodetection.
##
## Test that a direct `spirv:` module keeps the address route:
## engine autodetection survives, readback is exact:
##
## - bool kernel parameters lower to i32 push-constant members,
##   nonzero reads true, values check on MoltenVK
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_route_address.nim

import std/[os, osproc, strutils]
import workspace/crucible

const addSpv = spirv:
  proc addScalarKernel(output, a, b: ptr UncheckedArray[uint32], z: uint32) {.global.} =
    output[0] = a[0] + b[0] + z
    output[1] = a[1] + b[1] + z

const flagSpv = spirv:
  proc flagKernel(output: ptr UncheckedArray[uint32], flag: bool) {.global.} =
    if flag:
      output[0] = 7'u32
    else:
      output[0] = 3'u32

proc disassembly(spv: seq[uint32], name: string): string =
  # spirv-val the module before it reaches the engine: a mis-encoded word
  # sequence must fail here, not as a driver-side opaque launch error
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / (name & ".spv")
  # the file closes at block exit, before the subprocesses read it
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr spv[0], spv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut
  let spirvDis = findExe("spirv-dis")
  doAssert spirvDis.len > 0,
    "spirv-dis not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let (disOut, disCode) = execCmdEx(spirvDis & " " & spvPath)
  doAssert disCode == 0, "spirv-dis failed on " & name
  disOut

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # spirv-val the module before it reaches the engine: a mis-encoded word
  # sequence must fail here, not as a driver-side opaque launch error
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_route_address.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr addSpv[0], addSpv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(addSpv)

  var a: array[2, uint32] = [10'u32, 20'u32]
  var b: array[2, uint32] = [1'u32, 2'u32]
  let aDev = engine.allocDevice(uint32, a.len)
  let bDev = engine.allocDevice(uint32, b.len)
  let oDev = engine.allocDevice(uint32, 2)
  engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
  engine.copyToDeviceFromHost(bDev, addr b[0], b.len)

  engine.launchKernel("addScalarKernel", oDev, (aDev, bDev, 5'u32))

  var out32: array[2, uint32]
  engine.copyToHostFromDevice(addr out32[0], oDev, 2)
  doAssert out32[0] == 16, "slot 0 got " & $out32[0] & ", want 10 + 1 + 5 = 16"
  doAssert out32[1] == 27, "slot 1 got " & $out32[1] & ", want 20 + 2 + 5 = 27"
  echo "  address route over direct SPIR-V: [", out32[0], ", ", out32[1], "] exact readback"

  block boolParam:
    # the push-constant member is i32: the struct behind the PushConstant
    # pointer is the module push block. Its bool member is OpTypeInt 32 1
    # and never OpTypeBool.
    let disOut = disassembly(flagSpv, "t_spirv_route_address_bool")
    var pushStruct = ""
    for line in disOut.splitLines():
      if "OpTypePointer PushConstant" in line:
        let target = line.splitWhitespace()[^1]
        if disOut.contains(target & " = OpTypeStruct"):
          pushStruct = target
    doAssert pushStruct.len > 0, "no push-constant block struct in the disassembly"
    var sawStruct = false
    for line in disOut.splitLines():
      if pushStruct & " = OpTypeStruct" in line:
        sawStruct = true
        doAssert "%int" in line, "bool push-constant member is not i32: " & line
        doAssert "%bool" notin line, "OpTypeBool leaked into the push-constant block: " & line
    doAssert sawStruct, "push-constant struct definition not found"
    doAssert "OpINotEqual" in disOut, "the bool param load does not convert via i32 != 0"

    var engine = newEngine(runVulkan)
    engine.ingest(flagSpv)
    let oDev = engine.allocDevice(uint32, 1)

    engine.launchKernel("flagKernel", oDev, (true,))
    var outFlag: array[1, uint32]
    engine.copyToHostFromDevice(addr outFlag[0], oDev, 1)
    doAssert outFlag[0] == 7, "flag=true got " & $outFlag[0] & ", want 7"

    engine.launchKernel("flagKernel", oDev, (false,))
    engine.copyToHostFromDevice(addr outFlag[0], oDev, 1)
    doAssert outFlag[0] == 3, "flag=false got " & $outFlag[0] & ", want 3"
    echo "  bool kernel param: i32 push member, truthy [", outFlag[0], "] on MoltenVK"

when isMainModule:
  runTest()
  echo "t_spirv_route_address: all checks passed"
