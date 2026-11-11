# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Direct SPIR-V: nested indexing through a by-ref array parameter
## Test that a device function reads `m[i][j]` of a 2-D by-ref array
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_spirv/app/t_spirv_byref_array_2d.nim

import std/[os, osproc]
import workspace/crucible

const byrefArray2dSpv = spirv:
  proc matAt(m: array[4, array[8, float32]], row, col: int): float32 {.device.} =
    m[row][col] * 2.0'f32

  proc array2dKernel(output: ptr UncheckedArray[float32]) {.global.} =
    var buf: array[4, array[8, float32]]
    for i in 0 ..< 4:
      for j in 0 ..< 8:
        buf[i][j] = float32(i * 8 + j + 1)
    output[0] = matAt(buf, 2, 3)
    output[1] = matAt(buf, 0, 0)

proc runTest() =
  let spirvVal = findExe("spirv-val")
  doAssert spirvVal.len > 0,
    "spirv-val not found on PATH, install the SPIRV-Tools (brew install spirv-tools)"
  let outDir = "build/tests"
  createDir(outDir)
  let spvPath = outDir / "t_spirv_byref_array_2d.spv"
  block:
    var f = open(spvPath, fmWrite)
    defer: f.close()
    discard f.writeBuffer(addr byrefArray2dSpv[0], byrefArray2dSpv.len * 4)
  let (valOut, valCode) = execCmdEx(spirvVal & " --target-env vulkan1.4 " & spvPath)
  doAssert valCode == 0, "spirv-val rejected the executed module:\n" & valOut

  var engine = newEngine(runVulkan)
  engine.ingest(byrefArray2dSpv)
  let oDev = engine.allocDevice(float32, 2)
  var zeros: array[2, float32]
  engine.copyToDeviceFromHost(oDev, zeros)

  engine.launchKernel("array2dKernel", oDev, ())
  var got: array[2, float32]
  engine.copyToHostFromDevice(got.toOpenArray(0, 1), oDev)
  # buf (2, 3) is 20, buf (0, 0) is 1, the device fn doubles both reads
  doAssert got[0] == 40.0'f32,
    "the nested read reaches the by-ref array's cell (2, 3), got " & $got[0]
  doAssert got[1] == 2.0'f32,
    "the nested read of the untouched cell (0, 0) doubles it, got " & $got[1]
  echo "[OK] direct SPIR-V: nested indexing through a by-ref array parameter"

when isMainModule:
  runTest()
  echo "t_spirv_byref_array_2d: all checks passed"
