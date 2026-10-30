## HLSL: `{.exportc.}` entry-point name override, end to end.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted HLSL kernel
## entry and dxc compiles and launches the kernel under that name, and that a bare
## `{.exportc.}` kernel keeps the HLSL entry `main`.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/vulkan_via_hlsl/app/t_hlsl_exportc_entry.nim

import std/strutils
import workspace/crucible

const renamedHlsl = hlsl:
  proc renamedKernel(output, a: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    output[0] = a[0] + 1

const bareHlsl = hlsl:
  proc bareKernel(output, a: ptr UncheckedArray[uint32]) {.global, exportc.} =
    output[0] = a[0] + 2

static:
  # compile-time signals: a rename regression fails compilation here
  doAssert "void my_custom_entry(" in renamedHlsl
  doAssert "void renamedKernel(" notin renamedHlsl
  # HLSL names the bare entry `main`, its built-in entry-point name
  doAssert "void main(" in bareHlsl
  doAssert "void my_custom_entry(" notin bareHlsl

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  # Renamed HLSL compiles through dxc and launches under the exportc name
  var engine = newEngine(runVulkan)
  engine.ingest(renamedHlsl)
  var a: array[1, uint32] = [41'u32]
  let aDev = engine.allocDevice(uint32, 1)
  engine.copyToDeviceFromHost(aDev, addr a[0], 1)
  let oDev = engine.allocDevice(uint32, 1)
  engine.launchKernel("my_custom_entry", oDev, (aDev,))
  var res: array[1, uint32]
  engine.copyToHostFromDevice(addr res[0], oDev, 1)
  doAssert res[0] == 42, "renamed entry got " & $res[0] & ", want 42"

  # Bare HLSL keeps `main` as the entry
  var bareEngine = newEngine(runVulkan)
  bareEngine.ingest(bareHlsl)
  var b: array[1, uint32] = [40'u32]
  let bDev = bareEngine.allocDevice(uint32, 1)
  bareEngine.copyToDeviceFromHost(bDev, addr b[0], 1)
  let bOut = bareEngine.allocDevice(uint32, 1)
  bareEngine.launchKernel("main", bOut, (bDev,))
  var bareRes: array[1, uint32]
  bareEngine.copyToHostFromDevice(addr bareRes[0], bOut, 1)
  doAssert bareRes[0] == 42, "bare entry got " & $bareRes[0] & ", want 42"
  echo "  HLSL exportc entry: renamed and bare kernels exact on MoltenVK"

when isMainModule:
  runTest()
  echo "t_hlsl_exportc_entry: all checks passed"
