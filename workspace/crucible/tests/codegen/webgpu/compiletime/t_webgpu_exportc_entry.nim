## WebGPU: `{.exportc.}` entry-point name override, end to end.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted WGSL entry and the kernel launches under it,
## and that a bare `{.exportc.}` kernel keeps its proc name and still launches.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/compiletime/t_webgpu_exportc_entry.nim

import std/unittest
import workspace/crucible

const renamedWgsl = webgpu:
  proc renamedKernel(o, a: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    o[0] = a[0] + 1

const bareWgsl = webgpu:
  proc bareKernel(o, a: ptr UncheckedArray[uint32]) {.global, exportc.} =
    o[0] = a[0] + 2

proc runTest() =
  suite "WebGPU - {.exportc.} entry-point name override":

    test "both kernels launch under their exportc entry names":
      var renamedEngine = newEngine(runWebGPU)
      renamedEngine.ingest(renamedWgsl)
      var bareEngine = newEngine(runWebGPU)
      bareEngine.ingest(bareWgsl)

      var a: array[1, uint32] = [41'u32]
      let aDev = renamedEngine.allocDevice(uint32, 1)
      renamedEngine.copyToDeviceFromHost(aDev, a.toOpenArray(0, 1 - 1))
      let oDev = renamedEngine.allocDevice(uint32, 1)
      renamedEngine.launchKernel("my_custom_entry", oDev, (aDev,))
      var res: array[1, uint32]
      renamedEngine.copyToHostFromDevice(res.toOpenArray(0, 1 - 1), oDev)
      check res[0] == 42

      var b: array[1, uint32] = [40'u32]
      let bDev = bareEngine.allocDevice(uint32, 1)
      bareEngine.copyToDeviceFromHost(bDev, b.toOpenArray(0, 1 - 1))
      let oDevBare = bareEngine.allocDevice(uint32, 1)
      bareEngine.launchKernel("bareKernel", oDevBare, (bDev,))
      var resBare: array[1, uint32]
      bareEngine.copyToHostFromDevice(resBare.toOpenArray(0, 1 - 1), oDevBare)
      check resBare[0] == 42

when isMainModule:
  runTest()
