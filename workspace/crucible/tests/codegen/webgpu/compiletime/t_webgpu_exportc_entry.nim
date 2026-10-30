## WebGPU: `{.exportc.}` entry-point name override, end to end.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted WGSL entry and the kernel launches under it,
## and that a bare `{.exportc.}` kernel keeps its proc name and still launches.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/compiletime/t_webgpu_exportc_entry.nim

import std/[strutils, unittest]
import workspace/crucible

const renamedWgsl = webgpu:
  proc renamedKernel(o, a: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    o[0] = a[0] + 1

const bareWgsl = webgpu:
  proc bareKernel(o, a: ptr UncheckedArray[uint32]) {.global, exportc.} =
    o[0] = a[0] + 2

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - {.exportc.} entry-point name override":

    test "both kernels launch under their exportc entry names":
      # Compile-time signals: a rename regression fails compilation here.
      check "fn my_custom_entry(" in renamedWgsl
      check "fn renamedKernel(" notin renamedWgsl
      check "fn bareKernel(" in bareWgsl
      check "fn my_custom_entry(" notin bareWgsl
      var renamedEngine = newEngine(runWebGPU)
      renamedEngine.ingest(renamedWgsl)
      var bareEngine = newEngine(runWebGPU)
      bareEngine.ingest(bareWgsl)

      var a: array[1, uint32] = [41'u32]
      let aDev = renamedEngine.allocDevice(uint32, 1)
      renamedEngine.copyToDeviceFromHost(aDev, addr a[0], 1)
      let oDev = renamedEngine.allocDevice(uint32, 1)
      renamedEngine.launchKernel("my_custom_entry", oDev, (aDev,))
      var res: array[1, uint32]
      renamedEngine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

      var b: array[1, uint32] = [40'u32]
      let bDev = bareEngine.allocDevice(uint32, 1)
      bareEngine.copyToDeviceFromHost(bDev, addr b[0], 1)
      let oDevBare = bareEngine.allocDevice(uint32, 1)
      bareEngine.launchKernel("bareKernel", oDevBare, (bDev,))
      var resBare: array[1, uint32]
      bareEngine.copyToHostFromDevice(addr resBare[0], oDevBare, 1)
      check resBare[0] == 42

when isMainModule:
  runTest()
