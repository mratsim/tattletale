## Metal: `{.exportc.}` entry-point name override, end to end.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted kernel entry
## and the kernel launches under that name, and that a bare `{.exportc.}`
## kernel keeps its proc name and still launches.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_exportc_entry.nim

import std/[strutils, unittest]
import workspace/crucible

const renamedMsl = metal:
  proc renamedKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    C[0] = A[0] + 1

const bareMsl = metal:
  proc bareKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc.} =
    C[0] = A[0] + 2

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - {.exportc.} entry-point name override":

    test "exportc: \"name\" renames the emitted entry and launches under it":
      check "kernel void my_custom_entry(" in renamedMsl
      check "kernel void renamedKernel(" notin renamedMsl
      var engine = newEngine(runMetal)
      engine.ingest(renamedMsl)
      var a: array[1, uint32] = [41'u32]
      let aDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(aDev, addr a[0], 1)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("my_custom_entry", oDev, (aDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "bare exportc keeps the proc name and launches":
      check "kernel void bareKernel(" in bareMsl
      check "kernel void my_custom_entry(" notin bareMsl
      var engine = newEngine(runMetal)
      engine.ingest(bareMsl)
      var a: array[1, uint32] = [40'u32]
      let aDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(aDev, addr a[0], 1)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("bareKernel", oDev, (aDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

when isMainModule:
  runTest()
