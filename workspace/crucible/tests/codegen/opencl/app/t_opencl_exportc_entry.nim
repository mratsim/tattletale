## OpenCL: `{.exportc.}` entry-point name override, end to end.
##
## Test that `{.exportc: "my_custom_entry".}` renames the emitted OpenCL kernel
## entry and the kernel launches under that name, and that a bare `{.exportc.}`
## kernel keeps its proc name and still launches.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/opencl/app/t_opencl_exportc_entry.nim

import std/unittest
import workspace/crucible

const renamedCl = opencl:
  proc renamedKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc: "my_custom_entry".} =
    C[0] = A[0] + 1

const bareCl = opencl:
  proc bareKernel(C, A: ptr UncheckedArray[uint32]) {.global, exportc.} =
    C[0] = A[0] + 2

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "OpenCL - {.exportc.} entry-point name override":

    test "exportc: \"name\" launches the kernel under the override name":
      var engine = newEngine(runOpenCL)
      engine.ingest(renamedCl)
      var a: array[1, uint32] = [41'u32]
      let aDev = engine.allocDevice(uint32, 1)
      engine.copyToDeviceFromHost(aDev, addr a[0], 1)
      let oDev = engine.allocDevice(uint32, 1)
      engine.launchKernel("my_custom_entry", oDev, (aDev,))
      var res: array[1, uint32]
      engine.copyToHostFromDevice(addr res[0], oDev, 1)
      check res[0] == 42

    test "bare exportc keeps the proc name and launches":
      var engine = newEngine(runOpenCL)
      engine.ingest(bareCl)
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
