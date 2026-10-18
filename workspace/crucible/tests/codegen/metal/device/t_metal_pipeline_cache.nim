## Metal: engine pipeline cache (ingest-once / run-many). Re-ingesting a changed
## source invalidates the compiled library (one per ingest) and the pipeline
## states (one per kernel name).
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/device/t_metal_pipeline_cache.nim


import workspace/crucible

import std/unittest

# ── Kernels: elementwise scaling of a 4-element uint32 buffer ─────────────────

const kernelDouble = metal:
  proc doubleKernel(output: ptr UncheckedArray[uint32];
                    input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 2'u32
    output[1] = input[1] * 2'u32
    output[2] = input[2] * 2'u32
    output[3] = input[3] * 2'u32

const kernelTriple = metal:
  proc doubleKernel(output: ptr UncheckedArray[uint32];
                    input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 3'u32
    output[1] = input[1] * 3'u32
    output[2] = input[2] * 3'u32
    output[3] = input[3] * 3'u32

# ── Host side ─────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "Metal - pipeline cache":

    test "same kernel, different input arrays (byte-exact outputs)":
      var engine = bkMetal.init()
      engine.ingest(kernelDouble)
      echo kernelDouble
      var a = [1'u32, 2'u32, 3'u32, 4'u32]
      let aDev = engine.allocDevice(uint32, a.len)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      let oDev = engine.allocDevice(uint32, 4)
      var res: array[4, uint32]
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 2
      check res[1] == 4
      check res[2] == 6
      check res[3] == 8
      var b = [10'u32, 20'u32, 30'u32, 40'u32]
      engine.copyToDeviceFromHost(aDev, addr b[0], b.len)
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 20
      check res[1] == 40
      check res[2] == 60
      check res[3] == 80
      engine.freeDevice(aDev)
      engine.freeDevice(oDev)

    test "re-ingest invalidates cached pipelines":
      var engine = bkMetal.init()
      engine.ingest(kernelDouble)
      echo kernelDouble
      var a = [1'u32, 2'u32, 3'u32, 4'u32]
      let aDev = engine.allocDevice(uint32, a.len)
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len)
      let oDev = engine.allocDevice(uint32, 4)
      var res: array[4, uint32]
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 2
      check res[1] == 4
      engine.ingest(kernelTriple)
      echo kernelTriple
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, 4)
      check res[0] == 3
      check res[1] == 6
      check res[2] == 9
      check res[3] == 12
      engine.freeDevice(aDev)
      engine.freeDevice(oDev)

when isMainModule:
  runTest()
