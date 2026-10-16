## WebGPU pipeline cache (ingest-once / run-many): the pipeline artifacts survive repeated
## runs with the same (kernel, arg-shape) key and are invalidated by re-ingest.
## Pipeline creation pops a validation error scope around wgpuDeviceCreateComputePipeline,
## so a cache miss (first creation) must not hang the callback.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/device/t_webgpu_pipeline_cache.nim

import std/[unittest, strformat]
import workspace/crucible

# ── Kernels: elementwise scaling of a 4-element uint32 buffer ─────────────────

const kernelDouble = webgpu:
  proc doubleKernel(output: ptr UncheckedArray[uint32];
                    input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 2'u32
    output[1] = input[1] * 2'u32
    output[2] = input[2] * 2'u32
    output[3] = input[3] * 2'u32

const kernelTriple = webgpu:
  proc tripleKernel(output: ptr UncheckedArray[uint32];
                    input: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = input[0] * 3'u32
    output[1] = input[1] * 3'u32
    output[2] = input[2] * 3'u32
    output[3] = input[3] * 3'u32

# ── Host side ─────────────────────────────────────────────────────────────────

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - pipeline cache":

    test "same kernel, different input arrays (cache reuse)":
      var engine = bkWGSL.init()
      engine.ingest(kernelDouble)
      echo kernelDouble
      var res: array[4, uint32]
      let oDev = engine.allocDevice(res.len * sizeof(uint32))
      var a = [1'u32, 2'u32, 3'u32, 4'u32]
      let aDev = engine.allocDevice(a.len * sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, res.len * sizeof(uint32))
      check res[0] == 2
      check res[1] == 4
      check res[2] == 6
      check res[3] == 8
      var b = [10'u32, 20'u32, 30'u32, 40'u32]
      engine.copyToDeviceFromHost(aDev, addr b[0], b.len * sizeof(uint32))
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, res.len * sizeof(uint32))
      engine.freeDevice(aDev)
      engine.freeDevice(oDev)
      check res[0] == 20
      check res[1] == 40
      check res[2] == 60
      check res[3] == 80

    test "re-ingest invalidates cached pipelines":
      var engine = bkWGSL.init()
      engine.ingest(kernelDouble)
      echo kernelDouble
      var res: array[4, uint32]
      let oDev = engine.allocDevice(res.len * sizeof(uint32))
      var a = [1'u32, 2'u32, 3'u32, 4'u32]
      let aDev = engine.allocDevice(a.len * sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, res.len * sizeof(uint32))
      check res[0] == 2
      check res[1] == 4
      engine.ingest(kernelTriple)
      echo kernelTriple
      engine.launchKernel("tripleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, res.len * sizeof(uint32))
      engine.freeDevice(aDev)
      engine.freeDevice(oDev)
      check res[0] == 3
      check res[1] == 6
      check res[2] == 9
      check res[3] == 12

    test "first pipeline creation is a cache miss and must not hang the error-scope callback":
      # Pipeline creation pops a validation error scope around
      # wgpuDeviceCreateComputePipeline, so a fresh engine exercises the miss path.
      var engine = bkWGSL.init()
      engine.ingest(kernelDouble)
      var res: array[4, uint32]
      let oDev = engine.allocDevice(res.len * sizeof(uint32))
      var a = [1'u32, 2'u32, 3'u32, 4'u32]
      let aDev = engine.allocDevice(a.len * sizeof(uint32))
      engine.copyToDeviceFromHost(aDev, addr a[0], a.len * sizeof(uint32))
      engine.launchKernel("doubleKernel", oDev, (aDev,))
      engine.copyToHostFromDevice(addr res[0], oDev, res.len * sizeof(uint32))
      engine.freeDevice(aDev)
      engine.freeDevice(oDev)
      check res == [2'u32, 4, 6, 8]

when isMainModule:
  runTest()
