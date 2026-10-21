## WebGPU: address-space tagging of locals and pointer aliases, source-text asserts on the WGSL plus on-device runs.
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/webgpu/serialization/t_webgpu_address_space.nim

import std/[strutils, unittest]
import workspace/crucible

const addressSpaceWgsl = webgpu:
  var scratch {.smem.}: array[4, uint32]

  proc plainLocalKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    var x: uint32 = 7
    var arr: array[4, uint32]
    arr[0] = x
    output[0] = arr[0]

  proc paramAliasKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (1, 1, 1).} =
    var x: uint32 = 7
    let p = output
    output[0] = x
    output[1] = p[0]

  proc smemAliasKernel(output: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 1, 1).} =
    let tid = thread_position_in_threadgroup.x
    scratch[tid] = uint32(tid * 3)
    let p = addr scratch[tid]
    output[tid] = p[]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "WebGPU - address-space tagging":
    test "unannotated locals emit no address-space template and run":
      var engine = bkWebGPU.newEngine()
      engine.ingest(addressSpaceWgsl)
      let src = engine.getArtifact()
      check "<private>" notin src
      check "var x: u32 = 7;" in src
      check "var arr: array<u32, 4>;" in src
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("plainLocalKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 7'u32
    test "let alias of a storage param keeps ptr<storage, ...> and runs":
      var engine = bkWebGPU.newEngine()
      engine.ingest(addressSpaceWgsl)
      let src = engine.getArtifact()
      check "let p: ptr<storage, array<u32>, read_write> = (&output);" in src
      var res: array[2, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("paramAliasKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 7'u32
      check res[1] == 7'u32
    test "let alias of a smem var keeps ptr<workgroup, ...> and runs":
      var engine = bkWebGPU.newEngine()
      engine.ingest(addressSpaceWgsl)
      let src = engine.getArtifact()
      check "var<workgroup> scratch: array<u32, 4>;" in src
      check "let p: ptr<workgroup, u32> = (&scratch[" in src
      var res: array[4, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("smemAliasKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      for i in 0 ..< 4:
        check res[i] == uint32(i * 3)

when isMainModule:
  runTest()
