## NVRTC: address-space emission for locals.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/serialization/t_nvrtc_address_space.nim
import std/[strutils, unittest]
import workspace/crucible

const addressSpaceCuda = cuda:
  proc addressSpaceKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var x: uint32 = 7
    var scratch {.smem.}: array[8, uint32]
    scratch[0] = x
    output[0] = scratch[0]

proc runTest() =   # private: tests run in a proc so engines are destroyed at return
  suite "NVRTC - address-space emission for locals":
    test "unannotated locals emit no qualifier, smem keeps __shared__, and run":
      var engine = bkCuda.newEngine()
      engine.ingest(addressSpaceCuda)
      let src = engine.getArtifact()
      check "__local__" notin src
      check "unsigned int x = 7;" in src
      check "__shared__ unsigned int scratch[8];" in src
      var res: array[1, uint32]
      let oDev = engine.allocDevice(uint32, res.len)
      engine.launchKernel("addressSpaceKernel", oDev, ())
      engine.copyToHostFromDevice(addr res[0], oDev, res.len)
      check res[0] == 7'u32

when isMainModule:
  runTest()
