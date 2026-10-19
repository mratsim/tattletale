## CuTe: generic proc returning a composed generic type (B10)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/app/cute/t_nvrtc_cute_tuple_return.nim
import std/strformat
import workspace/crucible

const kernelCode = cuda:
  type
    Tile[N: static int] = object
      data: array[N, uint32]

  proc makeTile[N: static int](val: uint32): Tile[N] =
    for i in 0 .. N-1:
      result.data[i] = val

  proc pairReturnKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let a = makeTile[4](42'u32)
    output[0] = a.data[0]
    output[1] = a.data[3]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("pairReturnKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  doAssert buf[0] == 42, &"a[0]: {buf[0]}"
  doAssert buf[1] == 42, &"a[3]: {buf[1]}"
  echo "  OK — generic return type (B10)"

when isMainModule:
  runTest()
