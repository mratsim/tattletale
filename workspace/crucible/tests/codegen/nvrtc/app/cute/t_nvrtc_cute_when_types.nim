## CuTe: when branches producing different types (B26)
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/app/cute/t_nvrtc_cute_when_types.nim
import std/strformat
import workspace/crucible

type
  TileSmall = object
    val: uint32
  TileLarge = object
    val: uint32
    extra: array[4, uint32]

const kernelCode = cuda:
  proc whenTypesKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Use `when` directly in kernel body — it's resolved by Nim before Crucible.
    when 8 <= 16:
      let t = TileSmall(val: 42'u32)
      output[0] = t.val
    else:
      let t = TileLarge(val: 42'u32, extra: [0'u32, 0'u32, 0'u32, 0'u32])
      output[0] = t.val + 1'u32

    when 32 <= 16:
      let t2 = TileSmall(val: 42'u32)
      output[1] = t2.val
    else:
      let t2 = TileLarge(val: 42'u32, extra: [0'u32, 0'u32, 0'u32, 0'u32])
      output[1] = t2.val + 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[2, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel("whenTypesKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 42, &"small tile: {buf[0]}"
  doAssert buf[1] == 43, &"large tile: {buf[1]}"
  echo "  OK — when dispatch types (B26)"

when isMainModule:
  runTest()
