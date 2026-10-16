## NVRTC: Int[N] generic structs instantiate inside a `cuda:` block.
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/generics/t_nvrtc_int_generic.nim
import workspace/crucible

type
  MyInt*[V: static int] = object
    ## Empty struct — CuTe style compile-time int wrapper

  Layout2D*[M, N: static int] = object
    ## A CuTe-like 2D layout with compile-time known dimensions
    data: array[M * N, float32]

  MicroKernel*[TileM, TileN: static int] = object
    ## A config object like gemm_tiling.nim's MicroKernel
    tile: Layout2D[TileM, TileN]

const kernelCode = cuda:
  proc testIntGeneric(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = MyInt[32]()
    let y = MyInt[64]()
    output[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[1, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(sizeof(buf))
  engine.launchKernel("testIntGeneric", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, sizeof(buf))
  engine.freeDevice(oDev)
  doAssert buf[0] == 1
  echo "  OK (test_nvrtc_int_generic)"

when isMainModule:
  runTest()
