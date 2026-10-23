## NVRTC: generic instantiation shapes inside a `cuda:` block:
##   - parseGenericImpl over ObjectTy and StaticTy params
##   - static-array member fallback in initGpuGenericInst
##   - empty CuTe-style Int[N] structs with nested static-dim layouts
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/cuda/generics/t_cuda_generic_parse_impl.nim
import std/strformat
import workspace/crucible

type
  Vec2[T] = object
    ## ObjectTy: generic object with fields
    x, y: T

  Sized[N: static int] = object
    ## StaticTy: static int generic param
    val: uint32

  SizedArr[N: static int] = object
    ## StaticTy param sizing an array member via initGpuGenericInst fallback
    data: array[N, uint32]

  MyInt*[V: static int] = object
    ## Empty struct, CuTe style compile-time int wrapper
    discard

  Layout2D*[M, N: static int] = object
    ## CuTe-like 2D layout with compile-time known dimensions
    data: array[M * N, float32]

  MicroKernel*[TileM, TileN: static int] = object
    ## Config object like gemm_tiling.nim's MicroKernel
    tile: Layout2D[TileM, TileN]

const kernelCode = cuda:
  proc genericImplKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # ObjectTy generic
    let v = Vec2[uint32](x: 10'u32, y: 20'u32)
    output[0] = v.x + v.y
    # StaticTy generic
    let s = Sized[64](val: 42'u32)
    output[1] = s.val

  proc sizedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = SizedArr[4](data: [1'u32, 2'u32, 3'u32, 4'u32])
    output[0] = x.data[0]
    output[1] = x.data[3]

  proc testIntGeneric(output: ptr UncheckedArray[uint32]) {.global.} =
    let x = MyInt[32]()
    let y = MyInt[64]()
    output[0] = 1'u32

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var bufImpl: array[2, uint32]
  var bufFallback: array[2, uint32]
  var bufInt: array[1, uint32]
  var engine = newEngine(runCuda)
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDevImpl = engine.allocDevice(uint32, bufImpl.len)
  let oDevFallback = engine.allocDevice(uint32, bufFallback.len)
  let oDevInt = engine.allocDevice(uint32, bufInt.len)
  engine.launchKernel("genericImplKernel", oDevImpl, ())
  engine.launchKernel("sizedKernel", oDevFallback, ())
  engine.launchKernel("testIntGeneric", oDevInt, ())
  engine.copyToHostFromDevice(addr bufImpl[0], oDevImpl, bufImpl.len)
  engine.copyToHostFromDevice(addr bufFallback[0], oDevFallback, bufFallback.len)
  engine.copyToHostFromDevice(addr bufInt[0], oDevInt, bufInt.len)
  doAssert bufImpl[0] == 30, &"objectTy: got {bufImpl[0]}"
  doAssert bufImpl[1] == 42, &"staticTy: got {bufImpl[1]}"
  doAssert bufFallback[0] == 1, &"SizedArr[4].data[0]: got {bufFallback[0]}, expected 1"
  doAssert bufFallback[1] == 4, &"SizedArr[4].data[3]: got {bufFallback[1]}, expected 4"
  doAssert bufInt[0] == 1
  echo "  OK (test_cuda_generic_parse_impl, test_cuda_generic_non_generic_fallback, test_cuda_int_generic)"

when isMainModule:
  runTest()
