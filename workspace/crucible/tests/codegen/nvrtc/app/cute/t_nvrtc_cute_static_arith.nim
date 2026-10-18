## CuTe scaling: static-int arithmetic, static-param passthrough, large tile arrays
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/app/cute/t_nvrtc_cute_static_arith.nim
import std/strformat
import workspace/crucible

type
  # B01: static addition
  Summed[M, N: static int] = object
    data: array[M + N, uint32]

  # B02: static multiplication
  Multiplied[M, N: static int] = object
    data: array[M * N, uint32]

  # B03: complex (multiplication + addition)
  Complex[M, N, K: static int] = object
    data: array[M * N + K, uint32]

  # Static-param instantiation passthrough
  Tile[M, N: static int] = object
    val: uint32

  # Hardcoded tile arrays, large array literal compile + first/last read
  TileA = object
    data: array[64, uint32]   # 256 div 4
  TileB = object
    data: array[8, uint32]    # 256 div 32

const kernelCode = cuda:
  proc staticArithKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # B01: M + N = 3 + 2 = 5 elements
    let a = Summed[3, 2](data: [1'u32, 2'u32, 3'u32, 4'u32, 5'u32])
    output[0] = a.data[0]
    output[1] = a.data[4]

    # B02: M * N = 2 * 3 = 6 elements
    let b = Multiplied[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[2] = b.data[0]
    output[3] = b.data[5]

    # B03: M * N + K = 2*3 + 1 = 7 elements
    let c = Complex[2, 3, 1](data: [100'u32, 200'u32, 300'u32, 400'u32, 500'u32, 600'u32, 700'u32])
    output[4] = c.data[0]
    output[5] = c.data[6]

    # Static-param instantiation passthrough
    let p = Tile[16, 8](val: 16'u32)
    output[6] = p.val
    let q = Tile[32, 16](val: 32'u32)
    output[7] = q.val

    # Hardcoded tile arrays
    let tileA = TileA(data: [1'u32, 2'u32, 3'u32, 4'u32,
                              5'u32, 6'u32, 7'u32, 8'u32,
                              9'u32, 10'u32, 11'u32, 12'u32,
                              13'u32, 14'u32, 15'u32, 16'u32,
                              17'u32, 18'u32, 19'u32, 20'u32,
                              21'u32, 22'u32, 23'u32, 24'u32,
                              25'u32, 26'u32, 27'u32, 28'u32,
                              29'u32, 30'u32, 31'u32, 32'u32,
                              33'u32, 34'u32, 35'u32, 36'u32,
                              37'u32, 38'u32, 39'u32, 40'u32,
                              41'u32, 42'u32, 43'u32, 44'u32,
                              45'u32, 46'u32, 47'u32, 48'u32,
                              49'u32, 50'u32, 51'u32, 52'u32,
                              53'u32, 54'u32, 55'u32, 56'u32,
                              57'u32, 58'u32, 59'u32, 60'u32,
                              61'u32, 62'u32, 63'u32, 64'u32])
    output[8] = tileA.data[0]
    output[9] = tileA.data[63]
    let perWarp = TileB(data: [100'u32, 101'u32, 102'u32, 103'u32,
                                104'u32, 105'u32, 106'u32, 107'u32])
    output[10] = perWarp.data[0]

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  var buf: array[11, uint32]
  var engine = bkCuda.init()
  engine.ingest(kernelCode)
  echo "PTX: ", engine.getArtifact().len, " bytes"
  let oDev = engine.allocDevice(uint32, buf.len)
  engine.launchKernel("staticArithKernel", oDev, ())
  engine.copyToHostFromDevice(addr buf[0], oDev, buf.len)
  engine.freeDevice(oDev)
  doAssert buf[0] == 1,   &"summed[0]: {buf[0]}"
  doAssert buf[1] == 5,   &"summed[4]: {buf[1]}"
  doAssert buf[2] == 10,  &"mult[0]: {buf[2]}"
  doAssert buf[3] == 60,  &"mult[5]: {buf[3]}"
  doAssert buf[4] == 100, &"complex[0]: {buf[4]}"
  doAssert buf[5] == 700, &"complex[6]: {buf[5]}"
  doAssert buf[6] == 16,  &"Tile[16,8]: {buf[6]}"
  doAssert buf[7] == 32,  &"Tile[32,16]: {buf[7]}"
  doAssert buf[8] == 1,   &"tile[0]: {buf[8]}"
  doAssert buf[9] == 64,  &"tile[63]: {buf[9]}"
  doAssert buf[10] == 100, &"perWarp[0]: {buf[10]}"
  echo "  OK — static-int arithmetic (+, *, +*)"

when isMainModule:
  runTest()
