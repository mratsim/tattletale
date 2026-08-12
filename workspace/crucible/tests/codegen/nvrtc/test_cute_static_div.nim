## CuTe: static division for tile sizing (B06)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_static_div.nim
##
## Note: uses separate type names because gtObject struct naming
## doesn't yet differentiate generic params.
import std/strformat
import workspace/crucible

type
  TileA = object
    data: array[64, uint32]   # 256 div 4
  TileB = object
    data: array[8, uint32]    # 256 div 32

const kernelCode = cuda:
  proc divTileKernel(output: ptr UncheckedArray[uint32]) {.global.} =
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
    output[0] = tileA.data[0]
    output[1] = tileA.data[63]

    let perWarp = TileB(data: [100'u32, 101'u32, 102'u32, 103'u32,
                                104'u32, 105'u32, 106'u32, 107'u32])
    output[2] = perWarp.data[0]

var buf: array[3, uint32]
var engine = bkCuda.init()
engine.ingest(kernelCode)
echo "PTX: ", engine.getArtifact().len, " bytes"
engine.run("divTileKernel", buf, ())
doAssert buf[0] == 1,   &"tile[0]: {buf[0]}"
doAssert buf[1] == 64,  &"tile[63]: {buf[1]}"
doAssert buf[2] == 100, &"perWarp[0]: {buf[2]}"
echo "  OK — static division (B06)"
