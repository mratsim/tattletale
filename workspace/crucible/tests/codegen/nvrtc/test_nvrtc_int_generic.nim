## NVRTC: test Int[N] generic struct inside cuda: block
## Run: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_nvrtc_int_generic.nim
import workspace/crucible/src/codegen/nvrtc

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

var buf: array[1, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("testIntGeneric", buf, ())
doAssert buf[0] == 1
echo "  OK"
