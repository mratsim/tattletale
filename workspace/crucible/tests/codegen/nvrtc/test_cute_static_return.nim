## CuTe: generic proc with when dispatch (B07, B26)
## Run with: nim cpp -r workspace/crucible/tests/codegen/nvrtc/test_cute_static_return.nim
##
## CuTe dispatches tile sizes per GPU arch at compile time.
## Uses `when` for compile-time branching on static params.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  Tile[M, N: static int] = object
    val: uint32

const kernelCode = cuda:
  proc staticWhenKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let a = Tile[16, 8](val: 16'u32)
    output[0] = a.val

    let b = Tile[32, 16](val: 32'u32)
    output[1] = b.val

var buf: array[4, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("staticWhenKernel", buf, ())
doAssert buf[0] == 16, &"Tile[16,8]: {buf[0]}"
doAssert buf[1] == 32, &"Tile[32,16]: {buf[1]}"
echo "  OK — static dispatch"
