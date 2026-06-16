## NVRTC: generic type fields (nnkBracketExpr in type context)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_nvrtc_bracket_in_tuple.nim
##
## Coverage: nim_to_gpu.nim:507-519 (nnkBracketExpr field handling)
##
## Important for CuTe: generic types like `Vec2[float32]` for
## different precision layouts, and `MyInt[N]` for type-level ints.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  MyInt[V: static int] = object
    data: array[V, uint32]

  Vec2[T] = object
    x, y: T

const kernelCode = cuda:
  proc genericFieldKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # MyInt[N] — generic with static int array (nnkBracketExpr in type)
    let a = MyInt[4](data: [10'u32, 20'u32, 30'u32, 40'u32])
    output[0] = a.data[0]
    output[1] = a.data[3]

    # Vec2[uint32] — generic with primitive type arg (u32 not unsigned int)
    let v = Vec2[uint32](x: 5'u32, y: 6'u32)
    output[2] = v.x + v.y

var buf: array[3, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("genericFieldKernel", buf, ())
doAssert buf[0] == 10, &"data[0]: got {buf[0]}"
doAssert buf[1] == 40, &"data[3]: got {buf[1]}"
doAssert buf[2] == 11, &"Vec2[u32]: got {buf[2]}"
echo "  OK"
