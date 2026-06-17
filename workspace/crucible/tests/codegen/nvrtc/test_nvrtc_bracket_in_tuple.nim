## NVRTC: tuple with generic types (key CuTe pattern)
## Run with: nim cpp -d:cuda -r workspace/crucible/tests/nvrtc/test_nvrtc_bracket_in_tuple.nim
##
## Coverage: nim_to_gpu.nim:507-519 — nnkBracketExpr + else branch
## in parseTypeFields for tuple constructors.
##
## Important for CuTe: shapes, strides, and coordinates are tuples
## of generic types like `(Int<M>, Int<N>)` or `(float32, float32)`.
import std/strformat
import workspace/crucible/src/codegen/nvrtc

type
  MyInt[V: static int] = object
    data: array[V, uint32]

  Vec2[T] = object
    x, y: T

const kernelCode = cuda:
  proc tupleGenericKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    # Tuple with generic type construction (nnkObjConstr inside tuple)
    let t = (MyInt[4](data: [10'u32, 20'u32, 30'u32, 40'u32]), 99'u32)
    output[0] = t[0].data[0]
    output[1] = t[0].data[3]
    output[2] = t[1]
    # Vec2[uint32] — generic with primitive type arg
    let v = Vec2[uint32](x: 5'u32, y: 6'u32)
    output[3] = v.x + v.y

var buf: array[4, uint32]
var nv = initNvrtc(kernelCode)
nv.compile()
nv.getPtx()
echo "PTX: ", nv.ptx.len, " bytes"
nv.execute("tupleGenericKernel", buf, ())
doAssert buf[0] == 10, &"tuple[0].data[0]: got {buf[0]}"
doAssert buf[1] == 40, &"tuple[0].data[3]: got {buf[1]}"
doAssert buf[2] == 99, &"tuple[1]: got {buf[2]}"
doAssert buf[3] == 11, &"Vec2[u32]: got {buf[3]}"
echo "  OK"
