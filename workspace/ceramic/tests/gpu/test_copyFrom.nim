## copyFrom / size() inside cuda: — narrow down the AST pattern
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/ceramic/tests/gpu/test_copyFrom.nim

import std/[unittest]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/tensor_datatypes
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernel_copy_gpu

# All static — should work
const test1 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((Int[8](), Int[16]()))
    var tv = make_view(C, L)
    let sz = size(tv)
    for i in 0 ..< sz: discard

# shape with runtime component from kernel param
const test2 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]; N: int32) {.global.} =
    let sh = (Int[8](), Int[16](), int(N))
    let st = (Int[1](), Int[8](), int(N))
    let L = make_layout(sh, st)
    var tv = make_view(C, L)
    let sz = size(tv)
    for i in 0 ..< sz: discard

# copyFrom with all-static layout — should work
const test3 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = make_layout((Int[32](), Int[8]()))
    var tv = make_view(C, L)
    copyFrom(tv, tv)

# copyFrom with runtime-sized layout
const test4 = cuda:
  proc kernel(C: ptr UncheckedArray[float32]; N: int32) {.global.} =
    let sh = (Int[32](), Int[8](), int(N))
    let L = make_layout(sh, (Int[1](), Int[32](), int(N)))
    var tv = make_view(C, L)
    copyFrom(tv, tv)

suite "size() / copyFrom in cuda:":
  test "all-static size":
    discard cstring(test1)

  test "mixed static/dynamic size":
    discard cstring(test2)

  test "copyFrom all-static":
    discard cstring(test3)

  test "copyFrom with runtime component":
    discard cstring(test4)
