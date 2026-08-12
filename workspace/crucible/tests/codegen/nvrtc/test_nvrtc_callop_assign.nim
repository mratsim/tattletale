## Minimal reproduction of fillWith on TensorView inside cuda:
## without any ceramic dependency.
##
## Issue: for-loop with `()` operator assignment on TensorView inside cuda: block
##        — codegen emits `tv.data[pos]; = val;` (extra semicolon before `=`).
##
## Root cause: `()` template produces a stmt-list-expr with `let` bindings
##   followed by `tv.data[pos]`. The codegen emits the `let` statements
##   with `;` and then `tv.data[pos];` (statement not lvalue), breaking
##   the subsequent `= val`.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_callop_assign.nim

import std/[unittest, macros]
import workspace/crucible

type Int*[V: static int] = object
  discard

template toIntVal*[V: static int](x: Int[V]): int = V
template toIntVal*(x: int): int = x

type Layout*[Sh, St] = object
  shape*: Sh
  stride*: St

type TensorView*[T, Sh, St] = object
  data*: ptr UncheckedArray[T]
  layout*: Layout[Sh, St]

macro evalOnceAs*(alias: untyped{nkIdent}, expression: typed): untyped =
  result = newProc(
    name = genSym(nskTemplate, $alias),
    params = [getType(untyped)],
    body = expression,
    procType = nnkTemplateDef
  )

template crd2idx*(coord: int; shape, stride: typed): untyped =
  (coord mod toIntVal(shape[0])) * toIntVal(stride[0]) +
  (coord div toIntVal(shape[0])) * toIntVal(stride[1])

{.experimental: "callOperator".}

template `()`*[T, Sh, St](tv: var TensorView[T, Sh, St]; coord: int): var T =
  let pos = block:
    evalOnceAs(s, tv.layout.shape)
    evalOnceAs(d, tv.layout.stride)
    crd2idx(coord, s(), d())
  tv.data[pos]

template fillWith*[T, Sh, St](tv: var TensorView[T, Sh, St]; val: T) =
  block:
    evalOnceAs(s, tv.layout.shape)
    let n = toIntVal(s()[0]) * toIntVal(s()[1])
    for i in 0 ..< n:
      tv(i) = val

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let L = Layout[(Int[8], Int[16]), (Int[1], Int[8])](
      shape: (Int[8](), Int[16]()),
      stride: (Int[1](), Int[8]())
    )
    var tv = TensorView[float32, (Int[8], Int[16]), (Int[1], Int[8])](data: C, layout: L)
    fillWith(tv, 42.0'f32)

proc runTest() =   # private — tests run in a proc so engines are destroyed at return
  suite "NVRTC callop assign":
    test "fillWith via NVRTC":
      var buf: array[128, float32]
      var engine = bkCuda.init()
      engine.ingest(kernel)
      engine.run<<(1, 1)>>("kernel", buf, ())
      check buf[0] == 42.0'f32
      check buf[127] == 42.0'f32

when isMainModule:
  runTest()
