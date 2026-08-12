## Crucible: DotExpr in generic type resolution
##
## tv(0,0) → crd2idx(tv.layout, (0,0)) produces nnkBracketExpr(DotExpr, IntLit)
## which crucible's getGenericTypeName can't handle.
##
## Run:
##   cd tattletale
##   CUDA_HOME=... PATH=... nim cpp -r \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_tensorview_paren.nim

import std/[unittest, macros]
import workspace/crucible/src/codegen/gpu_compiler
import workspace/crucible/src/runtime/engines

# Int[V] and operators — no ceramic imports
type Int*[V: static int] = object
  discard

template `+`*[A, B: static int](a: Int[A], b: Int[B]): Int[A + B] =
  Int[A + B]()
template `*`*[A, B: static int](a: Int[A], b: Int[B]): Int[A * B] =
  Int[A * B]()
# Local helpers — toIntVal and tuple expander
template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

# Produces (t[0], t[1], ...) with live nnkBracketExpr(DotExpr, IntLit) nodes.
# Same pattern as ceramic's mapLeavesWith macro.
macro expandTuple(t: typed): untyped =
  let tType = t.getTypeInst()
  var elems: seq[NimNode]
  for i in 0 ..< tType.len:
    elems.add nnkBracketExpr.newTree(t, newLit i)
  result = nnkTupleConstr.newTree(elems)

# Types defined locally — no ceramic layout import needed
type Layout*[Sh, St] = object
  shape*: Sh
  stride*: St

type Wrapper* = object
  layout*: Layout[(Int[8], Int[16]), (Int[1], Int[8])]

const kernel = cuda:
  proc kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let w = Wrapper(
      layout: Layout[(Int[8], Int[16]), (Int[1], Int[8])](
        shape: (Int[8](), Int[16]()), stride: (Int[1](), Int[8]())))
    # Manual expansion of crd2idx with expandTuple instead of makeIntTuple:
    let P = (Int[0](), Int[0]())
    let D = expandTuple(w.layout.stride)
    let pos = Int[0]() + D[0] * P[0] + D[1] * P[1]
    C[0] = float32(toIntVal pos)

suite "Crucible - DotExpr in generic type resolution":
  test "crd2idx with DotExpr arg inside cuda:":
    let code = kernel
    var output: array[1, float32]
    var engine = bkCuda.init()
    engine.ingest(code)
    engine.run<<(1, 1)>>("kernel", output, ())
    check output[0] == 0.0
