## hoistFromExprs else-branch hoists preambles from expression blocks.

import std/macros
import workspace/crucible/src/codegen/nvrtc

type
  Layout = object
    shapeStride: (int32, int32, int32, int32)
  X2 = object
  Y2 = object

macro evalOnceAs(alias: untyped{nkIdent}, expression: typed): untyped =
  let aName = genSym(nskTemplate, $alias)
  result = newStmtList()
  result.add quote do:
    when `expression` is static:
      const ct_tmp {.genSym.} = `expression`
      template `aName`(): untyped = ct_tmp
    else:
      let rt_tmp {.genSym.} = `expression`
      template `aName`(): untyped = rt_tmp

macro varargs_to_par(args: varargs[untyped]): untyped =
  result = nnkPar.newTree()
  args.copyChildrenTo(result)

template slice(target: Layout; args: varargs[untyped]): Layout =
  block:
    evalOnceAs(t, target)
    target

template crd2idx(layout: Layout; coord: auto): int =
  block:
    evalOnceAs(t, layout)
    0

type TensorView*[T] = object
  data*: ptr UncheckedArray[T]
  layout*: Layout

func make_view[T](data: ptr UncheckedArray[T] or ptr T;
                  L: Layout): TensorView[T] =
  TensorView[T](data: cast[ptr UncheckedArray[T]](data), layout: L)

{.experimental: "callOperator".}

template `()`[T](tv: TensorView[T]; args: varargs[untyped]): untyped =
  block:
    evalOnceAs(coord, varargs_to_par(args))
    evalOnceAs(sub, slice(tv.layout, coord))
    evalOnceAs(offset, crd2idx(tv.layout, coord))
    make_view(tv.data, sub)

const kernel = cuda:
  proc gemmKernel(A: ptr UncheckedArray[float32], M, K: int32) {.global.} =
    let L = Layout(shapeStride: (M, K, 1, M))
    let v = make_view(A, L)
    for i in 0 ..< 3:
      discard v(X2(), Y2())(i)

when isMainModule:
  var nv = initNvrtc(kernel)
  nv.compile()
