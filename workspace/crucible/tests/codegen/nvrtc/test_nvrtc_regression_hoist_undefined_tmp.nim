## Class C: "tmp_N is undefined" after hoistFromExprs renames.
## Minimal repro — zero ceramic imports.

import std/macros
import workspace/crucible/src/codegen/nvrtc
import ./test_nvrtc_regression_hoist_undefined_tmp_helper

type TensorView*[T] = object
  data*: ptr UncheckedArray[T]

macro varargs_to_par*(args: varargs[untyped]): untyped =
  result = nnkPar.newTree()
  args.copyChildrenTo(result)

macro evalOnceAs*(alias: untyped{nkIdent}, expression: typed): untyped =
  let aName = genSym(nskTemplate, $alias)
  result = newStmtList()
  result.add quote do:
    when `expression` is static:
      const ct_tmp {.genSym.} = `expression`
      template `aName`(): untyped = ct_tmp
    else:
      let rt_tmp {.genSym.} = `expression`
      template `aName`(): untyped = rt_tmp

func make_view*[T](data: ptr UncheckedArray[T] or ptr T): TensorView[T] =
  TensorView[T](data: cast[ptr UncheckedArray[T]](data))

template `()`*(tv: TensorView; args: varargs[untyped]): untyped =
  block:
    evalOnceAs(coord, varargs_to_par(args))
    coord

const kernel = cuda:
  proc reproKernel(A: ptr UncheckedArray[float32]) {.global.} =
    let mA = make_view(A)
    for kTile in 0 ..< 3:
      let _ = mA(_, kTile)

when isMainModule:
  var nv = initNvrtc(kernel)
  nv.compile()
