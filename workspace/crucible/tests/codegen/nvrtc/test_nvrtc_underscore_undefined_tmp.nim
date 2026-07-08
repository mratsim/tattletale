## Verifies that `_` (underscore) used as a marker in layout indexing
## is properly handled: the `const _ = X_marker()` is resolved inline
## rather than creating an undefined `tmp_N` reference.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_hoist_undefined_tmp.nim

import std/macros
import workspace/crucible/src/codegen/nvrtc
import ./test_nvrtc_underscore_undefined_tmp_helper

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
  proc reproKernel(output: ptr UncheckedArray[float32]; A: ptr UncheckedArray[float32]) {.global.} =
    let mA = make_view(A)
    for kTile in 0 ..< 3:
      let _ = mA(_, kTile)
    output[0] = 42.0f

when isMainModule:
  var data: array[4, float32] = [1.0'f32, 2.0, 3.0, 4.0]
  var outBuf: array[1, float32]
  var nv = initNvrtc(kernel)
  nv.numBlocks = 1
  nv.threadsPerBlock = 1
  nv.compile()
  nv.getPtx()
  nv.execute("reproKernel", outBuf, (data,))
  doAssert outBuf[0] == 42.0'f32, "output[0] = " & $outBuf[0] & " (expected 42.0)"
  echo "  OK (test_nvrtc_underscore_undefined_tmp)"
