## Verifies that hoistFromExprs correctly hoists preambles from expression
## blocks without producing duplicate variable declarations.
## Chained `()` calls: v(X2(), Y2())(i) in a for-loop.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_hoist_vars_dedup.nim

import std/macros
import workspace/crucible/src/codegen/nvrtc

type X2 = object
type Y2 = object

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

template slice(target: auto; args: varargs[untyped]): auto =
  block:
    evalOnceAs(t, target)
    target

template crd2idx(layout: auto; coord: auto): int =
  block:
    evalOnceAs(t, layout)
    0

type TensorView*[T] = object
  data*: ptr UncheckedArray[T]
  layout*: (int32, int32, int32, int32)

{.experimental: "callOperator".}

template `()`[T](tv: TensorView[T]; args: varargs[untyped]): TensorView[T] =
  block:
    evalOnceAs(coord, varargs_to_par(args))
    evalOnceAs(sub, slice(tv.layout, coord))
    evalOnceAs(offset, crd2idx(tv.layout, coord))
    tv

const kernel = cuda:
  proc gemmKernel(M, K: int32; output: ptr UncheckedArray[float32]) {.global.} =
    let v = TensorView[float32](data: nil, layout: (M, K, int32(1), M))
    var tmp: array[1, TensorView[float32]]
    for i in 0 ..< 3:
      tmp[0] = v(X2(), Y2())(i)
    output[0] = 42.0f

when isMainModule:
  var outBuf: array[1, float32]
  var nv = initNvrtc(kernel)
  nv.numBlocks = 1
  nv.threadsPerBlock = 1
  nv.compile()
  nv.getPtx()
  nv.execute("gemmKernel", outBuf, (128'i32, 64'i32))
  doAssert outBuf[0] == 42.0'f32, "output[0] = " & $outBuf[0] & " (expected 42.0)"
  echo "  OK (test_nvrtc_hoist_vars_dedup)"
