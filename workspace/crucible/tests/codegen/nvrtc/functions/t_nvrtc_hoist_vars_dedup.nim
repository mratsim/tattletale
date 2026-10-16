## hoistFromExprs hoists preambles from expression blocks without producing
## duplicate variable declarations when chained `()` calls land in one
## for-loop scope. Two variants share the construct:
##   - evalOnceAs-based `()` over TensorView[T] with tuple layout
##   - genSym-const `()` over TensorView[T, Sh, St] with Layout shape/stride
##
## Run:
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests workspace/crucible/tests/codegen/nvrtc/functions/t_nvrtc_hoist_vars_dedup.nim

import std/macros
import workspace/crucible

{.experimental: "callOperator".}

type X2 = object
type Y2 = object

type
  Layout*[Sh, St] = object
    shape*: Sh
    stride*: St

  TensorView3*[T, Sh, St] = object
    data*: ptr UncheckedArray[T]
    layout*: Layout[Sh, St]

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

template `()`[T](tv: TensorView[T]; args: varargs[untyped]): TensorView[T] =
  block:
    evalOnceAs(coord, varargs_to_par(args))
    evalOnceAs(sub, slice(tv.layout, coord))
    evalOnceAs(offset, crd2idx(tv.layout, coord))
    tv

template foldZipWith_recurse(idx: static int, a, b: typed, state: typed, op: untyped): auto =
  const N = 3
  let field = foldZipWith(a[idx], b[idx], state, op)
  when idx == N - 1:
    field
  else:
    foldZipWith_recurse(idx + 1, a, b, field, op)

template foldZipWith(a, b, startingAcc: typed, op: untyped): auto =
  when typeof(a) is tuple:
    foldZipWith_recurse(0, a, b, startingAcc, op)
  else:
    block:
      let acc {.inject.} = startingAcc
      let it_a {.inject.} = a
      let it_b {.inject.} = b
      op

template crd2idxN(layout: Layout, coord: typed): auto =
  when coord is tuple:
    block:
      let P {.genSym.} = coord
      let D {.genSym.} = layout.stride
      foldZipWith(P, D, 0): 0
  else:
    0

template `()`(tv: TensorView3, args: varargs[untyped]): untyped =
  block:
    let coord {.genSym.} = varargs_to_par(args)
    let sub {.genSym.} = tv.layout
    let offset {.genSym.} = crd2idxN(tv.layout, coord)
    TensorView3[float32, typeof(tv.layout.shape), typeof(tv.layout.stride)](
      data: tv.data, layout: sub)

const kernel = cuda:
  proc gemmKernel(output: ptr UncheckedArray[float32]; M, K: int32) {.global.} =
    let v = TensorView[float32](data: nil, layout: (M, K, int32(1), M))
    var tmp: array[1, TensorView[float32]]
    for i in 0 ..< 3:
      tmp[0] = v(X2(), Y2())(i)
    output[0] = 42.0f

const kernelGensym = cuda:
  proc reproKernel(output: ptr UncheckedArray[float32]) {.global.} =
    let tv = TensorView3[float32, (int, int, int), (int, int, int)](
      data: nil,
      layout: Layout[(int, int, int), (int, int, int)](
        shape: (1, 1, 0),
        stride: (0, 0, 0)))
    for kTile in 0 ..< 3:
      let _ = tv(0, 0, kTile)(0)
    output[0] = 42.0f

proc runTest() =   # private — tests run in a proc so engines are destroyed at return

  echo "═══════════════════════════════════════════════════════════════════"
  echo kernel
  echo "═══════════════════════════════════════════════════════════════════"

  var outBuf: array[1, float32]
  var engine = bkCuda.init()
  engine.ingest(kernel)
  let oDev = engine.allocDevice(sizeof(outBuf))
  engine.launchKernel<<(1, 1)>>("gemmKernel", oDev, (128'i32, 64'i32))
  engine.copyToHostFromDevice(addr outBuf[0], oDev, sizeof(outBuf))
  engine.freeDevice(oDev)
  doAssert outBuf[0] == 42.0'f32, "output[0] = " & $outBuf[0] & " (expected 42.0)"
  echo "  OK (test_nvrtc_hoist_vars_dedup)"

  echo "═══════════════════════════════════════════════════════════════════"
  echo kernelGensym
  echo "═══════════════════════════════════════════════════════════════════"

  var outBuf2: array[1, float32]
  var engine2 = bkCuda.init()
  engine2.ingest(kernelGensym)
  let oDev2 = engine2.allocDevice(sizeof(outBuf2))
  engine2.launchKernel<<(1, 1)>>("reproKernel", oDev2, ())
  engine2.copyToHostFromDevice(addr outBuf2[0], oDev2, sizeof(outBuf2))
  engine2.freeDevice(oDev2)
  doAssert outBuf2[0] == 42.0'f32, "output[0] = " & $outBuf2[0] & " (expected 42.0)"
  echo "  OK (test_nvrtc_hoist_N_gensym_dup)"

when isMainModule:
  runTest()
