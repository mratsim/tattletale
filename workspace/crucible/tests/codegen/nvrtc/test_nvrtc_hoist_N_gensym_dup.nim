## Verifies that duplicate `N_gensym*` constexpr declarations no longer occur
## when two `()` expansions land in the same for-loop scope.
## The `()` operator chains: tv(0, 0, kTile)(0)
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_hoist_N_gensym_dup.nim

import std/macros
import workspace/crucible/src/codegen/nvrtc

{.experimental: "callOperator".}

type Layout*[Sh, St] = object
  shape*: Sh
  stride*: St

type TensorView*[T, Sh, St] = object
  data*: ptr UncheckedArray[T]
  layout*: Layout[Sh, St]

template foldZipWith_recurse(idx: static int; a, b: typed; state: typed; op: untyped): auto =
  const N = 3
  let field = foldZipWith(a[idx], b[idx], state, op)
  when idx == N - 1:
    field
  else:
    foldZipWith_recurse(idx + 1, a, b, field, op)

template foldZipWith(a, b, startingAcc: typed; op: untyped): auto =
  when typeof(a) is tuple:
    foldZipWith_recurse(0, a, b, startingAcc, op)
  else:
    block:
      let acc {.inject.} = startingAcc
      let it_a {.inject.} = a
      let it_b {.inject.} = b
      op

template crd2idx(layout: Layout; coord: typed): auto =
  when coord is tuple:
    block:
      let P {.genSym.} = coord
      let D {.genSym.} = layout.stride
      foldZipWith(P, D, 0): 0
  else:
    0

macro varargs_to_par(args: varargs[untyped]): untyped =
  result = nnkPar.newTree()
  args.copyChildrenTo(result)

template `()`(tv: TensorView; args: varargs[untyped]): untyped =
  block:
    let coord {.genSym.} = varargs_to_par(args)
    let sub {.genSym.} = tv.layout
    let offset {.genSym.} = crd2idx(tv.layout, coord)
    TensorView[float32, typeof(tv.layout.shape), typeof(tv.layout.stride)](
      data: tv.data, layout: sub)

const kernel = cuda:
  proc reproKernel(output: ptr UncheckedArray[float32]) {.global.} =
    let tv = TensorView[float32, (int, int, int), (int, int, int)](
      data: nil,
      layout: Layout[(int, int, int), (int, int, int)](
        shape: (1, 1, 0),
        stride: (0, 0, 0)))
    for kTile in 0 ..< 3:
      let _ = tv(0, 0, kTile)(0)
    output[0] = 42.0f

when isMainModule:

  echo "═══════════════════════════════════════════════════════════════════"
  echo kernel
  echo "═══════════════════════════════════════════════════════════════════"

  var outBuf: array[1, float32]
  var nv = initNvrtc(kernel)
  nv.numBlocks = 1
  nv.threadsPerBlock = 1
  nv.compile()
  nv.getPtx()
  nv.execute("reproKernel", outBuf, ())
  doAssert outBuf[0] == 42.0'f32, "output[0] = " & $outBuf[0] & " (expected 42.0)"
  echo "  OK (test_nvrtc_hoist_N_gensym_dup)"
