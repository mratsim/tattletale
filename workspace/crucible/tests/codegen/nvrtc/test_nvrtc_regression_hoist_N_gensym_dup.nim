## "N_gensym*" has already been declared in the current scope
##
## The codegen emits `constexpr int N_gensymXX = N` for template-local
## `const` declarations. When two `()` expansions land in the same
## `for` loop scope, the hoisting iteration-split renames them with
## a `_0` suffix. `dedupVar` skips `gpuConstexpr` nodes — no renaming
## occurs — so the same `N_gensymXX` name appears multiple times.
##
## Run:
##   cd tattletale
##   nim cpp -r --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip \
##     workspace/crucible/tests/codegen/nvrtc/test_nvrtc_regression_hoist_N_gensym_dup.nim 2>&1 | grep "error:"

import std/macros
import workspace/crucible/src/codegen/nvrtc

{.experimental: "callOperator".}

# ──────────────────────────────────────────────────
#  Layout / TensorView — type dispatch scaffolding
# ──────────────────────────────────────────────────

type Layout*[Sh, St] = object
  shape*: Sh
  stride*: St

type TensorView*[T, Sh, St] = object
  data*: ptr UncheckedArray[T]
  layout*: Layout[Sh, St]

# ──────────────────────────────────────────────────
#  foldZipWith — produces `constexpr int N_gensymXX`
#  via `const N = 3` in the template body.
# ──────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────
#  crd2idx — creates genSym'd P/D, calls foldZipWith
# ──────────────────────────────────────────────────

template crd2idx(layout: Layout; coord: typed): auto =
  when coord is tuple:
    block:
      let P {.genSym.} = coord
      let D {.genSym.} = layout.stride
      foldZipWith(P, D, 0): 0
  else:
    0

# ──────────────────────────────────────────────────
#  varargs_to_par — constructs a tuple from varargs
# ──────────────────────────────────────────────────

macro varargs_to_par(args: varargs[untyped]): untyped =
  result = nnkPar.newTree()
  args.copyChildrenTo(result)

# ──────────────────────────────────────────────────
#  `()` operator — entry point, creates genSym'd
#  variables that collide after hoisting.
# ──────────────────────────────────────────────────

template `()`(tv: TensorView; args: varargs[untyped]): untyped =
  block:
    let coord {.genSym.} = varargs_to_par(args)
    let sub {.genSym.} = tv.layout
    let offset {.genSym.} = crd2idx(tv.layout, coord)
    TensorView[float32, typeof(tv.layout.shape), typeof(tv.layout.stride)](
      data: tv.data, layout: sub)

# ──────────────────────────────────────────────────
#  Kernel — the chained `(0, 0, kTile)(0)` creates
#  two `()` expansions in the same for-loop scope.
# ──────────────────────────────────────────────────

const kernel = cuda:
  proc reproKernel() {.global.} =
    let tv = TensorView[float32, (int, int, int), (int, int, int)](
      data: nil,
      layout: Layout[(int, int, int), (int, int, int)](
        shape: (1, 1, 0),
        stride: (0, 0, 0)))
    for kTile in 0 ..< 3:
      let _ = tv(0, 0, kTile)(0)

when isMainModule:
  var nv = initNvrtc(kernel)
  nv.compile()
