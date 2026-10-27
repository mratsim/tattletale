# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Common utilities for tests over ceramic tensors.
##
## Provides:
## - `assertAllClose` for TensorView, element-wise tolerance check
##   with mismatch diagnostics

import std/[macros, math, random]
import workspace/crucible
import workspace/ceramic/src/ptr_arithmetic
import workspace/ceramic/src/macros/varargs_to_par
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_indexing_slicedice

# ═══════════════════════════════════════════════════════════════
#  Randomness
# ═══════════════════════════════════════════════════════════════

proc seed*(rng: var Rand, x: SomeInteger) =
  ## Re-seed a std/rand state, constantine-style method call
  rng = initRand(int64(x))

# ═══════════════════════════════════════════════════════════════
#  Host and bridge tensors
# ═══════════════════════════════════════════════════════════════

type
  HostTensor*[T, Sh, St] = object
    data*: ptr UncheckedArray[T]
    offset*: int
    layout*: Layout[Sh, St]

  BridgeTensor*[DevMem, Sh, St] = object
    data*: DevMem
    layout*: Layout[Sh, St]

proc `=destroy`*[T, Sh, St](ht: var HostTensor[T, Sh, St]) =
  if ht.data != nil:
    freeShared(ht.data)
    ht.data = nil

proc `=copy`*[T, Sh, St](dst: var HostTensor[T, Sh, St], src: HostTensor[T, Sh, St]) {.error: "HostTensor is move-only".}

proc popNode(tree: var NimNode): NimNode {.compileTime.} =
  result = tree[tree.len - 1]
  tree.del(tree.len - 1)

template `[]`*(ht: HostTensor, args: varargs[untyped]): untyped =
  ht.data[ht.offset + toInt crd2idx(ht.layout, varargs_to_par(args))]

macro `[]=`*(ht: HostTensor, args: varargs[untyped]): untyped =
  var a = args
  let val = popNode(a)
  let coord = getAst(varargs_to_par(a))
  result = quote do:
    `ht`.data[`ht`.offset + toInt crd2idx(`ht`.layout, `coord`)] = `val`

func originOffset[Sh, St](layout: Layout[Sh, St]): int =
  ## Buffer offset of the logical origin, one (shape-1)*|stride| term per negative-stride leaf
  let sh = layout.shape.flatten()
  let st = layout.stride.flatten()
  var strides: seq[int]
  for f in fields(st):
    strides.add(toInt(f))
  var i = 0
  for f in fields(sh):
    if strides[i] < 0:
      result += (toInt(f) - 1) * (-strides[i])
    i.inc

func makeHostTensor*[T](_: typedesc[T], shape: tuple, order: static StrideOrder): auto =
  let layout = make_layout(shape, order)
  HostTensor[T, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[T]](allocShared(toInt(layout.cosize()) * sizeof(T))),
    offset: originOffset(layout),
    layout: layout)

func makeHostTensor*[T](_: typedesc[T], shape, strides: tuple): auto =
  let layout = make_layout(shape, strides)
  HostTensor[T, typeof(shape), typeof(strides)](
    data: cast[ptr UncheckedArray[T]](allocShared(toInt(layout.cosize()) * sizeof(T))),
    offset: originOffset(layout),
    layout: layout)

template bounds[T](start, stopIncl: untyped): HSlice[T, T] =
  # Workaround `..` or Slice[T](...) not resolving T as a default value
  Slice[T](a: T(start), b: T(stopIncl))

proc randomHostTensor*[T](rng: var Rand, _: typedesc[T], shape, strides: tuple, bounds = bounds[float32](1, 2)): auto =
  let layout = make_layout(shape, strides)
  result = HostTensor[T, typeof(shape), typeof(strides)](
    data: cast[ptr UncheckedArray[T]](allocShared(toInt(layout.cosize()) * sizeof(T))),
    offset: originOffset(layout),
    layout: layout)
  for i in 0 ..< toInt(layout.size()):
    let crd = idx2crd(layout.shape, i)
    when T is float32 or T is float64:
      result.data[result.offset + toInt crd2idx(layout, crd)] = rng.rand(bounds)
    else:
      result.data[result.offset + toInt crd2idx(layout, crd)] = to(rng.rand(bounds.a .. bounds.b), T)

func len*[T, Sh, St](ht: HostTensor[T, Sh, St]): int =
  toInt(ht.layout.cosize())

func view*[T, Sh, St](ht: HostTensor[T, Sh, St]): TensorView[T, Sh, St] =
  TensorView[T, Sh, St](data: ht.data +% ht.offset, layout: ht.layout)

proc bridge*[E: HwEngine, T, Sh, St](engine: E, ht: HostTensor[T, Sh, St]): auto =
  let n = ht.layout.cosize()
  let dev = engine.allocDevice(T, n)
  engine.copyToDeviceFromHost(dev, ht.data +% ht.offset, n)
  BridgeTensor[typeof(dev), Sh, St](data: dev, layout: ht.layout)

# ═══════════════════════════════════════════════════════════════
#  Tensor assertions
# ═══════════════════════════════════════════════════════════════

func flatSize[L](l: L): int =
  ## Size as a runtime int, static `Int[N]` layouts included
  when typeof(size(l)) is Int:
    toInt(size(l))
  else:
    size(l)

type AllCloseReport* = object
  bad*: int
  worst*, firstGot*, firstWant*: float64
  firstCrd*: string

proc allClose*[T, ShA, StA, ShB, StB](
    actual: TensorView[T, ShA, StA], expected: TensorView[T, ShB, StB],
    rtol, abstol: float64): AllCloseReport =
  ## Element-wise closeness, |a - e| <= abstol + rtol*|e| per element,
  ## compared in float64, same shape required, strides may differ.
  let n = flatSize(actual.getLayout())
  doAssert n == flatSize(expected.getLayout()),
    "allClose: size mismatch, got " & $n &
    " want " & $flatSize(expected.getLayout())
  let actualL = actual.getLayout()
  let expectedL = expected.getLayout()
  for i in 0 ..< n:
    let crd = idx2crd(actual.shape, i)
    let a = to(actual.data[toInt crd2idx(actualL, crd)], float64)
    let e = to(expected.data[toInt crd2idx(expectedL, idx2crd(expected.shape, i))], float64)
    # NaN never matches, a NaN difference is not greater than any bound,
    # an infinite expected value must not inflate the relative bound
    var d = abs(a - e)
    if a != e and (a != a or e != e):
      d = Inf
    let re = if e == e and abs(e) != Inf: abs(e) else: 0.0
    if d > abstol + rtol * re:
      if result.bad == 0:
        result.firstCrd = $crd
        result.firstGot = a
        result.firstWant = e
      result.worst = max(result.worst, d)
      inc result.bad

template assertAllClose*(actual, expected: untyped, rtol: float64 = 1e-4, abstol: float64 = 1e-4, msg = "") =
  ## Assert two views are element-wise close, compared in float64:
  ## |a - e| <= abstol + rtol*|e| for every element.
  ##
  ## Args:
  ## - actual: the view produced by the code under test
  ## - expected: the reference view, same shape and strides
  ## - rtol, abstol: relative and absolute tolerance
  ## - msg: optional context echoed on failure
  block:
    let actualView = actual
    let expectedView = expected
    let report = allClose(actualView, expectedView, rtol, abstol)
    if report.bad > 0:
      echo "Assertion failed: allClose"
      echo "  '", astToStr(actual), "'"
      if msg.len > 0:
        echo "  ", msg
      echo "  ", report.bad, " mismatched elements of ",
           flatSize(actualView.getLayout()), ", worst |d| = ", report.worst
      echo "  first mismatch at ", report.firstCrd, ": got ", report.firstGot,
           " want ", report.firstWant, " (rtol ", rtol, ", abstol ", abstol, ")"
      raise newException(AssertionDefect, "[ceramic] allClose assertion failed")
