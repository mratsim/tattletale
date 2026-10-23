# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[math, random, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/kernels/k_tile_rmsnorm

export int_tuples
export layout_algebra
export tensors
export tile_algebra
export k_tile_rmsnorm

# ═══════════════════════════════════════════════════════════════════════
#  fp64 host reference
# ═══════════════════════════════════════════════════════════════════════

func refRmsnorm[T, ShS, StS, ShD, StD, ShW, StW](
    dst: TensorView[T, ShD, StD], src: TensorView[T, ShS, StS],
    weight: TensorView[T, ShW, StW], eps: float32, axis: static int) =
  ## RMSNorm with weight, one f64 row at a time through each tensor's own layout
  let
    rowSize = toInt(src.shape[axis])
    srcL = src.getLayout()
    dstL = dst.getLayout()
  for i in 0 ..< toInt(dst.size()):
    var crd = idx2crd(dst.shape, i)
    if crd[axis] != 0:
      continue
    var ss = 0.0
    for r in 0 ..< rowSize:
      crd[axis] = r
      let x = float64(src.data[toInt crd2idx(srcL, crd)])
      ss += x * x
    let scale = 1.0 / sqrt(ss / float64(rowSize) + float64(eps))
    for r in 0 ..< rowSize:
      crd[axis] = r
      let w = float64(weight.data[toInt crd2idx(weight.getLayout(), r)])
      dst.data[toInt crd2idx(dstL, crd)] = float32(
          float64(src.data[toInt crd2idx(srcL, crd)]) * scale * w)

func refRmsnorm[T, ShS, StS, ShD, StD](
    dst: TensorView[T, ShD, StD], src: TensorView[T, ShS, StS],
    eps: float32, axis: static int) =
  ## RMSNorm without weight, one f64 row at a time
  let
    rowSize = toInt(src.shape[axis])
    srcL = src.getLayout()
    dstL = dst.getLayout()
  for i in 0 ..< toInt(dst.size()):
    var crd = idx2crd(dst.shape, i)
    if crd[axis] != 0:
      continue
    var ss = 0.0
    for r in 0 ..< rowSize:
      crd[axis] = r
      let x = float64(src.data[toInt crd2idx(srcL, crd)])
      ss += x * x
    let scale = 1.0 / sqrt(ss / float64(rowSize) + float64(eps))
    for r in 0 ..< rowSize:
      crd[axis] = r
      dst.data[toInt crd2idx(dstL, crd)] = float32(
          float64(src.data[toInt crd2idx(srcL, crd)]) * scale)

# ═══════════════════════════════════════════════════════════════════════
#  Data preparation
# ═══════════════════════════════════════════════════════════════════════

proc randomHostVector*[T](rng: var Rand, _: typedesc[T], size: int, bounds: HSlice[float32, float32] = 1'f32 .. 2'f32): auto =
  ## Returns a random rank-1 host tensor of shape (size,), stride 1.
  ##
  ## Rank-1 layouts flatten single-element shape and stride tuples into scalars, the shared rank-N host tensor constructors reject that shape.
  ## This constructor serves the vector case.
  let layout = make_layout((size,), (1,))
  result = HostTensor[T, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[T]](allocShared(size * sizeof(T))),
    offset: 0,
    layout: layout)
  for i in 0 ..< size:
    result.data[i] = rng.rand(bounds)

func fillCanary[T, Sh, St](ht: var HostTensor[T, Sh, St]) =
  ## Fills every element, garbage slots included, with the sentinel 12345:
  ## the kernel under test must overwrite every slot and never read dst
  for i in 0 ..< ht.len():
    ht.data[i] = 12345'f32

# ═══════════════════════════════════════════════════════════════════════
#  The checks
# ═══════════════════════════════════════════════════════════════════════

proc checkRmsnorm*[E: HwEngine](rng: var Rand, engine: E, shape, srcStrides, dstStrides: tuple, eps: float32, axis: static int, withWeight: static bool, msg: string, srcBounds: HSlice[float32, float32] = 1'f32 .. 2'f32) =
  ## One rmsnorm launch over a random tensor of any rank against the f64 reference.
  ## Checked at rtol 1e-5 and abstol 1e-4, the worst absolute difference is echoed per case.
  ##
  ## The launch grid width is random, undersized and oversized grids alike exercise the grid-stride row loop.
  ##
  ## Args:
  ## - rng, engine: the seeded random state and the device engine
  ## - shape, srcStrides, dstStrides: source geometry and destination strides
  ## - msg: the case label echoed with the worst difference
  ##
  ## Scalars:
  ## - eps: the stability constant added to the mean of squares
  ## - axis: the axis normalized over
  ## - withWeight: static switch, true adds the per-axis weight check
  ## - srcBounds: the value range of the random source elements
  var src = rng.randomHostTensor(float32, shape, srcStrides, bounds = srcBounds)
  var got = makeHostTensor(float32, shape, dstStrides)
  var expect = makeHostTensor(float32, shape, dstStrides)
  got.fillCanary()
  # a non-compact dst layout carries gap slots the kernel and the reference
  # never touch: the sentinel there matches the canary-filled got
  expect.fillCanary()
  when withWeight:
    var wDims: seq[int]
    for f in fields(shape):
      wDims.add(toInt(f))
    let wSize = wDims[axis]
    var weight = rng.randomHostVector(float32, wSize, bounds = 0.5'f32 .. 1.5'f32)
    refRmsnorm(expect.view(), src.view(), weight.view(), eps, axis)
    let gotDev = engine.bridge(got)
    engine.launchDeviceFn rmsnorm(gotDev, engine.bridge(src), engine.bridge(weight), eps, axis) do:
      grid = (rng.rand(1 .. 8), 1)
      blk = (32, 1)
  else:
    refRmsnorm(expect.view(), src.view(), eps, axis)
    let gotDev = engine.bridge(got)
    engine.launchDeviceFn rmsnorm(gotDev, engine.bridge(src), eps, axis) do:
      grid = (rng.rand(1 .. 8), 1)
      blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, toInt(got.layout.cosize()))
  let report = allClose(got.view(), expect.view(), rtol = 1e-5, abstol = 1e-4)
  echo msg, ": worst |d| = ", formatFloat(report.worst, ffScientific, 3)
  doAssert report.bad == 0, msg & ": " & $report.bad & " mismatched elements"

func genRmsnormCases*[Sh: tuple](shapes: varargs[Sh]): seq[tuple[src, dst: tuple[shape, strides: Sh], axis, iters: int]] =
  ## Returns one case per (stride order, axis) per shape.
  ##
  ## - src strides: every compact order (row-major, column-major, middle fastest and middle slowest) plus each axis stride scaled 2x and 3x
  ## - dst: same shape, compact row-major
  ## - iters: scaled by element count

  func rowmajor(dims: seq[int]): seq[int] =
    result = newSeq[int](dims.len)
    var p = 1
    for i in countdown(dims.len - 1, 0):
      result[i] = p
      p *= dims[i]

  func asTuple(vals: seq[int]): Sh =
    var i = 0
    for f in fields(result):
      f = vals[i]
      i.inc

  func perms(vals: seq[int]): seq[seq[int]] =
    if vals.len <= 1:
      return @[vals]
    for i in 0 ..< vals.len:
      var rest = vals
      rest.delete(i)
      for p in perms(rest):
        result.add(@[vals[i]] & p)

  for shape in shapes:
    var dims: seq[int]
    var np = 1
    for f in fields(shape):
      dims.add(f)
      np *= f
    let iters = if np <= 200_000: 200 elif np <= 1_000_000: 50 else: 20
    var orders = perms(rowmajor(dims))
    for k in 0 ..< dims.len:
      var skip2 = rowmajor(dims)
      skip2[k] *= 2
      var skip3 = rowmajor(dims)
      skip3[k] *= 3
      orders.add(skip2)
      orders.add(skip3)
    for sdims in orders:
      for axis in 0 ..< dims.len:
        result.add((
          src: (shape: asTuple(dims), strides: asTuple(sdims)),
          dst: (shape: asTuple(dims), strides: asTuple(rowmajor(dims))),
          axis: axis, iters: iters))

const RmsnormCases2 = genRmsnormCases((8, 4), (5, 3), (2, 16))
const RmsnormCases3 = genRmsnormCases((2, 3, 4), (3, 5, 2), (2, 2, 6), (5, 2, 2))
const RmsnormCases4 = genRmsnormCases((2, 3, 4, 5), (4, 3, 2, 2), (3, 2, 2, 5))

proc run_rmsnorm_tests*[E: HwEngine](engine: E, moduleName: string) =
  var rng: Rand
  let seed = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seed))
  echo "\n------------------------------------------------------\n"
  echo moduleName, " xoroshiro128+ seed: ", seed

  # ── layout cases: 2D/3D/4D, contiguous, swapped and column-major
  #    sources, compact destinations, with and without weight ──
  template checkCases(cases: untyped, label: string) =
    staticFor ci, 0, cases.len:
      checkRmsnorm(rng, engine, cases[ci].src.shape, cases[ci].src.strides,
          cases[ci].dst.strides, 1e-5'f32, cases[ci].axis, false,
          label & " case " & $ci)
      checkRmsnorm(rng, engine, cases[ci].src.shape, cases[ci].src.strides,
          cases[ci].dst.strides, 1e-5'f32, cases[ci].axis, true,
          label & " case " & $ci & " weighted")
  checkCases(RmsnormCases2, "2D")
  checkCases(RmsnormCases3, "3D")
  checkCases(RmsnormCases4, "4D")

  # ── rows shorter than the lane width: idle lanes must contribute
  #    nothing to the sum of squares ──
  checkRmsnorm(rng, engine, (3, 7), (7, 1), (7, 1), 1e-5'f32, 1, false,
      "row shorter than one warp")
  checkRmsnorm(rng, engine, (3, 7), (7, 1), (7, 1), 1e-5'f32, 1, true,
      "row shorter than one warp, weighted")

  # ── row of exactly one lane width ──
  checkRmsnorm(rng, engine, (2, 32), (32, 1), (32, 1), 1e-5'f32, 1, false,
      "row of exactly one warp")
  checkRmsnorm(rng, engine, (2, 32), (32, 1), (32, 1), 1e-5'f32, 1, true,
      "row of exactly one warp, weighted")

  # ── ragged rows, shorter than one 128-element tile: past-the-end
  #    tile slots contribute nothing ──
  checkRmsnorm(rng, engine, (2, 100), (100, 1), (100, 1), 1e-5'f32, 1, false,
      "ragged row (100)")
  checkRmsnorm(rng, engine, (2, 100), (100, 1), (100, 1), 1e-5'f32, 1, true,
      "ragged row (100), weighted")
  checkRmsnorm(rng, engine, (2, 37), (37, 1), (37, 1), 1e-5'f32, 1, false,
      "ragged row (37)")
  checkRmsnorm(rng, engine, (2, 37), (37, 1), (37, 1), 1e-5'f32, 1, true,
      "ragged row (37), weighted")

  # ── size-1 axis: scale is rsqrt(x² + eps), the row value barely moves ──
  checkRmsnorm(rng, engine, (4, 1), (1, 1), (1, 1), 1e-5'f32, 1, false,
      "size-1 axis")
  checkRmsnorm(rng, engine, (4, 1), (1, 1), (1, 1), 1e-5'f32, 1, true,
      "size-1 axis, weighted")

  template checkPair(shape, strides, eps, axis, msg: untyped, bounds: HSlice[float32, float32] = 1'f32 .. 2'f32) =
    checkRmsnorm(rng, engine, shape, strides, strides, eps, axis, false, msg, bounds)
    checkRmsnorm(rng, engine, shape, strides, strides, eps, axis, true,
        msg & ", weighted", bounds)

  # ── wide dynamic range, and a small eps: the rsqrt must hold
  #    across orders of magnitude ──
  checkPair((2, 64), (64, 1), 1e-6'f32, 1, "wide dynamic range, eps 1e-6",
      -30'f32 .. 30'f32)

  # ── wide rows spanning many tiles: the tile walk across the row
  #    must cover every tile, ragged last tile included ──
  checkRmsnorm(rng, engine, (8, 4096), (4096, 1), (4096, 1), 1e-5'f32, 1, false,
      "wide row (8, 4096)", -3'f32 .. 3'f32)
  checkPair((4, 300), (300, 1), 1e-5'f32, 1, "ragged wide row (300)",
      -3'f32 .. 3'f32)

  # ── strided destination: a 2x-scaled stride on the non-reduced axis,
  #    the kernel writes through the real strides, gap slots untouched ──
  checkRmsnorm(rng, engine, (4, 16), (16, 1), (32, 1), 1e-5'f32, 1, false,
      "strided dst, 2x stride on the non-reduced axis")
  checkRmsnorm(rng, engine, (4, 16), (16, 1), (32, 1), 1e-5'f32, 1, true,
      "strided dst, 2x stride on the non-reduced axis, weighted")

  # ── the one-warp-per-row contract is enforced at the launch: a threadgroup
  #    of any other size is rejected, dst comes back NaN ──
  checkOffContractLaunch(engine, (2, 64), (64, 1), (31, 1),
      "guard, threadgroup narrower than the lane width (31)")
  checkOffContractLaunch(engine, (2, 64), (64, 1), (33, 1),
      "guard, threadgroup wider than the lane width (33)")
  checkOffContractLaunch(engine, (2, 64), (64, 1), (32, 4),
      "guard, threadgroup holding several warps (32, 4)")
