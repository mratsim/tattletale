# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[math, random, strformat, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/kernels/k_tile_sum
import workspace/ceramic/src/kernels/k_layout_fillwith_cpu

export int_tuples
export layout_algebra
export tensors
export k_tile_sum

export int_tuples, int_tuples_streams
export k_tile_sum

# ═══════════════════════════════════════════════════════════════
#  fp32 host reference
# ═══════════════════════════════════════════════════════════════

func refSum*[T, Sh, St](src: TensorView[T, Sh, St]): float32 =
  ## Sum of the tensor, f64 accumulation
  var acc = 0.0
  for i in 0 ..< src.size():
    acc += float64(src.data[i])
  float32(acc)

func refSum[T, ShS, StS, ShD, StD](dst: TensorView[T, ShD, StD], src: TensorView[T, ShS, StS], axis: static int) =
  ## Sums over `axis`, one f64 accumulator per output slot
  let dstSlots = make_layout(dst.shape)
  var acc = newSeq[float64](toInt(dst.size()))
  for i in 0 ..< src.size():
    var crd = idx2crd(src.shape, i)
    crd[axis] = 0
    acc[toInt(dstSlots.crd2idx(crd))] += float64(src(i))
  for j in 0 ..< dst.size():
    var crd = idx2crd(dst.shape, j)
    dst[crd] = float32(acc[toInt(dstSlots.crd2idx(crd))])

# ═══════════════════════════════════════════════════════════════
#  The checks
# ═══════════════════════════════════════════════════════════════

proc checkSum*[E: HwEngine](rng: var Rand, engine: E, rows, cols: static int, sSrc: static (int, int)) =
  ## One sum_full launch over a random (rows, cols) tensor
  var src = rng.randomHostTensor( float32, (rows, cols), sSrc)
  var got = makeHostTensor(float32, (1, 1), (1, 1))
  var gotView = got.view()
  fillWith_cpu(gotView, 0'f32)  # the accumulation contract: dst starts at zero
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)
  # the kernel's grid-stride loop completes the tile grid from any launch grid
  let
    gridX = rng.rand(1 .. 2 * ceil_div(rows, 32))
    gridY = rng.rand(1 .. 2 * ceil_div(cols, 32))
  engine.launchDeviceFn sum(gotDev, srcDev) do:
    grid = (gridX, gridY)
    blk = (32, 1)
  var expect = makeHostTensor(float32, (1, 1), (1, 1))
  expect.data[0] = refSum(src.view())
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, 1)
  assertAllClose(got.view(), expect.view(), rtol = 1e-5, abstol = 1e-4)

proc checkSumAxis*[E: HwEngine](rng: var Rand, engine: E, shape, srcStrides, dstStrides: tuple, axis: static int, msg: string) =
  ## One sum(axis) launch over a random tensor of any rank
  var src = rng.randomHostTensor(float32, shape, srcStrides)
  var dstShape = shape
  dstShape[axis] = 1
  let dstLayout = make_layout(dstShape, dstStrides)
  var got = makeHostTensor(float32, dstShape, dstLayout.stride)
  var gotView = got.view()
  fillWith_cpu(gotView, 12345.0'f32)  # canary: the kernel must not read dst
  var expect = makeHostTensor(float32, dstShape, dstLayout.stride)
  refSum(expect.view(), src.view(), axis)
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)
  # the kernel's grid-stride loop completes the tile grid from any launch grid
  let gridX = rng.rand(1 .. 8)
  engine.launchDeviceFn sum(gotDev, srcDev, axis) do:
    grid = (gridX, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, toInt(got.layout.cosize()))
  assertAllClose(got.view(), expect.view(), rtol = 1e-5, abstol = 1e-4, msg = msg)

proc run_sum_full_tests*[E: HwEngine](engine: E, moduleName: string) =
  var rng: Rand
  let seed = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seed))
  echo "\n------------------------------------------------------\n"
  echo moduleName, " xoroshiro128+ seed: ", seed

  # stride permutations within one covered tile: the (8, 4) block covers (8, 128)
  rng.checkSum( engine, 8, 128, (128, 1))
  rng.checkSum( engine, 8, 128, (1, 8))

  # ── 3D tensor: the (2, 2, 2) block covers (2, 2, 60) ──
  block:
    var src = rng.randomHostTensor( float32, (2, 2, 60), (120, 60, 1))
    var sumGot = makeHostTensor(float32, (1, 1, 1), (1, 1, 1))
    let srcDev = engine.bridge(src)
    var expectSum = makeHostTensor(float32, (1, 1, 1), (1, 1, 1))
    expectSum.data[0] = refSum(src.view())
    let sumDev = engine.bridge(sumGot)
    engine.launchDeviceFn sum(sumDev, engine.bridge(src)) do:
      grid = (1, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr sumGot.data[0], sumDev.data, 1)
    assertAllClose(sumGot.view(), expectSum.view(), msg = "3D sum full")

  # ── 4D tensor: the (2, 2, 2, 2) block covers (2, 2, 2, 50) ──
  block:
    var src = rng.randomHostTensor( float32, (2, 2, 2, 50), (200, 100, 50, 1))
    var sumGot = makeHostTensor(float32, (1, 1, 1, 1), (1, 1, 1, 1))
    let srcDev = engine.bridge(src)
    var expectSum = makeHostTensor(float32, (1, 1, 1, 1), (1, 1, 1, 1))
    expectSum.data[0] = refSum(src.view())
    let sumDev = engine.bridge(sumGot)
    engine.launchDeviceFn sum(sumDev, engine.bridge(src)) do:
      grid = (1, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr sumGot.data[0], sumDev.data, 1)
    assertAllClose(sumGot.view(), expectSum.view(), msg = "4D sum full")

func genAxisCases*[Sh: tuple](shapes: varargs[Sh]): seq[tuple[src, dst: tuple[shape, strides: Sh], axis, iters: int]] =
  ## Axis-reduction cases, one entry per (stride order, axis) per shape:
  ## - src strides: every compact order, row-major, column-major, middle
  ##   fastest and slowest included, plus each axis stride scaled 2x and 3x
  ## - dst: compact, axis size 1
  ## - iters scaled by element count

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
    # Compact-permuted planes: kept modes stay row-major, summed axis moved outside.
    # Memory stays dense but whether the slots form contiguous runs depends on the axis.
    # The kernel cannot assume a fixed stride between slots here.
    for axis in 0 ..< dims.len - 1:
      var kept: seq[int]
      for i in 0 ..< dims.len:
        if i != axis:
          kept.add(dims[i])
      let keptRm = rowmajor(kept)
      var sdims = newSeq[int](dims.len)
      var k = 0
      for i in 0 ..< dims.len:
        if i != axis:
          sdims[i] = keptRm[k]
          k.inc
      sdims[axis] = np div dims[axis]
      orders.add(sdims)
      var skip2 = sdims
      skip2[axis] *= 2
      orders.add(skip2)
    for sdims in orders:
      for axis in 0 ..< dims.len:
        var ddims = dims
        ddims[axis] = 1
        result.add((
          src: (shape: asTuple(dims), strides: asTuple(sdims)),
          dst: (shape: asTuple(ddims), strides: asTuple(rowmajor(ddims))),
          axis: axis, iters: iters))

const AxisCases2 = genAxisCases((8, 4), (5, 3), (2, 16))
const AxisCases3 = genAxisCases((2, 3, 4), (3, 5, 2), (2, 2, 6), (5, 2, 2))
const AxisCases4 = genAxisCases((2, 3, 4, 5), (4, 3, 2, 2), (3, 2, 2, 5))

proc run_sum_axis_tests*[E: HwEngine](engine: E, moduleName: string) =
  var rng: Rand
  let seed = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seed))
  echo "\n------------------------------------------------------\n"
  echo moduleName, " xoroshiro128+ seed: ", seed

  # stride permutations within one covered tile: the (8, 4) block covers (8, 128)
  rng.checkSumAxis(engine, (8, 128), (128, 1), (1, 1), 1, "2D tile permutation (128, 1)")
  rng.checkSumAxis(engine, (8, 128), (1, 8), (1, 1), 1, "2D tile permutation (1, 8)")

  # ── layout cases: 2D/3D/4D, contiguous, swapped and column-major
  #    sources, compact and mirrored destinations ──
  template checkAxisCases(cases: untyped, label: string) =
    staticFor ci, 0, cases.len:
      checkSumAxis(rng, engine, cases[ci].src.shape, cases[ci].src.strides,
          cases[ci].dst.strides, cases[ci].axis, label & " case " & $ci)
  checkAxisCases(AxisCases2, "2D")
  checkAxisCases(AxisCases3, "3D")
  checkAxisCases(AxisCases4, "4D")

  # ── partial tiles on the last axis: the (2, 2, 2) block covers (2, 2, 60) ──
  block:
    var src = rng.randomHostTensor( float32, (2, 2, 60), (120, 60, 1))
    var axisGot = makeHostTensor(float32, (2, 2, 1), (2, 1, 1))
    let srcDev = engine.bridge(src)
    var expectAxis = makeHostTensor(float32, (2, 2, 1), (2, 1, 1))
    refSum(expectAxis.view(), src.view(), 2)
    let axisDev = engine.bridge(axisGot)
    engine.launchDeviceFn sum(axisDev, engine.bridge(src), 2) do:
      grid = (1, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr axisGot.data[0], axisDev.data, 4)
    assertAllClose(axisGot.view(), expectAxis.view(), msg = "3D sum with axis")

  # ── partial tiles on the last axis: the (2, 2, 2, 2) block covers (2, 2, 2, 50) ──
  block:
    var src = rng.randomHostTensor( float32, (2, 2, 2, 50), (200, 100, 50, 1))
    var axisGot = makeHostTensor(float32, (2, 2, 2, 1), (4, 2, 1, 1))
    let srcDev = engine.bridge(src)
    var expectAxis = makeHostTensor(float32, (2, 2, 2, 1), (4, 2, 1, 1))
    refSum(expectAxis.view(), src.view(), 3)
    let axisDev = engine.bridge(axisGot)
    engine.launchDeviceFn sum(axisDev, engine.bridge(src), 3) do:
      grid = (1, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr axisGot.data[0], axisDev.data, 8)
    assertAllClose(axisGot.view(), expectAxis.view(), msg = "4D sum with axis")
