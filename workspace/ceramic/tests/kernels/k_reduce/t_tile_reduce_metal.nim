# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile reduce over Metal: strip reductions, accumulation and atAxisIndex
##
## Test that the two-phase tile reduce matches an f64 host reference on Metal.
## It covers compact, ragged and multi-tile strips with overwrite and accumulate
## semantics and checks that atAxisIndex cuts per-row strips end to end.
##
## Run:
##   nim c -r --hints:off --warnings:off --outdir:build/tests \
##     --nimcache:nimcache/tests \
##     workspace/ceramic/tests/kernels/k_reduce/t_tile_reduce_metal.nim

import std/[math, strutils]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/int_tuples/int_tuples_streams
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/layout_algebra/layout_compiletime
import workspace/ceramic/src/tensors
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/tile_algebra

export int_tuples, layout_algebra, tensors

# ═══════════════════════════════════════════════════════════════════════
#  f64 host reference
# ═══════════════════════════════════════════════════════════════════════

func refStripSum[T, Sh, St](src: TensorView[T, Sh, St]): float64 =
  ## f64 sum over the flat element order
  for k in 0 ..< src.size():
    result += float64(src[0, k])

func refStripMax[T, Sh, St](src: TensorView[T, Sh, St]): float64 =
  ## f64 max over the flat element order
  result = -Inf
  for k in 0 ..< src.size():
    result = max(result, float64(src[0, k]))

func refStripMin[T, Sh, St](src: TensorView[T, Sh, St]): float64 =
  ## f64 min over the flat element order
  result = Inf
  for k in 0 ..< src.size():
    result = min(result, float64(src[0, k]))

proc fillStrip[T, Sh, St](src: HostTensor[T, Sh, St], first, step: float32) =
  ## Fills the logical elements with an arithmetic progression
  for k in 0 ..< src.len():
    src[k] = first + step * float32(k)

proc fillHost[T, Sh, St](ht: var HostTensor[T, Sh, St], v: T) =
  ## Fills every element of the host tensor, canary value included
  for i in 0 ..< ht.len():
    ht.data[i] = v

proc checkClose(got, expect: HostTensor, msg: string) =
  ## Element-wise closeness at rtol 1e-5, abstol 1e-4, worst difference reported
  let report = allClose(got.view(), expect.view(), rtol = 1e-5, abstol = 1e-4)
  echo msg, ": worst |d| = ", formatFloat(report.worst, ffScientific, 3)
  doAssert report.bad == 0, msg & ": " & $report.bad & " mismatched elements"

# ═══════════════════════════════════════════════════════════════════════
#  Device kernels
# ═══════════════════════════════════════════════════════════════════════

proc stripReduceKernel(dst, src: distinct TensorView, op: static ReduceOp, clear: static bool) =
  ## Reduces src's first strip into dst[0], the value dst holds before
  ## the launch enters the accumulator
  ##
  ## Strips longer than one tile walk the tile grid, one register tile at a time:
  ## the first tile honors `clear`, every later tile combines onto the running total.
  let thrID = getLocalThreadID()
  var reg: tile(src.T)
  var acc = dst[0]
  let
    tiled = make_layout(src.size()).logical_divide(reg.getLayout())
    stripTiles = tiled.getTileGrid().size()
  for tileID in 0 ..< stripTiles:
    reg.load(src, thrID, tileID, padWith = reduceIdentity(op, src.T))
    if tileID == 0:
      when clear:
        acc.reduce(reg, op)
      else:
        acc.reduceAcc(reg, op)
    else:
      acc.reduceAcc(reg, op)
  if thrID == 0:
    dst[0] = acc

proc accumulateKernel(dst, a, b: distinct TensorView) =
  ## Two strip reductions onto one accumulator, the second accumulates
  let thrID = getLocalThreadID()
  var reg: tile(a.T)
  var acc = zero(a.T)
  reg.load(a, thrID, 0, padWith = reduceIdentity(roSum, a.T))
  acc.reduce(reg, roSum)
  reg.load(b, thrID, 0, padWith = reduceIdentity(roSum, b.T))
  acc.reduceAcc(reg, roSum)
  if thrID == 0:
    dst[0] = acc

proc rowReduceKernel(dst, src: distinct TensorView, op: static ReduceOp) =
  ## Per-row reduce over axis 2, one threadgroup per row, lane 0 stores
  let
    thrID = getLocalThreadID()
    rowFlat = getLocalTileID(dst.shape)
  let strip = atAxisIndex(src, 2, rowFlat)
  var reg: tile(strip.T)
  reg.load(strip, thrID, 0, padWith = reduceIdentity(op, strip.T))
  var outv = zero(strip.T)
  outv.reduce(reg, op)
  if thrID == 0:
    atAxisIndex(dst, 2, rowFlat)[0] = outv

# ═══════════════════════════════════════════════════════════════════════
#  The cases
# ═══════════════════════════════════════════════════════════════════════

proc stripCase[E: HwEngine](engine: E, msg: string, shape, strides: tuple, op: static ReduceOp, first, step: float32) =
  ## One clear = true strip reduction against the f64 host reference
  var src = makeHostTensor(float32, shape, strides)
  fillStrip(src, first, step)
  var got = makeHostTensor(float32, (1, 1), (1, 1))
  var expect = makeHostTensor(float32, (1, 1), (1, 1))
  case op
  of roSum: expect[0] = float32(refStripSum(src.view()))
  of roMax: expect[0] = float32(refStripMax(src.view()))
  of roMin: expect[0] = float32(refStripMin(src.view()))
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)
  engine.launchDeviceFn stripReduceKernel(gotDev, srcDev, op, true) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, 1)
  checkClose(got, expect, msg)

proc canaryCase[E: HwEngine](engine: E, msg: string, clear: static bool) =
  ## Sum over a compact strip with dst prefilled at the sentinel 12345,
  ## clear = true overwrites the sentinel, clear = false keeps it
  var src = makeHostTensor(float32, (1, 128), (128, 1))
  fillStrip(src, -64'f32, 1'f32)
  var got = makeHostTensor(float32, (1, 1), (1, 1))
  fillHost(got, 12345'f32)
  var expect = makeHostTensor(float32, (1, 1), (1, 1))
  let total = refStripSum(src.view())
  expect[0] = if clear: float32(total) else: float32(12345.0 + total)
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)
  engine.launchDeviceFn stripReduceKernel(gotDev, srcDev, roSum, clear) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, 1)
  checkClose(got, expect, msg)

proc accumulateCase[E: HwEngine](engine: E) =
  ## Two strip reductions onto one dst, the second accumulates onto the first
  var a = makeHostTensor(float32, (1, 128), (128, 1))
  var b = makeHostTensor(float32, (1, 100), (100, 1))
  fillStrip(a, -64'f32, 1'f32)
  fillStrip(b, 1'f32, 2'f32)
  var got = makeHostTensor(float32, (1, 1), (1, 1))
  var expect = makeHostTensor(float32, (1, 1), (1, 1))
  expect[0] = float32(refStripSum(a.view()) + refStripSum(b.view()))
  let
    aDev = engine.bridge(a)
    bDev = engine.bridge(b)
    gotDev = engine.bridge(got)
  engine.launchDeviceFn accumulateKernel(gotDev, aDev, bDev) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, 1)
  checkClose(got, expect, "accumulate, strip B onto strip A (clear = false)")

proc combineCase[E: HwEngine](engine: E, op: static ReduceOp) =
  ## Two strip reductions onto one dst: strip A reduces fresh (clear = true),
  ## strip B combines onto the same dst with clear = false
  var a = makeHostTensor(float32, (1, 128), (128, 1))
  var b = makeHostTensor(float32, (1, 300), (300, 1))
  var got = makeHostTensor(float32, (1, 1), (1, 1))
  var expect = makeHostTensor(float32, (1, 1), (1, 1))
  when op == roMax:
    # strip A must win the combine, a clear = false launch that ignores
    # the value dst holds would reduce strip B alone instead
    fillStrip(a, -64'f32, 8'f32)
    fillStrip(b, 1'f32, 2'f32)
    expect[0] = float32(max(refStripMax(a.view()), refStripMax(b.view())))
  else:
    fillStrip(a, -64'f32, 1'f32)
    fillStrip(b, 1'f32, 2'f32)
    expect[0] = float32(min(refStripMin(a.view()), refStripMin(b.view())))
  fillHost(got, 12345'f32)
  let
    aDev = engine.bridge(a)
    bDev = engine.bridge(b)
    gotDev = engine.bridge(got)
  engine.launchDeviceFn stripReduceKernel(gotDev, aDev, op, true) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.launchDeviceFn stripReduceKernel(gotDev, bDev, op, false) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, 1)
  when op == roMax:
    checkClose(got, expect, "max, strip B combines onto strip A (clear = false)")
  else:
    checkClose(got, expect, "min, strip B combines onto strip A (clear = false)")

proc atAxisIndexCase[E: HwEngine](engine: E, strides: tuple, msg: string) =
  ## Per-row sum and max over (3, 5, 7) along axis 2 at the given strides,
  ## every output slot prefilled at the sentinel 12345 and overwritten
  var src = makeHostTensor(float32, (3, 5, 7), strides)
  for i in 0 ..< 3:
    for j in 0 ..< 5:
      for k in 0 ..< 7:
        src[i, j, k] = float32(10*i + j - 3*k)
  var
    gotSum = makeHostTensor(float32, (3, 5, 1), (5, 1, 1))
    gotMax = makeHostTensor(float32, (3, 5, 1), (5, 1, 1))
    expectSum = makeHostTensor(float32, (3, 5, 1), (5, 1, 1))
    expectMax = makeHostTensor(float32, (3, 5, 1), (5, 1, 1))
  fillHost(gotSum, 12345'f32)
  fillHost(gotMax, 12345'f32)
  for i in 0 ..< 3:
    for j in 0 ..< 5:
      var s = 0.0
      var m = -Inf
      for k in 0 ..< 7:
        let v = float64(src[i, j, k])
        s += v
        m = max(m, v)
      expectSum[i, j, 0] = float32(s)
      expectMax[i, j, 0] = float32(m)
  let
    sumDev = engine.bridge(gotSum)
    maxDev = engine.bridge(gotMax)
  engine.launchDeviceFn rowReduceKernel(sumDev, engine.bridge(src), roSum) do:
    grid = (15, 1)
    blk = (32, 1)
  engine.launchDeviceFn rowReduceKernel(maxDev, engine.bridge(src), roMax) do:
    grid = (15, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr gotSum.data[0], sumDev.data, 15)
  engine.copyToHostFromDevice(addr gotMax.data[0], maxDev.data, 15)
  checkClose(gotSum, expectSum, msg & ", per-row sum")
  checkClose(gotMax, expectMax, msg & ", per-row max")

proc runTest =
  var engine = newEngine(runMetal)
  stripCase(engine, "sum, compact 128-element strip (one tile)",
      (1, 128), (128, 1), roSum, -64'f32, 1'f32)
  stripCase(engine, "sum, ragged 100-element strip",
      (1, 100), (100, 1), roSum, -50'f32, 1'f32)
  stripCase(engine, "max, ragged strip with all-negative elements",
      (1, 100), (100, 1), roMax, -1'f32, -1'f32)
  stripCase(engine, "min, ragged strip",
      (1, 100), (100, 1), roMin, 1'f32, 1'f32)
  stripCase(engine, "sum, 4096-element strip (32 tiles)",
      (1, 4096), (4096, 1), roSum, -64'f32, 1'f32)
  stripCase(engine, "max, ragged 300-element strip across the last tile",
      (1, 300), (300, 1), roMax, -64'f32, 1'f32)
  canaryCase(engine, "canary sentinel 12345, clear = true overwrites it", true)
  canaryCase(engine, "canary sentinel 12345, clear = false keeps it", false)
  accumulateCase(engine)
  combineCase(engine, roMax)
  combineCase(engine, roMin)
  atAxisIndexCase(engine, (35, 7, 1), "atAxisIndex over (3, 5, 7) row-major")
  atAxisIndexCase(engine, (70, 14, 2), "atAxisIndex over (3, 5, 7) with a 2x-scaled stride on axis 2")
  echo "t_tile_reduce_metal: all cases passed"

when isMainModule:
  runTest()
