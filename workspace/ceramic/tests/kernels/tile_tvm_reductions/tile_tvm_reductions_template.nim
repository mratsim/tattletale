# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile axis reduce through the atom's thread->value map, per backend
##
## Sum and max over one axis, the keepdims tile compared against host
## references through allClose. The k_sum suite covers the tensor kernel.

import std/[math, strutils]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/src/kernels/k_layout_fillwith_gpu
import workspace/ceramic/src/hardware/h_mma_registry

export int_tuples
export layout_algebra
export tensors
export tile_algebra

func refSum[T, Sh, St](dst, src: TensorView[T, Sh, St], axis: static int) =
  ## Axis sum over `src` into the keepdims tile `dst`: every slot of a kept
  ## coordinate stores its total, f64 accumulation
  for i in 0 ..< toInt(dst.size()):
    var crd = idx2crd(dst.shape, i)
    var acc = 0.0'f64
    for a in 0 ..< toInt(src.shape[axis]):
      var srcCrd = crd
      srcCrd[axis] = a
      acc += float64(src[srcCrd])
    dst[crd] = float32(acc)

func refMax[T, Sh, St](dst, src: TensorView[T, Sh, St], axis: static int) =
  ## Axis max over `src` into the keepdims tile `dst`
  for i in 0 ..< toInt(dst.size()):
    var crd = idx2crd(dst.shape, i)
    var acc = -Inf
    for a in 0 ..< toInt(src.shape[axis]):
      var srcCrd = crd
      srcCrd[axis] = a
      acc = max(acc, float64(src[srcCrd]))
    dst[crd] = float32(acc)

proc sumKernel*(dst, src: distinct TensorView, mma: static TiledMma, axis: static int) =
  ## Axis reduce of `src` into the keepdims `dst`:
  ## - the tensor is contiguous, the reduce axis first or last
  ## - the flattened (kept, axis) 2-mode view puts the reduce axis last,
  ##   the map's columns carry it, one threadgroup per whole view
  ## - tiles walk the view's columns, the totals store per tile block
  let
    sizeS = toInt(src.shape[axis])
    kept = toInt(src.size()) div sizeS
  when axis == 0:
    var
      mat = make_view(src.data, make_layout((kept, sizeS), (1, kept)))
      outMat = make_view(dst.data, make_layout((kept, sizeS), (1, kept)))
  else:
    var
      mat = make_view(src.data, make_layout((kept, sizeS), (sizeS, 1)))
      outMat = make_view(dst.data, make_layout((kept, sizeS), (sizeS, 1)))
  const tvm = mma.getThreadValueMap(opC)
  let thrID = getLocalThreadID()
  var reg: tile(float32, mma.getMmaLayout(opC))
  var part: tile(float32, mma.getMmaLayout(opC))
  const tileLen = toInt(mma.getMmaLayout(opC).cosize())
  const
    mt = mma.MT
    nt = mma.NT
  let
    keptTiles = (kept + mt - 1) div mt
    tiles = (sizeS + nt - 1) div nt
  for mT in 0 ..< keptTiles:
    var acc: tile(float32, mma.getMmaLayout(opC))
    acc.zeros()
    for t in 0 ..< tiles:
      reg.load(mat, tvm, thrID, (mT, t), padWith = zero(float32))
      sum(part, reg, 1, tvm)
      for i in 0 ..< tileLen:
        {.unroll.}
        acc.data[i] += part.data[i]
    for t in 0 ..< tiles:
      outMat.store(acc, tvm, thrID, (mT, t))

proc maxKernel*(dst, src: distinct TensorView, mma: static TiledMma, axis: static int) =
  ## Axis max of `src` into the keepdims `dst`, same contract as `sumKernel`.
  let
    sizeS = toInt(src.shape[axis])
    kept = toInt(src.size()) div sizeS
  when axis == 0:
    var
      mat = make_view(src.data, make_layout((kept, sizeS), (1, kept)))
      outMat = make_view(dst.data, make_layout((kept, sizeS), (1, kept)))
  else:
    var
      mat = make_view(src.data, make_layout((kept, sizeS), (sizeS, 1)))
      outMat = make_view(dst.data, make_layout((kept, sizeS), (sizeS, 1)))
  const tvm = mma.getThreadValueMap(opC)
  let thrID = getLocalThreadID()
  var reg: tile(float32, mma.getMmaLayout(opC))
  var part: tile(float32, mma.getMmaLayout(opC))
  const tileLen = toInt(mma.getMmaLayout(opC).cosize())
  const
    mt = mma.MT
    nt = mma.NT
  let
    keptTiles = (kept + mt - 1) div mt
    tiles = (sizeS + nt - 1) div nt
  for mT in 0 ..< keptTiles:
    var acc: tile(float32, mma.getMmaLayout(opC))
    acc.negInf()
    for t in 0 ..< tiles:
      reg.load(mat, tvm, thrID, (mT, t), padWith = negInf(float32))
      max(part, reg, 1, tvm)
      for i in 0 ..< tileLen:
        {.unroll.}
        acc.data[i] = max(acc.data[i], part.data[i])
    for t in 0 ..< tiles:
      outMat.store(acc, tvm, thrID, (mT, t))

proc checkSum[E: HwEngine](engine: E, axis: static int, shape: tuple) =
  ## Sum against the host reference over `shape`:
  ## - the reduce axis is first or last
  ## - one threadgroup reduces the whole flattened view
  var src = makeHostTensor(float32, shape, LayoutRight)
  for i in 0 ..< src.len:
    src.data[i] = float32(i) - 64.0'f32
  var
    got = makeHostTensor(float32, shape, LayoutRight)
    expect = makeHostTensor(float32, shape, LayoutRight)
  refSum(expect.view(), src.view(), axis)
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)

  engine.launchDeviceFn sumKernel(gotDev, srcDev,
      tiledMma(float32, float32, float32, 32, 32, 32), axis) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, got.len)
  assertAllClose(got.view(), expect.view(), msg = "sum " & $shape)

proc checkMax[E: HwEngine](engine: E, axis: static int, shape: tuple) =
  ## Max against the host reference over `shape`:
  ## - the reduce axis is first or last
  ## - one threadgroup reduces the whole flattened view
  var src = makeHostTensor(float32, shape, LayoutRight)
  for i in 0 ..< src.len:
    src.data[i] = float32(i) - 64.0'f32
  var
    got = makeHostTensor(float32, shape, LayoutRight)
    expect = makeHostTensor(float32, shape, LayoutRight)
  refMax(expect.view(), src.view(), axis)
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)

  engine.launchDeviceFn maxKernel(gotDev, srcDev,
      tiledMma(float32, float32, float32, 32, 32, 32), axis) do:
    grid = (1, 1)
    blk = (32, 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, got.len)
  assertAllClose(got.view(), expect.view(), msg = "max " & $shape)

proc run_tile_reduce_tests*[E: HwEngine](engine: E, moduleName: string) =
  checkSum(engine, 1, (32, 32))
  checkSum(engine, 1, (32, 100))
  checkSum(engine, 0, (32, 32))
  checkSum(engine, 2, (2, 32, 32))
  checkSum(engine, 2, (2, 32, 100))
  checkSum(engine, 3, (3, 2, 32, 32))
  checkSum(engine, 3, (2, 2, 32, 100))
  checkMax(engine, 1, (32, 32))
  checkMax(engine, 1, (32, 100))
  checkMax(engine, 0, (32, 32))
  checkMax(engine, 2, (2, 32, 32))
  checkMax(engine, 2, (2, 32, 100))
  checkMax(engine, 3, (3, 2, 32, 32))
  checkMax(engine, 3, (2, 2, 32, 100))
  echo moduleName & ": all cases passed"
