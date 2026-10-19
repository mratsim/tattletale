# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Backend-agnostic test driver for the tensor reductions.
##
## Test that the full and last-axis tensor sums match the host references.
##
## Run:
##   nim c -r --outdir:build/tests \
##     workspace/ceramic/tests/kernels/k_sum/t_kernel_sum.nim (from tattletale)

import std/[math, random, strformat, times]
import workspace/crucible
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/kernels/k_tile_sum

# ═══════════════════════════════════════════════════════════════
#  fp32 host reference
# ═══════════════════════════════════════════════════════════════

func refSum[T, Sh, St](src: TensorView[T, Sh, St]): float32 =
  ## Sum of the tensor, f64 accumulation
  var acc = 0.0
  for i in 0 ..< src.size():
    acc += float64(src.data[i])
  float32(acc)

func refSum[T, ShS, StS, ShD, StD](dst: TensorView[T, ShD, StD], src: TensorView[T, ShS, StS], axis: static int) =
  ## Sums over the last axis, one accumulator per output slice
  doAssert axis == src.shape.rank - 1
  when src.shape.rank == 2:
    for row in 0 ..< src.shape[0]:
      var acc = 0.0
      for col in 0 ..< src.shape[1]:
        acc += float64(src[row, col])
      dst[row, 0] = float32(acc)
  else:
    let lastExtent = src.shape[src.shape.rank - 1]
    for s in 0 ..< src.size() div lastExtent:
      var acc = 0.0
      for i in s * lastExtent ..< (s + 1) * lastExtent:
        acc += float64(src.data[i])
      dst.data[s] = float32(acc)

# ═══════════════════════════════════════════════════════════════
#  The checks
# ═══════════════════════════════════════════════════════════════

proc checkSum*[E: HwEngine](rng: var Rand, engine: E, rows, cols: static int, sSrc: static (int, int)) =
  ## One sum_full launch over a random (rows, cols) tensor
  var src = randomHostTensor(rng, float32, (rows, cols), sSrc)
  var got = makeHostTensor(float32, (1, 1), (1, 1))
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)
  # the kernel's grid-stride loop completes the tile grid from any launch grid
  let
    gridX = rng.rand(1 .. 2 * ceil_div(rows, 32))
    gridY = rng.rand(1 .. 2 * ceil_div(cols, 32))
  engine.launchDeviceFn sum_full((8, 4), gotDev, srcDev) do:
    grid = (gridX, gridY)
    blk = (32, 1)
  var expect = makeHostTensor(float32, (1, 1), (1, 1))
  expect.data[0] = refSum(src.view())
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, 1)
  assertAllClose(got.view(), expect.view(), rtol = 1e-5, abstol = 1e-4)

proc checkSumAxis*[E: HwEngine](rng: var Rand, engine: E, rows, cols: static int, sSrc: static (int, int)) =
  ## One sum_axis launch over a random (rows, cols) tensor
  var src = randomHostTensor(rng, float32, (rows, cols), sSrc)
  var got = makeHostTensor(float32, (rows, 1), (1, 1))
  let
    srcDev = engine.bridge(src)
    gotDev = engine.bridge(got)
  # the kernel's grid-stride loop completes the tile grid from any launch grid
  let
    gridX = rng.rand(1 .. 2 * ceil_div(rows, 32))
    gridY = rng.rand(1 .. 2 * ceil_div(cols, 32))
  engine.launchDeviceFn sum_axis((8, 4), gotDev, srcDev) do:
    grid = (gridX, gridY)
    blk = (32, 1)
  var expect = makeHostTensor(float32, (rows, 1), (1, 1))
  refSum(expect.view(), src.view(), 1)
  engine.copyToHostFromDevice(addr got.data[0], gotDev.data, rows)
  assertAllClose(got.view(), expect.view(), rtol = 1e-5, abstol = 1e-4)

# ═══════════════════════════════════════════════════════════════
#  The driver
# ═══════════════════════════════════════════════════════════════

proc run_sum_tests*[E: HwEngine](engine: E, moduleName: string) =
  var rng: Rand
  let seed = uint32(getTime().toUnix() and (1'i64 shl 32 - 1)) # unixTime mod 2^32
  rng.seed(int64(seed))
  echo "\n------------------------------------------------------\n"
  echo moduleName, " xoroshiro128+ seed: ", seed

  # stride permutations within one covered tile: the (8, 4) block covers (8, 128)
  checkSum(rng, engine, 8, 128, (128, 1))
  checkSum(rng, engine, 8, 128, (1, 8))
  checkSumAxis(rng, engine, 8, 128, (128, 1))
  checkSumAxis(rng, engine, 8, 128, (1, 8))

  # ── 3D tensor: the (2, 2, 2) block covers (2, 2, 64) ──
  block:
    var src = randomHostTensor(rng, float32, (2, 2, 60), (120, 60, 1))
    var sumGot = makeHostTensor(float32, (1, 1, 1), (1, 1, 1))
    var axisGot = makeHostTensor(float32, (2, 2, 1), (2, 1, 1))
    let srcDev = engine.bridge(src)
    var
      expectSum = makeHostTensor(float32, (1, 1, 1), (1, 1, 1))
      expectAxis = makeHostTensor(float32, (2, 2, 1), (2, 1, 1))
    expectSum.data[0] = refSum(src.view())
    refSum(expectAxis.view(), src.view(), 2)
    let sumDev = engine.bridge(sumGot)
    engine.launchDeviceFn sum_full((2, 2, 2), sumDev, engine.bridge(src)) do:
      grid = (1, 1)
      blk = (32, 1)
    let axisDev = engine.bridge(axisGot)
    engine.launchDeviceFn sum_axis((2, 2, 2), axisDev, engine.bridge(src)) do:
      grid = (1, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr sumGot.data[0], sumDev.data, 1)
    engine.copyToHostFromDevice(addr axisGot.data[0], axisDev.data, 4)
    assertAllClose(sumGot.view(), expectSum.view())
    assertAllClose(axisGot.view(), expectAxis.view(), msg = "3D sum with axis")

  # ── 4D tensor: the (2, 2, 2, 2) block covers (2, 2, 2, 64) ──
  block:
    var src = randomHostTensor(rng, float32, (2, 2, 2, 50), (200, 100, 50, 1))
    var sumGot = makeHostTensor(float32, (1, 1, 1, 1), (1, 1, 1, 1))
    var axisGot = makeHostTensor(float32, (2, 2, 2, 1), (2, 2, 1, 1))
    let srcDev = engine.bridge(src)
    var
      expectSum = makeHostTensor(float32, (1, 1, 1, 1), (1, 1, 1, 1))
      expectAxis = makeHostTensor(float32, (2, 2, 2, 1), (2, 2, 1, 1))
    expectSum.data[0] = refSum(src.view())
    refSum(expectAxis.view(), src.view(), 3)
    let sumDev = engine.bridge(sumGot)
    engine.launchDeviceFn sum_full((2, 2, 2, 2), sumDev, engine.bridge(src)) do:
      grid = (1, 1)
      blk = (32, 1)
    let axisDev = engine.bridge(axisGot)
    engine.launchDeviceFn sum_axis((2, 2, 2, 2), axisDev, engine.bridge(src)) do:
      grid = (1, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr sumGot.data[0], sumDev.data, 1)
    engine.copyToHostFromDevice(addr axisGot.data[0], axisDev.data, 8)
    assertAllClose(sumGot.view(), expectSum.view())
    assertAllClose(axisGot.view(), expectAxis.view(), msg = "4D sum with axis")
