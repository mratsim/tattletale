# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: axis-sum reduction bandwidth on Metal, GB/s.
# - shape (16, 16, cols), cols swept, the swept axis last
# - axis 2 reads contiguous lines, axis 0/1 read strided lines of 16
# - GB/s counts source bytes read (each element is in exactly one strip)
# Run:
#   nim c -r -d:release --outdir:build/bench \
#     workspace/ceramic/bench/reduce_sum_gpu/bench_sum_axis_metal.nim (from tattletale)

import std/[monotimes, strformat, os, strutils]
import workspace/crucible
import workspace/crucible/src/runtime/engines/metal
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_tile_sum
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/bench/reduce_sum_gpu/bench_sum_template

export bench_sum_template

const Cols = [1_000, 4_000, 16_000, 64_000]
const Rows = 16
const Mid = 16

proc fillAxisInputs(srcTV: TensorView) =
  for i in 0 ..< srcTV.size():
    srcTV[i] = float32(i mod 7) * 0.25'f32

func refSumAxis3(src: TensorView, axis: static int): seq[float32] =
  ## f64 accumulation per output slot, the arbiter for the axis sums
  let (r, m, c) = (src.shape[0], src.shape[1], src.shape[2])
  when axis == 0:
    result = newSeq[float32](m * c)
    for mm in 0 ..< m:
      for cc in 0 ..< c:
        var acc = 0.0
        for rr in 0 ..< r:
          acc += src[rr, mm, cc].float64
        result[mm * c + cc] = float32(acc)
  elif axis == 1:
    result = newSeq[float32](r * c)
    for rr in 0 ..< r:
      for cc in 0 ..< c:
        var acc = 0.0
        for mm in 0 ..< m:
          acc += src[rr, mm, cc].float64
        result[rr * c + cc] = float32(acc)
  else:
    result = newSeq[float32](r * m)
    for rr in 0 ..< r:
      for mm in 0 ..< m:
        var acc = 0.0
        for cc in 0 ..< c:
          acc += src[rr, mm, cc].float64
        result[rr * m + mm] = float32(acc)

proc sumAxisLoop(dst, src: distinct TensorView, axis: static int, iters: int) =
  ## Bench harness: iters axis sums in one launch. Each strip's total is
  ## stored by exactly one threadgroup, the stores are idempotent, no
  ## reset is needed between iterations.
  for _ in 0 ..< iters:
    sum(dst, src, axis)

proc benchAxisSum(engine: MetalEngine, axis: static int, cols, iters: int): auto =
  ## One axis-sum launch config, returned for the host cross-check
  let one = 1  # runtime int for the layouts to stay dynamic
  let shape = (Rows, Mid, cols)
  let dstShape = when axis == 0: (one, Mid, cols)
                 elif axis == 1: (Rows, one, cols)
                 else: (Rows, Mid, one)
  var
    srcHost = makeHostTensor(float32, shape, (Mid * cols, cols, 1))
    dstHost = makeHostTensor(float32, dstShape,
                             (dstShape[1] * dstShape[2], dstShape[2], 1))
  fillAxisInputs(srcHost.view())
  # strips: the output slots, one per kept coordinate
  let strips = when axis == 0: Mid * cols
               elif axis == 1: Rows * cols
               else: Rows * Mid
  let gx = min(strips, 1024)
  # warmup
  block:
    let
      srcDev = engine.bridge(srcHost)
      dstDev = engine.bridge(dstHost)
    engine.launchDeviceFn sumAxisLoop(dstDev, srcDev, axis, 3) do:
      grid = (gx, 1)
      blk = (32, 1)
  # timed, the launch overhead leaves the measurement
  let
    srcDev = engine.bridge(srcHost)
    dstDev = engine.bridge(dstHost)
  let benchStart = getMonoTime()
  engine.launchDeviceFn sumAxisLoop(dstDev, srcDev, axis, iters) do:
    grid = (gx, 1)
    blk = (32, 1)
  let benchStop = getMonoTime()
  let slots = when axis == 0: Mid * cols
              elif axis == 1: Rows * cols
              else: Rows * Mid
  engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, slots)
  let np = shape[0] * shape[1] * shape[2]
  reportSum("axis " & $axis, gx, np, iters, benchStart, benchStop)
  dstHost

proc runAxis(engine: MetalEngine, axis: static int, cols, iters: int) =
  ## One axis: bench + f64-reference cross-check
  var srcHost = makeHostTensor(float32, (Rows, Mid, cols), (Mid * cols, cols, 1))
  fillAxisInputs(srcHost.view())
  let expect = refSumAxis3(srcHost.view(), axis)
  let got = benchAxisSum(engine, axis, cols, iters)
  let slots = when axis == 0: Mid * cols
              elif axis == 1: Rows * cols
              else: Rows * Mid
  for i in 0 ..< slots:
    doAssert abs(got.data[i].float64 - expect[i].float64) <
      1e-2 * max(1.0, abs(expect[i].float64)),
      &"axis {axis} sum wrong at slot {i}, cols {cols}: got {got.data[i]}, expect {expect[i]}"

proc main() =
  var engine = bkMetal.newEngine()
  echo "axis sum, Metal, blk = (32, 1), shape (16, 16, cols)"
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  for cols in Cols:
    if wanted.len == 0 or cols in wanted:
      let np = Rows * Mid * cols
      let iters = if np <= 200_000: 200
                  elif np <= 1_000_000: 50
                  else: 20
      runAxis(engine, 0, cols, iters)
      runAxis(engine, 1, cols, iters)
      runAxis(engine, 2, cols, iters)
      separator()
  separator()

when isMainModule:
  main()
