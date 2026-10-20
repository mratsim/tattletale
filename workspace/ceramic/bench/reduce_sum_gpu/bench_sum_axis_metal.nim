# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: axis-sum reduction bandwidth on Metal, GB/s.
# - cube shape (n, n, n), n swept
# - two stride layouts per cube: contiguous (n^2, n, 1) and the last
#   two axes swapped (n^2, 1, n), so each reduced-axis stride
#   (1, n, n^2) appears in both layouts
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

const Cubes = [64, 128, 256]

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

proc benchAxisSum(engine: MetalEngine, axis: static int, n: int, swapped: static bool, iters: int): auto =
  ## One axis-sum launch config on an n-cube, returned for the host cross-check
  let one = 1  # runtime int for the layouts to stay dynamic
  let srcStrides = when swapped: (n * n, 1, n) else: (n * n, n, 1)
  let dstShape = when axis == 0: (one, n, n)
                 elif axis == 1: (n, one, n)
                 else: (n, n, one)
  let dstStrides = (dstShape[1] * dstShape[2], dstShape[2], 1)
  var
    srcHost = makeHostTensor(float32, (n, n, n), srcStrides)
    dstHost = makeHostTensor(float32, dstShape, dstStrides)
  fillAxisInputs(srcHost.view())
  let strips = n * n
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
  engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, n * n)
  let np = n * n * n
  reportSum(when swapped: "axis " & $axis & " sw" else: "axis " & $axis, gx, np, iters, benchStart, benchStop)
  dstHost

proc runAxis(engine: MetalEngine, axis: static int, n: int, swapped: static bool, iters: int) =
  ## One (axis, layout): bench + f64-reference cross-check
  let srcStrides = when swapped: (n * n, 1, n) else: (n * n, n, 1)
  var srcHost = makeHostTensor(float32, (n, n, n), srcStrides)
  fillAxisInputs(srcHost.view())
  let expect = refSumAxis3(srcHost.view(), axis)
  let got = benchAxisSum(engine, axis, n, swapped, iters)
  for i in 0 ..< n * n:
    doAssert abs(got.data[i].float64 - expect[i].float64) <
      1e-2 * max(1.0, abs(expect[i].float64)),
      &"axis {axis} sum wrong at slot {i}, n {n}: got {got.data[i]}, expect {expect[i]}"

proc main() =
  var engine = bkMetal.newEngine()
  echo "axis sum, Metal, blk = (32, 1), cube (n, n, n), contiguous and swapped strides"
  var wanted: seq[int]
  for p in commandLineParams():
    wanted.add parseInt(p)
  separator()
  for n in Cubes:
    if wanted.len == 0 or n in wanted:
      let np = n * n * n
      let iters = if np <= 200_000: 200
                  elif np <= 1_000_000: 50
                  else: 20
      runAxis(engine, 0, n, false, iters)
      runAxis(engine, 1, n, false, iters)
      runAxis(engine, 2, n, false, iters)
      runAxis(engine, 0, n, true, iters)
      runAxis(engine, 1, n, true, iters)
      runAxis(engine, 2, n, true, iters)
      separator()
  separator()

when isMainModule:
  main()
