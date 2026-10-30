# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: RMSNorm bandwidth on Metal, GB/s.
# runCase takes the source and dst shape and strides, the axis, and the weighted switch.
# Cases list what to run, nothing is derived.
# Correctness lives in the k_rmsnorm tests, not here.
#
# GB/s traffic model: 2 source passes plus 1 destination pass = 12 bytes per logical element.
# The weight vector is rowSize floats total and stays cache resident, uncounted.
#
# Timing: a warmup launch then 3 timed launches per case, the median per-iteration time is reported.
#
# Run:
#   nim c -r -d:release --outdir:build/bench workspace/ceramic/bench/rmsnorm_gpu/bench_rmsnorm_metal.nim (from tattletale)

import std/[algorithm, monotimes, strformat, strutils, times, typetraits]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_tile_rmsnorm
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/tests/ceramic_testutils

proc separator(length = 118) =
  writeLine(stdout, "-".repeat(length))

proc makeHostVector[T](_: typedesc[T], size: int): auto =
  ## Rank-1 host tensor of shape (size,), stride 1.
  ## Rank-1 layouts flatten the shape and stride tuples into scalars.
  ## The shared rank-N makeHostTensor rejects that shape.
  ## This constructor serves the weight vector case.
  let layout = make_layout((size,), (1,))
  HostTensor[T, typeof(layout.shape), typeof(layout.stride)](
    data: cast[ptr UncheckedArray[T]](allocShared(size * sizeof(T))),
    offset: 0,
    layout: layout)

proc fillPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Logical position mod 7, times 0.25 plus 0.5: slot permutations change sums
  when tupleLen(Sh) == 2:
    var logical = 0
    for i in 0 ..< tv.getLayout().shape[0]:
      for j in 0 ..< tv.getLayout().shape[1]:
        tv[i, j] = T(logical mod 7) * T(0.25) + T(0.5)
        logical.inc
  elif tupleLen(Sh) == 3:
    var logical = 0
    for i in 0 ..< tv.getLayout().shape[0]:
      for j in 0 ..< tv.getLayout().shape[1]:
        for k in 0 ..< tv.getLayout().shape[2]:
          tv[i, j, k] = T(logical mod 7) * T(0.25) + T(0.5)
          logical.inc
  else:
    var logical = 0
    for i in 0 ..< tv.getLayout().shape[0]:
      for j in 0 ..< tv.getLayout().shape[1]:
        for k in 0 ..< tv.getLayout().shape[2]:
          for l in 0 ..< tv.getLayout().shape[3]:
            tv[i, j, k, l] = T(logical mod 7) * T(0.25) + T(0.5)
            logical.inc

proc fillWeight[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Position mod 5, times 0.25 plus 1.0: per-axis weights in [1.0, 2.0],
  ## the same pattern the MLX runner fills.
  let data = cast[ptr UncheckedArray[T]](tv.data)
  for r in 0 ..< tv.size():
    data[r] = T(r mod 5) * T(0.25) + T(1.0)

proc rmsnormLoop(weighted: static bool, dst, src, weight: distinct TensorView, eps: float32, axis: static int, iters: int) =
  ## Bench harness: iters RMSNorm passes in one launch.
  ## Every row is overwritten each iteration, the stores are idempotent.
  ## No reset is needed between iterations.
  for _ in 0 ..< iters:
    when weighted:
      rmsnorm(dst, src, weight, eps, axis)
    else:
      rmsnorm(dst, src, eps, axis)

proc runCase(engine: MetalEngine, shape, srcStrides, dstStrides: tuple, axis: static int, iters: int, weighted: static bool) =
  ## One (shape, srcStrides, dstStrides, axis) case: one warmup launch then 3 timed launches.
  ## The median per-iteration time is reported as GB/s.
  var
    srcHost = makeHostTensor(float32, shape, srcStrides)
    dstHost = makeHostTensor(float32, shape, dstStrides)
  var
    srcView = srcHost.view()
    dstView = dstHost.view()
    weightHost = makeHostVector(float32, toInt(srcView.shape[axis]))
  fillPattern(srcView)
  # dst stays uninitialized, the kernel overwrites every element of every row
  block:
    var wv = weightHost.view()
    fillWeight(wv)
  let
    np = toInt(srcView.size())
    rows = np div toInt(srcView.shape[axis])
    gx = min((rows + 3) div 4, 8192)
  # warmup
  block:
    let
      srcDev = engine.bridge(srcHost)
      dstDev = engine.bridge(dstHost)
      weightDev = engine.bridge(weightHost)
    engine.launchDeviceFn rmsnormLoop(weighted, dstDev, srcDev, weightDev, 1e-5'f32, axis, 3) do:
      grid = (gx, 1)
      blk = (32, 1)
  # timed, the launch overhead leaves the measurement, the device to host
  # copy waits for the iteration before the next rep starts
  var timings: array[3, int64]
  staticFor rep, 0, 3:
    block:
      let
        srcDev = engine.bridge(srcHost)
        dstDev = engine.bridge(dstHost)
        weightDev = engine.bridge(weightHost)
      let benchStart = getMonoTime()
      engine.launchDeviceFn rmsnormLoop(weighted, dstDev, srcDev, weightDev, 1e-5'f32, axis, iters) do:
        grid = (gx, 1)
        blk = (32, 1)
      let benchStop = getMonoTime()
      engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, 1)
      timings[rep] = inNanoseconds(benchStop - benchStart) div iters
  sort(timings)
  let gbps = np.float * 12.0 / (timings[1].float / 1e9) / 1e9
  writeLine(stdout, &"{np:>10}  {rows:>7}  {gx:>5}  {iters:>6}   {gbps:.1f}")

const Cases2: array[6, tuple[src, dst: tuple[shape, strides: (int, int)], axis, iters: int]] = [
  # row-major, dense rows, the transformer RMSNorm
  ((shape: (4096, 4096), strides: (4096, 1)),
   (shape: (4096, 4096), strides: (4096, 1)), 1, 50),
  # column-major, rows 4096 elements apart
  ((shape: (4096, 4096), strides: (1, 4096)),
   (shape: (4096, 4096), strides: (4096, 1)), 1, 50),
  # row skip, rows 8192 elements apart
  ((shape: (4096, 4096), strides: (8192, 1)),
   (shape: (4096, 4096), strides: (4096, 1)), 1, 50),
  # small, launch bound
  ((shape: (64, 64), strides: (64, 1)),
   (shape: (64, 64), strides: (64, 1)), 1, 200),
  # large row-major
  ((shape: (16384, 16384), strides: (16384, 1)),
   (shape: (16384, 16384), strides: (16384, 1)), 1, 20),
  # gappy stride-2 rows
  ((shape: (4096, 4096), strides: (8192, 2)),
   (shape: (4096, 4096), strides: (4096, 1)), 1, 20)]

const Cases3: array[4, tuple[src, dst: tuple[shape, strides: (int, int, int)], axis, iters: int]] = [
  # row-major, dense rows of 512
  ((shape: (256, 128, 512), strides: (65536, 512, 1)),
   (shape: (256, 128, 512), strides: (65536, 512, 1)), 2, 20),
  # last-axis stride 128, elements scattered within a row
  ((shape: (256, 128, 512), strides: (65536, 1, 128)),
   (shape: (256, 128, 512), strides: (65536, 512, 1)), 2, 20),
  # middle stride slowest, dense rows of 512
  ((shape: (256, 128, 512), strides: (512, 131072, 1)),
   (shape: (256, 128, 512), strides: (65536, 512, 1)), 2, 20),
  # large row-major
  ((shape: (512, 512, 512), strides: (262144, 512, 1)),
   (shape: (512, 512, 512), strides: (262144, 512, 1)), 2, 20)]

const Cases4: array[1, tuple[src, dst: tuple[shape, strides: (int, int, int, int)], axis, iters: int]] = [
  # rank-4, dense rows of 128
  ((shape: (64, 32, 64, 128), strides: (8192, 524288, 128, 1)),
   (shape: (64, 32, 64, 128), strides: (8192, 524288, 128, 1)), 3, 20)]

proc main() =
  var engine = newEngine(runMetal)
  writeLine(stdout, "rmsnorm, Metal, blk = (32, 1), one warp per row, GB/s = logical elements x 12 bytes")
  writeLine(stdout, "        np      rows     gx   iters   GB/s")
  separator()
  template runCases(cases: untyped, weighted: static bool) =
    staticFor ci, 0, cases.len:
      runCase(engine, cases[ci].src.shape, cases[ci].src.strides,
          cases[ci].dst.strides, cases[ci].axis, cases[ci].iters, weighted)
  runCases(Cases2, true)
  runCases(Cases3, true)
  runCases(Cases4, true)
  separator()

when isMainModule:
  main()
