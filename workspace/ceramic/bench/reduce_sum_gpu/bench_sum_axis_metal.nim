# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: axis-sum reduction bandwidth on Metal, GB/s.
# runCase takes source shape and strides, dst shape and strides and axis.
# Cases list what to run, nothing is derived.
# Correctness lives in the k_sum tests, not here.
# GB/s counts source bytes read (each element is in exactly one strip).
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
import workspace/ceramic/src/kernels/k_layout_fillwith_cpu
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/bench/reduce_sum_gpu/bench_sum_template

export bench_sum_template

proc fillPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Logical position mod 7, times 0.25: slot permutations change sums
  for i in 0 ..< tv.size():
    tv(i) = T(i mod 7) * T(0.25)

proc sumAxisLoop(dst, src: distinct TensorView, axis: static int, iters: int) =
  ## Bench harness: iters axis sums in one launch. Each strip's total is
  ## stored by exactly one threadgroup, the stores are idempotent, no
  ## reset is needed between iterations.
  for _ in 0 ..< iters:
    sum(dst, src, axis)

proc runCase(engine: MetalEngine, shape, srcStrides, dstShape, dstStrides: tuple, axis: static int, iters: int) =
  ## One (shape, strides, dstShape, dstStrides, axis): timed launches
  var
    srcHost = makeHostTensor(float32, shape, srcStrides)
    dstHost = makeHostTensor(float32, dstShape, dstStrides)
  var
    srcView = srcHost.view()
    dstView = dstHost.view()
  fillPattern(srcView)
  fillWith_cpu(dstView, 0'f32)
  let
    np = toInt(srcView.size())
    nonReducedSize = toInt(dstView.size())
    gx = min((nonReducedSize + 3) div 4, 8192)
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
  engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, nonReducedSize)
  let label = &"axis {axis}"
  reportSum(label, gx, np, iters, benchStart, benchStop)

const Cases2: array[6, tuple[src, dst: tuple[shape, strides: (int, int)], axis, iters: int]] = [
  # row-major, reduced axis stride 1, dense strips
  ((shape: (4096, 4096), strides: (4096, 1)),
   (shape: (4096, 1), strides: (4096, 1)), 1, 50),
  # column-major, reduced axis stride 4096
  ((shape: (4096, 4096), strides: (1, 4096)),
   (shape: (4096, 1), strides: (4096, 1)), 1, 50),
  # reduced-axis skip, non-reduced plane contiguous
  ((shape: (4096, 4096), strides: (8192, 1)),
   (shape: (1, 4096), strides: (4096, 1)), 0, 50),
  # small, launch bound
  ((shape: (64, 64), strides: (64, 1)),
   (shape: (64, 1), strides: (64, 1)), 1, 200),
  # large row-major
  ((shape: (16384, 16384), strides: (16384, 1)),
   (shape: (16384, 1), strides: (16384, 1)), 1, 20),
  # gappy non-reduced plane, non-reduced slot stride 8192
  ((shape: (4096, 4096), strides: (8192, 2)),
   (shape: (4096, 1), strides: (4096, 1)), 1, 20)]

const Cases3: array[5, tuple[src, dst: tuple[shape, strides: (int, int, int)], axis, iters: int]] = [
  # row-major, reduced axis stride 65536, non-reduced plane contiguous
  ((shape: (256, 128, 512), strides: (65536, 512, 1)),
   (shape: (1, 128, 512), strides: (65536, 512, 1)), 0, 20),
  # middle stride fastest, memory dense
  ((shape: (256, 128, 512), strides: (65536, 1, 128)),
   (shape: (256, 1, 512), strides: (512, 512, 1)), 1, 20),
  # middle stride slowest, memory dense
  ((shape: (256, 128, 512), strides: (512, 131072, 1)),
   (shape: (256, 128, 1), strides: (128, 1, 1)), 2, 20),
  # large row-major
  ((shape: (512, 512, 512), strides: (262144, 512, 1)),
   (shape: (1, 512, 512), strides: (262144, 512, 1)), 0, 20),
  # gappy non-reduced plane, non-reduced rows 512 elements apart
  ((shape: (256, 128, 512), strides: (131072, 1024, 1)),
   (shape: (1, 128, 512), strides: (65536, 512, 1)), 0, 20)]

const Cases4: array[1, tuple[src, dst: tuple[shape, strides: (int, int, int, int)], axis, iters: int]] = [
  # rank-4, middle stride slowest
  ((shape: (64, 32, 64, 128), strides: (8192, 524288, 128, 1)),
   (shape: (64, 1, 64, 128), strides: (8192, 128, 128, 1)), 1, 20)]

proc main() =
  var engine = bkMetal.newEngine()
  echo "axis sum, Metal, blk = (32, 1)"
  separator()
  template runCases(cases: untyped) =
    staticFor ci, 0, cases.len:
      runCase(engine, cases[ci].src.shape, cases[ci].src.strides,
          cases[ci].dst.shape, cases[ci].dst.strides, cases[ci].axis, cases[ci].iters)
  runCases(Cases2)
  runCases(Cases3)
  runCases(Cases4)
  separator()

when isMainModule:
  main()
