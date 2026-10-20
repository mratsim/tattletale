# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import std/[monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/crucible/src/runtime/engines/metal
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_tile_sum
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/bench/gemm_gpu/bench_blueprint

export bench_blueprint
export int_tuples
export layout_algebra
export tensors
export ceramic_testutils

proc fillSumInputs*(srcTV: TensorView, np: int) =
  for i in 0 ..< np:
    srcTV[i] = float32(i mod 7) * 0.25'f32

proc sumExpect*(np: int): float64 =
  ## Host reference: the f64 sum of fillSumInputs' values
  ##
  ## - f32 accumulation loses 0.25 increments past ~1e7
  ## - the GPU tree sum is right, the f64 host reference is the arbiter
  for i in 0 ..< np:
    result += (i mod 7).float64 * 0.25'f64

proc sumLoop*(dst, src: distinct TensorView, iters: int) =
  ## Bench harness: iters full-sum reductions in one launch, launch
  ## overhead excluded from the measurement.
  ## - Accumulation is atomic, dst grows by the total per iteration
  ## - The bench resets dst before timing, the cross-check divides by iters
  for _ in 0 ..< iters:
    sum(dst, src)

proc reportSum*(policy: string, G, np, iters: int, startTime, stopTime: MonoTime) =
  let ns = inNanoseconds(stopTime - startTime) div iters
  let gbps = np.float * 4.0 / (ns.float / 1e9) / 1e9
  echo &"{np:>9}  {iters:>6}  {policy:>8}    {G:>5}   {gbps:.1f}"

template benchSum*(policy: string, G, np, iters: int, body: untyped) =
  let benchStart = getMonoTime()
  body
  let benchStop = getMonoTime()
  reportSum(policy, G, np, iters, benchStart, benchStop)

proc benchSumCeramic*(engine: MetalEngine, np, iters: int): auto =
  ## Shipped sum, atomic single-pass: each threadgroup folds a grid-stride
  ## strip and the leader banks it with one atomic add.
  let gx = min(ceil_div(np, 4096), 1024)
  let one = 1  # runtime int for the layouts to stay dynamic
  var
    srcHost = makeHostTensor(float32, (np, one), (one, np))
    dstHost = makeHostTensor(float32, (1, 1), (1, 1))
  fillSumInputs(srcHost.view(), np)
  # warmup, the kernel loops 3 times inside one launch
  block:
    let
      srcDev = engine.bridge(srcHost)
      dstDev = engine.bridge(dstHost)
    engine.launchDeviceFn sumLoop(dstDev, srcDev, 3) do:
      grid = (gx, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, 1)
  # timed, the launch overhead leaves the measurement
  dstHost.data[0] = 0
  let
    srcDev = engine.bridge(srcHost)
    dstDev = engine.bridge(dstHost)
  benchSum("atomic", gx, np, iters):
    engine.launchDeviceFn sumLoop(dstDev, srcDev, iters) do:
      grid = (gx, 1)
      blk = (32, 1)
  engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, 1)
  dstHost

proc benchOneThreadgroup*(engine: MetalEngine, np, iters: int): auto =
  ## One threadgroup folds the whole tensor with a grid-stride loop.
  let one = 1  # runtime int for the layouts to stay dynamic
  var
    srcHost = makeHostTensor(float32, (np, one), (one, np))
    dstHost = makeHostTensor(float32, (1, 1), (1, 1))
  fillSumInputs(srcHost.view(), np)
  # warmup, the kernel loops 3 times inside one launch
  block:
    let
      srcDev = engine.bridge(srcHost)
      dstDev = engine.bridge(dstHost)
    engine.launchDeviceFn sumLoop(dstDev, srcDev, 3) do:
      grid = (1, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, 1)
  # timed, the launch overhead leaves the measurement
  dstHost.data[0] = 0
  let
    srcDev = engine.bridge(srcHost)
    dstDev = engine.bridge(dstHost)
  benchSum("1 group", 1, np, iters):
    engine.launchDeviceFn sumLoop(dstDev, srcDev, iters) do:
      grid = (1, 1)
      blk = (32, 1)
  engine.copyToHostFromDevice(addr dstHost.data[0], dstDev.data, 1)
  dstHost
