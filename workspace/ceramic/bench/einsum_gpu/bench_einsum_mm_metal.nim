# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark: the einsum macro's mm recipe on Metal, effective TFLOP/s.
#
## D[i,k] = A[i,j] * B[j,k] through the k_tile_einsum macro's mma path
## (EinsumMmaTile = 16x16x32, one (16, 16) D tile per launch squadron).
#
## Flops: 2*M*N*K, one multiply-add per output element (bench_gemm convention).
#
## Methodology: one warmup launch of 2 contractions, then 3 timed reps of `iters` contractions each.
## The median rep's TFLOP/s is printed. Numbers are session-local medians. Re-record on a quiet GPU before citing.
##
## Run:
##   nim c -r -d:release --outdir:build/bench \
##     workspace/ceramic/bench/einsum_gpu/bench_einsum_mm_metal.nim (from tattletale)

import std/[algorithm, monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/tile_algebra
import workspace/ceramic/tests/ceramic_testutils
import workspace/ceramic/tests/kernels/k_einsum/kernel_einsum_template

const EinsumBenchMmaTile = 16 # EinsumMmaTile in k_tile_einsum, one (16, 16) D tile per squadron

proc fillMmPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Deterministic fill: element value = (linear position mod 13) * 0.25 + 0.5
  for i in 0 ..< tv.size():
    tv(i) = to(float32(i mod 13) * 0.25'f32 + 0.5'f32, T)

func computeMmFlops(M, N, K: int): float64 =
  ## 2*M*N*K, one multiply-add per output element
  2.0 * M.float * N.float * K.float

proc einsumMmLoop(D, A, B: distinct TensorView, iters: int) =
  for _ in 0 ..< iters:
    kEinsumMm(D, A, B)

proc correctnessLeg[E: HwEngine](engine: E) =
  ## Small-config check: the kernel output vs an f64 host-loop reference
  ## over the same f32 inputs
  const
    M = 64
    N = 70 # ragged against the 8-wide D tiles, exercises paddedLoad/guardedStore
    K = 96
  let one = 1 # runtime int, layouts stay dynamic
  var
    aHost = makeHostTensor(float32, (M, K), (K, one))
    bHost = makeHostTensor(float32, (K, N), (N, one))
    dHost = makeHostTensor(float32, (M, N), (N, one))
    expectHost = makeHostTensor(float64, (M, N), (N, one))
  var aView = aHost.view()
  var bView = bHost.view()
  fillMmPattern(aView)
  fillMmPattern(bView)
  for i in 0 ..< M:
    for k in 0 ..< N:
      var acc = 0.0'f64
      for j in 0 ..< K:
        acc += to(aView[i, j], float64) * to(bView[j, k], float64)
      expectHost.view()[i, k] = acc
  let
    aDev = engine.bridge(aHost)
    bDev = engine.bridge(bHost)
    dDev = engine.bridge(dHost)
  engine.launchDeviceFn kEinsumMm(dDev, aDev, bDev) do:
    grid = (ceil_div(M, EinsumBenchMmaTile), ceil_div(N, EinsumBenchMmaTile))
    blk = (32, 1)
  engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
  # widen the f32 kernel output so allClose compares both sides in f64
  var actualHost = makeHostTensor(float64, (M, N), (N, one))
  var dView = dHost.view()
  var actualView = actualHost.view()
  for i in 0 ..< M:
    for k in 0 ..< N:
      actualView[i, k] = to(dView[i, k], float64)
  assertAllClose(actualView, expectHost.view(), rtol = 1e-5, abstol = 1e-3,
    msg = "einsum mm vs f64 host loop")

proc runCase[E: HwEngine](engine: E, M, N, K, iters: int) =
  ## One (M, N, K) case, methodology per the module header
  let one = 1 # one is a runtime int, layouts stay dynamic
  var
    aHost = makeHostTensor(float32, (M, K), (K, one))
    bHost = makeHostTensor(float32, (K, N), (N, one))
    dHost = makeHostTensor(float32, (M, N), (N, one))
  var aView = aHost.view()
  var bView = bHost.view()
  fillMmPattern(aView)
  fillMmPattern(bView)
  let launchGrid = (ceil_div(M, EinsumBenchMmaTile), ceil_div(N, EinsumBenchMmaTile))
  # warmup
  block:
    let
      aDev = engine.bridge(aHost)
      bDev = engine.bridge(bHost)
      dDev = engine.bridge(dHost)
    engine.launchDeviceFn einsumMmLoop(dDev, aDev, bDev, 2) do:
      grid = launchGrid
      blk = (32, 1)
  # timed, fresh bridges per rep, copy back drains each launch
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    block:
      let
        aDev = engine.bridge(aHost)
        bDev = engine.bridge(bHost)
        dDev = engine.bridge(dHost)
      let benchStart = getMonoTime()
      engine.launchDeviceFn einsumMmLoop(dDev, aDev, bDev, iters) do:
        grid = launchGrid
        blk = (32, 1)
      engine.copyToHostFromDevice(addr dHost.data[0], dDev.data, dHost.len)
      timings[rep] = inNanoseconds(getMonoTime() - benchStart) div iters
  sort(timings)
  let
    med = timings[1].float64 / 1e9
    flops = computeMmFlops(M, N, K)
    tflops = flops / med / 1e12
  writeLine(stdout, &"M={M:>5} N={N:>5} K={K:>5}  median {med * 1e6:>10.1f} us  effective {tflops:>6.2f} TFLOP/s")

proc main() =
  writeLine(stdout, "Einsum mm bench, D[i,k] = A[i,j] * B[j,k] through k_tile_einsum (Metal)")
  writeLine(stdout, "WARNING: timings are session-local medians, re-record on a quiet GPU before citing")
  var engine = newEngine(runMetal)
  correctnessLeg(engine)
  # 1024: 12 MB of f32 operand+output footprint.
  # 2048: 48 MB of f32 operand+output footprint.
  runCase(engine, 1024, 1024, 1024, 20)
  runCase(engine, 2048, 2048, 2048, 10)

main()
