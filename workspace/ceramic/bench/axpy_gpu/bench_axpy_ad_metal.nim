# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: axpy bandwidth on Metal, GB/s.
#
# Measured cases:
#
# - forward, official kernel: k_tile_axpy.axpy launched as-is through
#   launchDeviceFn. y = alpha*x + y in place, one threadgroup per
#   128-element register tile, grid-strided over ceil(n/128) threadgroups.
# - AD pair, forward and backward: the same axpy math (dst = alpha*a + b)
#   instantiated for bwd_diff in the differentiable subset's map form,
#   flat buffers with thread t owning the strided slice i = t + k*W,
#   its backward allocating and returning gradients per call (convenience).
#
# k_tile_axpy.axpy is a device function over TensorViews, its body compiles
# only inside the GPU macro context.
# bwd_diff differentiates global map-form kernels, and only those kernels,
# so the AD pair instantiates the axpy math in that map form over flat buffers.
#
# Geometry cases, one threadgroup per AD launch (the pair's baked workgroup):
#
# - wg32: workgroup (32, 1), the official kernel's threadgroup width
# - wg1024: workgroup (1024, 1), the thread-parallel form
#
# GB/s traffic model:
#
# | case     | charged streams                    | bytes per element |
# | -------- | ---------------------------------- | ----------------- |
# | forward  | read a + read b + write dst        | 12                |
# | backward | read d_dst + write d_a + write d_b | 12                |
#
# Official-kernel y serves as b and dst (in-place accumulator).
# Timed backward span covers the whole convenience call, which moves more
# than the charged streams:
#
# - the derived backward also reads the primal a (the d_alpha partial term)
#   and its gradient writes accumulate, read-modify-write
# - the d_alpha reduce launch is counted, every backward with a scalar
#   gradient launches it
# - the convenience allocates and zeroes d_a, d_b and the scratch per call,
#   host-side fill plus device upload
#
# One scratch element per loop trip, one thread folds the scratch sequentially.
# Past 8M elements that fold exceeds the GPU's execution watchdog and Metal
# reports a GPU hang error on the command buffer.
# Backward cases end at 8M elements, past that the refusal is recorded, never worked around.
#
# Timing: a warmup call then 3 timed calls per case, the median is reported.
# Looped official forward runs `iters` axpy passes inside one launch,
# alpha 1.0 keeps the accumulated y finite across passes.
# Correctness lives in the k_axpy kernel tests and the autodiff axpy tests.
#
# Run (from tattletale):
#   nim c -r -d:release --maxLoopIterationsVM:100000000 \
#     --nimcache:nimcache/bench_axpy_ad \
#     --outdir:build/bench workspace/ceramic/bench/axpy_gpu/bench_axpy_ad_metal.nim

import std/[algorithm, monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/src/kernels/k_tile_axpy
import workspace/ceramic/src/kernels/k_layout_fillwith_cpu
import workspace/ceramic/src/macros/static_for
import workspace/ceramic/tests/ceramic_testutils

const Sizes = [1_024, 16_384, 65_536, 262_144, 1_048_576, 4_194_304, 8_388_608, 16_777_216]
  ## Element counts, 4 KB to 64 MB tensors of float32.
  ## Backward cases end at 8_388_608, see the traffic model in the header.

const BackwardMax = 8_388_608

proc separator(length = 78) =
  writeLine(stdout, "-".repeat(length))

func itersFor(n: int): int =
  ## Timed axpy passes per launch for the official forward case, sized so one
  ## command buffer fits well inside the GPU's execution watchdog.
  case n
  of 0 .. 262_144:
    200
  of 262_145 .. 1_048_576:
    50
  of 1_048_577 .. 4_194_304:
    20
  else:
    10

proc fillPattern[T, Sh, St](tv: var TensorView[T, Sh, St]) =
  ## Logical position mod 7, times 0.25 plus 0.5: slot permutations change sums
  for i in 0 ..< tv.size():
    tv(i) = T(i mod 7) * T(0.25) + T(0.5)

# ═══════════════════════════════════════════════════════════════════════
#  Forward row: the official k_tile_axpy kernel, launched as-is
# ═══════════════════════════════════════════════════════════════════════

proc axpyLoop(y, x: distinct TensorView, alpha: float32, iters: int) =
  ## Bench harness: iters axpy passes in one launch.
  ## Every pass overwrites y from its own previous value, the stores need no reset between passes.
  for _ in 0 ..< iters:
    axpy(y, x, alpha)

proc runOfficialForward(engine: MetalEngine, n: int) =
  ## One (n) case of the official kernel forward: a warmup launch, 3 timed launches, median per pass reported as GB/s.
  let one = 1  # a runtime int so each size sweep builds its own layout
  var
    xHost = makeHostTensor(float32, (n, one), (one, n))
    yHost = makeHostTensor(float32, (n, one), (one, n))
  block:
    var xv = xHost.view()
    fillPattern(xv)
  block:
    var yv = yHost.view()
    fillWith_cpu(yv, 1.0'f32)
  let
    gx = ceil_div(n, 128)
    iters = itersFor(n)
  # warmup
  block:
    let
      xDev = engine.bridge(xHost)
      yDev = engine.bridge(yHost)
    engine.launchDeviceFn axpyLoop(yDev, xDev, 1.0'f32, 3) do:
      grid = (gx, 1)
      blk = (32, 1)
    engine.copyToHostFromDevice(addr yHost.data[0], yDev.data, n)
  # timed, Metal launches commit and wait, the readback after the timed span
  # keeps the sibling benches' flush-between-reps shape
  var timings: array[3, int64]
  staticFor rep, 0, 3:
    block:
      let
        xDev = engine.bridge(xHost)
        yDev = engine.bridge(yHost)
      let benchStart = getMonoTime()
      engine.launchDeviceFn axpyLoop(yDev, xDev, 1.0'f32, iters) do:
        grid = (gx, 1)
        blk = (32, 1)
      let benchStop = getMonoTime()
      engine.copyToHostFromDevice(addr yHost.data[0], yDev.data, n)
      timings[rep] = inNanoseconds(benchStop - benchStart) div iters
  sort(timings)
  let gbps = n.float * 12.0 / (timings[1].float / 1e9) / 1e9
  writeLine(stdout, &"{n:>10}  {gx:>8}  {iters:>5}  {timings[1]:>10}  {gbps:>7.1f}")

# ═══════════════════════════════════════════════════════════════════════
#  AD rows: the axpy pair through setupAutoDiff, both geometry cases
# ═══════════════════════════════════════════════════════════════════════

proc axpyKernel(dst, a, b: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global, workgroup: (32, 1).} =
  ## Map form at the official kernel's threadgroup width: thread t owns the strided slice i = t + k*32, n is the per-thread trip count.
  let t = thread_position_in_threadgroup.x
  for k in 0 ..< n:
    let i = t + k * 32
    dst[i] = alpha * a[i] + b[i]

proc axpyKernelTP(dst, a, b: ptr UncheckedArray[float32], alpha: float32, n: int32) {.global, workgroup: (1024, 1).} =
  ## Thread-parallel form: thread t owns the strided slice i = t + k*1024.
  let t = thread_position_in_threadgroup.x
  for k in 0 ..< n:
    let i = t + k * 1024
    dst[i] = alpha * a[i] + b[i]

const axpyPair = cgMetal.bwd_diff(axpyKernel)
const axpyTPPair = cgMetal.bwd_diff(axpyKernelTP)

type AxpyBuf[DevMem] = object
  ## Host wrapper of one device buffer plus its length.
  ## The convenience backward reads that length for its grad allocation.
  p: DevMem
  n: int

proc medianNanos(timings: array[3, int64]): int64 =
  ## Median of 3 timings, the sibling benches' sort-and-take-the-middle
  var sorted = timings
  sort(sorted)
  sorted[1]

proc printRow(n: int, label: string, ns: int64, bytesPerElem = 12.0) =
  let gbps = n.float * bytesPerElem / (ns.float / 1e9) / 1e9
  writeLine(stdout, &"{n:>10}  {label:>7}  {ns:>10}  {gbps:>7.1f}")

proc benchAdForward[E: HwEngine, K: static BwdDiffKernels](
    engine: E, ad: AdHandle[E, K], label: string, n, threads: int) =
  ## One (n, geometry) case of the AD pair's forward: one warmup call, 3 timed calls, the median per-call time reported as GB/s.
  ## One launch covers n*threads elements: n strided slots per thread.
  let
    trip = int32(n div threads)
    aDev = engine.allocDevice(float32, n)
    bDev = engine.allocDevice(float32, n)
    dstDev = engine.allocDevice(float32, n)
  var aHost = newSeq[float32](n)
  var bHost = newSeq[float32](n)
  for i in 0 ..< n:
    aHost[i] = float32(i mod 7) * 0.25'f32 + 0.5'f32
    bHost[i] = float32(i mod 11) * 0.0625'f32 + 1.0'f32
  engine.copyToDeviceFromHost(aDev, addr aHost[0], n)
  engine.copyToDeviceFromHost(bDev, addr bHost[0], n)
  block:   # warmup
    ad.forward(dstDev, aDev, bDev, 2.0'f32, trip)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    let benchStart = getMonoTime()
    ad.forward(dstDev, aDev, bDev, 2.0'f32, trip)
    let benchStop = getMonoTime()
    timings[rep] = inNanoseconds(benchStop - benchStart)
  printRow(n, label, medianNanos(timings))

proc benchAdBackward[E: HwEngine, K: static BwdDiffKernels](
    engine: E, ad: AdHandle[E, K], label: string, n, threads: int) =
  ## One (n, geometry) case of the AD pair's convenience backward: one warmup call, 3 timed calls, the median per-call time reported as GB/s.
  ## The timed span is the whole call: gradient and scratch allocations, zeroing uploads, the backward launch and the d_alpha reduce launch.
  let
    trip = int32(n div threads)
    aDev = engine.allocDevice(float32, n)
    bDev = engine.allocDevice(float32, n)
    dstDev = engine.allocDevice(float32, n)
    dOutDev = engine.allocDevice(float32, n)
  var aHost = newSeq[float32](n)
  var dOutHost = newSeq[float32](n)
  for i in 0 ..< n:
    aHost[i] = float32(i mod 7) * 0.25'f32 + 0.5'f32
    dOutHost[i] = float32(i mod 5) * 0.5'f32 + 0.25'f32
  engine.copyToDeviceFromHost(aDev, addr aHost[0], n)
  engine.copyToDeviceFromHost(dOutDev, addr dOutHost[0], n)
  let
    bufA = AxpyBuf[typeof(aDev)](p: aDev, n: n)
    bufB = AxpyBuf[typeof(bDev)](p: bDev, n: n)
  block:   # warmup, the returned gradients drop at the block's end
    discard ad.backward(dOutDev, dstDev, bufA, bufB, 2.0'f32, trip)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    block:
      let benchStart = getMonoTime()
      block:
        discard ad.backward(dOutDev, dstDev, bufA, bufB, 2.0'f32, trip)
      let benchStop = getMonoTime()
      timings[rep] = inNanoseconds(benchStop - benchStart)
  printRow(n, label, medianNanos(timings))

# ═══════════════════════════════════════════════════════════════════════
#  The sweep
# ═══════════════════════════════════════════════════════════════════════

proc main() =
  var engine = newEngine(runMetal)
  let
    ad = engine.setupAutoDiff(axpyPair)
    adTP = engine.setupAutoDiff(axpyTPPair)

  writeLine(stdout, "axpy, Metal, GB/s = logical elements x 12 bytes")
  separator()
  writeLine(stdout, "forward, official k_tile_axpy kernel, blk = (32, 1), one threadgroup per tile")
  writeLine(stdout, "         n  tile-grid  iters   ns/pass    GB/s")
  for n in Sizes:
    runOfficialForward(engine, n)
  separator()

  writeLine(stdout, "forward, AD pair, one threadgroup per launch, n = elements div threads")
  writeLine(stdout, "         n     case        ns    GB/s")
  for n in Sizes:
    benchAdForward(engine, adTP, "wg1024", n, 1024)
    benchAdForward(engine, ad, "wg32", n, 32)
  separator()

  writeLine(stdout, "backward, AD pair convenience overload, one threadgroup + the d_alpha reduce")
  writeLine(stdout, "rows stop at 8M elements: past that the single-thread d_alpha fold exceeds")
  writeLine(stdout, "the GPU execution watchdog and Metal fails the command buffer (GPU hang error)")
  writeLine(stdout, "         n     case        ns    GB/s")
  for n in Sizes:
    if n > BackwardMax:
      continue
    benchAdBackward(engine, adTP, "wg1024", n, 1024)
    benchAdBackward(engine, ad, "wg32", n, 32)
  separator()

when isMainModule:
  main()
