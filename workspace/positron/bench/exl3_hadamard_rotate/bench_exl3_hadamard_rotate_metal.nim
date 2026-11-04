# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Benchmark: Hadamard rotate bandwidth on Metal, GB/s, fused pre/post scales.
# runCase takes the M and C counts and the iteration count.
# Cases list what to run, nothing is derived.
# Correctness lives in the hadamard tests, not here.
#
# GB/s traffic model: 2-byte fp16 read plus 2-byte fp16 write = 4 bytes per element.
# Scale vectors are column-length and stay cache resident, uncounted.
#
# Timing: a warmup launch then 3 timed launches per case. Median per-iteration time is reported.
#
# Run, from the repo root:
# - nim c -r -d:release --threads:off --outdir:build/bench \
#   workspace/positron/bench/exl3_hadamard_rotate/bench_exl3_hadamard_rotate_metal.nim

import std/[algorithm, monotimes, strformat, strutils, times]
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/tests/ceramic_testutils
import ../../tests/properties/properties
import workspace/positron/src/kernels/exl3_hadamard_rotate

const InvSqrt128 = 0.088388347648'f32

proc hadamardBenchFn(dst, src, preScale, postScale: distinct TensorView, norm: float32, iters: int32) =
  ## Bench harness: `iters` back-to-back rotations with both scale switches enabled.
  ## The norm takes a per-iteration epsilon: identical stores per iteration are
  ## collapsible by the device compiler, which would fake the bandwidth.
  let
    dstP = cast[ptr UncheckedArray[float16]](dst.data)
    srcP = cast[ptr UncheckedArray[float16]](src.data)
    preP = cast[ptr UncheckedArray[float16]](preScale.data)
    postP = cast[ptr UncheckedArray[float16]](postScale.data)
    M = int32(dst.shape[0])
    C = int32(dst.shape[1])
  for i in 0'i32 ..< iters:
    hadamard_rotate_128_fwd(dstP, srcP, preP, postP, M, C, norm + float32(i) * 1e-7'f32, true, true)

proc hadamardBenchCaveman(dst, src, preScale, postScale: distinct TensorView, norm: float32, iters: int32) =
  ## Caveman harness: `iters` back-to-back rotations with both scale switches enabled.
  ## The raw-pointer kernel reads the views' device pointers, M and C come from the dst shape.
  let
    dstP = cast[ptr UncheckedArray[float16]](dst.data)
    srcP = cast[ptr UncheckedArray[float16]](src.data)
    preP = cast[ptr UncheckedArray[float16]](preScale.data)
    postP = cast[ptr UncheckedArray[float16]](postScale.data)
    M = int32(dst.shape[0])
    C = int32(dst.shape[1])
  for i in 0'i32 ..< iters:
    hadamard_rotate_128_fwd(dstP, srcP, preP, postP, M, C, norm + float32(i) * 1e-7'f32, true, true)

proc separator(length = 100) =
  writeLine(stdout, "-".repeat(length))

proc fillScales(data: ptr UncheckedArray[uint16], size: int) =
  ## Position mod 5, times 0.25 plus 1.0, encoded fp16, matching the fill pattern of the ceramic benches
  for i in 0 ..< size:
    data[i] = fp32ToFp16(float32(i mod 5) * 0.25'f32 + 1.0'f32)

template stageViews(engine: MetalEngine, rows, cols, np: int, pattern, preHost, postHost: seq[uint16]): auto =
  ## Uploads the fp16 patterns to fresh device buffers and builds the four views.
  ## A template: DeviceMem handles are move-only and never cross a proc boundary.
  let
    srcDev = engine.allocDevice(float16, np)
    dstDev = engine.allocDevice(float16, np)
    preDev = engine.allocDevice(float16, cols)
    postDev = engine.allocDevice(float16, cols)
  engine.copyToDeviceFromHost(srcDev, cast[ptr float16](addr pattern[0]), np)
  engine.copyToDeviceFromHost(preDev, cast[ptr float16](addr preHost[0]), cols)
  engine.copyToDeviceFromHost(postDev, cast[ptr float16](addr postHost[0]), cols)
  let
    srcLay = make_layout((rows, cols), (cols, 1))
    lineLay = make_layout((cols,), (1,))
    # each BridgeTensor value is built inline over one handle
    srcView = BridgeTensor[typeof(srcDev), typeof(srcLay.shape), typeof(srcLay.stride)](
      data: srcDev, layout: srcLay)
    dstView = BridgeTensor[typeof(dstDev), typeof(srcLay.shape), typeof(srcLay.stride)](
      data: dstDev, layout: srcLay)
    preView = BridgeTensor[typeof(preDev), typeof(lineLay.shape), typeof(lineLay.stride)](
      data: preDev, layout: lineLay)
    postView = BridgeTensor[typeof(postDev), typeof(lineLay.shape), typeof(lineLay.stride)](
      data: postDev, layout: lineLay)
  (dstView, srcView, preView, postView)

template launchIters(engine: MetalEngine, rows, cols: int, itersVal: int32, dstView, srcView, preView, postView) =
  ## `itersVal` back-to-back rotations over the staged views, one launch.
  engine.launchDeviceFn hadamardBenchFn(
    dstView, srcView, preView, postView, InvSqrt128, itersVal) do:
    grid = (rows, cols div 128, 1)
    blk = (32, 1)

template launchIters(engine: MetalEngine, rows, cols: int, itersVal: int32, dstView, srcView, preView, postView) =
  ## `itersVal` back-to-back rotations over the staged views, one launch.
  engine.launchDeviceFn hadamardBenchCaveman(
    dstView, srcView, preView, postView, InvSqrt128, itersVal) do:
    grid = (rows, cols div 128, 1)
    blk = (32, 1)

proc timeIt(engine: MetalEngine, rows, cols, np, iters: int, pattern, preHost, postHost: seq[uint16]): int64 =
  ## Median per-iteration nanoseconds over 3 reps, the raw-pointer kernel.
  block:
    let (dstView, srcView, preView, postView) =
      stageViews(engine, rows, cols, np, pattern, preHost, postHost)
    launchIters(engine, rows, cols, 3'i32, dstView, srcView, preView, postView)
  var timings: array[3, int64]
  for rep in 0 ..< 3:
    block:
      let (dstView, srcView, preView, postView) =
        stageViews(engine, rows, cols, np, pattern, preHost, postHost)
      let benchStart = getMonoTime()
      launchIters(engine, rows, cols, int32(iters), dstView, srcView, preView, postView)
      let benchStop = getMonoTime()
      engine.copyToHostFromDevice(cast[ptr float16](addr pattern[0]), dstView.data, 1)
      timings[rep] = inNanoseconds(benchStop - benchStart) div iters
  sort(timings)
  timings[1]

proc runCase(engine: MetalEngine, rows, cols, iters: int) =
  ## One (M, C) case: one warmup launch then 3 timed launches,
  ## the median per-iteration time reported as GB/s.
  let np = rows * cols
  var pattern = newSeq[uint16](np)
  for i in 0 ..< np:
    pattern[i] = fp32ToFp16(float32(i mod 7) * 0.25'f32 + 0.5'f32)
  var preHost = newSeq[uint16](cols)
  var postHost = newSeq[uint16](cols)
  fillScales(cast[ptr UncheckedArray[uint16]](addr preHost[0]), cols)
  fillScales(cast[ptr UncheckedArray[uint16]](addr postHost[0]), cols)
  let
    ns = timeIt(engine, rows, cols, np, iters, pattern, preHost, postHost)
    gbps = np.float * 4.0 / (ns.float / 1e9) / 1e9
  writeLine(stdout, &"{np:>10}  {rows:>7}  {cols:>7}  {iters:>6}   {gbps:>9.1f}")

const Cases: array[6, tuple[rows, cols, iters: int]] = [
  # latency, M = 1
  (1, 2048, 200),
  # small, launch bound
  (32, 2048, 200),
  # transformer activation, 512 blocks along C
  (8192, 4096, 50),
  # medium
  (1024, 4096, 50),
  # wide, C = 16384
  (8192, 16384, 10),
  # largest working set runs last, an allocation failure on a small device costs no later case
  (65536, 16384, 5)]

proc main() =
  var engine = newEngine(runMetal)
  writeLine(stdout, "hadamard rotate, Metal, blk = (32, 1), one simdgroup per 128-block, GB/s = elements x 4 bytes")
  writeLine(stdout, "        np    rows    cols   iters   GB/s")
  separator()
  for c in Cases:
    runCase(engine, c.rows, c.cols, c.iters)
  separator()

when isMainModule:
  main()
