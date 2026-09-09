# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Per-op timing for the RoPE path, Qwen3.6-35B-A3B decode shapes on MPS.
##
## Shapes are the fixture-recorded geometry:
##   q (1, seq, 16, 256), k (1, seq, 2, 256), rotary_dim 64 (partial 0.25)
##   cos/sin cache (262144, 64)
##
## Standalone benchmark: run directly, not wired into test suites.

import
  std/monotimes,
  std/times,
  std/strformat,
  std/os,
  std/math,
  std/strutils,
  std/algorithm,
  workspace/libtorch,
  workspace/transformers/src/layers/rope

template measure(iters: int, body: untyped): int64 =
  ## Wall nanoseconds per iteration over `iters` calls.
  let startTime = getMonotime()
  for _ in 0 ..< iters:
    body
  let elapsed = (getMonotime() - startTime).inNanoseconds
  elapsed div iters

proc median(values: seq[int64]): int64 =
  var sorted = values
  sorted.sort()
  result = sorted[sorted.len div 2]

proc reportLine(name: string, ns: int64, bytes: int64) =
  ## Per-call line: microseconds, derived GB/s, share of the 410 GB/s roofline.
  let us = float64(ns) / 1000.0
  let gbs = float64(bytes) / (float64(ns) * 1e-9) / 1e9
  let pct = gbs / 410.0 * 100.0
  echo &"{name:<44} {us:>10.2f} us {gbs:>9.2f} GB/s {pct:>6.1f}%"

proc runBench() =
  let deviceName = if paramCount() > 0: paramStr(1).toLowerAscii() else: "mps"
  let device =
    case deviceName
    of "cpu": kCPU
    of "mps", "gpu": kMPS
    else: kMPS

  const
    NumQo = 16
    NumKv = 2
    HeadDim = 256
    RotaryDim = 64
    MaxSeqLen = 262144
    RopeTheta = 1e7
    DecodeSeq = 1
    PrefillSeq = 8
    Warmup = 20
    Iters = 100
    Runs = 5

  echo "RoPE benchmark, Qwen3.6-35B-A3B geometry"
  echo "========================================"
  echo &"date: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo &"device: {deviceName}"
  echo &"dtype: bfloat16 activations, cache bf16"
  echo &"q shape: (1, seq, {NumQo}, {HeadDim}), k shape: (1, seq, {NumKv}, {HeadDim})"
  echo &"rotary_dim: {RotaryDim} (partial_rotary_factor 0.25), cache ({MaxSeqLen}, {RotaryDim})"
  echo &"iters per run: {Iters}, runs: {Runs} (median reported)"
  echo()

  let rotary = RotaryPositionEmbedding.new(
    HeadDim, MaxSeqLen, RopeTheta, kBFloat16, device, rotary_dim = RotaryDim)

  # Decode-time activations
  var q = randn(1, DecodeSeq, NumQo, HeadDim, kBFloat16).to(device)
  var k = randn(1, DecodeSeq, NumKv, HeadDim, kBFloat16).to(device)
  var qPre = randn(1, PrefillSeq, NumQo, HeadDim, kBFloat16).to(device)
  var kPre = randn(1, PrefillSeq, NumKv, HeadDim, kBFloat16).to(device)

  # Fixed cos/sin rows for one decode position
  let posIds = [[int64(1024)]].toTensor().to(device)
  let (cosRow, sinRow) = rotary.ropeByPositions(posIds)

  var qRot, kRot: Tensor

  # Block 1: cache lookup, one position
  block:
    for _ in 0 ..< Warmup:
      let (c, s) = rotary.ropeByPositions(posIds)
      qRot = c
    var nsRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        let (c, s) = rotary.ropeByPositions(posIds)
        qRot = c
        kRot = s
      nsRuns.add ns
    var nsSyncRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        let (c, s) = rotary.ropeByPositions(posIds)
        qRot = c
        discard qRot.to(kCPU)
      nsSyncRuns.add ns
    discard qRot.to(kCPU)
    # bytes: cache row read 2 x 64 x 2 B, slice writes 2 x 64 x 2 B
    reportLine("ropeByPositions 1 pos (async wall)", median(nsRuns), 512)
    reportLine("ropeByPositions 1 pos (synced)", median(nsSyncRuns), 512)

  # Block 2: applyRope partial rotation, decode (seq 1)
  block:
    for _ in 0 ..< Warmup:
      (qRot, kRot) = rotary.applyRope(q, k, cosRow, sinRow)
    var nsRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        (qRot, kRot) = rotary.applyRope(q, k, cosRow, sinRow)
      nsRuns.add ns
    discard qRot.to(kCPU)
    var nsSyncRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        (qRot, kRot) = rotary.applyRope(q, k, cosRow, sinRow)
        discard qRot.to(kCPU)
      nsSyncRuns.add ns
    # bytes: q+k read and q_rot+k_rot write, cos/sin read
    let bytesDecode = 2 * (NumQo + NumKv) * HeadDim * 2 + 2 * RotaryDim * 2
    reportLine("applyRope partial, decode seq 1 (async wall)", median(nsRuns), bytesDecode)
    reportLine("applyRope partial, decode seq 1 (synced)", median(nsSyncRuns), bytesDecode)

  # Block 3: applyRope partial rotation, prefill (seq 8)
  block:
    let (cosPre, sinPre) = rotary.ropeByPositions(
      ([int64(0), 1, 2, 3, 4, 5, 6, 7]).toTensor().to(device))
    for _ in 0 ..< Warmup:
      (qRot, kRot) = rotary.applyRope(qPre, kPre, cosPre, sinPre)
    var nsRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        (qRot, kRot) = rotary.applyRope(qPre, kPre, cosPre, sinPre)
      nsRuns.add ns
    discard qRot.to(kCPU)
    var nsSyncRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        (qRot, kRot) = rotary.applyRope(qPre, kPre, cosPre, sinPre)
        discard qRot.to(kCPU)
      nsSyncRuns.add ns
    let bytesPre = 2 * (NumQo + NumKv) * PrefillSeq * HeadDim * 2 + 2 * PrefillSeq * RotaryDim * 2
    reportLine("applyRope partial, prefill seq 8 (async wall)", median(nsRuns), bytesPre)
    reportLine("applyRope partial, prefill seq 8 (synced)", median(nsSyncRuns), bytesPre)

when isMainModule:
  runBench()
