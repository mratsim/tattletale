# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Per-op timing for the Gated DeltaNet decode step, Qwen3.6-35B-A3B shapes on MPS.
##
## Layer under test: GatedDeltaNet.forward, the production decode path:
## projections, causal conv1d with state rewind, f32 delta-rule recurrence
## over seq 1, gated RMSNorm, out_proj, state write-back.
##
## Fixture-recorded geometry (layer 0 linear_attn):
##   hidden 2048, key heads 16 x 128, value heads 32 x 128,
##   conv_dim 8192, conv kernel 4
##
## Standalone benchmark: run directly, not wired into test suites.

import
  std/monotimes,
  std/times,
  std/strformat,
  std/os,
  std/strutils,
  std/math,
  std/algorithm,
  workspace/libtorch,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net,
  workspace/transformers/src/stateful/inference_context

{.experimental: "callOperator".}

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
  echo &"{name:<52} {us:>10.2f} us {gbs:>9.2f} GB/s {pct:>6.1f}%"

proc runBench() =
  let deviceName = if paramCount() > 0: paramStr(1).toLowerAscii() else: "mps"
  let device =
    case deviceName
    of "cpu": kCPU
    of "mps", "gpu": kMPS
    else: kMPS

  const
    Hidden = 2048
    NumKHeads = 16
    NumVHeads = 32
    HeadKDim = 128
    HeadVDim = 128
    ConvKernelSize = 4
    KeyDim = NumKHeads * HeadKDim          # 2048
    ValueDim = NumVHeads * HeadVDim        # 4096
    ConvDim = 2 * KeyDim + ValueDim        # 8192
    Warmup = 20
    Iters = 100
    Runs = 5

  echo "Gated DeltaNet benchmark, Qwen3.6-35B-A3B layer geometry"
  echo "========================================================"
  echo &"date: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo &"device: {deviceName}"
  echo &"dtype: bfloat16, SSM state f32"
  echo &"weights: in_proj_qkv ({ConvDim}, {Hidden}), in_proj_z ({ValueDim}, {Hidden}), in_proj_a ({NumVHeads}, {Hidden}), in_proj_b ({NumVHeads}, {Hidden}), conv1d ({ConvDim}, 1, {ConvKernelSize}), out_proj ({Hidden}, {ValueDim})"
  echo &"decode activations: x (1, 1, {Hidden}), SSM state ({NumVHeads}, {HeadKDim}, {HeadVDim}) f32"
  echo &"iters per run: {Iters}, runs: {Runs} (median reported)"
  echo()

  let opts = tensorOptions(kBFloat16, device)
  let qkvProj = Linear.init(randn(ConvDim, Hidden, opts))
  let zProj = Linear.init(randn(ValueDim, Hidden, opts))
  let aProj = Linear.init(randn(NumVHeads, Hidden, opts))
  let bProj = Linear.init(randn(NumVHeads, Hidden, opts))
  let oProj = Linear.init(randn(Hidden, ValueDim, opts))
  let convW = randn(ConvDim, 1, ConvKernelSize, opts)
  # a_log holds log decay, magnitudes near zero keep the decay stable
  let aLog = randn(NumVHeads, kBFloat16).to(device) * Scalar(0.01)
  let dtBias = randn(NumVHeads, kBFloat16).to(device) * Scalar(0.01)
  let norm = RmsNormGated.init(randn(HeadVDim, kBFloat16).to(device))

  let gdn = GatedDeltaNet.init(
    0, "bench.gdn",
    qkvProj, zProj, aProj, bProj,
    convW, aLog, dtBias,
    norm, oProj,
    NumKHeads, NumVHeads, HeadKDim, HeadVDim, ConvKernelSize)

  var ctx = InferenceContext.init(1, 1, NumKHeads, 4096, HeadKDim)
  let x = randn(1, 1, Hidden, opts)

  var gdnOut: Tensor
  for _ in 0 ..< Warmup:
    gdnOut = gdn(ctx, x)
  var nsRuns: seq[int64]
  for run in 0 ..< Runs:
    let ns = measure(Iters):
      gdnOut = gdn(ctx, x)
    nsRuns.add ns
  discard gdnOut.to(kCPU)
  var nsSyncRuns: seq[int64]
  for run in 0 ..< Runs:
    let ns = measure(Iters):
      gdnOut = gdn(ctx, x)
      discard gdnOut.to(kCPU)
    nsSyncRuns.add ns
  let ssmBytes = int64(NumVHeads) * HeadKDim * HeadVDim * 4

  # Bytes: projections + conv weight + norm + SSM state read + write
  # + conv state read + write, bf16 except SSM f32
  let weightBytes = int64(ConvDim + ValueDim + 2 * NumVHeads) * Hidden * 2 +
    int64(Hidden) * ValueDim * 2 + int64(ConvDim) * ConvKernelSize * 2
  let stateBytes = 2 * ssmBytes + 2 * int64(ConvDim) * 3 * 2
  reportLine("gdn decode fwd, seq 1 (async wall)", median(nsRuns), weightBytes + stateBytes)
  reportLine("gdn decode fwd, seq 1 (synced)", median(nsSyncRuns), weightBytes + stateBytes)

  # Block: projections only (in_proj_qkv + z + a + b), the weight-heavy half
  block:
    var mixed: Tensor
    for _ in 0 ..< Warmup:
      mixed = qkvProj(x)
    var projRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        mixed = qkvProj(x)
        let zP = zProj(x)
        let aP = aProj(x)
        let bP = bProj(x)
      projRuns.add ns
    discard mixed.to(kCPU)
    var projSyncRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        mixed = qkvProj(x)
        let zP = zProj(x)
        let aP = aProj(x)
        let bP = bProj(x)
        discard mixed.to(kCPU)
      projSyncRuns.add ns
    reportLine("gdn in_proj qkv+z+a+b (async wall)", median(projRuns),
      int64(ConvDim + ValueDim + 2 * NumVHeads) * Hidden * 2)
    reportLine("gdn in_proj qkv+z+a+b (synced)", median(projSyncRuns),
      int64(ConvDim + ValueDim + 2 * NumVHeads) * Hidden * 2)

when isMainModule:
  runBench()
