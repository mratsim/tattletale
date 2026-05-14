# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Benchmark RMSNorm implementations.
##
## Usage:
##   nim cpp -d:release --d:TTT_LIBTORCH_SOURCE=vendor bench_rmsnorm.nim
##   LD_LIBRARY_PATH=... ./bench_rmsnorm            # CPU with vendor libtorch
##   nim cpp -d:release -d:cuda --d:TTT_LIBTORCH_SOURCE=venv bench_rmsnorm.nim
##   LD_LIBRARY_PATH=... ./bench_rmsnorm cuda       # GPU (requires CUDA-enabled libtorch)

import
  std/monotimes,
  std/times,
  std/strformat,
  std/os,
  std/strutils,
  std/sequtils,
  workspace/libtorch,
  workspace/libtorch/src/raw_libtorch

# ── RMSNorm implementations ────────────────────────────────────────────

func rmsNormManualBF16(x, weight: Tensor, eps: float64): Tensor =
  let x2 = x * x
  let variance = x2.mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).sqrt().reciprocal()
  (x * rstd) * weight

func rmsNormManualFP32(x, weight: Tensor, eps: float64): Tensor =
  let x_fp = x.to(kFloat32)
  let variance = (x_fp * x_fp).mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).sqrt().reciprocal()
  let normed = (x_fp * rstd).to(x.scalarType())
  normed * weight

func rmsNormFusedBF16(x, weight: Tensor, eps: float64): Tensor =
  rms_norm(x, weight.size(0), weight, eps)

func rmsNormFusedFP32(x, weight: Tensor, eps: float64): Tensor =
  let weight_fp = weight.to(kFloat32)
  rms_norm(x.to(kFloat32), weight_fp.size(0), weight_fp, eps).to(x.scalarType())

func rmsNormFusedFP32BF16Weight(x, weight: Tensor, eps: float64): Tensor =
  let opts = TensorOptions.init()
                          .dtype(kBFloat16)
                          .device(if x.is_cuda(): kCUDA else: kCPU)
  let ones = ones(weight.size(0), opts)
  let normed = rms_norm(x.to(kFloat32), weight.size(0), ones, eps).to(x.scalarType())
  normed * weight

# ── Benchmark infrastructure ───────────────────────────────────────────

template measure*(iters: int, body: untyped): int64 =
  ## Measure execution time of `body` over `iters` iterations.
  ## Returns nanoseconds per iteration.
  let startTime = getMonotime()
  for _ in 0 ..< iters:
    body
  let elapsed = (getMonotime() - startTime).inNanoseconds
  elapsed div iters

proc reportLine*(name: string, ns: int64, diff: float64, refNs: int64) =
  ## Print a single benchmark result line.
  let ops = 1e9 / float64(ns)
  let vsRef = float64(refNs) / float64(ns)
  let mark = if diff < 1e-6: "✓" else: "✗"
  echo &"{name:<40} {ns:>8} {ops:>8.1f} {vsRef:>6.2f}x  {diff:.0e} {mark}"

proc benchRMSNorm() =
  let deviceName = if paramCount() > 0: paramStr(1).toLowerAscii() else: "cpu"
  let useCuda = deviceName == "cuda" or deviceName == "gpu"

  if useCuda:
    if not Torch.cuda_is_available():
      echo "ERROR: CUDA is not available. libtorch was built without CUDA support."
      quit(1)
    echo &"CUDA device: {Torch.deviceCount()} GPU(s) available"
  echo()

  const
    Batch = 2
    SeqLen = 2048
    Hidden = 1024
    Iters = 1000
    Warmup = 100

  echo "RMSNorm Benchmark"
  echo "================="
  echo &"Shape: ({Batch}, {SeqLen}, {Hidden})"
  echo &"Device: {deviceName}"
  echo &"Iters: {Iters}"
  echo()

  var x = randn(Batch, SeqLen, Hidden, kBFloat16)
  var weight = randn(Hidden, kBFloat16)
  if useCuda:
    x = x.cuda()
    weight = weight.cuda()
  const eps = 1e-6

  echo "Warmup..."
  for _ in 0 ..< Warmup:
    discard rmsNormManualBF16(x, weight, eps)
    discard rmsNormManualFP32(x, weight, eps)
    discard rmsNormFusedBF16(x, weight, eps)
    discard rmsNormFusedFP32(x, weight, eps)
    discard rmsNormFusedFP32BF16Weight(x, weight, eps)
  echo "Done."
  echo()

  # Header
  echo "Implementation                                  ns/op  ops/s  vs ref     diff"
  echo "-----------------------------------------------------------------------------"

  # Reference first (exact FP32 match)
  var rRef: Tensor
  let nsRef = measure(Iters):
    rRef = rmsNormManualFP32(x, weight, eps)
  let fp32Ref = rRef.to(kCPU).to(kFloat32)

  # Collect results
  var results: seq[tuple[name: string, ns: int64, diff: float64]]

  # Manual BF16
  var r1: Tensor
  let ns1 = measure(Iters):
    r1 = rmsNormManualBF16(x, weight, eps)
  let d1 = (r1.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
  results.add(("Manual BF16", ns1, d1))

  # Fused BF16
  var r3: Tensor
  let ns3 = measure(Iters):
    r3 = rmsNormFusedBF16(x, weight, eps)
  let d3 = (r3.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
  results.add(("Fused BF16", ns3, d3))

  # Fused FP32
  var r4: Tensor
  let ns4 = measure(Iters):
    r4 = rmsNormFusedFP32(x, weight, eps)
  let d4 = (r4.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
  results.add(("Fused FP32", ns4, d4))

  # Fused FP32 + BF16 wt
  var r5: Tensor
  let ns5 = measure(Iters):
    r5 = rmsNormFusedFP32BF16Weight(x, weight, eps)
  let d5 = (r5.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
  results.add(("Fused FP32 + BF16 weight", ns5, d5))

  # Print reference
  reportLine("Manual FP32 (ref)", nsRef, 0.0, nsRef)
  # Print others
  for e in results:
    reportLine(e.name, e.ns, e.diff, nsRef)

when isMainModule:
  benchRMSNorm()
