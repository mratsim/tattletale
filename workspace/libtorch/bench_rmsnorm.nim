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
  workspace/libtorch

# ── RMSNorm implementations ────────────────────────────────────────────


func rmsNormManualFP32(x, weight: Tensor, eps: float64): Tensor =
  let x_fp = x.to(kFloat32)
  let variance = (x_fp * x_fp).mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).sqrt().reciprocal()
  let normed = (x_fp * rstd).to(x.scalarType())
  normed * weight

func rmsNormManualBF16(x, weight: Tensor, eps: float64): Tensor =
  let x2 = x * x
  let variance = x2.mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).sqrt().reciprocal()
  (x * rstd) * weight

func rmsNormFusedBF16(x, weight: Tensor, eps: float64): Tensor =
  rms_norm(x, weight.size(0), weight, eps)

func rmsNormFusedFP32(x, weight: Tensor, eps: float64): Tensor =
  let weight_fp = weight.to(kFloat32)
  rms_norm(x.to(kFloat32), weight_fp.size(0), weight_fp, eps).to(x.scalarType())

# note: according to https://github.com/pytorch/pytorch/issues/167308
# nn.RMSNorm is supposed to dispatch to fused impl with mixed precision,
# but we get a warning
#   [TIMESTAMP layer_norm.cpp:344] Warning: Mismatch dtype between input and weight: input dtype = float, weight dtype = c10::BFloat16, Cannot dispatch to fused implementation. (function operator())

func rmsNormOptPow2RsqrtFP32(x, weight: Tensor, eps: float64): Tensor =
  ## Optimized FP32 RMSNorm
  ## Uses `.pow(2)` and `.rsqrt()`
  let x_fp = x.to(kFloat32)
  let variance = x_fp.pow(2).mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).rsqrt()
  let normed = (x_fp * rstd).to(x.scalarType())
  normed * weight

func rmsNormOptSqrRsqrtFP32(x, weight: Tensor, eps: float64): Tensor =
  ## Optimized FP32 RMSNorm
  ## Uses `.square()` and `.rsqrt()`
  let x_fp = x.to(kFloat32)
  let variance = x_fp.square().mean(axis = -1, keepdim = true)
  let rstd = variance.add(eps).rsqrt()
  let normed = (x_fp * rstd).to(x.scalarType())
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

  # Header
  echo "Implementation                                  ns/op  ops/s  vs ref     diff"
  echo "-----------------------------------------------------------------------------"

  # Reference first (exact FP32 match)
  var rRef: Tensor
  for _ in 0 ..< Warmup:
    rRef = rmsNormManualFP32(x, weight, eps)
  let nsRef = measure(Iters):
    rRef = rmsNormManualFP32(x, weight, eps)
  let fp32Ref = rRef.to(kCPU).to(kFloat32)
  reportLine("Naive FP32 (ref)", nsRef, 0.0, nsRef)

  block: # Manual BF16
    var r1: Tensor
    for _ in 0 ..< Warmup:
      r1 = rmsNormManualBF16(x, weight, eps)
    let ns1 = measure(Iters):
      r1 = rmsNormManualBF16(x, weight, eps)
    let d1 = (r1.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
    reportLine("Naive BF16", ns1, d1, nsRef)

  block: # Fused BF16
    var r3: Tensor
    for _ in 0 ..< Warmup:
      r3 = rmsNormFusedBF16(x, weight, eps)
    let ns3 = measure(Iters):
      r3 = rmsNormFusedBF16(x, weight, eps)
    let d3 = (r3.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
    reportLine("Fused BF16", ns3, d3, nsRef)

  block: # Fused FP32
    var r4: Tensor
    for _ in 0 ..< Warmup:
      r4 = rmsNormFusedFP32(x, weight, eps)
    let ns4 = measure(Iters):
      r4 = rmsNormFusedFP32(x, weight, eps)
    let d4 = (r4.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
    reportLine("Fused FP32", ns4, d4, nsRef)

  block: # Optimized FP32 (.pow(2) + .rsqrt())
    var r6: Tensor
    for _ in 0 ..< Warmup:
      r6 = rmsNormOptPow2RsqrtFP32(x, weight, eps)
    let ns6 = measure(Iters):
      r6 = rmsNormOptPow2RsqrtFP32(x, weight, eps)
    let d6 = (r6.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
    reportLine("Opt FP32 (pow(2)+rsqrt)", ns6, d6, nsRef)

  block: # Optimized FP32 (.square() + .rsqrt())
    var r7: Tensor
    for _ in 0 ..< Warmup:
      r7 = rmsNormOptSqrRsqrtFP32(x, weight, eps)
    let ns7 = measure(Iters):
      r7 = rmsNormOptSqrRsqrtFP32(x, weight, eps)
    let d7 = (r7.to(kCPU).to(kFloat32) - fp32Ref).abs().max().item(float)
    reportLine("Opt FP32 (square+rsqrt)", ns7, d7, nsRef)


when isMainModule:
  benchRMSNorm()
