# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on --hints:off --warnings:off \
##   --passC:"-std=c++20" --outdir:build/bench/rel --nimcache:nimcache/bench/rel \
##   workspace/libtorch/bench/cpu/bench_moe_decode.nim
## LD_LIBRARY_PATH="$(pwd)/.venv/lib:$(pwd)/.venv/lib/python3.14/site-packages/torch/lib" \
##   ./build/bench/rel/bench_moe_decode

## CPU spellings of the routed MoE decode dispatch at the Qwen3.6-35B-A3B
## shape, no model load, synthetic weights.
##
## Geometry (layer 0 mlp of the checkpoint):
##   T = 1 token, top_k K = 8, num_experts E = 256
##   routed width I = 512, hidden H = 2048
##   gate_up_proj [E, 2I, H] bf16, down_proj [E, H, I] bf16
##   routed weight bytes per token, K experts: (2I + I) * H * 2 = 50 MB
##
## Spellings:
##   v0 current  expertForwardDecode, gather + one bmm per stage,
##               copies the 50 MB of routed weights per call
##   v1 loop     per-expert narrow + GEMV loop, weights read in place,
##               f32 accumulator so the k terms round once like sum(0)
##   v2 loop-bf16  v1 with a bf16 accumulator, k roundings, drift control
##
## Block (b) times each spelling. Block (c) compares v1 and v2 against v0:
## max abs diff within 2 bf16 ulps at the output magnitude.
##
## Standalone benchmark: run directly, not wired into test suites.

import
  std/monotimes,
  std/times,
  std/strformat,
  std/os,
  std/math,
  std/algorithm,
  std/importutils,
  workspace/libtorch as F,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/ffn {.all.}

{.experimental: "callOperator".}

privateAccess(GatedBlockSparseFFN)

const
  NumExperts = 256
  TopK = 8
  RoutedWidth = 512
  Hidden = 2048
  Iters = 60

proc median(values: seq[int64]): int64 =
  var sorted = values
  sorted.sort()
  result = sorted[sorted.len div 2]

proc ms(ns: int64): float64 = ns.float64 / 1e6

proc ulpBf16(m: float64): float64 =
  ## One bf16 ulp at magnitude m: 8 significand bits, ulp 2^(e-8).
  if m <= 0.0:
    return 0.0
  pow(2.0, floor(log2(m)) - 8.0)

proc bench(label: string, ffn: GatedBlockSparseFFN,
    hidden, topKIndex, topKWeights: Tensor,
    step: proc (f: GatedBlockSparseFFN, h, i, w: Tensor): Tensor,
    refOut: Tensor): int64 =
  ## Times one decode spelling per call, then reports the max abs diff
  ## against refOut in bf16 ulps at the output magnitude.
  var sink: Tensor
  sink = step(ffn, hidden, topKIndex, topKWeights) # warmup
  discard sink.to(F.kCPU)
  var samples: seq[int64]
  for _ in 0 ..< 5:
    let t0 = getMonotime()
    for _ in 0 ..< Iters:
      sink = step(ffn, hidden, topKIndex, topKWeights)
    discard sink.to(F.kCPU)
    samples.add (getMonotime() - t0).inNanoseconds div Iters
  result = median(samples)
  if not refOut.isNil:
    let diff = (sink.to(F.kFloat32) - refOut.to(F.kFloat32)).abs().max().item(float64)
    let mag = sink.to(F.kFloat32).abs().max().item(float64)
    let u = if diff == 0.0: 0.0 else: diff / ulpBf16(mag)
    echo &"  {label}: {ms(result):.3f} ms/call, drift {u:.2f} bf16 ulp at mag {mag:.3e}"
  else:
    echo &"  {label}: {ms(result):.3f} ms/call (reference)"

proc decodeLoop(f: GatedBlockSparseFFN, hidden, topKIndex, topKWeights: Tensor,
    accDtype: ScalarKind): Tensor =
  ## Per-expert narrow + GEMV loop. Reading the expert ids on the host
  ## is free on CPU, no sync concern. The accumulator runs at accDtype,
  ## so the terms round once (f32, like sum(0)) or once per term (bf16).
  let topK = topKIndex.size(1)
  var acc = F.zeros(1, Hidden, accDtype)
  for pos in 0 ..< topK:
    let e = topKIndex[0, pos].item(int64).int
    let gateUpW = f.gateUpProj[e]                  # (2I, H) view, no copy
    let gu = F.matmul(hidden, gateUpW.t())         # (1, 2I) GEMV
    let chunks = F.chunk(gu, 2, -1)
    let act = F.silu(chunks[0]) * chunks[1]        # (1, I)
    let down = F.matmul(act, f.downProj[e].t())    # (1, H) GEMV
    let term = down * topKWeights[0, pos]
    acc = acc + term.to(accDtype)
  result = acc.to(hidden.scalarType())

proc main() =
  echo "MoE decode dispatch spellings, CPU, 35B shape"
  echo "========================================================================="
  echo &"T = 1, K = {TopK}, E = {NumExperts}, I = {RoutedWidth}, H = {Hidden}"
  echo()

  let gateUpProj = F.randn(NumExperts, 2 * RoutedWidth, Hidden, F.kBFloat16)
    .to(F.kCPU) * Scalar(0.02)
  let downProj = F.randn(NumExperts, Hidden, RoutedWidth, F.kBFloat16)
    .to(F.kCPU) * Scalar(0.02)
  let routerWeight = F.randn(NumExperts, Hidden, F.kBFloat16).to(F.kCPU)
  let sharedGateWeight = F.randn(1, Hidden, F.kBFloat16).to(F.kCPU)
  let sharedExpert = GatedDenseFFN.init(
    F.randn(RoutedWidth, Hidden, F.kBFloat16).to(F.kCPU) * Scalar(0.02),
    F.randn(RoutedWidth, Hidden, F.kBFloat16).to(F.kCPU) * Scalar(0.02),
    F.randn(Hidden, RoutedWidth, F.kBFloat16).to(F.kCPU) * Scalar(0.02))
  let ffn = GatedBlockSparseFFN.init(
    gateUpProj, downProj, routerWeight, sharedExpert, sharedGateWeight, TopK)

  let hidden = F.randn(1, Hidden, F.kBFloat16).to(F.kCPU)
  # Realistic routing through the production router.
  let routed = routeToExperts(hidden, routerWeight, TopK)
  let topKIndex = routed.indices
  let topKWeights = routed.weights

  let routedMb = TopK * (2 * RoutedWidth + RoutedWidth) * Hidden * 2 div (1024 * 1024)
  echo "[a] routed weight bytes touched per call: ", routedMb, " MB"
  echo "[b] decode spellings"
  let refOut = ffn.expertForwardDecode(hidden, topKIndex, topKWeights)
  discard bench("v0 current (gather + bmm)", ffn, hidden, topKIndex, topKWeights,
    proc (f: GatedBlockSparseFFN, h, i, w: Tensor): Tensor =
      f.expertForwardDecode(h, i, w), nil)
  discard bench("v1 narrow + GEMV loop, f32 acc", ffn, hidden, topKIndex, topKWeights,
    proc (f: GatedBlockSparseFFN, h, i, w: Tensor): Tensor =
      decodeLoop(f, h, i, w, F.kFloat32), refOut)
  discard bench("v2 narrow + GEMV loop, bf16 acc", ffn, hidden, topKIndex, topKWeights,
    proc (f: GatedBlockSparseFFN, h, i, w: Tensor): Tensor =
      decodeLoop(f, h, i, w, F.kBFloat16), refOut)

  echo()
  echo "[c] budget: v1 and v2 drift within 2 bf16 ulps at the output magnitude"

when isMainModule:
  main()
