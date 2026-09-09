# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Per-op timing for the full-attention decode path, Qwen3.6-35B-A3B shapes on MPS.
##
## Layer under test: RopeElementWiseGatedAttention[RmsNormOne] forward,
## the production decode path with paged KV write and gather.
##
## Fixture-recorded geometry (layer 3):
##   hidden 2048, q heads 16, kv heads 2, head_dim 256, partial rotary 64,
##   attn_output_gate, q_proj emits [q | gate] per head
##
## Decode-time activations: x (1, 1, 2048), context length N pages filled
## with random K/V, per-call cost measured against N.
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
  std/options,
  workspace/libtorch,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/norm,
  workspace/transformers/src/layers/attn_ssm/gated_attention,
  workspace/transformers/src/layers/rope,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/src/stateful/kvcache,
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
    NumQo = 16
    NumKv = 2
    HeadDim = 256
    RotaryDim = 64
    MaxSeqLen = 4096
    QProjRows = NumQo * 2 * HeadDim     # 8192, [q | gate] per head
    KProjRows = NumKv * HeadDim         # 512
    VProjRows = NumKv * HeadDim         # 512
    OProjIn = NumQo * HeadDim           # 4096
    Warmup = 20
    Iters = 100
    Runs = 5
  const Contexts = [64, 256, 1024, 2048]

  echo "Gated GQA attention benchmark, Qwen3.6-35B-A3B layer geometry"
  echo "============================================================="
  echo &"date: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo &"device: {deviceName}"
  echo &"dtype: bfloat16"
  echo &"weights: q_proj ({QProjRows}, {Hidden}), k_proj ({KProjRows}, {Hidden}), v_proj ({VProjRows}, {Hidden}), o_proj ({Hidden}, {OProjIn})"
  echo &"decode activations: x (1, 1, {Hidden}), query (1, 1, {NumQo}, {HeadDim}), kv cache {NumKv} heads x {HeadDim}"
  echo &"qk norm: RmsNormOne over {HeadDim}, rotary_dim {RotaryDim}, attn_output_gate on"
  echo &"iters per run: {Iters}, runs: {Runs} (median reported)"
  echo()

  let opts = tensorOptions(kBFloat16, device)
  let qProj = Linear.init(randn(QProjRows, Hidden, opts))
  let kProj = Linear.init(randn(KProjRows, Hidden, opts))
  let vProj = Linear.init(randn(VProjRows, Hidden, opts))
  let oProj = Linear.init(randn(Hidden, OProjIn, opts))
  let qNorm = RmsNormOne.init(randn(HeadDim, kBFloat16).to(device))
  let kNorm = RmsNormOne.init(randn(HeadDim, kBFloat16).to(device))
  let rotary = RotaryPositionEmbedding.new(
    HeadDim, MaxSeqLen, 1e7, kBFloat16, device, rotary_dim = RotaryDim)
  let layer = RopeElementWiseGatedAttention[RmsNormOne].init(
    0, "bench.attn", qProj, kProj, vProj, oProj,
    NumQo, NumKv, HeadDim, rotary, qNorm, kNorm)

  # Projection weight bytes: one read each per call
  let projBytes = int64(QProjRows + KProjRows + VProjRows + Hidden) * Hidden * 2
  # qk norm weight bytes
  let normBytes = 2 * HeadDim * 2

  for n in Contexts:
    # Fill N tokens of KV content through the public sequence path:
    # startSequence borrows the pages. The bench writes random K/V rows
    # and rewinds the write cursor to N.
    # n+1 tracked tokens so the page set covers the decode write at n
    var ids = newSeq[uint32](n + 1)
    for i in 0 ..< n + 1:
      ids[i] = uint32((7 * i + 3) mod 248320)
    var orc = Orchestrator.init(
      numLayers = 1, batchSize = 1, kvHeads = NumKv,
      maxSeq = MaxSeqLen, headDim = HeadDim,
      numPages = ceilDiv(n + 1, TokensPerPage) + 1, dtype = kBFloat16, device = device)
    orc.startSequence(ids)
    for p in 0 ..< orc.getInferenceContextMut().pages.len:
      let kFill = randn(TokensPerPage, NumKv, HeadDim, opts)
      let vFill = randn(TokensPerPage, NumKv, HeadDim, opts)
      orc.getInferenceContextMut().pages[p].k_view[0, 0 ..< TokensPerPage].copyFrom(kFill)
      orc.getInferenceContextMut().pages[p].v_view[0, 0 ..< TokensPerPage].copyFrom(vFill)

    orc.getInferenceContextMut().kv_position = n
    orc.getInferenceContextMut().position_ids = [[int64(n)]].toTensor().to(device)
    setRopeForPositions(orc.getInferenceContextMut(), rotary)

    let x = randn(1, 1, Hidden, opts)

    var attnOut: Tensor
    for _ in 0 ..< Warmup:
      attnOut = layer(orc.getInferenceContextMut(), x)
    var nsRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        attnOut = layer(orc.getInferenceContextMut(), x)
      nsRuns.add ns
    discard attnOut.to(kCPU)
    var nsSyncRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        attnOut = layer(orc.getInferenceContextMut(), x)
        discard attnOut.to(kCPU)
      nsSyncRuns.add ns

    # Bytes: projections + qk norm + kv cache read (N+1) + kv write 1
    # + gather read+write (N+1), all bf16
    let kvOne = int64(NumKv) * HeadDim * 2
    let kvTraffic = int64(n + 1) * kvOne * (2 + 1 + 2)
    let totalBytes = projBytes + normBytes + kvTraffic
    reportLine(&"attn decode fwd, ctx {n:>5} (async wall)", median(nsRuns), totalBytes)
    reportLine(&"attn decode fwd, ctx {n:>5} (synced)", median(nsSyncRuns), totalBytes)
    orc.endSequence()

when isMainModule:
  runBench()
