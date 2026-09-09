# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## nim cpp -d:release --stackTrace:on --lineTrace:on --lineDir:on --hints:off --warnings:off \
##   --passC:"-std=c++20" --outdir:build/bench/rel --nimcache:nimcache/bench/rel \
##   workspace/libtorch/bench/cpu/bench_gdn_recurrence.nim
## LD_LIBRARY_PATH="$(pwd)/.venv/lib:$(pwd)/.venv/lib/python3.14/site-packages/torch/lib" \
##   ./build/bench/rel/bench_gdn_recurrence

## CPU spellings of the gated delta-rule recurrence at the Qwen3.6-35B-A3B
## decode shape, no model load.
##
## Geometry (35B, text_config linear_* keys):
##   batch 1, value heads 32, head dim 128, one decode step (T = 1)
##   SSM state (1, 32, 128, 128) f32, 2 MB per copy
##
## Block (a) times the production gatedDeltaRuleRecurrence proc whole,
## l2norm and layout prep included, as the layer calls it at T = 1.
## Block (b) isolates the per-step state math with the f32 tensors
## pre-transformed, and compares op sequences:
##   v0 current   the production loop body, mul + sum reductions
##   v1 bmm       bmm for the k and q projections, bmm outer product
##                for the state update, gate folded after each reduce
##   v2 bmm+fold  v1 with the output read decomposed to g*(q.s) + (q.k)*d,
##                one state read fewer, larger rounding drift
##   v3 reorder   elementwise like v0, gate applied after the kv reduce
##
## Block (c) reports max abs differences of each variant against v0,
## in fp32 ulps, state and output. The fixture state budget is 4 fp32
## ulps, measured against the chunked reference form. A variant drifting
## past about one ulp against v0 is a poor production candidate.
##
## Standalone benchmark: run directly, not wired into test suites.

import
  std/monotimes,
  std/times,
  std/strformat,
  std/os,
  std/math,
  std/algorithm,
  workspace/libtorch as F,
  workspace/transformers/src/layers/attn_ssm/gated_delta_net {.all.}

{.experimental: "callOperator".}

const
  NumHeads = 32
  HeadDim = 128
  ProdIters = 30   # whole production proc, about 3 ms per call
  StepIters = 200  # isolated per-step spellings, sub-millisecond

proc median(values: seq[int64]): int64 =
  var sorted = values
  sorted.sort()
  result = sorted[sorted.len div 2]

proc ms(ns: int64): float64 = ns.float64 / 1e6

proc ulpFp32(m: float64): float64 =
  ## One fp32 ulp at magnitude m.
  if m <= 0.0:
    return 0.0
  pow(2.0, floor(log2(m)) - 23.0)

proc ulps(diff, mag: float64): float64 =
  ## Difference expressed in fp32 ulps at magnitude mag.
  if diff == 0.0:
    0.0
  else:
    diff / ulpFp32(mag)

type
  StepResult = tuple[s: Tensor, o: Tensor]

proc benchWhole(label: string, iters: int,
    q, k, v, g, beta, initialS: Tensor) =
  ## Times the production proc whole, per call.
  var sink: (Tensor, Tensor)
  sink = gatedDeltaRuleRecurrence(q, k, v, g, beta, initialS) # warmup
  discard sink[0].to(F.kCPU)
  discard sink[1].to(F.kCPU)
  var samples: seq[int64]
  for _ in 0 ..< 5:
    let t0 = getMonotime()
    for _ in 0 ..< iters:
      sink = gatedDeltaRuleRecurrence(q, k, v, g, beta, initialS)
    discard sink[0].to(F.kCPU)
    discard sink[1].to(F.kCPU)
    samples.add (getMonotime() - t0).inNanoseconds div iters
  echo &"  {label}: {ms(median(samples)):.3f} ms/call (median of 5 x {iters})"

proc benchStep(label: string, iters: int, step: proc (): StepResult,
    refR: StepResult): int64 =
  ## Times one per-step spelling, then diffs it against the reference
  ## result in fp32 ulps. Returns the per-call nanoseconds.
  var sink: StepResult
  sink = step() # warmup
  discard sink.s.to(F.kCPU)
  discard sink.o.to(F.kCPU)
  var samples: seq[int64]
  for _ in 0 ..< 5:
    let t0 = getMonotime()
    for _ in 0 ..< iters:
      sink = step()
    discard sink.s.to(F.kCPU)
    discard sink.o.to(F.kCPU)
    samples.add (getMonotime() - t0).inNanoseconds div iters
  result = median(samples)
  let sDiff = (sink.s - refR.s).abs().max().item(float64)
  let oDiff = (sink.o - refR.o).abs().max().item(float64)
  let sMag = sink.s.abs().max().item(float64)
  let oMag = sink.o.abs().max().item(float64)
  echo &"  {label}: {ms(result):.3f} ms/call, state drift " &
    &"{ulps(sDiff, sMag):.1f} ulp, output drift {ulps(oDiff, oMag):.1f} ulp"

proc main() =
  echo "GDN delta-rule recurrence spellings, CPU, 35B decode shape"
  echo "========================================================================="
  echo &"state (1, {NumHeads}, {HeadDim}, {HeadDim}) f32, T = 1, batch 1"
  echo()

  # Production-shaped inputs: q, k, v (1, 1, H, dim) bf16, g (1, 1, H) f32,
  # beta (1, 1, H) bf16, initial state (1, H, Dk, Dv) f32.
  let q = F.randn(1, 1, NumHeads, HeadDim, F.kBFloat16).to(F.kCPU)
  let k = F.randn(1, 1, NumHeads, HeadDim, F.kBFloat16).to(F.kCPU)
  let v = F.randn(1, 1, NumHeads, HeadDim, F.kBFloat16).to(F.kCPU)
  let g = F.randn(1, 1, NumHeads, F.kFloat32).to(F.kCPU) * Scalar(0.1)
  let beta = F.randn(1, 1, NumHeads, F.kBFloat16).to(F.kCPU)
  let initialS = F.randn(1, NumHeads, HeadDim, HeadDim, F.kFloat32).to(F.kCPU)

  echo "[a] production gatedDeltaRuleRecurrence proc, whole call"
  benchWhole("production proc", ProdIters, q, k, v, g, beta, initialS)
  echo()

  # Pre-transformed f32 step tensors, shared by all spellings.
  # Shapes mirror the production loop body at t: kT and qT are (1, H, D, 1)
  # column slices, vT is (1, H, Dv), gates are (1, H, 1) rows.
  let qB = (q.to(F.kFloat32) * Scalar(1.0 / sqrt(HeadDim.float64)))
    .view(1, NumHeads, 1, HeadDim)                    # (1, H, 1, Dk), bmm form
  let qT = qB.view(1, NumHeads, HeadDim).unsqueeze(-1) # (1, H, Dk, 1), mul form
  let kT = k.to(F.kFloat32).view(1, NumHeads, HeadDim).unsqueeze(-1)
  let vT = v.to(F.kFloat32).view(1, NumHeads, HeadDim)
  let betaT = beta.to(F.kFloat32).view(1, NumHeads, 1)
  let gT = g.view(1, NumHeads).exp().unsqueeze(-1)     # (1, H, 1)
  let gT4 = gT.unsqueeze(-1)                           # (1, H, 1, 1)

  echo "[b] per-step state math, f32 tensors pre-transformed"
  proc stepV0(): StepResult =
    ## Production loop body at T = 1, coreOut alloc and slice write included.
    var st = initialS * gT4
    let kvMem = (st * kT).sum(axis = -2)
    let delta = (vT - kvMem) * betaT
    st = st + kT * delta.unsqueeze(-2)
    var coreOut = F.zeros(1, NumHeads, 1, HeadDim, F.kFloat32)
    coreOut[_, _, 0, _] = (st * qT).sum(axis = -2)
    (st, coreOut)
  let refR = stepV0()
  discard benchStep("v0 current (mul + sum)", StepIters, stepV0, refR)
  discard benchStep("v1 bmm reductions", StepIters,
    proc (): StepResult =
      let s3 = initialS.view(NumHeads, HeadDim, HeadDim)
      let kT3 = kT.view(NumHeads, HeadDim, 1)
      let qB3 = qB.view(NumHeads, 1, HeadDim)
      let kvMem = F.bmm(kT3.transpose(1, 2), s3)
        .view(1, NumHeads, HeadDim) * gT
      let delta = (vT - kvMem) * betaT
      let outer = F.bmm(kT3, delta.view(NumHeads, 1, HeadDim))
      let sNew = initialS * gT4 + outer.view(1, NumHeads, HeadDim, HeadDim)
      (sNew, F.bmm(qB3, sNew.view(NumHeads, HeadDim, HeadDim))
        .view(1, NumHeads, 1, HeadDim)),
    refR)
  discard benchStep("v2 bmm + folded output", StepIters,
    proc (): StepResult =
      let s3 = initialS.view(NumHeads, HeadDim, HeadDim)
      let kT3 = kT.view(NumHeads, HeadDim, 1)
      let qB3 = qB.view(NumHeads, 1, HeadDim)
      let kvMem = F.bmm(kT3.transpose(1, 2), s3)
        .view(1, NumHeads, HeadDim) * gT
      let delta = (vT - kvMem) * betaT
      let outer = F.bmm(kT3, delta.view(NumHeads, 1, HeadDim))
      let sNew = initialS * gT4 + outer.view(1, NumHeads, HeadDim, HeadDim)
      let qsRaw = F.bmm(qB3, s3).view(1, NumHeads, HeadDim) * gT
      let qk = F.bmm(qB3, kT3).view(1, NumHeads).unsqueeze(-1)
      (sNew, qsRaw + qk * delta),
    refR)
  discard benchStep("v3 elementwise reorder", StepIters,
    proc (): StepResult =
      let kvMem = (initialS * kT).sum(axis = -2) * gT
      let delta = (vT - kvMem) * betaT
      let sNew = initialS * gT4 + kT * delta.unsqueeze(-2)
      (sNew, (sNew * qT).sum(axis = -2).view(1, NumHeads, 1, HeadDim)),
    refR)

  echo()
  echo "[c] read the drift column against the 4 fp32 ulp fixture budget"

when isMainModule:
  main()
