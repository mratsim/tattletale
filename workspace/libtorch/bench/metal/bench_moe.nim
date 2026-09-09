# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Per-op timing for the routed MoE FFN, Qwen3.6-35B-A3B decode shapes on MPS.
##
## Fixture-recorded geometry (layer 0 mlp):
##   hidden 2048, num_experts 256, num_experts_per_tok 8,
##   moe_intermediate_size 512, shared_expert_intermediate_size 512
##   gate_up_proj [256, 1024, 2048], down_proj [256, 2048, 512]
##
## Shape labels below: T = tokens, K = num_experts_per_tok (top_k),
## E = num_experts.
##
## Attribution blocks:
##   - routed-weights evidence: expert hit audit, time scaling in top_k
##     and in num_experts
##   - item-sync isolation: the real forward carries 16 .item() host
##     syncs per layer, measured against a full-path replica with 2
##     syncs per layer under identical op order
##   - batched-dispatch prototype: gather plus one batched matmul pair
##     per stage, no per-expert aten calls, no host sync
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
  std/importutils,
  workspace/libtorch,
  workspace/transformers/src/layers/linear,
  workspace/transformers/src/layers/ffn

{.experimental: "callOperator".}

privateAccess(GatedBlockSparseFFN)

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
  echo &"{name:<56} {us:>10.2f} us {gbs:>9.2f} GB/s {pct:>6.1f}%"

proc benchTensor(runs: int, iters: int, bytes: int64, name: string,
    op: proc(): Tensor): tuple[asyncNs, syncNs: int64] =
  ## Warm up, then time `op` over `iters` per run, async wall and synced wall.
  var r: Tensor
  for _ in 0 ..< 20:
    r = op()
  var nsRuns, nsSyncRuns: seq[int64]
  for run in 0 ..< runs:
    let ns = measure(iters):
      r = op()
    nsRuns.add ns
  discard r.to(kCPU)
  for run in 0 ..< runs:
    let ns = measure(iters):
      r = op()
      discard r.to(kCPU)
    nsSyncRuns.add ns
  reportLine(name & " (async wall)", median(nsRuns), bytes)
  reportLine(name & " (synced)", median(nsSyncRuns), bytes)
  result = (median(nsRuns), median(nsSyncRuns))

proc moeForwardReplica(
    routerWeight, gateUpProj, downProj: Tensor,
    sharedExpert: GatedDenseFFN,
    sharedGateWeight: Tensor,
    x: Tensor,
    topK, hiddenSize: int): Tensor =
  ## Full-path replica of GatedBlockSparseFFN.forward with 2 host syncs
  ## (one index copy, one weight copy) instead of 16 .item() syncs.
  ## Op order matches the real path: route, shared gate, shared expert,
  ## routed loop in ascending expert order, scatter into a zeroed
  ## accumulator, add.
  let opts = tensorOptions(x.scalarType(), x.deviceType())
  let (indices, weights) = routeToExperts(x, routerWeight, topK)
  let hostIdx = indices.to(kCPU)
  let hostW = weights.to(kCPU)
  var acc = zeros(1, hiddenSize, opts)
  for k in 0 ..< topK:
    let e = int(hostIdx[0, k].item(int64))
    let tokenIdx = [int64(0)].toTensor().to(x.deviceType())
    let currentStates = index_select(x, 0, tokenIdx)
    let gateUpWeight = gateUpProj[e]
    let gateUpOut = matmul(currentStates, gateUpWeight.t())
    let chunks = chunk(gateUpOut, 2, -1)
    let act = silu(chunks[0]) * chunks[1]
    let downWeight = downProj[e]
    let currentHidden = matmul(act, downWeight.t())
    let col = hostW[0, k].item(float32)
    let weightCol = [[col]].toTensor().reshape(1, 1).to(x.deviceType(),
      x.scalarType())
    let weighted = currentHidden * weightCol
    let idx2d = tokenIdx.unsqueeze(1).expand(1, hiddenSize, implicit = false)
    acc = scatter_add(acc, 0, idx2d, weighted)
  let sharedGate = sigmoid(matmul(x, sharedGateWeight.t()))
  let sharedGated = sharedGate * sharedExpert.forward(x)
  result = acc + sharedGated

proc moeForwardBatched(
    routerWeight, gateUpProj, downProj: Tensor,
    sharedExpert: GatedDenseFFN,
    sharedGateWeight: Tensor,
    x: Tensor,
    topK, hiddenSize: int): Tensor =
  ## Batched-dispatch prototype for the decode shape (T = 1).
  ##
  ## Replaces the per-expert aten loop. Op sequence: one index_select
  ## gather over the routed experts, one batched matmul per projection
  ## stage, one elementwise multiply by the routing weights, one sum
  ## over experts.
  ## Zero .item() host syncs: routing indices stay on device.
  ##
  ## Gather doubles the routed weight traffic: one index_select copy,
  ## then one matmul read. Priced below as 2x expert bytes.
  ##
  ## Expected input:
  ## - x [1, H] at the multiply dtype
  ## - topK >= 1, expert ids per token read from routeToExperts directly
  ##
  ## Output:
  ## - [1, H]: routed sum over experts plus the gated shared expert,
  ##   same contract as GatedBlockSparseFFN.forward at T = 1
  let opts = tensorOptions(x.scalarType(), x.deviceType())
  let (indices, weights) = routeToExperts(x, routerWeight, topK)
  let flatIdx = indices.reshape(topK)
  let gatheredGateUp = index_select(gateUpProj, 0, flatIdx)
  let gatheredDown = index_select(downProj, 0, flatIdx)
  let xs = x.unsqueeze(0).expand(topK, 1, hiddenSize, implicit = false)
    .contiguous()
  let gateUpOut = matmul(xs, gatheredGateUp.transpose(1, 2))
  let chunks = chunk(gateUpOut, 2, -1)
  let act = silu(chunks[0]) * chunks[1]
  let downOut = matmul(act, gatheredDown.transpose(1, 2))
  let weightCol = weights.transpose(0, 1).unsqueeze(2)
    .to(x.scalarType())
  let summed = (downOut * weightCol).sum(0)
  let sharedGate = sigmoid(matmul(x, sharedGateWeight.t()))
  let sharedGated = sharedGate * sharedExpert.forward(x)
  result = summed + sharedGated

proc buildMoE(numExperts, topK, hidden, inter: int, opts: TensorOptions): GatedBlockSparseFFN =
  let gateUpProj = randn(numExperts, 2 * inter, hidden, opts)
  let downProj = randn(numExperts, hidden, inter, opts)
  let routerWeight = randn(numExperts, hidden, opts)
  let sharedExpert = GatedDenseFFN.init(
    randn(inter, hidden, opts), randn(inter, hidden, opts),
    randn(hidden, inter, opts))
  let sharedGateWeight = randn(1, hidden, opts)
  GatedBlockSparseFFN.init(
    gateUpProj, downProj, routerWeight, sharedExpert,
    sharedGateWeight, topK)

proc runBench() =
  let deviceName = if paramCount() > 0: paramStr(1).toLowerAscii() else: "mps"
  let device =
    case deviceName
    of "cpu": kCPU
    of "mps", "gpu": kMPS
    else: kMPS

  const
    Hidden = 2048
    Inter = 512
    TopK = 8
    NumExperts = 256
    Warmup = 20
    Iters = 100
    Runs = 5

  echo "Routed MoE FFN benchmark, Qwen3.6-35B-A3B layer geometry"
  echo "========================================================"
  echo &"date: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo &"device: {deviceName}"
  echo &"dtype: bfloat16"
  echo &"weights: gate_up_proj [E, {2 * Inter}, {Hidden}], down_proj [E, {Hidden}, {Inter}], router [E, {Hidden}], shared expert {Inter}"
  echo &"decode activations: hidden (1, {Hidden}), top_k {TopK} of {NumExperts} experts"
  echo &"iters per run: {Iters}, runs: {Runs} (median reported)"
  echo()

  let opts = tensorOptions(kBFloat16, device)
  let ffn = buildMoE(NumExperts, TopK, Hidden, Inter, opts)
  let x = randn(1, Hidden, opts)

  # Per-expert weight bytes: gate_up row 2I x H plus down row H x I, bf16
  let expertBytes = int64(2 * Inter) * Hidden * 2 + int64(Hidden) * Inter * 2
  # Shared expert weight bytes
  let sharedBytes = 3 * int64(Inter) * Hidden * 2
  let routerBytes = int64(NumExperts) * Hidden * 2

  # Block 1: router only
  block:
    var indices, weights: Tensor
    for _ in 0 ..< Warmup:
      (indices, weights) = routeToExperts(x, ffn.routerWeight, TopK)
    var nsRuns, nsSyncRuns: seq[int64]
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        (indices, weights) = routeToExperts(x, ffn.routerWeight, TopK)
      nsRuns.add ns
    discard indices.to(kCPU)
    for run in 0 ..< Runs:
      let ns = measure(Iters):
        (indices, weights) = routeToExperts(x, ffn.routerWeight, TopK)
        discard indices.to(kCPU)
      nsSyncRuns.add ns
    reportLine("routeToExperts T=1 K=8 (async wall)", median(nsRuns), routerBytes)
    reportLine("routeToExperts T=1 K=8 (synced)", median(nsSyncRuns), routerBytes)

    # Expert hit audit for one decode token
    let hostIdx = indices.to(kCPU)
    var hitSet: seq[int64]
    for k in 0 ..< TopK:
      let e = int(hostIdx[0, k].item(int64))
      if e notin hitSet:
        hitSet.add e
    echo &"audit: distinct experts hit by one decode token: {hitSet.len} of {NumExperts}, ids {hitSet}"
    echo()

  # Block 2: shared expert only
  discard benchTensor(Runs, Iters, sharedBytes, "shared expert T=1",
    proc(): Tensor = ffn.sharedExpert.forward(x))

  # Block 3: real full forward, K = 8, E = 256
  var fullAsync, fullSync: int64
  block:
    let fullBytes = routerBytes + sharedBytes + int64(TopK) * expertBytes
    (fullAsync, fullSync) = benchTensor(Runs, Iters, fullBytes,
      "moe full forward T=1 K=8 E=256",
      proc(): Tensor = ffn.forward(x))

  # Block 4: full-path replica, 2 syncs, same op order
  var repAsync, repSync: int64
  block:
    let fullBytes = routerBytes + sharedBytes + int64(TopK) * expertBytes
    (repAsync, repSync) = benchTensor(Runs, Iters, fullBytes,
      "moe replica 2-sync T=1 K=8 E=256",
      proc(): Tensor = moeForwardReplica(
        ffn.routerWeight, ffn.gateUpProj, ffn.downProj,
        ffn.sharedExpert, ffn.sharedGateWeight, x, TopK, Hidden))

  echo()
  let dAsync = float64(fullAsync - repAsync) / 1000.0
  let dSync = float64(fullSync - repSync) / 1000.0
  echo &"item-sync penalty per layer (real minus replica): async {dAsync:.1f} us, synced {dSync:.1f} us"
  echo &"estimated per-forward penalty, 40 moe layers: async {dAsync * 40:.1f} us, synced {dSync * 40:.1f} us"
  echo()

  # Block 5: time scaling in top_k through the real forward path,
  # E = 256. Linear growth in top_k matches the routed bytes law.
  for k in [1, 2, 4, 8]:
    block:
      let ffnK = buildMoE(NumExperts, k, Hidden, Inter, opts)
      let fullBytes = routerBytes + sharedBytes + int64(k) * expertBytes
      discard benchTensor(Runs, Iters, fullBytes,
        &"moe full forward T=1 K={k} E=256",
        proc(): Tensor = ffnK.forward(x))

  # Block 6: time scaling in num_experts through the real forward
  # path, top_k = 8. Near-flat time across expert counts means
  # routed-only weight touching.
  for e in [512, 1024]:
    block:
      let ffnBig = buildMoE(e, TopK, Hidden, Inter, opts)
      let routerBig = int64(e) * Hidden * 2
      let fullBytes = routerBig + sharedBytes + int64(TopK) * expertBytes
      discard benchTensor(Runs, Iters, fullBytes,
        &"moe full forward T=1 K=8 E={e}",
        proc(): Tensor = ffnBig.forward(x))

  # Block 7: batched-dispatch prototype, gather + 2 batched matmuls,
  # zero .item() syncs. Gather prices 2x expert bytes (copy + matmul read).
  var batchedSync: int64
  block:
    let fullBytes = routerBytes + sharedBytes + 2 * int64(TopK) * expertBytes
    discard benchTensor(Runs, Iters, fullBytes,
      "moe batched prototype T=1 K=8 E=256",
      proc(): Tensor = moeForwardBatched(
        ffn.routerWeight, ffn.gateUpProj, ffn.downProj,
        ffn.sharedExpert, ffn.sharedGateWeight, x, TopK, Hidden))

  # Correctness check: batched prototype against the real forward at T=1.
  # fp32 spelling of both sides keeps the bf16 round-trip out of the diff.
  block:
    let realOut = ffn.forward(x).to(kFloat32).to(kCPU)
    let batchedOut = moeForwardBatched(
      ffn.routerWeight, ffn.gateUpProj, ffn.downProj,
      ffn.sharedExpert, ffn.sharedGateWeight, x, TopK, Hidden
    ).to(kFloat32).to(kCPU)
    let repOut = moeForwardReplica(
      ffn.routerWeight, ffn.gateUpProj, ffn.downProj,
      ffn.sharedExpert, ffn.sharedGateWeight, x, TopK, Hidden
    ).to(kFloat32).to(kCPU)
    let diff = (realOut - batchedOut).abs().max().item(float32)
    let diffRep = (realOut - repOut).abs().max().item(float32)
    let absSum = (realOut - batchedOut).abs().sum().item(float32)
    echo &"correctness: max abs diff, real forward vs batched prototype: {diff:.3e}, vs 2-sync replica: {diffRep:.3e}, abs-sum diff: {absSum:.3e}"
    echo &"outputs share the scale: real max {realOut.max().item(float32):.4f}, batched max {batchedOut.max().item(float32):.4f}"

when isMainModule:
  runBench()
