# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Stage-attributed decode step through the wired Qwen3.6-35B-A3B Nim
## stack on CPU.
##
## Real checkpoint weights, tests/hf_models/Qwen3.6-35B-A3B, 67 GB bf16.
## Production forward path via the Orchestrator, mirroring
## qwen35_moe.forward op-for-op with stage timers: (a) embedding/entry,
## (b) per-decoder-layer, (c) final norm + lm_head, (d) argmax + .item().
## The stage sum is reconciled against the whole-step wall.
##
## Modes (first CLI arg, default "probe"):
##   probe - load the model, prefill 8, ONE timed decode step showing
##           the full per-layer-index breakdown.
##   timed - 1 warmup step, then TTT_STAGE_STEPS (env, default 4) timed
##           steps with per-stage and per-layer-index medians.
##   drill - no model load: kernel-level drill-downs at real decode
##           shapes: grouped conv1d bf16 vs f32, thread scaling,
##           lm_head GEMV bf16 vs f32 vocab slice, GDN recurrence step.
##
## Measurement only: no optimizations, no fixes. 300 s cap per step
## (partial steps are recorded and flagged).
##
## Standalone benchmark: run directly, not wired into test suites.

import
  std/monotimes,
  std/times,
  std/strformat,
  std/os,
  std/strutils,
  std/algorithm,
  std/sequtils,
  std/options,
  std/importutils,
  workspace/libtorch as F,
  workspace/transformers/src/models,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/models/loading/layer_kinds,
  workspace/transformers/src/layers {.all.},
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/orchestrator

{.experimental: "callOperator".}

# Diagnostic only: thread-scaling attribution for the CPU kernels.
proc atSetNumThreads(n: cint) {.importc: "at::set_num_threads",
  header: "<ATen/Parallel.h>".}
proc atGetNumThreads(): cint {.importc: "at::get_num_threads",
  header: "<ATen/Parallel.h>".}

const
  ModelDir = currentSourcePath().parentDir() / ".." / ".." / ".." /
    "transformers" / "tests" / "hf_models" / "Qwen3.6-35B-A3B"
  MaxContextLen = 256
  PromptLen = 8
  StepCapNs = 300_000_000_000'i64 # 300 s per single step
  VocabSlice = 8192               # vocab slice for the f32 GEMV drill

proc total(values: seq[int64]): int64 =
  for v in values:
    result += v

proc median(values: seq[int64]): int64 =
  var sorted = values
  sorted.sort()
  result = sorted[sorted.len div 2]

proc ms(ns: int64): float64 = ns.float64 / 1e6

proc stamp() =
  echo "---- run stamp ----"
  echo &"date: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo &"device: cpu (M4 Max, unified memory)"
  echo &"dtype: bfloat16 activations/weights, SSM state f32"
  echo &"weights: real checkpoint {ModelDir}"
  echo &"pid: {getCurrentProcessId()}"

type
  StepStages = object
    embedNs: int64
    layersNs: int64
    headNs: int64   # final norm + lm_head
    argmaxNs: int64
    wallNs: int64
    partial: bool

proc stagedForward(
    model: Qwen35MoeModel,
    ctx: var InferenceContext,
    inputIds: Tensor,
    layerNs: var seq[int64]): (Tensor, StepStages) =
  ## Mirrors qwen35_moe.forward (embed -> rope -> layers -> norm+lm_head)
  ## with stage timers. layerNs[i] accumulates per-layer time.
  privateAccess(Qwen35MoeModel)
  var stages: StepStages
  let t0 = getMonotime()
  var x = model.embedTokens.forward(inputIds)
  let t1 = getMonotime()
  stages.embedNs = (t1 - t0).inNanoseconds

  ctx.setRopeForPositions(model.rotary)

  var residual: Option[Tensor]
  for i, layer in model.layers:
    let tl0 = getMonotime()
    let layerOut = layer(ctx, x, residual)
    let tl1 = getMonotime()
    layerNs[i] += (tl1 - tl0).inNanoseconds
    x = layerOut[0]
    residual = some(layerOut[1])
    if (tl1 - t0).inNanoseconds > StepCapNs:
      stages.partial = true
      stages.layersNs = (tl1 - t1).inNanoseconds
      stages.wallNs = (tl1 - t0).inNanoseconds
      return (x, stages) # partial: head stage not reached
  let t2 = getMonotime()

  let finalResidual = residual.get(x)
  let normed = model.norm.forward(x + finalResidual)
  let logits = model.lmHead.forward(normed)
  let t3 = getMonotime()

  stages.layersNs = (t2 - t1).inNanoseconds
  stages.headNs = (t3 - t2).inNanoseconds
  stages.wallNs = (t3 - t0).inNanoseconds
  result = (logits, stages)

proc runStagedStep(
    model: Qwen35MoeModel,
    orc: var Orchestrator,
    ids: var seq[int],
    row: var Tensor,
    layerNs: var seq[int64],
    stepIdx: int): StepStages =
  ## One full decode step: token append, staged forward, argmax+.item(),
  ## KV position update.
  let chosen = row.argmax().item(int)
  ids.add chosen
  orc.appendToken(ids.len - 1, chosen.uint32, F.kCPU)

  let t0 = getMonotime()
  let (stepLogits, stages) = stagedForward(
    model, orc.getInferenceContextMut(), F.toTensor([[chosen]]).to(F.kCPU), layerNs)
  let t1 = getMonotime()

  orc.setKvPosition(ids.len)
  row = stepLogits.squeeze(0).squeeze(0)

  # argmax + .item() on THIS step's logits (stage d)
  let t2 = getMonotime()
  discard row.argmax().item(int)
  let t3 = getMonotime()
  var s = stages
  s.argmaxNs = (t3 - t2).inNanoseconds
  s.wallNs = (t1 - t0).inNanoseconds
  if s.partial:
    echo &"[step {stepIdx}] PARTIAL: 300 s per-step cap hit, head stage not reached"
  result = s

proc reportStage(stages: StepStages, layerNs: seq[int64], cfg: Qwen35MoeConfig) =
  let wall = stages.wallNs.float64
  echo &"  whole-step wall            {ms(stages.wallNs):>12.1f} ms"
  echo &"  (a) embedding/entry        {ms(stages.embedNs):>12.1f} ms  {100*stages.embedNs.float64/wall:>5.1f}%"
  echo &"  (b) all layers             {ms(stages.layersNs):>12.1f} ms  {100*stages.layersNs.float64/wall:>5.1f}%"
  echo &"  (c) final norm + lm_head   {ms(stages.headNs):>12.1f} ms  {100*stages.headNs.float64/wall:>5.1f}%"
  echo &"  (d) argmax + .item()       {ms(stages.argmaxNs):>12.1f} ms  {100*stages.argmaxNs.float64/wall:>5.1f}%"
  let stageSum = stages.embedNs + stages.layersNs + stages.headNs + stages.argmaxNs
  echo &"  stage sum (a+b+c+d)        {ms(stageSum):>12.1f} ms  ({100*stageSum.float64/wall:.1f}% of wall)"
  echo()
  echo "  per-layer times (ms):"
  for i in 0 ..< cfg.layerTypes.len:
    let kind = if cfg.layerTypes[i] == alkGatedDeltaNet: "gdn" else: "attn"
    echo &"    layer {i:>2} ({kind}) {ms(layerNs[i]):>12.1f}"
  let layerMed = median(layerNs.filterIt(it > 0))
  echo &"  median layer: {ms(layerMed):.1f} ms"

proc setupModelAndOrchestrator(): (Qwen35MoeModel, Orchestrator, Qwen35MoeConfig, seq[int], Tensor) =
  echo "Loading model (67 GB)..."
  let loadStart = getMonotime()
  let model = loadQwen35MoeModelRaw($(ModelDir), F.kCPU)
  echo &"Model loaded in {(getMonotime() - loadStart).inSeconds} s"
  let cfg = model.config
  echo &"config: layers {cfg.numHiddenLayers}, hidden {cfg.hiddenSize}, vocab {cfg.vocabSize}, gdn layers {cfg.layerTypes.countIt(it == alkGatedDeltaNet)}"

  let numPoolPages = computeNumPages(MaxContextLen, concurrentRequests = 1)
  var orc = Orchestrator.init(
    cfg.numHiddenLayers, 1, cfg.numKeyValueHeads, MaxContextLen,
    cfg.headDim, numPoolPages, F.kBFloat16, F.kCPU)

  var ids: seq[int] = @[]
  for i in 0 ..< PromptLen:
    ids.add (7 * i + 9707) mod cfg.vocabSize
  orc.startSequence(ids.mapIt(it.uint32))
  let inputIds = F.toTensor([ids]).to(F.kCPU)
  let t0 = getMonotime()
  let logits = model.forward(orc.getInferenceContextMut(), inputIds)
  echo &"prefill ({PromptLen} tokens) wall {ms((getMonotime() - t0).inNanoseconds):.1f} ms (excluded from stage attribution)"
  orc.setKvPosition(ids.len)
  let row = logits.narrow(1, ids.len - 1, 1).squeeze(1).squeeze(0)
  result = (model, orc, cfg, ids, row)

proc runProbe() =
  stamp()
  echo "mode: probe (ONE timed decode step, full per-layer breakdown)"
  echo()
  var (model, orc, cfg, ids0, row0) = setupModelAndOrchestrator()

  var layerNs = newSeq[int64](cfg.numHiddenLayers)
  var ids = ids0
  var row = row0
  echo()
  echo "probe step:"
  let stages = runStagedStep(model, orc, ids, row, layerNs, 0)
  reportStage(stages, layerNs, cfg)

proc runTimed() =
  let timedSteps =
    if getEnv("TTT_STAGE_STEPS").len > 0: getEnv("TTT_STAGE_STEPS").parseInt()
    else: 4
  stamp()
  echo &"mode: timed (1 warmup + {timedSteps} timed steps)"
  echo()
  var (model, orc, cfg, ids0, row0) = setupModelAndOrchestrator()

  var ids = ids0
  var row = row0

  var warmLayerNs = newSeq[int64](cfg.numHiddenLayers)
  echo "warmup step..."
  discard runStagedStep(model, orc, ids, row, warmLayerNs, -1)

  var
    embeds, layersTots, heads, argmaxes, walls: seq[int64]
    layerMatrix = newSeqWith(cfg.numHiddenLayers, newSeq[int64]())
  for step in 0 ..< timedSteps:
    var layerNs = newSeq[int64](cfg.numHiddenLayers)
    let s = runStagedStep(model, orc, ids, row, layerNs, step)
    embeds.add s.embedNs; layersTots.add s.layersNs; heads.add s.headNs
    argmaxes.add s.argmaxNs; walls.add s.wallNs
    for i in 0 ..< cfg.numHiddenLayers:
      layerMatrix[i].add layerNs[i]

  echo()
  echo &"per-stage medians over {timedSteps} steps:"
  let wall = median(walls)
  let e = median(embeds); let l = median(layersTots)
  let h = median(heads); let a = median(argmaxes)
  echo &"  whole-step wall            {ms(wall):>12.1f} ms"
  echo &"  (a) embedding/entry        {ms(e):>12.1f} ms  {100*e.float64/wall.float64:>5.1f}%"
  echo &"  (b) all layers             {ms(l):>12.1f} ms  {100*l.float64/wall.float64:>5.1f}%"
  echo &"  (c) final norm + lm_head   {ms(h):>12.1f} ms  {100*h.float64/wall.float64:>5.1f}%"
  echo &"  (d) argmax + .item()       {ms(a):>12.1f} ms  {100*a.float64/wall.float64:>5.1f}%"
  let stageSum = e + l + h + a
  echo &"  stage sum (a+b+c+d)        {ms(stageSum):>12.1f} ms  ({100*stageSum.float64/wall.float64:.1f}% of wall)"
  echo()
  echo "per-layer-index medians (ms):"
  for i in 0 ..< cfg.numHiddenLayers:
    let kind = if cfg.layerTypes[i] == alkGatedDeltaNet: "gdn" else: "attn"
    echo &"  layer {i:>2} ({kind}) {ms(median(layerMatrix[i])):>12.1f}"
  let allLayer = layerMatrix.mapIt(median(it))
  echo()
  echo &"median layer across stack: {ms(median(allLayer)):.1f} ms"
  var gdnList: seq[int64] = @[]
  var attnList: seq[int64] = @[]
  for i in 0 ..< cfg.numHiddenLayers:
    if cfg.layerTypes[i] == alkGatedDeltaNet: gdnList.add median(layerMatrix[i])
    else: attnList.add median(layerMatrix[i])
  echo &"gdn  layers ({gdnList.len}): median {ms(median(gdnList)):.1f} ms, sum/step {ms(total(gdnList)):.1f} ms"
  echo &"attn layers ({attnList.len}): median {ms(median(attnList)):.1f} ms, sum/step {ms(total(attnList)):.1f} ms"

proc benchConv1dGrouped(label: string, dtype: ScalarKind) =
  ## Depthwise grouped conv1d exactly as the GDN decode path spells it:
  ## input (1, conv_dim=8192, K=4), weight (8192, 1, 4), groups=8192.
  let cat = F.randn(1, 8192, 4, dtype).to(F.kCPU)
  let w = F.randn(8192, 1, 4, dtype).to(F.kCPU)
  var sink: Tensor
  discard F.silu(F.conv1d(cat, w, padding = [0], groups = 8192)).to(F.kCPU) # warmup
  let t0 = getMonotime()
  for i in 0 ..< 3:
    sink = F.silu(F.conv1d(cat, w, padding = [0], groups = 8192))
  discard sink.to(F.kCPU)
  let perCallNs = (getMonotime() - t0).inNanoseconds div 3
  echo &"  {label}: conv1d groups=8192 (1,8192,4) K=4: {ms(perCallNs):.1f} ms/call (3 calls, synced)"

proc benchRecurrenceStep(label: string, dtype: ScalarKind) =
  ## One step of the sequential delta-rule recurrence at 35B decode shapes,
  ## all-f32 core as the production code spells it (H=32, Dk=Dv=128).
  let q32 = F.randn(1, 32, 1, 128, F.kFloat32).to(F.kCPU)
  let k32 = F.randn(1, 32, 1, 128, F.kFloat32).to(F.kCPU)
  let v32 = F.randn(1, 32, 1, 128, F.kFloat32).to(F.kCPU)
  let g32 = F.randn(1, 32, 1, F.kFloat32).to(F.kCPU)
  let beta32 = F.randn(1, 32, 1, F.kFloat32).to(F.kCPU)
  let qScaled = q32 * Scalar(0.08838834764831845)
  var s = F.randn(1, 32, 128, 128, F.kFloat32).to(F.kCPU)
  let gT = g32.exp().unsqueeze(-1).unsqueeze(-1)
  let betaT = beta32.unsqueeze(-1)
  let kT = k32.unsqueeze(-1)
  let qT = qScaled.unsqueeze(-1)
  var sink: Tensor
  proc step(): Tensor =
    var st = s * gT
    let kvMem = (st * kT).sum(axis = -2)
    let delta = (v32 - kvMem) * betaT
    st = st + kT * delta.unsqueeze(-2)
    result = (st * qT).sum(axis = -2)
  sink = step().to(F.kCPU) # warmup
  let t0 = getMonotime()
  for i in 0 ..< 10:
    sink = step()
  discard sink.to(F.kCPU)
  let perCallNs = (getMonotime() - t0).inNanoseconds div 10
  echo &"  {label}: delta-rule recurrence 1 step f32 (1,32,128,128 state): {ms(perCallNs):.2f} ms/call (10 calls, synced)"

proc benchLmHead(label: string, weight: Tensor, weightDtypeBytes: int, x: Tensor) =
  var sink: Tensor
  sink = F.linear(x, weight).to(F.kCPU) # warmup
  let t0 = getMonotime()
  for i in 0 ..< 3:
    sink = F.linear(x, weight)
  discard sink.to(F.kCPU)
  let perCallNs = (getMonotime() - t0).inNanoseconds div 3
  let gbytes = (weight.numel() * weightDtypeBytes).float64 / 1e9
  echo &"  {label}: {ms(perCallNs):.1f} ms/call ({gbytes:.2f} GB weight read, 3 calls, synced)"

proc runDrill() =
  echo "kernel drill-downs, CPU, decode shapes (no model load, synthetic tensors)"
  echo "========================================================================="
  stamp()
  echo()
  echo &"threads (torch default): " & $atGetNumThreads()
  echo()

  echo "[1] GDN conv1d (grouped, 8192 groups) — the dominant GDN sub-op"
  benchConv1dGrouped("bf16", F.kBFloat16)
  benchConv1dGrouped("f32 ", F.kFloat32)
  echo()

  echo "[2] GDN delta-rule recurrence (f32 core, production spelling)"
  benchRecurrenceStep("recurrence", F.kFloat32)
  echo()

  echo "[3] conv1d f32 thread scaling (at::set_num_threads, restored after)"
  let defaultThreads = atGetNumThreads()
  for n in [1, 2, 4]:
    atSetNumThreads(n.cint)
    benchConv1dGrouped(&"f32 threads={n}", F.kFloat32)
  atSetNumThreads(defaultThreads)
  echo()

  echo "[4] lm_head GEMV bf16 vs f32 (vocab slice, weights converted once outside timing)"
  let wBf16 = F.randn(VocabSlice, 2048, F.kBFloat16).to(F.kCPU)
  let w32 = wBf16.to(F.kFloat32) # converted once, outside timed region
  let xBf16 = F.randn(1, 1, 2048, F.kBFloat16).to(F.kCPU)
  let x32 = xBf16.to(F.kFloat32)
  benchLmHead("lm_head slice bf16", wBf16, 2, xBf16)
  benchLmHead("lm_head slice f32  ", w32, 4, x32)
  echo()

  echo "[5] in_proj-style GEMV bf16 vs f32 (GDN in_proj_qkv shape 8192x2048)"
  let wiBf16 = F.randn(8192, 2048, F.kBFloat16).to(F.kCPU)
  let wi32 = wiBf16.to(F.kFloat32)
  benchLmHead("in_proj qkv bf16", wiBf16, 2, xBf16)
  benchLmHead("in_proj qkv f32 ", wi32, 4, x32)

proc main() =
  let mode =
    if paramCount() > 0: paramStr(1).toLowerAscii()
    else: "probe"
  case mode
  of "probe": runProbe()
  of "timed": runTimed()
  of "drill": runDrill()
  else:
    echo &"unknown mode '{mode}', expected probe|timed|drill"
    quit(1)

when isMainModule:
  main()
