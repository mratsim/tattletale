# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Full single decode step through the wired Qwen3.6-35B-A3B Nim stack on MPS.
##
## Real checkpoint weights, tests/hf_models/Qwen3.6-35B-A3B, 67 GB bf16
## under the 693-tensor loader contract. Production forward path: paged
## KV cache, GDN state, routed MoE, lm_head.
##
## Pass A: realistic step, argmax sampling with its .item() sync.
## Pass B: forward only, fixed token, no argmax, fresh orchestrator.
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
  std/stats,
  workspace/libtorch as F,
  workspace/transformers/src/models,
  workspace/transformers/src/models/qwen35_moe {.all.},
  workspace/transformers/src/stateful/orchestrator

{.experimental: "callOperator".}

const
  ModelDir = currentSourcePath().parentDir() / ".." / ".." / ".." /
    "transformers" / "tests" / "hf_models" / "Qwen3.6-35B-A3B"
  MaxContextLen = 256
  PromptLen = 8
  WarmupSteps = 3
  TimedSteps = 16
  Runs = 5

proc median(values: seq[int64]): int64 =
  var sorted = values
  sorted.sort()
  result = sorted[sorted.len div 2]

proc reportSteps(name: string, stepNs: seq[seq[int64]]) =
  ## Per-run median and the overall median across runs, plus tok/s.
  var runMedians: seq[float64]
  var allNs: seq[int64]
  for run, ns in stepNs:
    let med = median(ns).float64 / 1000.0
    runMedians.add med
    echo &"  {name} run {run}: median {med:>9.1f} us/step, " &
      &"mean {ns.mapIt(it.float64).mean/1000.0:>9.1f} us, min {min(ns).float64/1000.0:>9.1f} us, max {max(ns).float64/1000.0:>9.1f} us"
  for ns in stepNs:
    for v in ns:
      allNs.add v
  let overall = median(allNs).float64 / 1000.0
  let tok = 1e6 / overall
  echo &"  {name} OVERALL median {overall:.1f} us/step = {tok:.2f} tok/s"

proc runSequence(model: AnyModel, argmaxSample: bool): seq[seq[int64]] =
  ## Prefill a short prompt, then time decode steps. Returns per-run
  ## step times in nanoseconds.
  let cfg = model.getConfig()
  let device = model.getDeviceKind()
  let numPoolPages = computeNumPages(MaxContextLen, concurrentRequests = 1)
  var orc = Orchestrator.init(
    cfg.num_hidden_layers, 1, cfg.num_key_value_heads, MaxContextLen,
    cfg.head_dim, numPoolPages, F.kBFloat16, device)

  var ids: seq[int] = @[]
  for i in 0 ..< PromptLen:
    ids.add (7 * i + 9707) mod cfg.vocab_size

  orc.startSequence(ids.mapIt(it.uint32))
  let inputIds = F.toTensor([ids]).to(device)
  let logits = model.forward(orc.getInferenceContextMut(), inputIds)
  orc.setKvPosition(ids.len)
  var row = logits.narrow(1, ids.len - 1, 1).squeeze(1).squeeze(0)

  var stepNs: seq[seq[int64]]
  for run in 0 ..< Runs:
    var runTimes: seq[int64]
    for step in 0 ..< (WarmupSteps + TimedSteps):
      var chosen: int
      if argmaxSample:
        chosen = row.argmax().item(int)
      else:
        chosen = 1000 + step mod 97
      ids.add chosen
      orc.appendToken(ids.len - 1, chosen.uint32, device)
      let t0 = getMonotime()
      let stepLogits = model.forward(orc.getInferenceContextMut(),
        F.toTensor([[chosen]]).to(device))
      let t1 = getMonotime()
      orc.setKvPosition(ids.len)
      row = stepLogits.squeeze(0).squeeze(0)
      if step >= WarmupSteps:
        runTimes.add (t1 - t0).inNanoseconds
    stepNs.add runTimes
  result = stepNs

proc main() =
  # Chains replay on the Metal Performance Shaders device, with a PyTorch
  # fallback for the kernels Metal does not implement.
  putEnv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

  echo "Full decode step benchmark, Qwen3.6-35B-A3B, real checkpoint"
  echo "============================================================"
  echo &"date: ", now().format("yyyy-MM-dd HH:mm:ss")
  echo &"device: mps"
  echo &"dtype: bfloat16, SSM state f32"
  echo &"weights: real checkpoint {ModelDir}"
  echo &"decode activations: x (1, 1, 2048), batch 1, context grows {PromptLen} to {PromptLen + Runs * (WarmupSteps + TimedSteps)}"
  echo &"runs: {Runs} x {TimedSteps} timed steps (median reported), {WarmupSteps} warmup steps per run"
  echo()

  echo "Loading model (67 GB)..."
  let loadStart = getMonotime()
  let model = loadModel($(ModelDir), kMPS)
  let cfg = model.getConfig()
  echo &"Model loaded in {(getMonotime() - loadStart).inSeconds} s"
  echo &"config: layers {cfg.num_hidden_layers}, hidden {cfg.hidden_size}, vocab {cfg.vocab_size}, kv_heads {cfg.num_key_value_heads}, head_dim {cfg.head_dim}"
  echo()

  echo "Pass A: realistic step (argmax + .item() sync per step)"
  let stepNsA = runSequence(model, argmaxSample = true)
  reportSteps("passA", stepNsA)
  echo()

  echo "Pass B: forward only (fixed token, no argmax)"
  let stepNsB = runSequence(model, argmaxSample = false)
  reportSteps("passB", stepNsB)

when isMainModule:
  main()
