# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Qwen3 model smoke (Metal): real checkpoint end to end.
##
## Test that the ported Qwen3 stack runs the real Qwen3-0.6B checkpoint:
##
## - one prefill forward: bf16 logits (1, seq, 151936) all finite, top-5 sane
## - 3 greedy decode steps through generate() with pool grafting
## - a second generate() round over the same prompt: the LPM/COW path
##   reproduces round 1 token for token
##
## Checkpoint: TTT_QWEN3_PATH env override, default /Users/pi/Documents/Programming/MODELS/Qwen3-0.6B.
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/wipM08model workspace/tattleformers/tests/t_model_qwen3_metal.nim

import
  std/os,
  std/options,
  std/math,
  std/algorithm,
  std/sequtils,
  workspace/crucible,
  workspace/ceramic/src/tensors,
  workspace/ceramic/src/layout_algebra,
  workspace/tensorcery/src/device_view,
  workspace/tensorcery/src/tensor_datatypes,
  workspace/toktoktok/src/bpe_codec,
  workspace/tattleformers/src/stateful/orchestrator,
  workspace/tattleformers/src/layers/embedding,
  workspace/tattleformers/src/layers/linear,
  workspace/tattleformers/src/layers/norm,
  workspace/tattleformers/src/layers/rope,
  workspace/tattleformers/src/layers/ffn,
  workspace/tattleformers/src/layers/lmhead,
  workspace/tattleformers/src/layers/attention,
  workspace/tattleformers/src/layers/decoder_layers,
  workspace/tattleformers/src/ops_1i1o/rmsnorm,
  workspace/tattleformers/src/ops_2i1o/add,
  workspace/tattleformers/src/ops_2i1o/silu_mul,
  workspace/tattleformers/src/ops_Ni1o/attention,
  workspace/tattleformers/src/models

{.experimental: "callOperator".}

proc k_primer*(z, x: distinct DeviceView, shift: static int) =
  ## z = x + shift, one element. Compile-time primer for the erased face.
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let s = xv[0].to(float32) + float32(shift)
  zv[0] = s.to(bfloat16)

const ModelPathDefault = "/Users/pi/Documents/Programming/MODELS/Qwen3-0.6B"
const MaxCtx = 1024 # pool + RoPE sizing override for the smoke run

proc top5(host: seq[bfloat16], seqLen, vocab: int): seq[(int, float32)] =
  ## Top-5 next tokens over the last sequence position, first-max tie-break.
  let off = (seqLen - 1) * vocab
  var idx = newSeq[int](vocab)
  for j in 0 ..< vocab:
    idx[j] = j
  proc val(j: int): float32 = host[off + j].to(float32)
  idx.sort(proc (a, b: int): int =
    let d = val(b) - val(a)
    if d > 0: 1
    elif d < 0: -1
    else: a - b)
  result = @[]
  for k in 0 ..< 5:
    result.add((idx[k], val(idx[k])))

proc runMain[E: HwEngine](engine: E, modelPath: string) =
  block: # primer enqueue: textually precede the first erased-Tensor call
    let
      zDev = engine.allocDevice(bfloat16, 1)
      xDev = engine.allocDevice(bfloat16, 1)
      zb = TensorBridge[typeof(zDev), (int,), (int,)](
        data: zDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
      xb = TensorBridge[typeof(xDev), (int,), (int,)](
        data: xDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
    var batch = engine.recordBatch()
    batch.enqueue<<(1, 1)>> k_primer(zb, xb, 3)
    batch.submit()

  let tokenizer = loadHFTokenizer(modelPath / "tokenizer.json")
  let model = loadModel(engine, modelPath)
  let cfg = model.config
  doAssert cfg.vocab_size == 151936, "vocab size " & $cfg.vocab_size
  doAssert cfg.num_hidden_layers == 28, "layer count"
  doAssert cfg.tie_word_embeddings, "Qwen3-0.6B ships a tied LM head"

  let prompt = "The capital of France is"
  let promptIds = tokenizer.encode(prompt)
  let seqLen = promptIds.len
  let vocab = cfg.vocab_size

  # ── one prefill forward on the real checkpoint ─────────────────────
  block:
    let numPoolPages = computeNumPages(MaxCtx, concurrentRequests = 1)
    var orc = Orchestrator.init(engine, cfg.num_hidden_layers, batch_size = 1,
      cfg.num_key_value_heads, MaxCtx, cfg.head_dim, numPoolPages)
    orc.startSequence(promptIds.mapIt(uint32(it)))
    doAssert orc.getInferenceContextMut().cached_tokens == 0,
      "fresh trie, no LPM match"
    let logits = model.forward(orc.getInferenceContextMut(), promptIds)
    var host = newSeq[bfloat16](seqLen * vocab)
    copyToHostFromDevice(host, logits)
    # all logits finite in f32 terms: no NaN, no inf after bf16 rounding
    for i in 0 ..< host.len:
      let v = host[i].to(float32)
      doAssert classify(v) != fcNaN, "NaN logit at " & $i
      doAssert classify(v) != fcInf, "inf logit at " & $i
    let top = top5(host, seqLen, vocab)
    stderr.writeLine "prompt: " & prompt
    for k in 0 ..< top.len:
      let piece = tokenizer.decodeToString(@[top[k][0]])
      stderr.writeLine "top-" & $(k + 1) & ": " & $top[k][0] & " (" &
        piece & ") = " & $top[k][1]
    orc.endSequence()

  # ── 3 greedy steps through generate() ──────────────────────────────
  let round1 = generate(engine, model, promptIds, maxTokens = 3,
    maxContextLen = MaxCtx)
  doAssert round1.len == seqLen + 3,
    "greedy steps ran, got " & $round1.len & " tokens"
  doAssert round1[0 ..< seqLen] == promptIds, "prompt preserved"
  stderr.writeLine "round 1: " & tokenizer.decodeToString(round1)

  # ── round 2 over the same prompt: LPM + COW reproduce round 1 ──────
  let round2 = generate(engine, model, promptIds, maxTokens = 3,
    maxContextLen = MaxCtx)
  doAssert round2 == round1,
    "the LPM/COW restart reproduces round 1 token for token"
  stderr.writeLine "round 2 (LPM/COW restart): " & tokenizer.decodeToString(round2)

  stderr.writeLine "qwen3 model smoke (Metal): all checks passed"

when isMainModule:
  let modelPath = getEnv("TTT_QWEN3_PATH", ModelPathDefault)
  runMain(gMetalEngine(), modelPath)
