# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Qwen3 model smoke (Metal): the real checkpoint end to end.
##
## Test that the ported Qwen3 stack runs the real Qwen3-0.6B checkpoint through the Tensor face:
##
## - one prefill forward: bf16 logits (1, seq, 151936) all finite, top-5
##   next tokens match the recorded reference, ids exact and values
##   within the bf16 rounding band
## - 3 greedy steps through generate() with pool grafting
## - a second generate() round over the same prompt: the LPM/COW path
##   reproduces round 1 token for token
## - generate()'s limit contracts raise before the first token
##
## Checkpoint: TTT_QWEN3_PATH env override, default /Users/pi/Documents/Programming/MODELS/Qwen3-0.6B.
##
## Run from the worktree root:
##   nim c -r -d:release --hints:off --warnings:off --nimcache:nimcache/ttf --outdir:build/ttf --path:"$PWD" \
##     workspace/tattleformers/tests/stateful/test_model_qwen3.nim

import
  std/os,
  std/math,
  std/algorithm,
  std/sequtils,
  workspace/tensorcery/src/tensorcery,
  workspace/toktoktok/src/bpe_codec,
  workspace/tattleformers/src/stateful/orchestrator,
  workspace/tattleformers/src/stateful/page_pool,
  workspace/tattleformers/src/models

const
  ModelPathDefault = "/Users/pi/Documents/Programming/MODELS/Qwen3-0.6B"
  MaxCtx = 1024 # pool + RoPE sizing override for the smoke run
  # Reference run's recorded top-5 next tokens over the last prefill position, with their bf16 logits.
  RecordedTop5 = [(12095, 17.5'f32), (7407, 14.25'f32), (279, 14.125'f32),
                  (1112, 13.75'f32), (30743, 13.625'f32)]
  RecordedContinuation = "The capital of France is Paris. The"
  LogitBand = 1.0'f32
    ## Absolute band over the recorded top-5 logits. Token ids are
    ## the exact contract. The band absorbs the base's bf16 kernel
    ## rounding deltas (tile GEMM between-tile accumulation, rsqrt form).

proc top5(logits: Tensor, seqLen, vocab: int): seq[(int, float32)] =
  ## Top-5 next tokens over the last sequence position, first-max tie-break.
  let host = toSeq(logits, bfloat16)
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

proc runMain(modelPath: string) =
  let tokenizer = loadHFTokenizer(modelPath / "tokenizer.json")
  let model = loadModel(modelPath)
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
    var orc = Orchestrator.init(cfg.num_hidden_layers,
      cfg.num_key_value_heads, MaxCtx, cfg.head_dim, numPoolPages)
    orc.startSequence(promptIds.mapIt(uint32(it)))
    doAssert orc.getInferenceContextMut().cached_tokens == 0,
      "fresh trie, no LPM match"
    let logits = model.forward(orc.getInferenceContextMut(), promptIds)
    let ls = logits.shape()
    doAssert ls.len == 3 and ls[0] == 1 and ls[1] == seqLen and ls[2] == vocab,
      "logits shape (1, seq, vocab)"
    # all logits finite in f32 terms: no NaN, no inf after bf16 rounding
    let host = toSeq(logits, bfloat16)
    for v in host:
      let f = v.to(float32)
      doAssert classify(f) != fcNaN, "NaN logit"
      doAssert classify(f) != fcInf, "inf logit"
    let top = top5(logits, seqLen, vocab)
    stderr.writeLine "prompt: " & prompt
    for k in 0 ..< top.len:
      let piece = tokenizer.decodeToString(@[top[k][0]])
      stderr.writeLine "top-" & $(k + 1) & ": " & $top[k][0] & " (" &
        piece & ") = " & $top[k][1]
      doAssert top[k][0] == RecordedTop5[k][0],
        "top-" & $(k + 1) & " token id " & $top[k][0] &
        ", recorded " & $RecordedTop5[k][0]
      doAssert abs(top[k][1] - RecordedTop5[k][1]) <= LogitBand,
        "top-" & $(k + 1) & " logit " & $top[k][1] &
        ", recorded " & $RecordedTop5[k][1] & ", band " & $LogitBand
    orc.endSequence()

  # ── 3 greedy steps through generate() ──────────────────────────────
  let round1 = generate(model, promptIds, maxTokens = 3, maxContextLen = MaxCtx)
  doAssert round1.len == seqLen + 3,
    "greedy steps ran, got " & $round1.len & " tokens"
  doAssert round1[0 ..< seqLen] == promptIds, "prompt preserved"
  stderr.writeLine "round 1: " & tokenizer.decodeToString(round1)
  doAssert tokenizer.decodeToString(round1) == RecordedContinuation,
    "greedy continuation matches the recorded reference"

  # ── round 2 over the same prompt: LPM + COW reproduce round 1 ──────
  let round2 = generate(model, promptIds, maxTokens = 3, maxContextLen = MaxCtx)
  doAssert round2 == round1,
    "the LPM/COW restart reproduces round 1 token for token"
  stderr.writeLine "round 2 (LPM/COW restart): " & tokenizer.decodeToString(round2)

  # ── generate()'s limit contracts raise before the first token ──────
  doAssertRaises AssertionDefect:
    discard generate(model, promptIds, maxTokens = 0, maxContextLen = MaxCtx)
  doAssertRaises AssertionDefect:
    discard generate(model, newSeqWith(MaxCtx + 1, 1), maxTokens = 1,
      maxContextLen = MaxCtx)
  doAssertRaises AssertionDefect:
    discard generate(model, promptIds, maxTokens = MaxCtx,
      maxContextLen = MaxCtx)

  stderr.writeLine "qwen3 model smoke (Metal): all checks passed"

when isMainModule:
  runMain(getEnv("TTT_QWEN3_PATH", ModelPathDefault))
