# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Model entry points: registry-based loading and greedy generation.
##
## Import order matters:
## 1. all_interfaces defines the ModelRegistry table
## 2. Individual models (qwen3) populate it via static blocks at import
## 3. loadModel below dispatches through the populated registry

import
  std/json,
  std/os,
  std/tables,
  std/sequtils,
  workspace/ceramic/src/tensors,
  workspace/tensorcery/src/tensor_datatypes,
  ./models/all_interfaces,
  ./models/qwen3,
  ./stateful/orchestrator

export all_interfaces, qwen3

proc loadModel*(engine: AnyEngine, modelPath: string): Qwen3Model =
  ## Load a checkpoint through the registry.
  ##
  ## Args:
  ##
  ## - engine: allocates every weight tensor's device buffer
  ## - modelPath: checkpoint directory holding config.json + model.safetensors
  ##
  ## The config.json `architectures[0]` name selects the loader.
  ##
  ## The {.compileTime.} registry var becomes a const at runtime through
  ## static(), the compile-time to runtime boundary pass.
  const registry = static(ModelRegistry)

  let cfg = (modelPath / "config.json").parseFile()
  let archs = cfg["architectures"]

  doAssert archs.len != 0, "[ttt] No architectures found in config.json"
  doAssert archs.len <= 1, "[ttt] Multiple architectures not supported"

  let arch = archs[0].getStr()
  # tables-qualified: std/json exports same-name hasKey/[] overloads
  doAssert tables.hasKey(registry, arch), "[ttt] Unknown architecture: " & arch

  cast[Qwen3Model](tables.getOrDefault(registry, arch)(engine, modelPath))

proc greedyArgmax(logits: Tensor): int =
  ## Greedy sample: host readback of the logits, first-max argmax.
  ##
  ## Input: logits of shape (batch, seq, vocab), the last sequence
  ## position decides. Ties break to the lowest token id (first max wins).
  let shape = logits.node.getImpl().shape()
  let seqLen = shape[1]
  let vocab = shape[2]
  var host = newSeq[bfloat16](seqLen * vocab)
  copyToHostFromDevice(host, logits)
  let off = (seqLen - 1) * vocab
  var best = 0
  var bestVal = host[off].to(float32)
  for j in 1 ..< vocab:
    let v = host[off + j].to(float32)
    if v > bestVal:
      best = j
      bestVal = v
  best

proc generate*(engine: AnyEngine, model: Qwen3Model, promptIds: seq[int], maxTokens: int, maxContextLen: int = -1): seq[int] =
  ## Greedy generation over the paged KV cache.
  ##
  ## 1. startSequence on a fresh orchestrator (LPM + COW + page borrowing)
  ## 2. one prefill forward over the whole prompt
  ## 3. appendToken + one-token forwards, kv_position advanced after each
  ## 4. endSequence grafts the sequence's pages into the trie
  ##
  ## Args:
  ## - promptIds: tokenized prompt
  ## - maxTokens: generation step budget
  ## - maxContextLen: pool and RoPE sizing override, the config's
  ##   max_position_embeddings governs when negative. The full value
  ##   eager-allocates a multi-gigabyte pool, a smoke run passes a small one.
  ##
  ## Returns:
  ##   The full token sequence: the prompt plus the generated tokens.
  ##   Generation stops at the config's eos_token_id.
  let cfg = model.config
  let maxCtx = if maxContextLen < 0: cfg.max_position_embeddings else: maxContextLen
  let numPoolPages = computeNumPages(maxCtx, concurrentRequests = 1)
  var orc = Orchestrator.init(engine, cfg.num_hidden_layers, batch_size = 1,
    cfg.num_key_value_heads, maxCtx, cfg.head_dim, numPoolPages)
  defer: orc.endSequence()

  doAssert promptIds.len <= maxCtx,
    "[ttt] Prompt length exceeds max context length: " & $promptIds.len & " > " & $maxCtx

  let startPos = promptIds.len

  orc.startSequence(promptIds.mapIt(uint32(it))) # mapIt: std/sequtils

  # === PREFILL: one forward on the full prompt ===
  var ids = promptIds
  let logits = model.forward(orc.getInferenceContextMut(), ids)
  # kv_position reflects the prefill token total for decode page allocation
  orc.setKvPosition(ids.len)
  var nextToken = greedyArgmax(logits)
  ids.add(nextToken)
  if nextToken == cfg.eos_token_id:
    return ids

  # === DECODE LOOP: one token per forward ===
  while ids.len < startPos + maxTokens and ids.len < maxCtx:
    orc.appendToken(ids.len - 1, uint32(nextToken))
    let stepLogits = model.forward(orc.getInferenceContextMut(), @[nextToken])
    # Advance kv_position AFTER the forward: the attention layer read it
    # as the write offset (equal to position_ids[0]) during this call.
    orc.setKvPosition(ids.len)
    nextToken = greedyArgmax(stepLogits)
    ids.add(nextToken)
    if nextToken == cfg.eos_token_id:
      break

  result = ids
