# Tattletale
# Copyright (c) 2026 Mamy Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Stateful-machinery helpers for the suites, each wiring state to its pool.
##
## - the inference context comes with its page pool ref, the borrowed pages
##   stay alive while the returned pool ref is held
## - the replay orchestrator pre-allocates a bounded decode-context KV pool

import
  std/math,
  workspace/libtorch as F,
  workspace/transformers/src/models,
  workspace/transformers/src/stateful/orchestrator,
  workspace/transformers/src/stateful/inference_context,
  workspace/transformers/src/stateful/kvcache,
  workspace/transformers/src/stateful/page_pool

proc newKVContext*(
    numLayers, kvHeads, headDim: int,
    maxSeq = 512, device = F.kCPU
  ): (InferenceContext, PagePool) =
  ## Returns a fresh InferenceContext with a page pool of `maxSeq` tokens,
  ## one borrowed page per required page, the pages alive while
  ## the returned pool ref stays alive.
  var ctx = InferenceContext.init(
    num_layers = numLayers, batch_size = 1,
    kv_heads = kvHeads, max_seq = maxSeq, head_dim = headDim)
  let pool = PagePool.init(
    64, num_layers = numLayers, kv_heads = kvHeads, head_dim = headDim,
    dtype = F.kBFloat16, device = device)
  let numPages = ceilDiv(maxSeq, TokensPerPage)
  for i in 0 ..< numPages:
    ctx.pages.add(pool.borrow())
  (ctx, pool)

proc newOrchestrator*(model: AnyModel, dtype = F.kBFloat16, maxContextLen = 256): Orchestrator =
  ## Returns an Orchestrator whose KV pool is sized for one replay, geometry
  ## read off the model config, the pool placed on the model device.
  ##
  ## Expected input:
  ## - dtype, the orchestrator KV pool dtype of the recorded family
  ## - maxContextLen, the replay decode-context ceiling, a KV pool sized
  ##   from the model max_position_embeddings would pre-allocate gigabytes
  ##   for a tens-of-tokens chain
  ##
  ## Postcondition:
  ## - the returned orchestrator owns its page pool, the caller runs
  ##   endSequence once the sequence is done
  ##
  ## The pool slots carry the widest per-head KV width on the checkpoint
  ## (kvHeadDimMax, zero when head_dim governs). A dual-width checkpoint's
  ## full layers write full-width rows into the shared pool.
  let cfg = model.getConfig()
  let device = model.getDeviceKind()
  let numPoolPages = computeNumPages(maxContextLen, concurrentRequests = 1)
  let poolHeadDim =
    if cfg.kvHeadDimMax > 0: cfg.kvHeadDimMax else: cfg.head_dim
  Orchestrator.init(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, maxContextLen, poolHeadDim, numPoolPages, dtype, device)

proc newMlaOrchestrator*(maxSeq, kvLoraRank, qkRopeHeadDim: int, numLayers = 1, device = F.kCPU): Orchestrator =
  ## Batch-1 pool on the MLA split geometry:
  ## - the K buffer kv_lora_rank channels wide, one head
  ## - the V buffer the qk_rope_head_dim plane, one head
  ## - `numLayers` sizes the pool past the mixer's real wiring layer
  ##   index where the template keeps a nonzero index
  Orchestrator.init(
    num_layers = numLayers, batch_size = 1, k_kv_heads = 1,
    k_head_dim = kvLoraRank,
    v_kv_heads = 1, v_head_dim = qkRopeHeadDim, max_seq = maxSeq,
    num_pages = 4, dtype = kBFloat16, device = device)

proc newMlaCacheCtx*(maxSeq, kvLoraRank, qkRopeHeadDim: int,
    tokens: seq[uint32], numLayers = 1,
    device = F.kCPU): tuple[orc: Orchestrator, ctx: InferenceContext] =
  ## Returns a fresh orchestrator plus its active context:
  ## - the sequence opens on `tokens`, the fixture pass the caller replays
  ## - the context borrows the pool pages, alive while the returned
  ##   pool ref stays alive
  var orc = newMlaOrchestrator(maxSeq, kvLoraRank, qkRopeHeadDim,
    numLayers, device)
  orc.startSequence(tokens)
  result = (orc, orc.getInferenceContextMut())
